#!/usr/bin/env python3
"""Microbenchmarks for FF + attention optimization investigation.

Runs targeted MLX benchmarks at the EXACT shapes used by the LTX-2.3
distilled video transformer block (T=8784 latent tokens, D=4096 hidden,
H=32 heads of D_h=128, FF inner=16384) to give green/red signals on
candidate optimizations BEFORE committing to engineering work.

Each bench:
- Pre-warms with a few discarded iterations (compile + first-call costs).
- Runs N timed iterations with mx.eval barriers for honest per-call wall.
- Reports n, total, mean, p50, p90, p99, max in ms.

Reasoning is in `docs/PERFORMANCE_NOTES.md`; this script provides the
concrete numbers those entries cite.

Usage (from LTX-2-MLX repo root):
    python scripts/bench_ff_microbench.py                     # run all
    python scripts/bench_ff_microbench.py gelu                # run one
    python scripts/bench_ff_microbench.py qkv,ff_chain        # run several
    ITERS=50 python scripts/bench_ff_microbench.py            # more iters

Available benches:
    gelu                - standalone nn.gelu_approx at FF intermediate shape
    qkv_separate_vs_packed - 3 separate Q/K/V matmuls vs 1 packed matmul
    ff_chain            - matmul -> gelu -> matmul (naive vs lazy vs mx.compile)
    sdpa_floor          - mx.fast.scaled_dot_product_attention at video_self_attn shape
    sdpa_t_sweep        - SDPA at neighbor T values around 8784 to find
                          tile-alignment cliffs (is padding to a friendlier T
                          a free lever?)
    sdpa_d_sweep        - SDPA at varying head_dim across MLX's bk=32/bk=16
                          kernel-selection boundary at D=128.  Tests the FA-2
                          tile-size hypothesis (does MLX's bk=32 path at
                          D=120 achieve more TFlops/s than its bk=16 path
                          at D=128?) WITHOUT writing a custom kernel.
    quant_matmul        - BF16 vs mxfp8 matmul at LTX FF/attention shapes;
                          measures per-shape regression and effective TFlops/s
                          (explains why end-to-end mxfp8 loses post-AdaLN-fix)
    bf16_layout         - BF16 naive (x @ W.T) vs pretranspose (mx.addmm against
                          mx.contiguous(W.T)) at LTX video FF/attention shapes;
                          isolates where the --video-ff-layout /
                          --video-attn-layout pretranspose wins come from.
    bf16_layout_audio   - Same as bf16_layout but at LTX AUDIO shapes
                          (T=502, K/N=2048, FF_inner=8192).  Tests whether the
                          "only project_out wins" pattern from video also holds
                          at the smaller audio dimensions, or whether audio
                          should keep its existing pretranspose defaults.
    fused_ffn_feasibility - Bound the per-call wall AND peak-memory savings a
                          hypothetical fused (matmul->GELU->matmul) Metal kernel -
                          or a streamed inner-dim variant on top of MLX -
                          could ever deliver vs the stock path.  Reports stock
                          wall, two GEMM-floor scenarios (additive sum and
                          chained-no-GELU), GELU-pass cost in isolation, peak
                          transient memory above persistent weights, and a
                          recoverable-wall vs steel_gemm-ceiling verdict.
    pointwise_bw        - Achievable BF16 pointwise bandwidth ceiling on this
                          hardware.  Sweeps `x*1.0`, `x+x`, `x+y` at tensor
                          sizes from 1 MB to 512 MB plus LTX-specific shapes
                          (residual 72 MB, FF hidden 288 MB).  Calibration
                          probe - use the peak GB/s to grade every other
                          bandwidth-bound op below.
    adaln_residual      - AdaLN modulation + gated-residual chain at residual
                          stream shape (1, T, D).  Compares inline vs
                          @mx.compile (production default) for AdaLN alone,
                          residual_gate alone, and the full attention-side
                          chain.  Quantifies compile-fusion headroom and
                          per-step extrapolation (4 chains/block x 96 calls
                          per stage-1 run).
    rope                - `apply_split_rotary_emb` at LTX video Q/K shape
                          (B=1, H=32, T=8784, D=128).  Compares production
                          (FP32 cos/sin + cast back to BF16) vs all-BF16
                          freqs, plus bare-multiply and bare-concat floors.
                          Extrapolates per-step savings if BF16 freqs are
                          meaningfully cheaper.
    vae_ops             - Per-op breakdown of the native-conv3d VAE decoder
                          hot path (nn.Conv3d, mx.fast.rms_norm pixel-norm,
                          nn.silu, full resnet block) at three BFHWC shapes
                          spanning the decoder progression (bottleneck ->
                          mid -> late).  Bounds the payoff of the pending
                          upstream mx.conv_general int-overflow fix.

Each bench prints a result block. Compare numbers across modes within a
single bench, not across benches (different shapes / FLOPs).

Notes:
- Pre-transposes weights via mx.contiguous(weight.T) to match production
  (matches the `--video-ff-layout` and `--video-attn-layout` default stack).
- Uses BF16 throughout to match production compute dtype.
- mx.eval barriers + time.monotonic_ns timing - same methodology as
  scripts/sdpa_dtype_probe.py.
- Compile flag (LTX_DISABLE_COMPILED_*) is NOT toggled here; these are
  microbenches of bare ops, not the full transformer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable

import mlx.core as mx
import mlx.nn as nn


# --- shapes (from LTX-2.3 distilled at 1024x576x481, stage 1 latent) ---
B = 1
T = 8784            # latent tokens (61 frames x 9 x 16 spatial)
D = 4096            # hidden dim
H = 32              # attention heads
D_HEAD = 128        # head dim (H x D_HEAD = D)
FF_INNER = 16384    # FF intermediate (4x D)
DTYPE = mx.bfloat16

# --- timing helpers ---


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(round((p / 100.0) * (len(s) - 1)))
    return s[i]


def _report(label: str, durs_ms: list[float], notes: str = "") -> None:
    n = len(durs_ms)
    total = sum(durs_ms)
    mean = total / n if n else 0.0
    print(
        f"  {label:<46s} n={n:>3d}  total={total:>9.2f}ms  "
        f"mean={mean:>7.3f}  p50={_pct(durs_ms, 50):>7.3f}  "
        f"p90={_pct(durs_ms, 90):>7.3f}  p99={_pct(durs_ms, 99):>7.3f}  "
        f"max={max(durs_ms):>7.3f}"
    )
    if notes:
        print(f"  {'':>46s} {notes}")


def _time_call(fn: Callable[[], mx.array], warmup: int, iters: int) -> list[float]:
    """Time fn() for `iters` iterations after `warmup` discarded ones.

    fn() must return an mx.array.  We call mx.eval(out) inside the timed
    region so per-call wall reflects GPU work, not just dispatch.
    """
    # Warmup
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    # Timed
    durs_ms: list[float] = []
    for _ in range(iters):
        t0 = time.monotonic_ns()
        out = fn()
        mx.eval(out)
        durs_ms.append((time.monotonic_ns() - t0) / 1e6)

    return durs_ms


def _get_active_mem() -> int:
    return mx.get_active_memory() if hasattr(mx, "get_active_memory") else mx.metal.get_active_memory()


def _get_peak_mem() -> int:
    return mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else mx.metal.get_peak_memory()


def _reset_peak_mem() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    else:
        mx.metal.reset_peak_memory()


def _clear_cache() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    else:
        mx.metal.clear_cache()


def _time_and_peak(fn: Callable[[], mx.array], warmup: int, iters: int) -> tuple[list[float], int]:
    """Time fn() and measure the peak transient GPU memory the call triggers.

    Pattern: warmup, clear cache, snapshot active baseline, reset peak,
    run one probe call, measure peak - baseline.  Then run timed iters.

    The returned peak is "additional bytes above the persistent baseline at
    entry" - i.e. transient working-set for one call, not absolute allocation.
    """
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
        del out

    _clear_cache()
    baseline = _get_active_mem()
    _reset_peak_mem()

    out = fn()
    mx.eval(out)
    peak_delta = max(_get_peak_mem() - baseline, 0)
    del out

    durs_ms: list[float] = []
    for _ in range(iters):
        t0 = time.monotonic_ns()
        out = fn()
        mx.eval(out)
        durs_ms.append((time.monotonic_ns() - t0) / 1e6)
        del out

    return durs_ms, peak_delta


# --- weight helpers ---


def _bf16_linear_pretransposed(in_dim: int, out_dim: int, bias: bool = True):
    """Build a (W_T, b) pair matching our pretranspose layout cache."""
    # Initialize like nn.Linear does (small random) but in BF16
    weight = mx.random.normal(shape=(out_dim, in_dim)).astype(DTYPE) * (1.0 / in_dim ** 0.5)
    weight_t = mx.contiguous(weight.T)  # (in_dim, out_dim) contiguous
    bias_v = mx.zeros((out_dim,), dtype=DTYPE) if bias else None
    return weight_t, bias_v


def _linear(weight_t: mx.array, bias_v: mx.array | None, x: mx.array) -> mx.array:
    """Production-style linear: mx.addmm(bias, x, weight.T)."""
    if bias_v is not None:
        return mx.addmm(bias_v, x, weight_t)
    return x @ weight_t


# --- benches ---


def bench_gelu(warmup: int, iters: int) -> None:
    """Standalone gelu_approx at the FF intermediate shape.

    Question: is the GELU pass bandwidth-bound at ~125 ms/step ceiling
    per the PERFORMANCE_NOTES entry, or much cheaper?

    Shape: (1, 8784, 16384) BF16 = 288 MB.  Read+write = 576 MB.  At
    ~300 GB/s effective M1 Max HBM bandwidth: ~1.9 ms theoretical floor
    per call.  Per stage-1 run (96 calls = 48 blocks x 2 steps for our
    sync-mode capture): ~180 ms theoretical floor.
    """
    print()
    print("=" * 100)
    print(f"BENCH: gelu_approx at (1, {T}, {FF_INNER}) BF16  ({T*FF_INNER*2/1e6:.1f} MB tensor)")
    print("=" * 100)

    x = mx.random.normal(shape=(B, T, FF_INNER)).astype(DTYPE)
    mx.eval(x)

    durs = _time_call(lambda: nn.gelu_approx(x), warmup=warmup, iters=iters)
    _report("nn.gelu_approx", durs)

    # Sanity: pure read (no compute) to estimate bandwidth floor.
    durs_copy = _time_call(lambda: x * 1.0, warmup=warmup, iters=iters)
    _report("x * 1.0 (bandwidth-only sanity)", durs_copy,
            notes="lower bound: pure read+write through pointwise kernel")

    # Scale-to-run estimate
    per_run_total = (sum(durs) / iters) * 96  # 48 blocks x 2 steps
    per_step_total = per_run_total / 2
    print()
    print(f"  Extrapolation: {sum(durs)/iters:.2f} ms x 96 calls "
          f"(48 blocks x 2 steps) = {per_run_total:.0f} ms / 2-step run = "
          f"{per_step_total:.0f} ms / step")
    print(f"  => GELU-pass elimination upper bound on per-step savings: "
          f"~{per_step_total:.0f} ms ({100*per_step_total/45500:.2f} % of 45.5 s/step)")


def bench_qkv_separate_vs_packed(warmup: int, iters: int) -> None:
    """Q/K/V projections: 3 separate matmuls vs 1 packed matmul.

    Question: does packing (1, T, D) @ (D, 3D) + split beat three
    (1, T, D) @ (D, D) calls?  Bandwidth argument: read x once instead
    of 3x.  Dispatch argument: one larger matmul kernel vs three smaller.
    """
    print()
    print("=" * 100)
    print(f"BENCH: Q/K/V projections at (1, {T}, {D}) -> (1, {T}, {D})  "
          f"H={H} D_head={D_HEAD}  BF16")
    print("=" * 100)

    x = mx.random.normal(shape=(B, T, D)).astype(DTYPE)
    mx.eval(x)

    # Separate: 3 individual pretransposed matmuls
    w_q_t, b_q = _bf16_linear_pretransposed(D, D)
    w_k_t, b_k = _bf16_linear_pretransposed(D, D)
    w_v_t, b_v = _bf16_linear_pretransposed(D, D)
    mx.eval(w_q_t, b_q, w_k_t, b_k, w_v_t, b_v)

    def separate():
        q = _linear(w_q_t, b_q, x)
        k = _linear(w_k_t, b_k, x)
        v = _linear(w_v_t, b_v, x)
        return q + k + v  # force all three to materialize via sum

    durs_separate = _time_call(separate, warmup=warmup, iters=iters)
    _report("3x separate addmm (current code path)", durs_separate)

    # Packed: 1 matmul with concatenated weights, then split
    w_packed_t = mx.contiguous(mx.concatenate([w_q_t, w_k_t, w_v_t], axis=1))
    b_packed = mx.concatenate([b_q, b_k, b_v], axis=0)
    mx.eval(w_packed_t, b_packed)

    def packed():
        out = mx.addmm(b_packed, x, w_packed_t)
        q = out[:, :, :D]
        k = out[:, :, D:2*D]
        v = out[:, :, 2*D:]
        return q + k + v  # match the separate-mode aggregation

    durs_packed = _time_call(packed, warmup=warmup, iters=iters)
    _report("1x packed addmm + split", durs_packed)

    sep_mean = sum(durs_separate) / iters
    pack_mean = sum(durs_packed) / iters
    delta_ms = sep_mean - pack_mean
    delta_pct = 100 * delta_ms / sep_mean if sep_mean else 0.0
    print()
    print(f"  Delta: separate {sep_mean:.3f} ms -> packed {pack_mean:.3f} ms "
          f"= {delta_ms:+.3f} ms ({delta_pct:+.1f} %)")
    if delta_ms > 0:
        per_run = delta_ms * 96  # 48 blocks x 2 steps
        print(f"  Per-run savings: {delta_ms:.3f} ms x 96 attention calls "
              f"(48 blocks x 2 steps) = {per_run:.0f} ms = "
              f"{per_run/2:.0f} ms / step ({100*per_run/2/45500:.2f} % of 45.5 s/step)")
    else:
        print("  Packed is SLOWER - not worth pursuing.")


def bench_ff_chain(warmup: int, iters: int) -> None:
    """Full FF chain: matmul -> gelu -> matmul, three variants.

    Question: does mx.compile fuse this?  The MLX source says no
    (matmul not in fusable set), but worth verifying empirically.

    Variants:
    - naive: three eval boundaries (matmul, gelu, matmul) - uppermost cost
    - lazy: one mx.eval at the end of the chain (production code path)
    - compiled: wrap the whole chain in mx.compile
    """
    print()
    print("=" * 100)
    print(f"BENCH: FF chain  (1, {T}, {D}) -> ({D} -> {FF_INNER}) -> gelu -> "
          f"({FF_INNER} -> {D}) -> (1, {T}, {D})  BF16")
    print("=" * 100)

    x = mx.random.normal(shape=(B, T, D)).astype(DTYPE)
    w_in_t, b_in = _bf16_linear_pretransposed(D, FF_INNER)
    w_out_t, b_out = _bf16_linear_pretransposed(FF_INNER, D)
    mx.eval(x, w_in_t, b_in, w_out_t, b_out)

    # Naive: forces eval at each step (worst case - measures kernel dispatch
    # overhead and serial dependence)
    def naive():
        h = _linear(w_in_t, b_in, x)
        mx.eval(h)
        h = nn.gelu_approx(h)
        mx.eval(h)
        out = _linear(w_out_t, b_out, h)
        return out

    durs_naive = _time_call(naive, warmup=warmup, iters=iters)
    _report("naive (eval per op)", durs_naive,
            notes="matmul / gelu / matmul with eval barriers between")

    # Lazy: production code path - one eval at end, MLX scheduler decides
    def lazy():
        h = _linear(w_in_t, b_in, x)
        h = nn.gelu_approx(h)
        out = _linear(w_out_t, b_out, h)
        return out

    durs_lazy = _time_call(lazy, warmup=warmup, iters=iters)
    _report("lazy (production path)", durs_lazy)

    # Compiled: wrap the chain in mx.compile.  Agent says compile does NOT
    # fuse matmul+activation; this empirically tests that claim.
    @mx.compile
    def _compiled_ff(x, w_in_t, b_in, w_out_t, b_out):
        h = mx.addmm(b_in, x, w_in_t)
        h = nn.gelu_approx(h)
        out = mx.addmm(b_out, h, w_out_t)
        return out

    # Warmup the compiled function (first call pays compile cost)
    for _ in range(3):
        out = _compiled_ff(x, w_in_t, b_in, w_out_t, b_out)
        mx.eval(out)

    durs_compiled = _time_call(
        lambda: _compiled_ff(x, w_in_t, b_in, w_out_t, b_out),
        warmup=warmup, iters=iters,
    )
    _report("mx.compile wrapped", durs_compiled,
            notes="if ~equal to lazy, compile is just caching the graph, not fusing matmul+gelu")

    lazy_mean = sum(durs_lazy) / iters
    compiled_mean = sum(durs_compiled) / iters
    delta = lazy_mean - compiled_mean
    print()
    print(f"  lazy {lazy_mean:.3f} ms vs compiled {compiled_mean:.3f} ms "
          f"= {delta:+.3f} ms ({100*delta/lazy_mean:+.2f} %)")
    if abs(delta) < lazy_mean * 0.02:
        print("  Conclusion: ~equal. mx.compile does NOT fuse matmul+activation. "
              "Confirms MLX source inspection (is_fusable() excludes matmul).")
    elif delta > 0:
        print("  Conclusion: compile is faster - possible kernel fusion or "
              "better scheduling. Worth investigating further.")
    else:
        print("  Conclusion: compile is slower (within noise?). Re-run with more iters.")


def bench_sdpa_floor(warmup: int, iters: int) -> None:
    """SDPA at the video_self_attn shape - confirms the 230 ms/call floor.

    Cross-validates our per-call probe and the MLX SDPA tile-floor
    analysis.  If this comes out very different from 230 ms, the
    tile-floor hypothesis or the per-call probe attribution is off.
    """
    print()
    print("=" * 100)
    print(f"BENCH: mx.fast.scaled_dot_product_attention at "
          f"({B}, {H}, {T}, {D_HEAD}) BF16, no mask")
    print("=" * 100)

    q = mx.random.normal(shape=(B, H, T, D_HEAD)).astype(DTYPE)
    k = mx.random.normal(shape=(B, H, T, D_HEAD)).astype(DTYPE)
    v = mx.random.normal(shape=(B, H, T, D_HEAD)).astype(DTYPE)
    mx.eval(q, k, v)

    scale = 1.0 / (D_HEAD ** 0.5)

    durs = _time_call(
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None),
        warmup=warmup, iters=iters,
    )
    _report(f"sdpa BF16 self-attn T={T}", durs,
            notes="expected ~230 ms/call per the 05-17 per-call probe + MLX tile floor")


def bench_quant_matmul(warmup: int, iters: int) -> None:
    """BF16 vs mxfp8 matmul at LTX FF/attention shapes.

    Explains the per-shape source of the end-to-end mxfp8 regression
    measured in scripts/bench_mxfp8_draft.sh.  For each of the four
    matmul shapes that dominate per-block work, times:

    1. **BF16 naive** -- x @ W.T, no pretranspose cache (worst-case BF16)
    2. **BF16 pretranspose** -- mx.addmm(b, x, W_T_contiguous), production default
    3. **mxfp8 quantized** -- nn.QuantizedLinear(mode="mxfp8"), what
       --video-ff-quantize / --transformer-cache-quantize produces

    Reports per-call mean ms + effective TFlops/s (assuming 2 x M x K x N
    FMAs).  M1 Max BF16 theoretical peak is ~10.4 TFlops/s; INT8 is
    nominally ~21 TFlops/s, but the dequant overhead in mx.quantized_matmul
    typically lands well below that.

    Shapes (M = T tokens, K = input dim, N = output dim):
    - FF.project_in  : 8784 x 4096 x 16384  (BF16 weight ~128 MB)
    - FF.project_out : 8784 x 16384 x 4096  (BF16 weight ~128 MB)
    - attn.to_q/k/v  : 8784 x 4096 x 4096   (BF16 weight ~32 MB)
    - attn.to_out    : 8784 x 4096 x 4096   (BF16 weight ~32 MB)
    """
    print()
    print("=" * 100)
    print(f"BENCH: BF16 vs mxfp8 matmul at LTX FF/attention shapes  "
          f"(input batch (1, T={T}, K), BF16)")
    print("=" * 100)

    shapes = [
        ("FF.project_in   (K=4096  N=16384)", T,  4096, 16384),
        ("FF.project_out  (K=16384 N=4096)",  T, 16384,  4096),
        ("attn.to_q/k/v   (K=4096  N=4096)",  T,  4096,  4096),
        ("attn.to_out     (K=4096  N=4096)",  T,  4096,  4096),
    ]

    # Header
    print(f"  {'shape':<36s}  {'mode':<28s}  {'mean_ms':>8s}  "
          f"{'p99_ms':>8s}  {'TFlops/s':>9s}  {'vs BF16+pre':>12s}")
    print(f"  {'-'*36}  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*12}")

    for shape_name, M, K, N in shapes:
        x = mx.random.normal(shape=(B, M, K)).astype(DTYPE)
        # Standard nn.Linear weight orientation: (N, K).  Init small.
        W_NK = (mx.random.normal(shape=(N, K)).astype(DTYPE) * (1.0 / K ** 0.5))
        bias = mx.zeros((N,), dtype=DTYPE)
        mx.eval(x, W_NK, bias)

        flops = 2 * M * K * N  # 2x for FMA per output element

        # Mode 1: BF16 naive -- x @ W.T (W is (N, K), so W.T is (K, N))
        # Standard nn.Linear path without our pretranspose cache.
        def naive(_x=x, _W=W_NK, _b=bias):
            return mx.addmm(_b, _x, _W.T)
        durs_naive = _time_call(naive, warmup=warmup, iters=iters)

        # Mode 2: BF16 pretranspose -- production default
        # mx.contiguous(W.T) gives a (K, N) contiguous weight matrix, then
        # mx.addmm(b, x, W_T) skips the implicit transpose.
        W_T_contig = mx.contiguous(W_NK.T)
        mx.eval(W_T_contig)
        def pretrans(_x=x, _W=W_T_contig, _b=bias):
            return mx.addmm(_b, _x, _W)
        durs_pretrans = _time_call(pretrans, warmup=warmup, iters=iters)
        mean_pretrans = sum(durs_pretrans) / len(durs_pretrans)

        # Mode 3: mxfp8 quantized -- nn.QuantizedLinear with mode="mxfp8"
        # This is what --video-ff-quantize / mxfp8-blocks produces in memory.
        linear = nn.Linear(K, N, bias=True)
        linear.weight = W_NK
        linear.bias = bias
        qlinear = nn.QuantizedLinear.from_linear(linear, mode="mxfp8")
        mx.eval(qlinear.weight, qlinear.scales)
        if "biases" in qlinear:
            mx.eval(qlinear.biases)
        durs_quant = _time_call(lambda _q=qlinear, _x=x: _q(_x), warmup=warmup, iters=iters)

        # Report
        for mode_label, durs in [
            ("BF16 naive (no layout)",       durs_naive),
            ("BF16 pretranspose (prod)",     durs_pretrans),
            ("mxfp8 nn.QuantizedLinear",     durs_quant),
        ]:
            mean_ms = sum(durs) / len(durs)
            p99 = _pct(durs, 99)
            tflops = flops / (mean_ms * 1e-3) / 1e12
            if mode_label.startswith("BF16 pretranspose"):
                vs_pre = "(reference)"
            else:
                ratio_pct = 100 * (mean_ms - mean_pretrans) / mean_pretrans
                vs_pre = f"{ratio_pct:+.1f}%"
            print(f"  {shape_name:<36s}  {mode_label:<28s}  "
                  f"{mean_ms:>8.2f}  {p99:>8.2f}  {tflops:>9.2f}  {vs_pre:>12s}")
        print()

        # Free memory before next shape
        del x, W_NK, bias, W_T_contig, qlinear
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()


def bench_bf16_layout(warmup: int, iters: int) -> None:
    """BF16 naive (x @ W.T) vs pretranspose (mx.addmm against contiguous W.T).

    Isolates the per-shape layout win.  The production default stack carries
    `project_in:pretranspose,project_out:pretranspose` for video FF AND
    `to_q/to_k/to_v/to_out:pretranspose` for video attention -- but the
    isolated-matmul win is asymmetric across shapes:
    - "tall-skinny output" matmuls (K large, N small, like FF project_out
      16384->4096): pretranspose can be a massive win because the implicit
      W.T transpose moves a lot of bytes.
    - "wide output" matmuls (K small, N large, like FF project_in 4096->16384):
      pretranspose is essentially neutral.
    - Square 4096x4096 attention matmuls: tied within noise.

    Reports per-call mean + TFlops/s achieved.  Compare across shapes to
    see where pretranspose actually earns its keep.
    """
    print()
    print("=" * 100)
    print(f"BENCH: BF16 naive vs pretranspose at LTX FF/attention shapes  "
          f"(input batch (1, T={T}, K), BF16)")
    print("=" * 100)

    shapes = [
        ("FF.project_in   (K=4096  N=16384)", T,  4096, 16384),
        ("FF.project_out  (K=16384 N=4096)",  T, 16384,  4096),
        ("attn.to_q/k/v   (K=4096  N=4096)",  T,  4096,  4096),
        ("attn.to_out     (K=4096  N=4096)",  T,  4096,  4096),
    ]

    print(f"  {'shape':<36s}  {'mode':<26s}  {'mean_ms':>8s}  "
          f"{'p99_ms':>8s}  {'TFlops/s':>9s}  {'vs naive':>10s}")
    print(f"  {'-'*36}  {'-'*26}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}")

    for shape_name, M, K, N in shapes:
        x = mx.random.normal(shape=(B, M, K)).astype(DTYPE)
        # Standard nn.Linear weight orientation: (N, K).  Init small.
        W_NK = (mx.random.normal(shape=(N, K)).astype(DTYPE) * (1.0 / K ** 0.5))
        bias = mx.zeros((N,), dtype=DTYPE)
        mx.eval(x, W_NK, bias)

        flops = 2 * M * K * N  # 2x for FMA per output element

        # Mode 1: BF16 naive -- mx.addmm(b, x, W.T).  Standard nn.Linear path,
        # no pretranspose cache.  W is (N, K), W.T is (K, N) -- implicit transpose.
        def naive(_x=x, _W=W_NK, _b=bias):
            return mx.addmm(_b, _x, _W.T)
        durs_naive = _time_call(naive, warmup=warmup, iters=iters)
        mean_naive = sum(durs_naive) / len(durs_naive)

        # Mode 2: BF16 pretranspose -- production default
        # mx.contiguous(W.T) gives a (K, N) contiguous weight matrix, then
        # mx.addmm(b, x, W_T) calls steel_gemm directly without an implicit
        # transpose op.
        W_T_contig = mx.contiguous(W_NK.T)
        mx.eval(W_T_contig)
        def pretrans(_x=x, _W=W_T_contig, _b=bias):
            return mx.addmm(_b, _x, _W)
        durs_pretrans = _time_call(pretrans, warmup=warmup, iters=iters)

        # Report
        for mode_label, durs in [
            ("BF16 naive (no layout)",       durs_naive),
            ("BF16 pretranspose (prod)",     durs_pretrans),
        ]:
            mean_ms = sum(durs) / len(durs)
            p99 = _pct(durs, 99)
            tflops = flops / (mean_ms * 1e-3) / 1e12
            if mode_label.startswith("BF16 naive"):
                vs_naive = "(reference)"
            else:
                ratio_pct = 100 * (mean_ms - mean_naive) / mean_naive
                vs_naive = f"{ratio_pct:+.1f}%"
            print(f"  {shape_name:<36s}  {mode_label:<26s}  "
                  f"{mean_ms:>8.2f}  {p99:>8.2f}  {tflops:>9.2f}  {vs_naive:>10s}")
        print()

        # Free memory before next shape
        del x, W_NK, bias, W_T_contig
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()


def bench_bf16_layout_audio(warmup: int, iters: int) -> None:
    """BF16 naive vs pretranspose at LTX-2.3 AUDIO FF/attention shapes.

    Audio dims are roughly half of video:
    - T_audio        = 502   (vs T_video        = 8784)  -- ~17x smaller
    - D_audio        = 2048  (vs D_video        = 4096)  -- half
    - D_head_audio   = 64    (vs D_head_video   = 128)   -- half
    - FF_inner_audio = 8192  (vs FF_inner_video = 16384) -- half

    Audio uses its own pretranspose default stack
    (`LTX_DISABLE_AUDIO_PRETRANSPOSE=1` to opt out).  Original
    measurement was -11 % at small T (256x256x25) pre-AdaLN-fix.
    This bench tests whether the "only project_out wins" pattern
    that holds for video (per bench_bf16_layout) also applies at
    the smaller audio dims, or whether audio's defaults need
    different treatment.

    Shapes tested (M = T=502, K = input dim, N = output dim):
    - audio.FF.project_in  : 502 x 2048 x 8192   (BF16 weight ~32 MB)
    - audio.FF.project_out : 502 x 8192 x 2048   (BF16 weight ~32 MB)
    - audio.attn.to_q/k/v  : 502 x 2048 x 2048   (BF16 weight ~8 MB)
    - audio.attn.to_out    : 502 x 2048 x 2048   (BF16 weight ~8 MB)
    """
    print()
    print("=" * 100)
    print("BENCH: BF16 naive vs pretranspose at LTX AUDIO FF/attention shapes  "
          "(input batch (1, T=502, K), BF16)")
    print("=" * 100)

    T_AUDIO = 502
    D_AUDIO = 2048
    FF_INNER_AUDIO = 8192

    shapes = [
        ("audio.FF.project_in   (K=2048  N=8192)", T_AUDIO,  D_AUDIO,        FF_INNER_AUDIO),
        ("audio.FF.project_out  (K=8192  N=2048)", T_AUDIO,  FF_INNER_AUDIO, D_AUDIO),
        ("audio.attn.to_q/k/v   (K=2048  N=2048)", T_AUDIO,  D_AUDIO,        D_AUDIO),
        ("audio.attn.to_out     (K=2048  N=2048)", T_AUDIO,  D_AUDIO,        D_AUDIO),
    ]

    print(f"  {'shape':<40s}  {'mode':<26s}  {'mean_ms':>8s}  "
          f"{'p99_ms':>8s}  {'TFlops/s':>9s}  {'vs naive':>10s}")
    print(f"  {'-'*40}  {'-'*26}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*10}")

    for shape_name, M, K, N in shapes:
        x = mx.random.normal(shape=(B, M, K)).astype(DTYPE)
        W_NK = (mx.random.normal(shape=(N, K)).astype(DTYPE) * (1.0 / K ** 0.5))
        bias = mx.zeros((N,), dtype=DTYPE)
        mx.eval(x, W_NK, bias)

        flops = 2 * M * K * N

        # Mode 1: BF16 naive
        def naive(_x=x, _W=W_NK, _b=bias):
            return mx.addmm(_b, _x, _W.T)
        durs_naive = _time_call(naive, warmup=warmup, iters=iters)
        mean_naive = sum(durs_naive) / len(durs_naive)

        # Mode 2: BF16 pretranspose
        W_T_contig = mx.contiguous(W_NK.T)
        mx.eval(W_T_contig)
        def pretrans(_x=x, _W=W_T_contig, _b=bias):
            return mx.addmm(_b, _x, _W)
        durs_pretrans = _time_call(pretrans, warmup=warmup, iters=iters)

        for mode_label, durs in [
            ("BF16 naive (no layout)",   durs_naive),
            ("BF16 pretranspose (prod)", durs_pretrans),
        ]:
            mean_ms = sum(durs) / len(durs)
            p99 = _pct(durs, 99)
            tflops = flops / (mean_ms * 1e-3) / 1e12
            if mode_label.startswith("BF16 naive"):
                vs_naive = "(reference)"
            else:
                ratio_pct = 100 * (mean_ms - mean_naive) / mean_naive
                vs_naive = f"{ratio_pct:+.1f}%"
            print(f"  {shape_name:<40s}  {mode_label:<26s}  "
                  f"{mean_ms:>8.2f}  {p99:>8.2f}  {tflops:>9.2f}  {vs_naive:>10s}")
        print()

        del x, W_NK, bias, W_T_contig
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()


def bench_sdpa_t_sweep(warmup: int, iters: int) -> None:
    """SDPA at neighbor T values around 8784, to find tile-alignment cliffs.

    Question: T=8784 is only 16-aligned (8784/16=549, 8784/32=274.5).
    MLX's sdpa_full uses bq=32 bk=16 on M1 Max non-NAX.  A T that's not
    a multiple of 32 forces a partial tile on the Q axis.  Is padding to
    a friendlier alignment (e.g. 8832 = 64x138 = 128x69, or 9216 =
    128x72) actually faster, despite the extra tokens?

    If a friendlier T runs faster despite more FLOPs, padding becomes a
    free lever via input zero-padding + output slicing.

    Sweep covers T = 8704..9216 with different alignment profiles:
    | T    | / 16 | / 32 | / 64 | / 128 | notes               |
    | ---- | ---- | ---- | ---- | ----- | ------------------- |
    | 8704 | 544  | 272  | 136  | 68    | 128-aligned         |
    | 8736 | 546  | 273  |  -   |  -    | 32 only             |
    | 8768 | 548  | 274  | 137  |  -    | 64-aligned          |
    | 8784 | 549  |  -   |  -   |  -    | OUR ACTUAL (16 only)|
    | 8800 | 550  | 275  |  -   |  -    | 32 only             |
    | 8832 | 552  | 276  | 138  | 69    | 128-aligned (+48)   |
    | 8896 | 556  | 278  | 139  |  -    | 64 only             |
    | 8960 | 560  | 280  | 140  | 70    | 128-aligned (+176)  |
    | 9216 | 576  | 288  | 144  | 72    | 128-aligned (+432)  |

    Normalized cost (ms / T^2 to remove the O(T^2) scaling) makes the
    cliffs visible - a perfectly-scaling kernel would have constant
    normalized cost across all T.
    """
    print()
    print("=" * 100)
    print(f"BENCH: SDPA T-sweep at (1, {H}, T, {D_HEAD}) BF16, no mask")
    print("=" * 100)
    print(f"  {'T':>5}  {'aligned':>14}  {'mean_ms':>8}  {'p99_ms':>8}  "
          f"{'max_ms':>8}  {'norm_ns_T2':>11}  {'vs_T=8784':>10}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*10}")

    T_values = [8704, 8736, 8768, 8784, 8800, 8832, 8896, 8960, 9216]
    scale = 1.0 / (D_HEAD ** 0.5)

    rows: list[tuple] = []

    for T_val in T_values:
        # Compute alignment factors
        aligns = [str(n) for n in (16, 32, 64, 128) if T_val % n == 0]
        align_str = ",".join(aligns) if aligns else "none"

        q = mx.random.normal(shape=(B, H, T_val, D_HEAD)).astype(DTYPE)
        k = mx.random.normal(shape=(B, H, T_val, D_HEAD)).astype(DTYPE)
        v = mx.random.normal(shape=(B, H, T_val, D_HEAD)).astype(DTYPE)
        mx.eval(q, k, v)

        # Closure captures T_val via default-arg trick (avoids late-binding bug)
        def _call(_q=q, _k=k, _v=v):
            return mx.fast.scaled_dot_product_attention(_q, _k, _v, scale=scale, mask=None)

        durs = _time_call(_call, warmup=warmup, iters=iters)
        mean_ms = sum(durs) / len(durs)
        # Normalize by T^2 (SDPA QK and attn*V are both O(T^2 D H), constant factors aside)
        norm_ns_per_T2 = (mean_ms * 1e6) / (T_val * T_val)  # ns per (T tokens)^2

        if T_val == 8784:
            baseline_mean = mean_ms

        rows.append((T_val, align_str, mean_ms, _pct(durs, 99), max(durs), norm_ns_per_T2))

        # Free memory before next T
        del q, k, v
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:
            mx.metal.clear_cache()

    # Print after collecting all (so we can include vs-baseline column)
    for T_val, align_str, mean_ms, p99, mx_ms, norm in rows:
        if baseline_mean is not None and T_val != 8784:
            vs_baseline = f"{100*(mean_ms - baseline_mean)/baseline_mean:+.2f}%"
        elif T_val == 8784:
            vs_baseline = "(baseline)"
        else:
            vs_baseline = "n/a"
        print(f"  {T_val:>5}  {align_str:>14}  {mean_ms:>8.2f}  {p99:>8.2f}  "
              f"{mx_ms:>8.2f}  {norm:>11.4f}  {vs_baseline:>10}")

    # Identify the best PADDING target - must be T >= 8784 (can only pad UP).
    # Truncation is not valid: it would drop tokens (lose data).
    print()
    actual_mean = baseline_mean
    pad_candidates = [(T, align, m) for T, align, m, _, _, _ in rows
                      if T > 8784]
    fast_lower = [(T, align, m) for T, align, m, _, _, _ in rows
                  if T < 8784 and m < actual_mean]

    if pad_candidates:
        best_pad = min(pad_candidates, key=lambda r: r[2])
        best_T, best_align, best_mean = best_pad
        delta = best_mean - actual_mean
        if delta < 0:
            savings_per_call = -delta
            # video_self_attn fires 48 blocks x 8 steps = 384 calls per stage 1
            per_stage1 = savings_per_call * 384
            print(f"  WIN: padding T=8784 -> T={best_T} (aligned {best_align}) "
                  f"saves {savings_per_call:.2f} ms/call.")
            print(f"  Stage-1 extrapolation: x 384 calls = {per_stage1:.0f} ms saved "
                  f"({100*per_stage1/364400:.2f}% of 364 s stage-1 wall).")
        else:
            print(f"  No padding-up target beats T=8784.  Best pad candidate "
                  f"T={best_T} (aligned {best_align}) is {delta:+.2f} ms/call "
                  f"SLOWER.  Padding is NOT a lever for this shape.")

    if fast_lower:
        # Inform but don't claim win - these are sub-baseline T values that we
        # can't reach without truncating tokens.
        print(f"  (Note: T values < 8784 do run faster, e.g. "
              f"{', '.join(f'T={T} {(m-actual_mean):+.1f}ms' for T, _, m in fast_lower[:3])} "
              f"- but truncating tokens is not a valid optimization.)")

    # Normalized-cost note: identifies any per-tile-alignment penalty at T=8784
    norm_baseline = [r[5] for r in rows if r[0] == 8784][0]
    best_norm_T, best_norm_align, _, _, _, best_norm = min(rows, key=lambda r: r[5])
    norm_penalty_pct = 100 * (norm_baseline - best_norm) / best_norm
    if norm_penalty_pct > 0.5:
        print(f"  T=8784 carries a {norm_penalty_pct:.2f}% per-token^2 alignment "
              f"penalty vs the best T={best_norm_T} ({best_norm_align}-aligned) at "
              f"{best_norm:.4f} ns/T^2.  Real but dominated by the extra-token "
              f"cost of any padding target above.")


def bench_sdpa_d_sweep(warmup: int, iters: int) -> None:
    """SDPA at varying head_dim across MLX's kernel-selection boundary.

    Tests the FA-2 tile-size hypothesis WITHOUT writing a custom kernel.
    MLX's sdpa_full dispatch (`scaled_dot_product_attention.cpp:198`) picks:

        bq = 32; bk = bd < 128 ? 32 : 16;

    So at D=120 we get bk=32 (matches pmetal's K-dim choice for d=128);
    at D=128 we get bk=16.  If MLX's bk=16 path leaves compute on the
    table at our shape neighborhood, we expect a KINK in per-FLOP
    efficiency at the D=128 boundary -- D=120 should achieve meaningfully
    higher TFlops/s than D=128 in MLX's own kernels.

    Outcomes:
    - **Kink at D=128 (sharp drop in TFlops/s, jump in ns/FLOP):**
      MLX's bk=16 path leaves compute on the table -> FA-2 hypothesis
      is REAL -> custom kernel with bigger K-tile worth writing.
    - **Smooth curve through D=128:** MLX's tile choice is roughly
      shape-insensitive -> FA-2 lift is at the bottom of the
      1.1-1.3x range -> kernel work probably not worth it.
    - **D=120 already at GEMM ceiling (~7.95 TFlops/s achieved here
      earlier):** MLX's bk=32 path is hardware-bound -> even matching
      pmetal's tile choice at D=128 may not help, because MLX's bk=16
      kernel may have OTHER inefficiencies beyond tile size.

    FLOPs counted as 2 matmuls (QK^T + PV), 2 FLOPs per MAC:
        total_FLOPs = 4 * B * H * T**2 * D

    Compute TFlops/s and ns_per_FLOP for direct comparison across D.
    M1 Max BF16 hardware peak is ~10 TFlops/s; MLX `steel_gemm` on the
    FF shape (different shape) achieves ~7.95 TFlops/s = ~80% of peak.
    """
    print()
    print("=" * 100)
    print(f"BENCH: SDPA D-sweep at ({B}, {H}, {T}, D) BF16, no mask"
          f"  -- testing MLX bk=32 vs bk=16 kernel-selection cliff")
    print("=" * 100)
    print(f"  {'D':>4}  {'MLX_bk':>6}  {'mean_ms':>8}  {'p99_ms':>8}  "
          f"{'TFlops/s':>9}  {'ns/FLOP':>8}  {'vs_D=128':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*8}  "
          f"{'-'*9}  {'-'*8}  {'-'*10}")

    # D values straddling MLX's kernel-selection boundary at D=128.
    # bk=32 below 128, bk=16 at/above 128.
    D_values = [64, 80, 96, 112, 120, 128, 136, 160, 192, 256]

    baseline_tflops = None  # at D=128
    rows: list[tuple] = []

    for D_val in D_values:
        bk = 32 if D_val < 128 else 16
        scale = 1.0 / (D_val ** 0.5)

        try:
            q = mx.random.normal(shape=(B, H, T, D_val)).astype(DTYPE)
            k = mx.random.normal(shape=(B, H, T, D_val)).astype(DTYPE)
            v = mx.random.normal(shape=(B, H, T, D_val)).astype(DTYPE)
            mx.eval(q, k, v)

            def _call(_q=q, _k=k, _v=v, _scale=scale):
                return mx.fast.scaled_dot_product_attention(
                    _q, _k, _v, scale=_scale, mask=None,
                )

            durs = _time_call(_call, warmup=warmup, iters=iters)
        except Exception as e:
            rows.append((D_val, bk, None, None, None, None, f"FAILED: {type(e).__name__}"))
            continue
        finally:
            try:
                del q, k, v
            except NameError:
                pass
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()

        mean_ms = sum(durs) / len(durs)
        p99 = _pct(durs, 99)
        # FLOPs: 4 * B * H * T^2 * D  (2 matmuls, 2 FLOPs per MAC)
        flops = 4.0 * B * H * T * T * D_val
        tflops = (flops / 1e12) / (mean_ms / 1000.0)
        ns_per_flop = (mean_ms * 1e6) / flops  # ns per FLOP

        if D_val == 128:
            baseline_tflops = tflops

        rows.append((D_val, bk, mean_ms, p99, tflops, ns_per_flop, None))

    # Print rows
    for D_val, bk, mean_ms, p99, tflops, ns_per_flop, err in rows:
        if err is not None:
            print(f"  {D_val:>4}  bk={bk:<3}  {err}")
            continue
        if baseline_tflops is not None and D_val != 128:
            # Positive % = D_val achieves more TFlops/s than D=128 baseline
            vs_baseline = f"{100*(tflops - baseline_tflops)/baseline_tflops:+.1f}%"
        elif D_val == 128:
            vs_baseline = "(baseline)"
        else:
            vs_baseline = "n/a"
        print(f"  {D_val:>4}  bk={bk:<3}  {mean_ms:>8.2f}  {p99:>8.2f}  "
              f"{tflops:>9.2f}  {ns_per_flop:>8.4f}  {vs_baseline:>10}")

    # Cliff detection: compare D=120 (bk=32) vs D=128 (bk=16).
    # If MLX's bk=32 kernel achieves significantly more TFlops/s than bk=16,
    # that's direct evidence for the FA-2 tile-size hypothesis.
    print()
    d120 = next((r for r in rows if r[0] == 120 and r[4] is not None), None)
    d128 = next((r for r in rows if r[0] == 128 and r[4] is not None), None)

    if d120 is not None and d128 is not None:
        _, _, _, _, tflops_120, _, _ = d120
        _, _, _, _, tflops_128, _, _ = d128
        delta_pct = 100 * (tflops_120 - tflops_128) / tflops_128
        if delta_pct > 10:
            print(f"  KINK DETECTED at D=128 boundary: D=120 (bk=32) achieves "
                  f"{tflops_120:.2f} TFlops/s vs D=128 (bk=16) {tflops_128:.2f} "
                  f"TFlops/s -- {delta_pct:+.1f}% per-FLOP efficiency win for "
                  f"the bk=32 kernel.")
            print(f"  >>> FA-2 hypothesis SUPPORTED.  MLX's bk=16 kernel at "
                  f"D=128 leaves compute on the table.  A custom kernel "
                  f"matching the bk=32 throughput at D=128 could yield "
                  f"~{delta_pct:.0f}% SDPA speedup at our shape.")
        elif delta_pct > 2:
            print(f"  Small kink at D=128: D=120 (bk=32) {tflops_120:.2f} vs "
                  f"D=128 (bk=16) {tflops_128:.2f} TFlops/s ({delta_pct:+.1f}%).  "
                  f"Marginal -- the FA-2 lift may exist but is on the low end "
                  f"of the 1.1-1.3x estimate.")
        elif delta_pct > -2:
            print(f"  NO kink at D=128 boundary: D=120 (bk=32) {tflops_120:.2f} vs "
                  f"D=128 (bk=16) {tflops_128:.2f} TFlops/s "
                  f"({delta_pct:+.1f}%, within noise).")
            print("  >>> FA-2 hypothesis WEAKENED.  MLX's tile choice is roughly "
                  "shape-insensitive across the boundary.  Custom-kernel lift "
                  "likely under 10% -- probably not worth 2-5 days of work.")
        else:
            print(f"  D=128 (bk=16) is FASTER per FLOP than D=120 (bk=32): "
                  f"{tflops_128:.2f} vs {tflops_120:.2f} TFlops/s ({-delta_pct:.1f}% "
                  f"opposite direction).  Unexpected -- inverts the hypothesis.")

    # Achieved vs peak summary at baseline D=128
    if baseline_tflops is not None:
        m1_max_bf16_peak = 10.0  # TFlops/s approx
        gemm_ceiling = 7.95       # achieved by steel_gemm at FF shape
        pct_peak = 100 * baseline_tflops / m1_max_bf16_peak
        pct_gemm = 100 * baseline_tflops / gemm_ceiling
        print(f"  At D=128: MLX SDPA achieves {baseline_tflops:.2f} TFlops/s = "
              f"{pct_peak:.0f}% of M1 Max BF16 peak (~10) = "
              f"{pct_gemm:.0f}% of MLX steel_gemm ceiling (7.95 at FF shape).")
        max_lift = gemm_ceiling / baseline_tflops
        end_to_end_lift_pct = (1 - 1.0/max_lift) * 0.22 * 100  # SDPA ~22% of step
        print(f"  Theoretical maximum FA-2 lift (matching GEMM ceiling): "
              f"{max_lift:.2f}x per-SDPA-call, ~{end_to_end_lift_pct:.1f}% "
              f"end-to-end (SDPA is ~22% of step time).")


def bench_fused_ffn_feasibility(warmup: int, iters: int) -> None:
    """Bound the savings a hypothetical fused BF16 FFN Metal kernel could deliver.

    Question: is writing a fused (matmul -> GELU -> matmul) Metal kernel - or a
    streamed inner-dim variant on top of MLX - worth the engineering?

    A real fused FFN kernel can NOT beat the irreducible work: it must still
    do both BF16 GEMMs in series (project_out depends on project_in's output).
    Everything else is on the table to eliminate:
      * GELU pass over the full ``T x inner_dim`` hidden (read+write).
      * Kernel dispatch between the two GEMMs.
      * Materializing the hidden tensor in HBM (if the kernel keeps it in
        registers/threadgroup memory).

    This bench measures stock FF wall + peak transient memory at LTX video
    shapes, plus three reference floors:
      A. project_in + project_out as isolated GEMMs (sum)        - additive floor
      B. project_in -> project_out chained, no GELU              - chained floor
      C. GELU-pass cost at the hidden shape                      - what the GELU
         elimination alone could remove

    Anything a fused kernel saves is bounded by ``stock - chained_floor``,
    plus optionally the hidden-tensor HBM bandwidth (not isolated here - read
    it as the gap between A and B, or treat C as a proxy upper bound).

    Compute roofline: if the stock GEMMs already run near the steel_gemm
    ~7.95 TFlops/s ceiling at these shapes, the only way a kernel wins is by
    saving GELU + HBM traffic, which is bounded above.
    """
    print()
    print("=" * 100)
    print(f"BENCH: fused BF16 FFN feasibility  "
          f"(B={B} T={T} D={D} INNER={FF_INNER} BF16)")
    print("=" * 100)

    x = mx.random.normal(shape=(B, T, D)).astype(DTYPE)
    w_in_t, b_in = _bf16_linear_pretransposed(D, FF_INNER)
    w_out_t, b_out = _bf16_linear_pretransposed(FF_INNER, D)
    # Pre-built activated hidden, for the project_out-alone and GELU-alone benches.
    h_seed = nn.gelu_approx(mx.addmm(b_in, x, w_in_t))
    mx.eval(x, w_in_t, b_in, w_out_t, b_out, h_seed)

    in_mb = x.nbytes / 1e6
    hidden_mb = h_seed.nbytes / 1e6
    w_in_mb = w_in_t.nbytes / 1e6
    w_out_mb = w_out_t.nbytes / 1e6
    print(f"  input activation     (1,{T},{D}) BF16  : {in_mb:>7.1f} MB")
    print(f"  hidden activation    (1,{T},{FF_INNER}) BF16  : {hidden_mb:>7.1f} MB")
    print(f"  project_in weight    ({D},{FF_INNER}) BF16   : {w_in_mb:>7.1f} MB")
    print(f"  project_out weight   ({FF_INNER},{D}) BF16   : {w_out_mb:>7.1f} MB")
    print()

    # ---- modes ----
    def stock():
        h = mx.addmm(b_in, x, w_in_t)
        h = nn.gelu_approx(h)
        return mx.addmm(b_out, h, w_out_t)

    def gemm_in_only():
        return mx.addmm(b_in, x, w_in_t)

    def gemm_out_only():
        return mx.addmm(b_out, h_seed, w_out_t)

    def gelu_only():
        return nn.gelu_approx(h_seed)

    def gemms_chained_no_gelu():
        h = mx.addmm(b_in, x, w_in_t)
        return mx.addmm(b_out, h, w_out_t)

    modes = [
        ("stock FF (matmul->gelu->matmul)", stock,
         2 * T * D * FF_INNER + 2 * T * FF_INNER * D),
        ("project_in matmul only",        gemm_in_only,  2 * T * D * FF_INNER),
        ("project_out matmul only",       gemm_out_only, 2 * T * FF_INNER * D),
        ("gelu_approx at hidden shape",   gelu_only,     0),
        ("both matmuls chained, no GELU", gemms_chained_no_gelu,
         2 * T * D * FF_INNER + 2 * T * FF_INNER * D),
    ]

    # Per-mode bandwidth estimate (read inputs + write output, BF16 = 2 B).
    # For GELU: read+write the hidden.
    mode_bytes = {
        "stock FF (matmul->gelu->matmul)":
            x.nbytes + w_in_t.nbytes + h_seed.nbytes * 2 + w_out_t.nbytes + x.nbytes,
        "project_in matmul only":
            x.nbytes + w_in_t.nbytes + h_seed.nbytes,
        "project_out matmul only":
            h_seed.nbytes + w_out_t.nbytes + x.nbytes,
        "gelu_approx at hidden shape":
            h_seed.nbytes * 2,
        "both matmuls chained, no GELU":
            x.nbytes + w_in_t.nbytes + h_seed.nbytes + w_out_t.nbytes + x.nbytes,
    }

    print(f"  {'mode':<34s}  {'mean_ms':>8s}  {'p50_ms':>8s}  "
          f"{'p99_ms':>8s}  {'peak_MB':>9s}  {'TFlops/s':>9s}  {'GB/s':>6s}")
    print(f"  {'-'*34}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*6}")

    results: dict[str, tuple[float, float, int]] = {}
    for label, fn, flops in modes:
        durs, peak = _time_and_peak(fn, warmup, iters)
        mean = sum(durs) / len(durs)
        p50 = _pct(durs, 50)
        p99 = _pct(durs, 99)
        tflops = (flops / (mean * 1e-3) / 1e12) if flops else 0.0
        tf_str = f"{tflops:.2f}" if flops else "n/a"
        gbs = mode_bytes.get(label, 0) / (mean * 1e-3) / 1e9
        print(f"  {label:<34s}  {mean:>8.3f}  {p50:>8.3f}  "
              f"{p99:>8.3f}  {peak/1e6:>9.1f}  {tf_str:>9s}  {gbs:>6.1f}")
        results[label] = (mean, p99, peak)

    mean_stock, _, peak_stock = results["stock FF (matmul->gelu->matmul)"]
    mean_gin, _, _            = results["project_in matmul only"]
    mean_gout, _, peak_gout   = results["project_out matmul only"]
    mean_gelu, _, _           = results["gelu_approx at hidden shape"]
    mean_gg, _, peak_gg       = results["both matmuls chained, no GELU"]

    # Floors
    floor_additive = mean_gin + mean_gout
    floor_chained = mean_gg
    recoverable_chained = mean_stock - floor_chained
    recoverable_pct = 100 * recoverable_chained / mean_stock

    print()
    print("  WALL CEILING (per FF call)")
    print(f"    stock                                    = {mean_stock:>7.3f} ms")
    print(f"    floor A: project_in + project_out (sum)  = {floor_additive:>7.3f} ms")
    print(f"    floor B: both matmuls chained, no GELU   = {floor_chained:>7.3f} ms  "
          f"(upper bound on fused-kernel floor; still writes/reads hidden via HBM)")
    print(f"    GELU-pass cost (hidden read+write)       = {mean_gelu:>7.3f} ms")
    print(f"    recoverable (stock - floor B)            = {recoverable_chained:>+7.3f} ms  "
          f"({recoverable_pct:+.1f}% of stock)")
    # A truly tiled fused kernel could also elide the hidden HBM round-trip
    # embedded in floor B.  Estimate that bandwidth cost from the GELU row:
    # achieved GB/s x 2 (read+write of the hidden tensor) ~ same as mean_gelu.
    # So a perfect tiled fused floor is roughly floor B - mean_gelu.
    fused_floor_estimate = max(floor_chained - mean_gelu, 0.0)
    fused_recoverable = mean_stock - fused_floor_estimate
    fused_recoverable_pct = 100 * fused_recoverable / mean_stock if mean_stock else 0.0
    print(f"    floor C: B minus hidden HBM round-trip   = {fused_floor_estimate:>7.3f} ms  "
          f"(perfect tiled fused-kernel floor; loses if tiling drops GEMM efficiency)")
    print(f"    recoverable (stock - floor C)            = {fused_recoverable:>+7.3f} ms  "
          f"({fused_recoverable_pct:+.1f}% of stock)")

    print()
    print("  MEMORY CEILING (transient peak above persistent weights/inputs)")
    print(f"    stock peak                               = {peak_stock/1e6:>7.1f} MB  "
          f"(GELU produces a fresh hidden - old & new both alive briefly)")
    print(f"    both-matmuls-chained peak                = {peak_gg/1e6:>7.1f} MB  "
          f"(only one hidden alive at a time)")
    output_floor_mb = x.nbytes / 1e6  # FF output is (B, T, dim_out=D) = same as input
    print(f"    absolute floor (output buffer only)      = {output_floor_mb:>7.1f} MB  "
          f"(fused kernel keeping hidden in registers)")
    print(f"    stock -> chained recoverable              = "
          f"{(peak_stock - peak_gg)/1e6:>+7.1f} MB  (eliminate the GELU double-alloc)")
    print(f"    stock -> fused floor recoverable          = "
          f"{(peak_stock - x.nbytes)/1e6:>+7.1f} MB  (also eliminate hidden materialization)")

    print()
    print("  COMPUTE ROOFLINE  (steel_gemm ceiling ~ 7.95 TFlops/s, BF16 peak ~ 10)")
    for label in ("stock FF (matmul->gelu->matmul)", "project_in matmul only",
                  "project_out matmul only", "both matmuls chained, no GELU"):
        mean, _, _ = results[label]
        # Recover flops from the modes table by re-mapping label -> flops
        flops = next(f for ll, _, f in modes if ll == label)
        tflops = flops / (mean * 1e-3) / 1e12
        pct_gemm = 100 * tflops / 7.95
        print(f"    {label:<38s}  {tflops:>5.2f} TFlops/s   {pct_gemm:>5.0f}% of GEMM ceiling")

    # Per-run extrapolation: 48 blocks x 2 steps = 96 FF calls per 2-step run.
    # Report both the conservative (floor B) and optimistic (floor C) savings.
    print()
    per_run_b = recoverable_chained * 96
    per_step_b = per_run_b / 2
    per_run_c = fused_recoverable * 96
    per_step_c = per_run_c / 2
    print("  EXTRAPOLATION (48 blocks x 2 steps = 96 FF calls per 2-step run)")
    print(f"    vs floor B (conservative)  per step = {per_step_b:>+6.0f} ms  "
          f"({100*per_step_b/45500:+.2f}% of 45.5 s/step)")
    print(f"    vs floor C (optimistic)    per step = {per_step_c:>+6.0f} ms  "
          f"({100*per_step_c/45500:+.2f}% of 45.5 s/step)")

    # Verdict
    print()
    if recoverable_pct < 10:
        verdict = ("NOT WORTH IT - stock FF is already near its irreducible floor.  "
                   "A fused kernel could save at most a fraction of the GELU pass.")
    elif recoverable_pct < 25:
        verdict = ("MARGINAL - recoverable wall exists but is bounded.  Weigh the "
                   f"per-step savings ({per_step_b:+.0f} ms) against multi-day kernel work.")
    else:
        verdict = ("PROMISING - meaningful wall and memory headroom above the chained "
                   "GEMM floor.  A one-block fused-kernel prototype is justified.")
    print(f"  VERDICT: {verdict}")


def bench_pointwise_bw(warmup: int, iters: int) -> None:
    """Measure achievable BF16 pointwise bandwidth on this M1 Max as a ceiling.

    Calibration probe.  Pointwise/elementwise ops in the transformer (GELU,
    AdaLN modulation, residual adds, RoPE) are bandwidth-bound; without a
    known ceiling we can't tell whether a measured op runs at 50 GB/s
    because it's near peak or because it's leaving 5x on the table.

    Sweeps three op patterns at sizes spanning dispatch-bound (1 MB) through
    HBM-bound (1 GB), plus the specific LTX tensor sizes:

      * residual: (1, 8784, 4096) BF16 = 72 MB (used in AdaLN, residual add)
      * FF hidden: (1, 8784, 16384) BF16 = 288 MB (FF inner, GELU)

    Op patterns:
      * x * 1.0   - 1 read + 1 write  (pure-bandwidth read/write floor)
      * x + x     - 1 read + 1 write  (single-input add)
      * x + y     - 2 reads + 1 write (two-input add; ~1.5x bytes vs single)

    Reports achieved GB/s per (size, pattern).  M1 Max nominal HBM bandwidth
    is ~400 GB/s; benign access typically achieves 250-350 GB/s.  Pointwise
    on small tensors is dispatch-bound and will be much lower.
    """
    print()
    print("=" * 100)
    print("BENCH: BF16 pointwise bandwidth ceiling on this hardware")
    print("=" * 100)

    # Sweep sizes (in millions of elements) - covers dispatch-bound to HBM-bound.
    # Each entry: (label, shape).  All BF16 (2 B/element).
    shapes = [
        ("size  ~1 MB",       (1, 524288)),
        ("size  ~4 MB",       (1, 2_097_152)),
        ("size ~16 MB",       (1, 8_388_608)),
        ("size ~64 MB",       (1, 33_554_432)),
        ("size  72 MB (resid)", (1, T, D)),                # LTX residual stream
        ("size 288 MB (FF h)",  (1, T, FF_INNER)),         # LTX FF hidden
        ("size 512 MB",       (1, 268_435_456)),
    ]

    print(f"  {'tensor':<22s}  {'pattern':<12s}  "
          f"{'mean_ms':>8s}  {'p99_ms':>8s}  {'bytes_MB':>9s}  {'GB/s':>7s}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}")

    best_gbs = 0.0
    best_label = ""

    for label, shape in shapes:
        try:
            x = mx.random.normal(shape=shape).astype(DTYPE)
            y = mx.random.normal(shape=shape).astype(DTYPE)
            mx.eval(x, y)
        except Exception as e:
            print(f"  {label:<22s}  FAILED to allocate: {type(e).__name__}: {e}")
            continue

        elements = 1
        for d in shape:
            elements *= d
        bytes_one = elements * 2  # BF16

        patterns = [
            ("x * 1.0", lambda _x=x: _x * 1.0,    bytes_one * 2),   # 1R + 1W
            ("x + x",   lambda _x=x: _x + _x,     bytes_one * 2),   # 1R + 1W
            ("x + y",   lambda _x=x, _y=y: _x + _y, bytes_one * 3), # 2R + 1W
        ]

        for pat_label, fn, total_bytes in patterns:
            durs = _time_call(fn, warmup, iters)
            mean = sum(durs) / len(durs)
            p99 = _pct(durs, 99)
            gbs = total_bytes / (mean * 1e-3) / 1e9
            print(f"  {label:<22s}  {pat_label:<12s}  "
                  f"{mean:>8.3f}  {p99:>8.3f}  {total_bytes/1e6:>9.1f}  {gbs:>7.1f}")
            if gbs > best_gbs:
                best_gbs = gbs
                best_label = f"{label} / {pat_label}"

        del x, y
        _clear_cache()
        print()

    print(f"  PEAK ACHIEVED: {best_gbs:.1f} GB/s  ({best_label})")
    print("  (M1 Max nominal HBM = ~400 GB/s; pointwise benign-access target ~250-350 GB/s)")
    print("  USE: divide any bandwidth-bound op's MB/s by this peak to grade its efficiency.")


def bench_adaln_residual(warmup: int, iters: int) -> None:
    """AdaLN modulation + gated-residual chain at residual stream shape.

    Per transformer block:
      * 2 AdaLN modulations (attention prelude + FF prelude)
      * 2 gated residual adds (attention post-add + FF post-add)
    => ~4 pointwise chains per block x 48 blocks x 2 steps = ~384 calls
       per stage-1 2-step run.

    The AdaLN/RoPE dtype-cast fix (commit 2026-05-17) already saved
    -16.8 % bakery by keeping the modulation output in BF16.  The
    question this bench answers: is there additional wall time recoverable
    by fusing the modulation+residual sequence further (custom kernel,
    different compile boundaries)?

    Modes timed at production shapes (B=1, T=8784, D=4096, BF16):
      * `mx.fast.rms_norm`(x) alone                          - irreducible norm cost
      * AdaLN inline:   ``rms_norm; (normed*(1+scale)+shift).astype(bf16)``
      * AdaLN compiled (production default, `mx.compile`'d)
      * Residual gate inline: ``(x + residual*gate).astype(bf16)``
      * Residual gate compiled (production default)
      * Full AdaLN+attn-residual chain: rms_norm -> mod -> (fake_attn_out) -> gated_add
        (inline vs compiled - measures compile-fusion headroom across the chain)

    Scale/shift/gate are (1, 1, D) FP32 (per-token broadcasts from the FP32
    scale_shift_table); inputs and the cast-back output are BF16.

    Extrapolation prints per-step savings if the compiled variant beats
    inline by N % so the engineering-vs-savings call is honest.
    """
    print()
    print("=" * 100)
    print(f"BENCH: AdaLN + residual pointwise chain at (1, {T}, {D}) BF16")
    print("=" * 100)

    # Production scale/shift/gate dtype is FP32 (from scale_shift_table).
    # The cast-back at the end of each chain returns to BF16.  Match that.
    x = mx.random.normal(shape=(B, T, D)).astype(DTYPE)
    residual = mx.random.normal(shape=(B, T, D)).astype(DTYPE)
    scale = mx.random.normal(shape=(B, 1, D)).astype(mx.float32) * 0.1
    shift = mx.random.normal(shape=(B, 1, D)).astype(mx.float32) * 0.1
    gate  = mx.random.normal(shape=(B, 1, D)).astype(mx.float32) * 0.1
    mx.eval(x, residual, scale, shift, gate)

    eps = 1e-6

    # Inline (no compile) variants
    def adaln_inline(_x=x, _s=scale, _b=shift, _e=eps):
        normed = mx.fast.rms_norm(_x, None, _e)
        return (normed * (1 + _s) + _b).astype(_x.dtype)

    def residual_inline(_x=x, _r=residual, _g=gate):
        return (_x + _r * _g).astype(_x.dtype)

    def rmsnorm_only(_x=x, _e=eps):
        return mx.fast.rms_norm(_x, None, _e)

    # Compiled variants (production defaults)
    _adaln_compiled = mx.compile(
        lambda xx, ss, bb, ee: (mx.fast.rms_norm(xx, None, ee) * (1 + ss) + bb).astype(xx.dtype)
    )
    _residual_compiled = mx.compile(
        lambda xx, rr, gg: (xx + rr * gg).astype(xx.dtype)
    )

    # Warm up compiled functions (first call pays the trace cost)
    for _ in range(3):
        mx.eval(_adaln_compiled(x, scale, shift, eps))
        mx.eval(_residual_compiled(x, residual, gate))

    def adaln_compiled():
        return _adaln_compiled(x, scale, shift, eps)

    def residual_compiled():
        return _residual_compiled(x, residual, gate)

    # Full chain (AdaLN + fake-attention-out + residual gate) inline vs compiled.
    # The fake "attention output" here is just `residual` reused so the chain
    # is end-to-end measurable without actually running SDPA.
    def chain_inline(_x=x, _attn=residual, _s=scale, _b=shift, _g=gate, _e=eps):
        normed = mx.fast.rms_norm(_x, None, _e)
        modulated = (normed * (1 + _s) + _b).astype(_x.dtype)
        # In production, `modulated` would go into attention; for the bench,
        # treat `_attn` as the precomputed attn output (avoids running SDPA).
        return (_x + _attn * _g).astype(_x.dtype) + modulated * 0  # keep dep on modulated

    _chain_compiled = mx.compile(
        lambda xx, attn, ss, bb, gg, ee: (
            (xx + attn * gg).astype(xx.dtype)
            + ((mx.fast.rms_norm(xx, None, ee) * (1 + ss) + bb).astype(xx.dtype) * 0)
        )
    )
    for _ in range(3):
        mx.eval(_chain_compiled(x, residual, scale, shift, gate, eps))

    def chain_compiled():
        return _chain_compiled(x, residual, scale, shift, gate, eps)

    # Bytes accounting (approx): rms_norm = 2x residual bytes (R+W);
    # adaln modulation adds 2x more (one read of normed, one write of result);
    # residual adds 3x residual bytes (2R + 1W).  scale/shift/gate are tiny
    # broadcasts, ignored.
    residual_bytes = x.nbytes  # 72 MB
    bytes_map = {
        "mx.fast.rms_norm only":             residual_bytes * 2,
        "AdaLN inline (rms_norm+mod+cast)":  residual_bytes * 4,
        "AdaLN compiled (production)":       residual_bytes * 4,
        "residual_gate inline":              residual_bytes * 3,
        "residual_gate compiled (prod)":     residual_bytes * 3,
        "chain inline (adaln+resid)":        residual_bytes * 6,
        "chain compiled (adaln+resid)":      residual_bytes * 6,
    }

    print(f"  {'mode':<38s}  {'mean_ms':>8s}  {'p50_ms':>8s}  "
          f"{'p99_ms':>8s}  {'bytes_MB':>9s}  {'GB/s':>7s}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}")

    modes = [
        ("mx.fast.rms_norm only",            rmsnorm_only),
        ("AdaLN inline (rms_norm+mod+cast)", adaln_inline),
        ("AdaLN compiled (production)",      adaln_compiled),
        ("residual_gate inline",             residual_inline),
        ("residual_gate compiled (prod)",    residual_compiled),
        ("chain inline (adaln+resid)",       chain_inline),
        ("chain compiled (adaln+resid)",     chain_compiled),
    ]

    results: dict[str, float] = {}
    for label, fn in modes:
        durs = _time_call(fn, warmup, iters)
        mean = sum(durs) / len(durs)
        p50 = _pct(durs, 50)
        p99 = _pct(durs, 99)
        b = bytes_map[label]
        gbs = b / (mean * 1e-3) / 1e9
        print(f"  {label:<38s}  {mean:>8.3f}  {p50:>8.3f}  "
              f"{p99:>8.3f}  {b/1e6:>9.1f}  {gbs:>7.1f}")
        results[label] = mean

    print()
    print("  COMPILE FUSION HEADROOM (production uses compiled variants):")
    adaln_savings = results["AdaLN inline (rms_norm+mod+cast)"] - results["AdaLN compiled (production)"]
    resid_savings = results["residual_gate inline"] - results["residual_gate compiled (prod)"]
    chain_savings = results["chain inline (adaln+resid)"] - results["chain compiled (adaln+resid)"]
    for label, savings_ms in [
        ("AdaLN", adaln_savings),
        ("residual_gate", resid_savings),
        ("chain (AdaLN + residual)", chain_savings),
    ]:
        sign = "FASTER" if savings_ms > 0 else "SLOWER"
        print(f"    {label:<28s}  compile vs inline: {savings_ms:+.3f} ms ({sign})")

    # Per-run extrapolation: 4 pointwise chains per block x 48 blocks x 2 steps
    per_run_chains = 4 * 48 * 2
    print()
    print(f"  EXTRAPOLATION ({per_run_chains} chain-equivalent calls per stage-1 2-step run)")
    chain_compiled_mean = results["chain compiled (adaln+resid)"]
    print(f"    chain compiled @ {chain_compiled_mean:.3f} ms x {per_run_chains}"
          f" = {chain_compiled_mean * per_run_chains / 1000:.2f} s / run "
          f"({100*chain_compiled_mean*per_run_chains/2/45500:.2f}% of 45.5 s/step)")
    if chain_savings > 0:
        per_step_save = chain_savings * per_run_chains / 2
        print(f"    if inline ran instead, would lose {per_step_save:.0f} ms / step "
              f"({100*per_step_save/45500:.2f}% of step) - KEEP compile")
    else:
        per_step_pen = -chain_savings * per_run_chains / 2
        print(f"    inline would SAVE {per_step_pen:.0f} ms / step "
              f"({100*per_step_pen/45500:.2f}% of step) - RETEST compile boundary")

    # Headroom vs pointwise ceiling (run bench_pointwise_bw to get the actual peak)
    print()
    chain_gbs = bytes_map["chain compiled (adaln+resid)"] / (chain_compiled_mean * 1e-3) / 1e9
    print(f"  HEADROOM: chain compiled at {chain_gbs:.1f} GB/s.  "
          f"Compare to `pointwise_bw` peak; if much lower, a custom Metal "
          f"AdaLN+residual kernel could recover the gap.")


def bench_rope(warmup: int, iters: int) -> None:
    """Split RoPE application at LTX video Q/K shape.

    Production path: `apply_split_rotary_emb` from `LTX_2_MLX/.../rope.py`,
    which calls the `@mx.compile`'d `_apply_split_rope_4d`.  Input is BF16
    of shape (B, H, T, D); cos/sin are FP32 of shape (B, H, T, D//2).

    The FP32 cos/sin is intentional (sincos precision), and the kernel casts
    the result back to BF16 to avoid promoting downstream SDPA to FP32 (the
    fix that landed the -16.8% AdaLN/RoPE win on 2026-05-17).

    Per stage-1 run: 48 blocks x 2 steps x 2 (Q and K) = 192 RoPE calls.

    Modes:
      * production: BF16 input, FP32 cos/sin, cast back to BF16
      * all-BF16 freqs: same compute path but with BF16 cos/sin (lower
        precision - informational only, NOT a same-math substitute)
      * bare 4D multiply at (B, H, T, D//2) - pointwise floor
      * raw concatenate at (B, H, T, D) - concat cost in isolation

    Reports wall and bandwidth; extrapolates per-step savings if a fully-BF16
    path is meaningfully cheaper than the production FP32 freqs path.
    """
    print()
    print("=" * 100)
    print(f"BENCH: split RoPE at ({B}, {H}, {T}, {D_HEAD}) BF16 input, FP32 cos/sin")
    print("=" * 100)

    # Try to import the production rope function so we measure the EXACT path.
    # The bench runs as `python scripts/bench_ff_microbench.py ...` from the
    # repo root, but the LTX_2_MLX package may not be installed in this venv -
    # add the repo root to sys.path so the import resolves.
    rope_fn = None
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _repo_root = _Path(__file__).resolve().parent.parent
        if str(_repo_root) not in _sys.path:
            _sys.path.insert(0, str(_repo_root))
        from LTX_2_MLX.model.transformer.rope import apply_split_rotary_emb
        rope_fn = apply_split_rotary_emb
        print("  using production: LTX_2_MLX.model.transformer.rope.apply_split_rotary_emb")
    except Exception as e:
        print(f"  WARN: could not import production rope ({type(e).__name__}: {e}); using local impl")

    # Local fallback / for the all-BF16 comparison
    def _local_split_rope(x, cos_freqs, sin_freqs):
        input_dtype = x.dtype
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return mx.concatenate(
            [x1 * cos_freqs - x2 * sin_freqs, x1 * sin_freqs + x2 * cos_freqs],
            axis=-1,
        ).astype(input_dtype)

    if rope_fn is None:
        rope_fn = _local_split_rope

    half = D_HEAD // 2
    x_bf16 = mx.random.normal(shape=(B, H, T, D_HEAD)).astype(DTYPE)
    cos_fp32 = mx.random.normal(shape=(B, H, T, half)).astype(mx.float32)
    sin_fp32 = mx.random.normal(shape=(B, H, T, half)).astype(mx.float32)
    cos_bf16 = cos_fp32.astype(DTYPE)
    sin_bf16 = sin_fp32.astype(DTYPE)
    mx.eval(x_bf16, cos_fp32, sin_fp32, cos_bf16, sin_bf16)

    x_bytes = x_bf16.nbytes
    cos_fp32_bytes = cos_fp32.nbytes
    cos_bf16_bytes = cos_bf16.nbytes

    # Bytes accounting per call (approx):
    #   x read = x_bytes
    #   cos read + sin read = 2 * cos_bytes
    #   intermediate x1*cos, x2*sin etc - internal to compiled kernel
    #   output write = x_bytes
    bytes_prod   = x_bytes + 2 * cos_fp32_bytes + x_bytes   # FP32 freqs
    bytes_bf16   = x_bytes + 2 * cos_bf16_bytes + x_bytes   # BF16 freqs
    bytes_mul    = x_bytes // 2 + cos_bf16_bytes + x_bytes // 2  # bare (B,H,T,D//2) BF16 multiply
    bytes_concat = x_bytes + x_bytes  # read+write at full shape

    def prod_call():
        return rope_fn(x_bf16, cos_fp32, sin_fp32)

    def bf16_freqs_call():
        return rope_fn(x_bf16, cos_bf16, sin_bf16)

    half_bf16 = x_bf16[..., :half]
    mx.eval(half_bf16)

    def bare_mul():
        return half_bf16 * cos_bf16

    def bare_concat():
        return mx.concatenate([half_bf16, half_bf16], axis=-1)

    modes = [
        ("production (FP32 cos/sin, cast back)", prod_call,        bytes_prod),
        ("all-BF16 freqs (lower precision)",     bf16_freqs_call,  bytes_bf16),
        ("bare BF16 multiply at (B,H,T,D//2)",   bare_mul,         bytes_mul),
        ("bare concatenate at (B,H,T,D)",        bare_concat,      bytes_concat),
    ]

    print(f"  {'mode':<40s}  {'mean_ms':>8s}  {'p50_ms':>8s}  "
          f"{'p99_ms':>8s}  {'bytes_MB':>9s}  {'GB/s':>7s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}")

    results: dict[str, float] = {}
    for label, fn, total_bytes in modes:
        durs = _time_call(fn, warmup, iters)
        mean = sum(durs) / len(durs)
        p50 = _pct(durs, 50)
        p99 = _pct(durs, 99)
        gbs = total_bytes / (mean * 1e-3) / 1e9
        print(f"  {label:<40s}  {mean:>8.3f}  {p50:>8.3f}  "
              f"{p99:>8.3f}  {total_bytes/1e6:>9.1f}  {gbs:>7.1f}")
        results[label] = mean

    prod_mean = results["production (FP32 cos/sin, cast back)"]
    bf16_mean = results["all-BF16 freqs (lower precision)"]
    delta = prod_mean - bf16_mean
    delta_pct = 100 * delta / prod_mean if prod_mean else 0.0

    # Per-stage extrapolation: 48 blocks x 2 steps x 2 (Q + K) = 192 RoPE calls
    per_run_calls = 48 * 2 * 2
    print()
    print(f"  PRODUCTION vs ALL-BF16 FREQS: {delta:+.3f} ms ({delta_pct:+.1f}% of production)")
    if delta_pct > 5:
        per_run_save = delta * per_run_calls
        per_step_save = per_run_save / 2
        print(f"    BF16 freqs would save {per_step_save:.0f} ms / step "
              f"({100*per_step_save/45500:.2f}% of 45.5 s/step) IF precision held.")
        print("    Worth a sincos-precision experiment: keep cos/sin FP32 at build time")
        print("    but cast to BF16 before per-block RoPE.  Compare bakery output.")
    else:
        print("    FP32 cos/sin is essentially free here.  No action.")

    print()
    print(f"  EXTRAPOLATION ({per_run_calls} RoPE calls per stage-1 2-step run)")
    print(f"    production @ {prod_mean:.3f} ms x {per_run_calls}"
          f" = {prod_mean * per_run_calls / 1000:.2f} s / run "
          f"({100*prod_mean*per_run_calls/2/45500:.2f}% of 45.5 s/step)")

    prod_gbs = bytes_prod / (prod_mean * 1e-3) / 1e9
    print()
    print(f"  HEADROOM: production RoPE at {prod_gbs:.1f} GB/s.  Compare to "
          f"`pointwise_bw` peak; if much lower, a custom Metal RoPE kernel "
          f"(or merging RoPE into the QKV matmul epilogue) could recover the gap.")


def bench_vae_ops(warmup: int, iters: int) -> None:
    """Per-op breakdown of the native-conv3d VAE decoder hot path.

    The VAE decoder runs as ~7 % of total wall on bakery, but an upstream
    `mx.conv_general` int-indexing-overflow fix is pending; this bench
    quantifies what fraction of decode each op currently takes so the
    fix's payoff is bounded in advance.

    Ops timed at three representative BFHWC shapes spanning the decoder
    progression (bottleneck -> mid -> late):

      * `nn.Conv3d(in=C, out=C, k=3)` -- the dominant cost
      * `mx.fast.rms_norm` over the channel axis (pixel norm in `ops.py`)
      * `nn.silu` -- pointwise activation in every resnet block
      * Full resnet pattern: norm -> silu -> conv -> norm -> silu -> conv -> add

    Shape progression mirrors `NativeConv3dVideoDecoder` defaults
    (base_channels=128, feature_channels=1024 at bottleneck, halving on
    each spatial upsample, with per-tile spatial extent following the
    `tile_size_in_pixels >= 64` invariant from `tiling.py`).

    Bottleneck:  (1,  4,  8,  8, 1024)  -- ~0.5 MB BF16 - pure dispatch regime
    Mid:         (1,  8, 32, 32,  256)  -- ~4   MB
    Late:        (1, 16, 64, 64,   64)  -- ~8   MB

    Reports per-op mean/p99 + GB/s.  Use the `pointwise_bw` peak to grade
    pointwise efficiency; for Conv3d, compare achieved TFlops/s to the
    ~7.95 `steel_gemm` ceiling (Conv3d in MLX im2col-style typically lands
    well below GEMM ceiling - that's the int-overflow-fix opportunity).
    """
    print()
    print("=" * 100)
    print("BENCH: VAE decoder native-conv3d op breakdown (BFHWC, BF16)")
    print("=" * 100)

    shapes = [
        ("bottleneck (1,4,8,8,1024)",   (1,  4,  8,  8, 1024)),
        ("mid        (1,8,32,32,256)",  (1,  8, 32, 32,  256)),
        ("late       (1,16,64,64,64)",  (1, 16, 64, 64,   64)),
    ]

    print(f"  {'shape':<28s}  {'op':<28s}  {'mean_ms':>8s}  "
          f"{'p99_ms':>8s}  {'tensor_MB':>10s}  {'GB/s':>7s}  {'TFlops/s':>9s}")
    print(f"  {'-'*28}  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*9}")

    for shape_label, shape in shapes:
        B_, F_, H_, W_, C_ = shape
        x = mx.random.normal(shape=shape).astype(DTYPE)
        mx.eval(x)
        tensor_mb = x.nbytes / 1e6

        # ---- Conv3d (same in/out channels, k=3) ----
        # MLX nn.Conv3d expects BTHWC layout (B, T, H, W, C) -- matches BFHWC.
        conv = nn.Conv3d(in_channels=C_, out_channels=C_, kernel_size=3, padding=1)
        conv.set_dtype(DTYPE)
        mx.eval(conv.weight, conv.bias if "bias" in conv else x)
        # FLOPs: B * F * H * W * C_out * (k^3 * C_in * 2)
        conv_flops = B_ * F_ * H_ * W_ * C_ * (27 * C_ * 2)
        # Approx bytes: input + weights + output (kernel small relative to acts)
        conv_bytes = x.nbytes + conv.weight.nbytes + x.nbytes

        durs_conv = _time_call(lambda _c=conv, _x=x: _c(_x), warmup, iters)
        mean = sum(durs_conv) / len(durs_conv)
        gbs = conv_bytes / (mean * 1e-3) / 1e9
        tflops = conv_flops / (mean * 1e-3) / 1e12
        print(f"  {shape_label:<28s}  {'nn.Conv3d (k=3, C->C)':<28s}  "
              f"{mean:>8.3f}  {_pct(durs_conv,99):>8.3f}  "
              f"{tensor_mb:>10.2f}  {gbs:>7.1f}  {tflops:>9.3f}")

        # ---- rms_norm over channel axis (pixel norm pattern from ops.py) ----
        # Channel-last RMS norm = mx.fast.rms_norm at last dim (which is what
        # mx.fast.rms_norm does by default on the last axis).
        durs_norm = _time_call(
            lambda _x=x: mx.fast.rms_norm(_x, None, 1e-8),
            warmup, iters,
        )
        mean_n = sum(durs_norm) / len(durs_norm)
        gbs_n = (x.nbytes * 2) / (mean_n * 1e-3) / 1e9
        print(f"  {shape_label:<28s}  {'mx.fast.rms_norm (pixel)':<28s}  "
              f"{mean_n:>8.3f}  {_pct(durs_norm,99):>8.3f}  "
              f"{tensor_mb:>10.2f}  {gbs_n:>7.1f}  {'n/a':>9s}")

        # ---- silu ----
        durs_silu = _time_call(lambda _x=x: nn.silu(_x), warmup, iters)
        mean_s = sum(durs_silu) / len(durs_silu)
        gbs_s = (x.nbytes * 2) / (mean_s * 1e-3) / 1e9
        print(f"  {shape_label:<28s}  {'nn.silu':<28s}  "
              f"{mean_s:>8.3f}  {_pct(durs_silu,99):>8.3f}  "
              f"{tensor_mb:>10.2f}  {gbs_s:>7.1f}  {'n/a':>9s}")

        # ---- full resnet block pattern: norm -> silu -> conv -> norm -> silu -> conv -> add ----
        conv2 = nn.Conv3d(in_channels=C_, out_channels=C_, kernel_size=3, padding=1)
        conv2.set_dtype(DTYPE)
        mx.eval(conv2.weight)

        def resnet_block(_x=x, _c1=conv, _c2=conv2, _eps=1e-8):
            h = nn.silu(mx.fast.rms_norm(_x, None, _eps))
            h = _c1(h)
            h = nn.silu(mx.fast.rms_norm(h, None, _eps))
            h = _c2(h)
            return _x + h

        durs_rb = _time_call(resnet_block, warmup, iters)
        mean_rb = sum(durs_rb) / len(durs_rb)
        # rough byte count: input read 3x, 2 conv weights, output write - dominated by conv
        rb_bytes = x.nbytes * 6 + conv.weight.nbytes * 2
        gbs_rb = rb_bytes / (mean_rb * 1e-3) / 1e9
        # FLOPs: 2 convs
        rb_flops = conv_flops * 2
        rb_tflops = rb_flops / (mean_rb * 1e-3) / 1e12
        print(f"  {shape_label:<28s}  {'resnet block (full)':<28s}  "
              f"{mean_rb:>8.3f}  {_pct(durs_rb,99):>8.3f}  "
              f"{tensor_mb:>10.2f}  {gbs_rb:>7.1f}  {rb_tflops:>9.3f}")

        # Free
        del x, conv, conv2
        _clear_cache()
        print()

    print("  INTERPRETATION:")
    print("    * Conv3d typically dominates resnet-block time (~80-95%).")
    print("    * If achieved Conv3d TFlops/s is well below the ~7.95 `steel_gemm`")
    print("      ceiling, the upstream int-overflow fix could lift it.")
    print("    * pixel-norm + silu are pointwise; grade vs `pointwise_bw` peak.")
    print("    * VAE decode is ~7 % of total wall.  A 2x conv speedup would")
    print("      buy ~3.5 % of total - small but real if the fix lands free.")


# --- driver ---


BENCHES = {
    "gelu": bench_gelu,
    "qkv": bench_qkv_separate_vs_packed,
    "qkv_separate_vs_packed": bench_qkv_separate_vs_packed,
    "ff_chain": bench_ff_chain,
    "sdpa_floor": bench_sdpa_floor,
    "sdpa": bench_sdpa_floor,
    "sdpa_t_sweep": bench_sdpa_t_sweep,
    "sdpa_sweep": bench_sdpa_t_sweep,
    "sdpa_d_sweep": bench_sdpa_d_sweep,
    "sdpa_d": bench_sdpa_d_sweep,
    "d_sweep": bench_sdpa_d_sweep,
    "quant_matmul": bench_quant_matmul,
    "quant": bench_quant_matmul,
    "bf16_layout": bench_bf16_layout,
    "layout": bench_bf16_layout,
    "bf16_layout_audio": bench_bf16_layout_audio,
    "layout_audio": bench_bf16_layout_audio,
    "audio_layout": bench_bf16_layout_audio,
    "fused_ffn_feasibility": bench_fused_ffn_feasibility,
    "fused_ffn": bench_fused_ffn_feasibility,
    "fused": bench_fused_ffn_feasibility,
    "feasibility": bench_fused_ffn_feasibility,
    "pointwise_bw": bench_pointwise_bw,
    "pointwise": bench_pointwise_bw,
    "bw": bench_pointwise_bw,
    "bandwidth": bench_pointwise_bw,
    "adaln_residual": bench_adaln_residual,
    "adaln": bench_adaln_residual,
    "residual": bench_adaln_residual,
    "rope": bench_rope,
    "vae_ops": bench_vae_ops,
    "vae": bench_vae_ops,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benches", nargs="?", default=None,
                    help="Comma-separated bench names (default: all). "
                         "Available: " + ", ".join(sorted(BENCHES.keys())))
    args = ap.parse_args()

    iters = int(os.environ.get("ITERS", "20"))
    warmup = int(os.environ.get("WARMUP", "5"))

    requested: list[str]
    if args.benches:
        requested = [b.strip() for b in args.benches.split(",")]
        unknown = [b for b in requested if b not in BENCHES]
        if unknown:
            print(f"ERROR: unknown bench(es): {unknown}.  "
                  f"Available: {sorted(BENCHES.keys())}", file=sys.stderr)
            return 1
    else:
        # All unique benches (dedupe aliases)
        seen: set = set()
        requested = []
        for name, fn in BENCHES.items():
            if fn not in seen:
                seen.add(fn)
                requested.append(name)

    print(f"# FF/attention microbenches  iters={iters}  warmup={warmup}  "
          f"shapes: T={T} D={D} H={H} D_head={D_HEAD} FF_inner={FF_INNER}  BF16")

    for name in requested:
        fn = BENCHES[name]
        try:
            fn(warmup=warmup, iters=iters)
        except Exception as exc:
            print(f"\nBENCH {name} FAILED: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
