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
    ff_chain            - matmul → gelu → matmul (naive vs lazy vs mx.compile)
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

Each bench prints a result block. Compare numbers across modes within a
single bench, not across benches (different shapes / FLOPs).

Notes:
- Pre-transposes weights via mx.contiguous(weight.T) to match production
  (matches the `--video-ff-layout` and `--video-attn-layout` default stack).
- Uses BF16 throughout to match production compute dtype.
- mx.eval barriers + time.monotonic_ns timing — same methodology as
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
T = 8784            # latent tokens (61 frames × 9 × 16 spatial)
D = 4096            # hidden dim
H = 32              # attention heads
D_HEAD = 128        # head dim (H × D_HEAD = D)
FF_INNER = 16384    # FF intermediate (4× D)
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
    per call.  Per stage-1 run (96 calls = 48 blocks × 2 steps for our
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
    per_run_total = (sum(durs) / iters) * 96  # 48 blocks × 2 steps
    per_step_total = per_run_total / 2
    print()
    print(f"  Extrapolation: {sum(durs)/iters:.2f} ms × 96 calls "
          f"(48 blocks × 2 steps) = {per_run_total:.0f} ms / 2-step run = "
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
    print(f"  Delta: separate {sep_mean:.3f} ms → packed {pack_mean:.3f} ms "
          f"= {delta_ms:+.3f} ms ({delta_pct:+.1f} %)")
    if delta_ms > 0:
        per_run = delta_ms * 96  # 48 blocks × 2 steps
        print(f"  Per-run savings: {delta_ms:.3f} ms × 96 attention calls "
              f"(48 blocks × 2 steps) = {per_run:.0f} ms = "
              f"{per_run/2:.0f} ms / step ({100*per_run/2/45500:.2f} % of 45.5 s/step)")
    else:
        print(f"  Packed is SLOWER — not worth pursuing.")


def bench_ff_chain(warmup: int, iters: int) -> None:
    """Full FF chain: matmul → gelu → matmul, three variants.

    Question: does mx.compile fuse this?  The MLX source says no
    (matmul not in fusable set), but worth verifying empirically.

    Variants:
    - naive: three eval boundaries (matmul, gelu, matmul) — uppermost cost
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

    # Naive: forces eval at each step (worst case — measures kernel dispatch
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

    # Lazy: production code path — one eval at end, MLX scheduler decides
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
        print(f"  Conclusion: ~equal. mx.compile does NOT fuse matmul+activation. "
              f"Confirms MLX source inspection (is_fusable() excludes matmul).")
    elif delta > 0:
        print(f"  Conclusion: compile is faster — possible kernel fusion or "
              f"better scheduling. Worth investigating further.")
    else:
        print(f"  Conclusion: compile is slower (within noise?). Re-run with more iters.")


def bench_sdpa_floor(warmup: int, iters: int) -> None:
    """SDPA at the video_self_attn shape — confirms the 230 ms/call floor.

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

    Reports per-call mean ms + effective TFlops/s (assuming 2 × M × K × N
    FMAs).  M1 Max BF16 theoretical peak is ~10.4 TFlops/s; INT8 is
    nominally ~21 TFlops/s, but the dequant overhead in mx.quantized_matmul
    typically lands well below that.

    Shapes (M = T tokens, K = input dim, N = output dim):
    - FF.project_in  : 8784 × 4096 × 16384  (BF16 weight ~128 MB)
    - FF.project_out : 8784 × 16384 × 4096  (BF16 weight ~128 MB)
    - attn.to_q/k/v  : 8784 × 4096 × 4096   (BF16 weight ~32 MB)
    - attn.to_out    : 8784 × 4096 × 4096   (BF16 weight ~32 MB)
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
        mean_naive = sum(durs_naive) / len(durs_naive)

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
        mean_quant = sum(durs_quant) / len(durs_quant)

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
      16384→4096): pretranspose can be a massive win because the implicit
      W.T transpose moves a lot of bytes.
    - "wide output" matmuls (K small, N large, like FF project_in 4096→16384):
      pretranspose is essentially neutral.
    - Square 4096×4096 attention matmuls: tied within noise.

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
        mean_pretrans = sum(durs_pretrans) / len(durs_pretrans)

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
    - audio.FF.project_in  : 502 × 2048 × 8192   (BF16 weight ~32 MB)
    - audio.FF.project_out : 502 × 8192 × 2048   (BF16 weight ~32 MB)
    - audio.attn.to_q/k/v  : 502 × 2048 × 2048   (BF16 weight ~8 MB)
    - audio.attn.to_out    : 502 × 2048 × 2048   (BF16 weight ~8 MB)
    """
    print()
    print("=" * 100)
    print(f"BENCH: BF16 naive vs pretranspose at LTX AUDIO FF/attention shapes  "
          f"(input batch (1, T=502, K), BF16)")
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
        mean_pretrans = sum(durs_pretrans) / len(durs_pretrans)

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
    a friendlier alignment (e.g. 8832 = 64×138 = 128×69, or 9216 =
    128×72) actually faster, despite the extra tokens?

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

    Normalized cost (ms / T² to remove the O(T²) scaling) makes the
    cliffs visible — a perfectly-scaling kernel would have constant
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

    baseline_mean = None
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
        # Normalize by T² (SDPA QK and attn*V are both O(T² D H), constant factors aside)
        norm_ns_per_T2 = (mean_ms * 1e6) / (T_val * T_val)  # ns per (T tokens)²

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

    # Identify the best PADDING target — must be T >= 8784 (can only pad UP).
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
            # video_self_attn fires 48 blocks × 8 steps = 384 calls per stage 1
            per_stage1 = savings_per_call * 384
            print(f"  WIN: padding T=8784 → T={best_T} (aligned {best_align}) "
                  f"saves {savings_per_call:.2f} ms/call.")
            print(f"  Stage-1 extrapolation: × 384 calls = {per_stage1:.0f} ms saved "
                  f"({100*per_stage1/364400:.2f}% of 364 s stage-1 wall).")
        else:
            print(f"  No padding-up target beats T=8784.  Best pad candidate "
                  f"T={best_T} (aligned {best_align}) is {delta:+.2f} ms/call "
                  f"SLOWER.  Padding is NOT a lever for this shape.")

    if fast_lower:
        # Inform but don't claim win — these are sub-baseline T values that we
        # can't reach without truncating tokens.
        print(f"  (Note: T values < 8784 do run faster, e.g. "
              f"{', '.join(f'T={T} {(m-actual_mean):+.1f}ms' for T, _, m in fast_lower[:3])} "
              f"— but truncating tokens is not a valid optimization.)")

    # Normalized-cost note: identifies any per-tile-alignment penalty at T=8784
    norm_baseline = [r[5] for r in rows if r[0] == 8784][0]
    best_norm_T, best_norm_align, _, _, _, best_norm = min(rows, key=lambda r: r[5])
    norm_penalty_pct = 100 * (norm_baseline - best_norm) / best_norm
    if norm_penalty_pct > 0.5:
        print(f"  T=8784 carries a {norm_penalty_pct:.2f}% per-token² alignment "
              f"penalty vs the best T={best_norm_T} ({best_norm_align}-aligned) at "
              f"{best_norm:.4f} ns/T².  Real but dominated by the extra-token "
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
    baseline_mean = None
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
            baseline_mean = mean_ms

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
            print(f"  >>> FA-2 hypothesis WEAKENED.  MLX's tile choice is roughly "
                  f"shape-insensitive across the boundary.  Custom-kernel lift "
                  f"likely under 10% -- probably not worth 2-5 days of work.")
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
