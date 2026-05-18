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


# --- driver ---


BENCHES = {
    "gelu": bench_gelu,
    "qkv": bench_qkv_separate_vs_packed,
    "qkv_separate_vs_packed": bench_qkv_separate_vs_packed,
    "ff_chain": bench_ff_chain,
    "sdpa_floor": bench_sdpa_floor,
    "sdpa": bench_sdpa_floor,
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
