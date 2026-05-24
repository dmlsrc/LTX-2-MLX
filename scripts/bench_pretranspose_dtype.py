"""Pretranspose vs naive matmul at LTX-2.3 production FF/attn shapes, BF16 AND FP16.

Standalone — only needs `mlx`. Run from any venv:

    python scripts/bench_pretranspose_dtype.py
    python scripts/bench_pretranspose_dtype.py --iters 100

What this isolates:

  Naive layout    (stock MLX Linear): weight stored (out_features, in_features)
                                       contiguous; op is `x @ weight.T`
                                       (.T is a strided view).

  Pretranspose    (LTX-2-MLX cache):  weight stored mx.contiguous(weight.T) =
                                       (in_features, out_features) contiguous;
                                       op is `x @ pretransposed` directly
                                       (no .T view).

For each shape we run all four combinations:
    (BF16, naive)  (BF16, pretranspose)  (FP16, naive)  (FP16, pretranspose)

Output per shape:
  - 2x2 wall-time/throughput table
  - Within-dtype pretranspose delta (does pretranspose still win?)
  - Within-layout FP16/BF16 ratio (is FP16 faster, and does layout affect that?)

Why this lives alongside `bench_ff_microbench.py`:
  - `bench_ff_microbench.py bf16_layout` tests only BF16 across the same shapes.
    This script extends to FP16, which surfaced a key finding: FP16 has a
    DEEPER kernel-selection cliff at project_out (K=16384) than BF16, so
    `--video-ff-dtype float16` mandates `project_out:pretranspose`.
  - Video shapes match `bench_ff_microbench.py bf16_layout`.
  - Audio shapes match `bench_ff_microbench.py bf16_layout_audio`.

Reference: `PERFORMANCE.md` "2026-05-23: FF compute in FP16" entry and
`PERFORMANCE_NOTES.md` "FF FP16 + FP16 kernel cliff" entry.

Output: redirect to keep it, e.g.:
    python scripts/bench_pretranspose_dtype.py \
      | tee "$SHARED_TEMP_DIR/trace_analysis/bench_pretranspose_dtype_$(date +%Y%m%d_%H%M%S).log"
"""

import argparse
import time

import mlx.core as mx


def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def run_shape(label: str, M: int, K: int, N: int, iters: int, warmup: int) -> None:
    """For one shape: run all 4 cells (BF16/FP16 x naive/pretranspose) and print."""
    print()
    print("=" * 86)
    print(f"  {label}")
    print(f"  M={M}  K={K}  N={N}   (input ({M},{K}) -> output ({M},{N}))")
    print("=" * 86)

    flops = 2 * M * K * N

    # Reuse the same underlying random values across dtypes so timing differences
    # are dtype/layout, not content.
    rng_seed = mx.random.key(0)
    x_bf = mx.random.normal((M, K), key=rng_seed).astype(mx.bfloat16)
    w_naive_bf = mx.random.normal((N, K), key=mx.random.split(rng_seed)[1]).astype(mx.bfloat16)
    w_pre_bf = mx.contiguous(w_naive_bf.T)
    x_fp = x_bf.astype(mx.float16)
    w_naive_fp = w_naive_bf.astype(mx.float16)
    w_pre_fp = w_pre_bf.astype(mx.float16)
    mx.eval(x_bf, w_naive_bf, w_pre_bf, x_fp, w_naive_fp, w_pre_fp)

    cells: dict[tuple[str, str], tuple[float, float]] = {}
    for dtype_name, x, w_naive, w_pre in (
        ("BF16", x_bf, w_naive_bf, w_pre_bf),
        ("FP16", x_fp, w_naive_fp, w_pre_fp),
    ):
        for layout, expr in (
            ("naive",       lambda x=x, w=w_naive: x @ w.T),
            ("pretranspose", lambda x=x, w=w_pre:   x @ w),
        ):
            ms = bench(expr, iters, warmup)
            tflops = flops / (ms / 1000.0) / 1e12
            cells[(dtype_name, layout)] = (ms, tflops)

    # Print 2x2 table
    print(f"  {'':18s} {'naive':>18s} {'pretranspose':>18s}    {'pretrans Δ':>12s}")
    print(f"  {'':18s} {'(x @ w.T)':>18s} {'(x @ w_pt)':>18s}    {'vs naive':>12s}")
    print(f"  {'-'*18} {'-'*18} {'-'*18}    {'-'*12}")
    for dtype_name in ("BF16", "FP16"):
        n_ms, n_tf = cells[(dtype_name, "naive")]
        p_ms, p_tf = cells[(dtype_name, "pretranspose")]
        delta_pct = (n_ms - p_ms) / n_ms * 100.0
        verdict = "PRETRANS" if delta_pct > 1.0 else ("NAIVE" if delta_pct < -1.0 else "tied")
        print(f"  {dtype_name:>8s} ms       {n_ms:>15.2f}    {p_ms:>15.2f}     {delta_pct:>+8.1f}%  {verdict}")
        print(f"  {dtype_name:>8s} TFlops/s {n_tf:>15.2f}    {p_tf:>15.2f}")

    # FP16/BF16 ratios per layout
    print()
    for layout in ("naive", "pretranspose"):
        bf_ms = cells[("BF16", layout)][0]
        fp_ms = cells[("FP16", layout)][0]
        ratio = bf_ms / fp_ms
        delta = (bf_ms - fp_ms) / bf_ms * 100.0
        print(f"  FP16/BF16 @ {layout:>12s}: BF16/FP16 = {ratio:.2f}x  (FP16 saves {delta:+.1f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=14640, help="batch*tokens for VIDEO shapes (default 14640 = LTX stage-2 video)")
    p.add_argument("--m-audio", type=int, default=502, help="batch*tokens for AUDIO shapes (default 502 = LTX bakery T_audio)")
    p.add_argument("--iters", type=int, default=50, help="timed iterations per cell")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations per cell")
    p.add_argument(
        "--k-sweep",
        action="store_true",
        help=(
            "After the named shapes, sweep K at fine granularity over the project_out "
            "direction (K -> N=4096) to bracket the kernel-selection cliff threshold.  "
            "Default K-list: 10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432.  "
            "Adds ~2 minutes at default iters/warmup."
        ),
    )
    p.add_argument(
        "--k-sweep-only",
        action="store_true",
        help="Run ONLY the K-sweep (skip the named video/audio shapes).  Useful for quick re-runs.",
    )
    p.add_argument(
        "--k-sweep-list",
        type=str,
        default=None,
        help=(
            "Comma-separated K values for the sweep, overrides the default list.  "
            "Example: --k-sweep-list 14336,15360,16384,17408"
        ),
    )
    args = p.parse_args()

    print(f"\nLTX-2.3 distilled production shapes")
    print(f"  video M={args.m}    audio M={args.m_audio}")
    print(f"iters={args.iters} warmup={args.warmup}")
    print()
    print("Naive    : stock MLX Linear layout, op is `x @ weight.T` with strided view")
    print("Pretrans : LTX-2-MLX cache layout, op is `x @ pretransposed` (contiguous, no .T)")
    print()
    print("Each shape runs all 4 cells: {BF16, FP16} x {naive, pretranspose}")
    print("Reports both within-dtype layout effect and within-layout dtype effect.")
    print()
    print("Historical 2026-05-17 BF16 finding: project_out is +35% pretranspose")
    print("(kernel cliff at K=16384), project_in neutral, attn tied within 1%.")
    print("Open question this run answers: does FP16 share or escape the cliff?")

    if not args.k_sweep_only:
        run_shape(
            "FF.project_in    (production: K=4096 N=16384)",
            args.m, 4096, 16384, args.iters, args.warmup,
        )
        run_shape(
            "FF.project_out   (production: K=16384 N=4096)   <- historical BF16 35% cliff",
            args.m, 16384, 4096, args.iters, args.warmup,
        )
        run_shape(
            "attn.to_q/k/v    (K=4096 N=4096, square)",
            args.m, 4096, 4096, args.iters, args.warmup,
        )

        # Diagnostic shapes to confirm shape-sensitivity.
        run_shape(
            "DIAGNOSTIC: K=4096 N=10240",
            args.m, 4096, 10240, args.iters, args.warmup,
        )
        run_shape(
            "DIAGNOSTIC: K=10240 N=4096",
            args.m, 10240, 4096, args.iters, args.warmup,
        )

        # AUDIO shapes (smaller M=502, smaller K_max=8192).  Historical note from
        # PERFORMANCE_NOTES.md: audio pretranspose was neutral within 1% at bakery
        # T=502 in BF16, but -11.2% (a win) at small-T workloads (256x256x25)
        # because dispatch cost matters more there.  Open question: does FP16
        # have a smaller cliff at K=8192 the way video does at K=16384?
        print()
        print("#" * 86)
        print("#  AUDIO shapes (M = T_audio = 502, half-size hidden + FF dims)")
        print("#" * 86)

        run_shape(
            "audio.FF.project_in    (K=2048 N=8192)",
            args.m_audio, 2048, 8192, args.iters, args.warmup,
        )
        run_shape(
            "audio.FF.project_out   (K=8192 N=2048)   <- candidate audio cliff",
            args.m_audio, 8192, 2048, args.iters, args.warmup,
        )
        run_shape(
            "audio.attn.to_q/k/v    (K=2048 N=2048, square)",
            args.m_audio, 2048, 2048, args.iters, args.warmup,
        )

    # K-SWEEP: bracket the cliff threshold by fine-grained K samples at the
    # project_out direction (K -> N=4096, M=14640).  The two known datapoints
    # from `bench_pretranspose_dtype.py` (without --k-sweep) are K=10240 (~+2 %
    # pretranspose BF16, +14 % FP16) and K=16384 (+28 % BF16, +48 % FP16).
    # Goal: find where in between the cliff starts and how steeply it deepens.
    if args.k_sweep or args.k_sweep_only:
        if args.k_sweep_list:
            k_list = [int(x) for x in args.k_sweep_list.split(",") if x.strip()]
        else:
            k_list = [10240, 11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432]
        print()
        print("#" * 86)
        print(f"#  K-SWEEP at project_out direction:  M={args.m}, N=4096, K varies")
        print(f"#  K values: {k_list}")
        print("#" * 86)
        for k in k_list:
            run_shape(
                f"K-SWEEP: K={k:>5d} N=4096",
                args.m, k, 4096, args.iters, args.warmup,
            )

    print()
    print("=" * 86)
    print("Interpretation:")
    print("  - 'pretrans Δ' answers: at this dtype, is pretranspose still pulling weight?")
    print("  - 'FP16/BF16 ratio' answers: at this layout, how much does FP16 save?")
    print("  - If FP16 has its own cliff, expect a different layout verdict for FP16")
    print("    rows than BF16 rows on FF.project_out.")
    print("=" * 86)
    print()


if __name__ == "__main__":
    main()
