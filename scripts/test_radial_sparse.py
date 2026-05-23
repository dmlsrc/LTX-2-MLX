"""Equivalence + speed test for radial_sparse_attention (path b).

Validates:
- Block-sparse SDPA produces math-equivalent output to dense SDPA + token mask
  (modulo BF16 rounding-order drift; cos sim >= 0.999 expected).
- Reports per-call timing for sparse vs dense at LTX shapes.

Run:
    PYTHONPATH=. python scripts/test_radial_sparse.py
"""
from __future__ import annotations

import time

import mlx.core as mx

from LTX_2_MLX.model.transformer.radial_mask import (
    radial_block_mask,
    radial_mask,
)
from LTX_2_MLX.model.transformer.radial_sparse import (
    precompute_active_kblocks,
    radial_sparse_attention,
)


def _allclose_report(a: mx.array, b: mx.array, label: str, tol_cos: float = 0.999) -> None:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    cos = float(
        (af * bf).sum().item()
        / (mx.sqrt((af * af).sum()).item() * mx.sqrt((bf * bf).sum()).item() + 1e-12)
    )
    mae = float(mx.abs(af - bf).mean().item())
    max_err = float(mx.abs(af - bf).max().item())
    status = "PASS" if cos >= tol_cos else "FAIL"
    print(
        f"  [{status}] {label}: cos={cos:.6f}  mae={mae:.5f}  max_err={max_err:.5f}"
    )
    if cos < tol_cos:
        raise SystemExit(1)


def _bench_call(fn, *, warmup: int = 2, iters: int = 5) -> float:
    """Return mean wall time in seconds across `iters` iterations after warmup."""
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def smoke(seq_len: int, num_frame: int, label: str, heads: int = 32, dim: int = 128) -> None:
    print(f"\n=== {label}: T={seq_len}, F={num_frame}, H={heads}, D={dim} ===")

    # Random Q/K/V — independent across seeds for both paths.
    rng = mx.random.key(42)
    rng, k1, k2, k3 = mx.random.split(rng, 4)
    q = mx.random.normal((1, heads, seq_len, dim), key=k1, dtype=mx.bfloat16)
    k = mx.random.normal((1, heads, seq_len, dim), key=k2, dtype=mx.bfloat16)
    v = mx.random.normal((1, heads, seq_len, dim), key=k3, dtype=mx.bfloat16)
    scale = 1.0 / (dim ** 0.5)

    # Path-a reference: full QK^T with token-level boolean mask.
    token_mask = radial_mask(seq_len, num_frame, decay_factor=0.5, model_type="ltx")
    # Path-b candidate.
    block_mask = radial_block_mask(seq_len, num_frame, decay_factor=0.5, model_type="ltx")
    active = precompute_active_kblocks(seq_len, num_frame, decay_factor=0.5, model_type="ltx")
    density = float(block_mask.astype(mx.float32).sum().item()) / float(block_mask.size)
    print(f"  block_mask shape={tuple(block_mask.shape)}  density={density:.2%}")

    # Equivalence check
    out_dense_masked = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=token_mask  # bool pass-through; MLX SDPA handles natively
    )
    out_sparse = radial_sparse_attention(
        q, k, v, scale=scale, block_mask=block_mask, active_per_qblock=active
    )
    mx.eval(out_dense_masked, out_sparse)
    _allclose_report(out_dense_masked, out_sparse, "sparse == dense+token_mask")

    # Speed comparison
    def run_dense_no_mask():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    def run_dense_masked():
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=token_mask  # bool pass-through; MLX SDPA handles natively
        )

    def run_sparse():
        return radial_sparse_attention(
            q, k, v, scale=scale, block_mask=block_mask, active_per_qblock=active
        )

    t_dense = _bench_call(run_dense_no_mask)
    t_masked = _bench_call(run_dense_masked)
    t_sparse = _bench_call(run_sparse)
    speedup_vs_dense = t_dense / t_sparse if t_sparse > 0 else 0.0
    print(
        f"  dense no-mask  : {t_dense*1000:8.2f} ms\n"
        f"  dense + token mask (path a): {t_masked*1000:8.2f} ms\n"
        f"  block-sparse (path b)      : {t_sparse*1000:8.2f} ms  "
        f"({speedup_vs_dense:.2f}x vs dense)"
    )


def main() -> None:
    # Bakery stage 1 attention shape — the primary target.
    smoke(seq_len=8784, num_frame=61, label="bakery stage 1 (288x512)")

    # 5s smoke shape (matches our path-a A/B runs).
    smoke(seq_len=2304, num_frame=16, label="5s smoke (288x512)")

    # Bakery stage 2 — the prize, but 1.2 GB token mask, so skipped unless explicit.
    # smoke(seq_len=35136, num_frame=61, label="bakery stage 2 (576x1024)")

    print("\nAll equivalence checks passed.")


if __name__ == "__main__":
    main()
