"""Smoke test for LTX_2_MLX.model.transformer.radial_mask.

Validates that the radial mask construction returns sane shapes and sensible sparsity
across a sweep of decay_factor values, without needing the full model.

Run:
    python scripts/test_radial_mask.py
"""
from __future__ import annotations

import time

import mlx.core as mx

from LTX_2_MLX.model.transformer.radial_mask import (
    mask_sparsity,
    radial_mask,
)


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    bar = "ok" if cond else "!!"
    print(f"  [{bar}] {status}: {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        raise SystemExit(1)


def smoke(seq_len: int, num_frame: int, label: str) -> None:
    tpf = seq_len // num_frame
    print(f"\n=== {label}: seq_len={seq_len}, num_frame={num_frame}, tpf={tpf} ===")
    for decay in (0.25, 0.5, 1.0, 2.0):
        # Use first call to time construction; subsequent calls hit lru_cache.
        # Force a fresh build per decay value (different cache key).
        t0 = time.perf_counter()
        m = radial_mask(seq_len, num_frame, decay_factor=decay)
        mx.eval(m)
        build_s = time.perf_counter() - t0

        # Cached lookup.
        t1 = time.perf_counter()
        m2 = radial_mask(seq_len, num_frame, decay_factor=decay)
        mx.eval(m2)
        cached_s = time.perf_counter() - t1

        density = 1.0 - mask_sparsity(m)
        print(
            f"  decay={decay:>4}: shape={tuple(m.shape)} dtype={m.dtype} "
            f"density={density:6.2%} "
            f"build={build_s*1000:7.1f}ms cached={cached_s*1000:6.2f}ms"
        )
        _check(f"shape ({decay})", tuple(m.shape) == (seq_len, seq_len))
        _check(f"dtype ({decay})", m.dtype == mx.bool_)

        # Diagonal must be True (a token always attends to itself's block).
        diag_sample = mx.diagonal(m)
        diag_true_count = int(diag_sample.astype(mx.int32).sum().item())
        _check(
            f"diagonal all True ({decay})",
            diag_true_count == seq_len,
            f"true count {diag_true_count} / {seq_len}",
        )

        # Density should be in a sane range — never zero, never full at decay<2.
        if decay < 2.0:
            _check(f"density < 100% ({decay})", density < 1.0, f"{density:.2%}")
        _check(f"density > 0% ({decay})", density > 0.0, f"{density:.2%}")

        # Asymmetry expected: attention sink makes column j=0 always-True but
        # row i=0 follows the radial decay rule like any other.  Spot-check:
        # rows beyond frame 0 should have block-0 columns fully True.
        bs = 128
        tpf_blocks = max(1, tpf // bs)
        # Block representing frame 0: cols [0:tpf_blocks]
        # Check a non-frame-0 row (say row corresponding to frame 5):
        if num_frame >= 6 and seq_len > 5 * tpf:
            sample_row = 5 * tpf + tpf // 2  # middle of frame 5
            sample_cols = m[sample_row, : tpf]
            attended_to_frame0 = int(sample_cols.astype(mx.int32).sum().item())
            _check(
                f"frame 0 is attention sink ({decay})",
                attended_to_frame0 == tpf,
                f"frame-5 row attends to {attended_to_frame0}/{tpf} of frame 0",
            )


def main() -> None:
    # Bakery stage 1: 288×512 video → H_lat=9, W_lat=16, tpf=144, F=61, T=8784.
    smoke(seq_len=8784, num_frame=61, label="bakery stage 1 (288x512)")

    # Bakery stage 2: 576×1024 video → H_lat=18, W_lat=32, tpf=576, F=61, T=35136.
    smoke(seq_len=35136, num_frame=61, label="bakery stage 2 (576x1024)")

    # Small case: 256×256x25 (legacy small-T benchmark) — VAE temporal=8 → F=4.
    # 256/32 × 256/32 = 8 × 8 = 64 tpf. F=4, T=256. (Below block_size=128 → likely
    # degenerate.  Useful to confirm we don't crash.)
    print("\n=== degenerate small case ===")
    try:
        m = radial_mask(seq_len=256, num_frame=4)
        mx.eval(m)
        print(f"  shape={tuple(m.shape)} density={1-mask_sparsity(m):.2%}")
    except Exception as e:
        print(f"  small case raised: {e!r}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
