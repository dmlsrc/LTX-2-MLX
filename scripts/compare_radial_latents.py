"""Compare latent .npz sidecars from baseline vs radial-attention A/B runs.

Usage:
    python scripts/compare_radial_latents.py BASELINE.npz CANDIDATE.npz [CANDIDATE2.npz ...]

Reports cosine similarity, relative L2, and MAE per latent key.  Used in
`docs/PERFORMANCE.md`'s radial A/B sessions.
"""
from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np


def load_floats(path: str, key: str) -> np.ndarray:
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f"{path}: no key {key!r} (have: {sorted(data.files)})")
    arr = data[key]
    if arr.dtype.kind != "f":
        raise TypeError(f"{path}::{key} dtype {arr.dtype} is not float")
    return arr.astype(np.float32)


def compare(base: np.ndarray, cand: np.ndarray) -> tuple[float, float, float]:
    a = base.ravel()
    b = cand.ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {base.shape} vs {cand.shape}")
    base_norm = float(np.linalg.norm(a)) + 1e-12
    cand_norm = float(np.linalg.norm(b)) + 1e-12
    cos = float(np.dot(a, b) / (base_norm * cand_norm))
    rel_l2 = float(np.linalg.norm(a - b) / base_norm)
    mae = float(np.mean(np.abs(a - b)))
    return cos, rel_l2, mae


def label_from_path(path: str) -> str:
    base = os.path.basename(path).replace(".npz", "")
    return base[:70]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline latent .npz")
    parser.add_argument("candidates", nargs="+", help="One or more radial latent .npz")
    parser.add_argument(
        "--key",
        default="final_video_latent",
        help="Latent key inside the .npz (default: final_video_latent)",
    )
    args = parser.parse_args(argv)

    base = load_floats(args.baseline, args.key)
    print(f"baseline: {label_from_path(args.baseline)}")
    print(f"  shape={base.shape} dtype={base.dtype}")
    print()
    width = max(len(label_from_path(p)) for p in args.candidates)
    print(f"{'candidate':<{width}}  {'cos':>8s}  {'relL2':>7s}  {'mae':>7s}  verdict")
    print("-" * (width + 40))
    for cand_path in args.candidates:
        try:
            cand = load_floats(cand_path, args.key)
            cos, rel, mae = compare(base, cand)
        except (KeyError, TypeError, ValueError) as e:
            print(f"{label_from_path(cand_path):<{width}}  ERR: {e}")
            continue
        # Quality verdict bands tuned for distilled LTX 2.3 latents:
        # ≥0.999 = bit-identical-ish (BF16 noise floor)
        # ≥0.99  = strong match, likely indistinguishable visually
        # ≥0.95  = acceptable for quality probe
        # <0.95  = material degradation
        if cos >= 0.999:
            verdict = "≈ baseline"
        elif cos >= 0.99:
            verdict = "STRONG match"
        elif cos >= 0.95:
            verdict = "acceptable"
        elif cos >= 0.90:
            verdict = "MARGINAL"
        else:
            verdict = "DEGRADED"
        print(
            f"{label_from_path(cand_path):<{width}}  "
            f"{cos:8.5f}  {rel:7.4f}  {mae:7.5f}  {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
