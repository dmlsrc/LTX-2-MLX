"""Resolve a bundled-model variant token (or a path) to a concrete weights file.

The learned upscalers ship several checkpoints in their package `weights/` dir, so
callers can pass a short token like ``x4plus`` instead of a full .safetensors path.
Each net keeps its own ``_VARIANTS`` map and a one-line ``resolve_weights`` wrapper
around this helper.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_weights(spec: Any, variants: dict, weights_dir: Path, default: str) -> Path:
    """Turn a weights spec into a concrete file path.

    - ``None`` / empty  -> the ``default`` variant's file.
    - a known variant token -> that variant's file (FileNotFoundError if it is missing,
      e.g. a not-bundled checkpoint that has not been downloaded/converted yet).
    - an existing path, or a bare filename present in ``weights_dir`` -> that file.
    - anything else -> FileNotFoundError listing the valid tokens.
    """
    if spec is None or spec == "":
        spec = default
    spec = str(spec)
    if spec in variants:
        vp = weights_dir / variants[spec]
        if vp.is_file():
            return vp
        raise FileNotFoundError(
            f"weights variant {spec!r} maps to {vp}, which does not exist")
    p = Path(spec).expanduser()
    if p.is_file():
        return p
    if (weights_dir / spec).is_file():
        return weights_dir / spec
    raise FileNotFoundError(
        f"weights {spec!r}: not a known variant {sorted(variants)} and not an "
        f"existing file (also looked in {weights_dir})")
