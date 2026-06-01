"""MLX STEEL attention tile override for supported LTX-2.3 shapes.

The default path uses a compact MLX STEEL subset specialized for the no-mask
LTX-2.3 hot path.  Unsupported shapes fall back to stock MLX SDPA in the
caller; ``LTX_STEEL_ATTN_IMPL=retile`` keeps the full vendored MLX snapshot
available as a local bisect fallback.
"""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from ._steel_attention_ltx import (
    HEADER as _LTX_HEADER,
    SOURCE as _LTX_SOURCE,
)
from ._steel_attention_vendor import (
    HEADER as _VENDORED_HEADER,
    SOURCE as _VENDORED_SOURCE,
)


@dataclass(frozen=True)
class _TileConfig:
    bq: int
    bk: int
    bd: int
    wm: int
    wn: int = 1

    @property
    def threads(self) -> tuple[int, int, int]:
        return (32, self.wm, self.wn)


_CONFIGS = {
    64: _TileConfig(bq=64, bk=32, bd=64, wm=8),
    128: _TileConfig(bq=64, bk=32, bd=128, wm=8),
}
_PROBE_ENABLED = bool(os.environ.get("LTX_STEEL_ATTN_PROBE"))
# D64 no-mask shapes were neutral in isolation but won in full 8+3 AV runs.
# Keep a local escape hatch for quick bisects without editing code.
_ENABLE_D64 = not bool(os.environ.get("LTX_STEEL_ATTN_DISABLE_D64"))
_PROBE_COUNTS = {
    "hit_d64": 0,
    "hit_d128": 0,
    "fallback": 0,
}
_PROBE_REASONS: dict[str, int] = {}
_PROBE_SAMPLES: dict[str, str] = {}
_KERNEL_CACHE: dict[str, object] = {}
_VALID_IMPLS = {"compact", "retile"}


def _shape_sample(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array],
) -> str:
    mask_shape = None if mask is None else tuple(mask.shape)
    return (
        f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"mask={mask_shape}"
    )


def _probe_hit(config: _TileConfig) -> None:
    if _PROBE_ENABLED:
        _PROBE_COUNTS[f"hit_d{config.bd}"] += 1


def _probe_fallback(
    reason: str,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: Optional[mx.array],
) -> None:
    if not _PROBE_ENABLED:
        return
    _PROBE_COUNTS["fallback"] += 1
    _PROBE_REASONS[reason] = _PROBE_REASONS.get(reason, 0) + 1
    _PROBE_SAMPLES.setdefault(reason, _shape_sample(q, k, v, mask))


def _print_probe_summary() -> None:
    if not _PROBE_ENABLED:
        return
    total = sum(_PROBE_COUNTS.values())
    if total == 0:
        return
    print("\n[LTX_STEEL_ATTN_PROBE] selection summary:")
    for name in ("hit_d128", "hit_d64", "fallback"):
        n = _PROBE_COUNTS[name]
        pct = (100.0 * n / total) if total else 0.0
        print(f"  {name:<10} {n:>8}  ({pct:5.1f}%)")
    if _PROBE_REASONS:
        print("  fallback reasons:")
        for reason, count in sorted(
            _PROBE_REASONS.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"    {reason:<14} {count:>8}  sample: {_PROBE_SAMPLES[reason]}")


if _PROBE_ENABLED:
    atexit.register(_print_probe_summary)


def _kernel_impl() -> str:
    impl = os.environ.get("LTX_STEEL_ATTN_IMPL", "compact").strip().lower()
    if impl not in _VALID_IMPLS:
        valid = ", ".join(sorted(_VALID_IMPLS))
        raise ValueError(f"Unsupported LTX_STEEL_ATTN_IMPL={impl!r}; use {valid}.")
    return impl


def _kernel(impl: str):
    if impl not in _KERNEL_CACHE:
        if impl == "compact":
            header = _LTX_HEADER
            source = _LTX_SOURCE
        else:
            header = _VENDORED_HEADER
            source = _VENDORED_SOURCE
        _KERNEL_CACHE[impl] = mx.fast.metal_kernel(
            name=f"ltx_steel_attention_bq64_bk32_{impl}",
            input_names=["Q", "K", "V"],
            output_names=["O"],
            source=source,
            header=header,
            ensure_row_contiguous=False,
        )
    return _KERNEL_CACHE[impl]


def _scale_supported(scale: Optional[float], dim: int) -> bool:
    expected = 1.0 / (dim**0.5)
    return scale is None or abs(float(scale) - expected) < 1e-7


def _select_config(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float],
    mask: Optional[mx.array],
) -> tuple[Optional[_TileConfig], str]:
    if mask is not None:
        return None, "mask"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None, "ndim"
    if q.dtype not in (mx.bfloat16, mx.float16):
        return None, "dtype"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return None, "dtype_mismatch"
    if q.shape[0] != 1:
        return None, "batch"
    if q.shape[1] != 32 or k.shape[1] != 32 or v.shape[1] != 32:
        return None, "heads"
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        return None, "dim_mismatch"
    config = _CONFIGS.get(q.shape[-1])
    if config is None:
        return None, "dim"
    if config.bd == 64 and not _ENABLE_D64:
        return None, "d64_disabled"
    if k.shape[2] != v.shape[2]:
        return None, "kv_len"
    if q.shape[2] < 512 or k.shape[2] < 512:
        return None, "seq"
    if not _scale_supported(scale, q.shape[-1]):
        return None, "scale"
    return config, ""


def maybe_steel_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Return custom STEEL attention output when this call matches the gate."""
    config, reason = _select_config(q, k, v, scale, mask)
    if config is None:
        _probe_fallback(reason, q, k, v, mask)
        return None

    impl = _kernel_impl()
    batch, heads, seq, dim = q.shape
    n_q_tiles = (seq + config.bq - 1) // config.bq
    out = _kernel(impl)(
        inputs=[q, k, v],
        output_shapes=[(batch, seq, heads, dim)],
        output_dtypes=[q.dtype],
        grid=(n_q_tiles * 32, heads * config.wm, batch * config.wn),
        threadgroup=config.threads,
        template=[
            ("T", q.dtype),
            ("BQ", config.bq),
            ("BK", config.bk),
            ("BD", config.bd),
            ("WM", config.wm),
            ("WN", config.wn),
            ("AlignQ", (seq % config.bq) == 0),
            ("AlignK", (k.shape[2] % config.bk) == 0),
            ("DoCausal", False),
        ],
    )[0]
    _probe_hit(config)
    return out.transpose(0, 2, 1, 3)
