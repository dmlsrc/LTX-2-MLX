"""MLX STEEL attention tile override for supported LTX-2.3 shapes.

The default path uses a lean BF16-only MLX STEEL subset specialized for the
no-mask LTX-2.3 hot path.  Unsupported shapes and FP16 calls fall back to stock
MLX SDPA in the caller; ``LTX_STEEL_ATTN_IMPL=compact`` and ``retile`` keep the
older compact subset and full vendored MLX snapshot available for bisects.
"""

from __future__ import annotations

import atexit
import os
from typing import Optional

import mlx.core as mx


_BQ = 64
_BK = 32
_WM = 8
_THREADGROUP = (32, _WM, 1)

_SUPPORTED_DIMS = {64, 128}
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
_VALID_IMPLS = {"lean", "compact", "retile"}


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


def _probe_hit(bd: int) -> None:
    if _PROBE_ENABLED:
        _PROBE_COUNTS[f"hit_d{bd}"] += 1


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
    impl = os.environ.get("LTX_STEEL_ATTN_IMPL", "lean").strip().lower()
    if impl not in _VALID_IMPLS:
        valid = ", ".join(sorted(_VALID_IMPLS))
        raise ValueError(f"Unsupported LTX_STEEL_ATTN_IMPL={impl!r}; use {valid}.")
    return impl


def _kernel_resources(impl: str) -> tuple[str, str]:
    if impl == "lean":
        from ._steel_attention_ltx_lean import HEADER, SOURCE
    elif impl == "compact":
        from ._steel_attention_ltx import HEADER, SOURCE
    else:
        from ._steel_attention_vendor import HEADER, SOURCE
    return HEADER, SOURCE


def _kernel(impl: str):
    if impl not in _KERNEL_CACHE:
        header, source = _kernel_resources(impl)
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
    impl: str,
) -> tuple[Optional[int], str]:
    if mask is not None:
        return None, "mask"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return None, "ndim"
    if q.dtype not in (mx.bfloat16, mx.float16):
        return None, "dtype"
    if impl == "lean" and q.dtype != mx.bfloat16:
        return None, "dtype_lean_bf16"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return None, "dtype_mismatch"
    if q.shape[0] != 1:
        return None, "batch"
    if q.shape[1] != 32 or k.shape[1] != 32 or v.shape[1] != 32:
        return None, "heads"
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        return None, "dim_mismatch"
    bd = q.shape[-1]
    if bd not in _SUPPORTED_DIMS:
        return None, "dim"
    if bd == 64 and not _ENABLE_D64:
        return None, "d64_disabled"
    if k.shape[2] != v.shape[2]:
        return None, "kv_len"
    if q.shape[2] < 512 or k.shape[2] < 512:
        return None, "seq"
    if not _scale_supported(scale, q.shape[-1]):
        return None, "scale"
    return bd, ""


def maybe_steel_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Return custom STEEL attention output when this call matches the gate."""
    impl = _kernel_impl()
    bd, reason = _select_config(q, k, v, scale, mask, impl)
    if bd is None:
        _probe_fallback(reason, q, k, v, mask)
        return None

    batch, heads, seq, dim = q.shape
    n_q_tiles = (seq + _BQ - 1) // _BQ
    align_q = (seq % _BQ) == 0
    align_k = (k.shape[2] % _BK) == 0
    template = [
        ("BD", bd),
        ("AlignQ", align_q),
        ("AlignK", align_k),
    ]
    if impl != "lean":
        template = [
            ("T", q.dtype),
            ("BQ", _BQ),
            ("BK", _BK),
            ("BD", bd),
            ("WM", _WM),
            ("WN", 1),
            ("AlignQ", align_q),
            ("AlignK", align_k),
            ("DoCausal", False),
        ]
    out = _kernel(impl)(
        inputs=[q, k, v],
        output_shapes=[(batch, seq, heads, dim)],
        output_dtypes=[q.dtype],
        grid=(n_q_tiles * 32, heads * _WM, batch),
        threadgroup=_THREADGROUP,
        template=template,
    )[0]
    _probe_hit(bd)
    return out.transpose(0, 2, 1, 3)
