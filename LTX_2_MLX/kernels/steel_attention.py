"""Opt-in MLX STEEL attention tile override for local A/B testing.

This deliberately reads MLX's own local reference checkout and emits a
custom ``mx.fast.metal_kernel`` wrapper around the regular STEEL attention
body.  It is gated by the caller and intended for local perf experiments, not
as a portable packaged kernel.
"""

from __future__ import annotations

import atexit
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx


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
_HEADER_CACHE: dict[Path, str] = {}
_KERNEL_CACHE: dict[Path, object] = {}


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


def _reference_dir() -> Path:
    candidates: list[Path] = []
    if env_path := os.environ.get("LTX_STEEL_ATTN_MLX_REFERENCE"):
        candidates.append(Path(env_path))
    if env_path := os.environ.get("MLX_REFERENCE_DIR"):
        candidates.append(Path(env_path))
    candidates.append(Path("/Users/Shared/huggingface/reference/mlx"))

    for candidate in candidates:
        if (
            candidate
            / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
        ).exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "LTX_STEEL_ATTN=1 requires a local MLX reference checkout; tried "
        f"{tried}."
    )


def _find_matching_brace(source: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Could not find matching brace.")


def _remove_if_block(source: str, condition: str) -> str:
    start = source.find(condition)
    while start != -1:
        open_index = source.find("{", start)
        if open_index == -1:
            raise ValueError(f"Could not find opening brace for {condition}.")
        close_index = _find_matching_brace(source, open_index)
        source = source[:start] + source[close_index + 1 :]
        start = source.find(condition)
    return source


def _attention_body(root: Path) -> str:
    source = (
        root / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
    ).read_text()
    signature_start = source.find("void attention(")
    if signature_start == -1:
        raise ValueError("Could not find STEEL attention signature.")
    body_start = source.find("{", signature_start)
    if body_start == -1:
        raise ValueError("Could not find STEEL attention body.")
    body_end = _find_matching_brace(source, body_start)
    body = source[body_start + 1 : body_end]
    body = _remove_if_block(body, "if (has_mask)")
    body = _remove_if_block(body, "if (has_sinks)")
    for old, new in {
        "params->": "params.",
        "align_Q": "AlignQ",
        "align_K": "AlignK",
        "do_causal": "DoCausal",
    }.items():
        body = body.replace(old, new)
    return body


def _inline_header(root: Path) -> str:
    include_re = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"')
    seen: set[str] = set()
    seen_angle: set[str] = set()

    def inline(rel: str) -> str:
        if rel == "mlx/backend/metal/kernels/utils.h":
            return "\n// skipped custom-kernel auto include utils.h\n"
        path = root / rel
        key = str(path)
        if key in seen:
            return f"\n// skipped duplicate include {rel}\n"
        seen.add(key)

        chunks = [f"\n// BEGIN {rel}\n"]
        for line in path.read_text().splitlines():
            match = include_re.match(line)
            if match:
                chunks.append(inline(match.group(1)))
                continue
            if line.strip() == "#pragma once":
                continue
            if line.startswith("#include <"):
                if line not in seen_angle:
                    seen_angle.add(line)
                    chunks.append(line + "\n")
                continue
            chunks.append(line + "\n")
        chunks.append(f"// END {rel}\n")
        return "".join(chunks)

    return (
        "// LTX opt-in MLX STEEL attention wrapper header.\n"
        "#define MLX_METAL_JIT 1\n"
        + inline("mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h")
    )


def _source(root: Path) -> str:
    body = textwrap.indent(_attention_body(root), "  ").lstrip()
    prefix = r"""
  const int B = Q_shape[0];
  const int H = Q_shape[1];
  const int qL = Q_shape[2];
  const int D = Q_shape[3];
  const int kL = K_shape[2];

  AttnParams params;
  params.B = B;
  params.H = H;
  params.D = D;
  params.qL = qL;
  params.kL = kL;
  params.gqa_factor = H / K_shape[1];
  params.scale = 1.0f / sqrt(float(D));
  params.NQ = (qL + BQ - 1) / BQ;
  params.NK = (kL + BK - 1) / BK;
  params.NQ_aligned = qL / BQ;
  params.NK_aligned = kL / BK;
  params.qL_rem = qL - params.NQ_aligned * BQ;
  params.kL_rem = kL - params.NK_aligned * BK;
  params.qL_off = 0;
  params.Q_strides[0] = Q_strides[0];
  params.Q_strides[1] = Q_strides[1];
  params.Q_strides[2] = Q_strides[2];
  params.K_strides[0] = K_strides[0];
  params.K_strides[1] = K_strides[1];
  params.K_strides[2] = K_strides[2];
  params.V_strides[0] = V_strides[0];
  params.V_strides[1] = V_strides[1];
  params.V_strides[2] = V_strides[2];
  params.O_strides[0] = qL * H * D;
  params.O_strides[1] = D;
  params.O_strides[2] = H * D;

  uint simd_lane_id = thread_index_in_simdgroup;
  uint simd_group_id = simdgroup_index_in_threadgroup;
  uint3 tid = threadgroup_position_in_grid;
  uint3 lid = thread_position_in_threadgroup;

  using AccumType = float;

"""
    return prefix + body


def _kernel(root: Path):
    root = root.resolve()
    if root not in _KERNEL_CACHE:
        if root not in _HEADER_CACHE:
            _HEADER_CACHE[root] = _inline_header(root)
        _KERNEL_CACHE[root] = mx.fast.metal_kernel(
            name="ltx_steel_attention_bq64_bk32",
            input_names=["Q", "K", "V"],
            output_names=["O"],
            source=_source(root),
            header=_HEADER_CACHE[root],
            ensure_row_contiguous=False,
        )
    return _KERNEL_CACHE[root]


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
    """Return custom STEEL attention output when this call matches the A/B gate."""
    config, reason = _select_config(q, k, v, scale, mask)
    if config is None:
        _probe_fallback(reason, q, k, v, mask)
        return None

    root = _reference_dir()
    batch, heads, seq, dim = q.shape
    n_q_tiles = (seq + config.bq - 1) // config.bq
    out = _kernel(root)(
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
