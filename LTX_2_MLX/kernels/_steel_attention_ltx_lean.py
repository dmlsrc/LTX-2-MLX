"""Lean BF16-only MLX STEEL subset resources for LTX-2.3 attention."""

from pathlib import Path

from ._steel_attention_vendor import MLX_LICENSE, MLX_SOURCE_COMMIT

_RESOURCE_DIR = Path(__file__).with_name("metal")
__all__ = ["HEADER", "SOURCE", "MLX_LICENSE", "MLX_SOURCE_COMMIT"]


def _read_metal(name: str) -> str:
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


HEADER = _read_metal("steel_attention_ltx_lean_header.metal")
SOURCE = _read_metal("steel_attention_ltx_lean_body.metal")
