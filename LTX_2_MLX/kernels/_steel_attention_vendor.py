"""Vendored MLX STEEL attention source snapshot.

Generated from Apple MLX commit 2165dc08d7b33258260aa849d39f087d50e62962.
The embedded Metal source is derived from MLX's STEEL attention headers and
is kept as the full-source retile fallback for local bisects.  The default
runtime path uses the compact LTX-specific Metal resources in `metal/`.
"""

from pathlib import Path

_RESOURCE_DIR = Path(__file__).with_name("metal")
__all__ = ["HEADER", "SOURCE"]


def _read_metal(name: str) -> str:
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


HEADER = _read_metal("steel_attention_vendor_header.metal")
SOURCE = _read_metal("steel_attention_vendor_body.metal")
