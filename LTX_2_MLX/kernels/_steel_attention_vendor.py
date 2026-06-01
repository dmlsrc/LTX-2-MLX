"""Vendored MLX STEEL attention source snapshot.

Generated from Apple MLX commit 2165dc08d7b33258260aa849d39f087d50e62962.
The embedded Metal source is derived from MLX's STEEL attention headers and
is kept as the full-source retile fallback for local bisects.  The default
runtime path uses the compact LTX-specific Metal resources in `metal/`.
"""

from pathlib import Path

MLX_LICENSE = 'MIT License\n\nCopyright © 2023 Apple Inc.\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.'
MLX_SOURCE_COMMIT = '2165dc08d7b33258260aa849d39f087d50e62962'

_RESOURCE_DIR = Path(__file__).with_name("metal")
__all__ = ["HEADER", "SOURCE", "MLX_LICENSE", "MLX_SOURCE_COMMIT"]


def _read_metal(name: str) -> str:
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


HEADER = _read_metal("steel_attention_vendor_header.metal")
SOURCE = _read_metal("steel_attention_vendor_body.metal")
