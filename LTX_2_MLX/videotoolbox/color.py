"""Source color detection + propagation.

The encoder must tag the output to match the *source* color, not a hard-coded
BT.709. Tagged containers carry explicit primaries/transfer/matrix; we read and
propagate those (reliable). Untagged containers are genuinely ambiguous -- no
tool can know for sure -- so we fall back to a documented default (BT.709,
overridable) rather than silently inheriting VideoToolbox's decode-time guess.

The container's full-range flag is a real field (present even when color is
untagged), so range is always read directly.
"""
from __future__ import annotations

from typing import Any

from ._compat import CoreMedia, Quartz, av

# --- canonical CV constant triples (primaries, transfer, matrix) -------------
_709 = (Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2,
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2)
_601 = (Quartz.kCVImageBufferColorPrimaries_SMPTE_C,            # NTSC 601 primaries
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,      # same SDR gamma as 709
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4)
_2020 = (Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,
         Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,     # SDR 2020 uses ~709 gamma
         Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020)
_OVERRIDES = {"bt709": _709, "bt601": _601, "bt2020": _2020}

# AV writer constants, matched to a CV value by shared CFString value (with a
# BT.709 fallback for anything AV doesn't expose, e.g. the SMPTE_C transfer).
_AV_PRIMS = [getattr(av, k) for k in dir(av) if k.startswith("AVVideoColorPrimaries_")]
_AV_TRANS = [getattr(av, k) for k in dir(av) if k.startswith("AVVideoTransferFunction_")]
_AV_MATS = [getattr(av, k) for k in dir(av) if k.startswith("AVVideoYCbCrMatrix_")]


def _match(cv_val: Any, av_list: list, fallback: Any) -> Any:
    return next((a for a in av_list if a == cv_val), fallback)


def read_source_color(format_desc: Any) -> dict:
    """Explicit color tags from a CM format description (None where untagged)."""
    ext = CoreMedia.CMFormatDescriptionGetExtensions(format_desc) or {}
    by = {str(k): ext[k] for k in ext}
    prim = by.get("CVImageBufferColorPrimaries")
    return {
        "primaries": prim,
        "transfer": by.get("CVImageBufferTransferFunction"),
        "matrix": by.get("CVImageBufferYCbCrMatrix"),
        "full_range": bool(by.get("FullRangeVideo", False)),
        "tagged": prim is not None,
    }


def resolve(src: dict, override: str = "auto") -> tuple:
    """(primaries, transfer, matrix, full_range) as CV constants.

    override 'auto' = explicit source tags, else BT.709 for untagged. Any other
    override forces that colorimetry (the source's range flag is still honored).
    """
    if override != "auto":
        prim, trans, mat = _OVERRIDES[override]
        return prim, trans, mat, src["full_range"]
    if src["tagged"]:
        return src["primaries"], src["transfer"], src["matrix"], src["full_range"]
    return (*_709, src["full_range"])


def cv_triple(resolved: tuple) -> tuple:
    """Just the (primaries, transfer, matrix) CV constants (for VTPixelTransfer)."""
    return resolved[0], resolved[1], resolved[2]


def av_color_properties(resolved: tuple) -> dict:
    """AVVideoColorPropertiesKey dict for the writer, mapped from CV constants."""
    prim, trans, mat, _ = resolved
    return {
        av.AVVideoColorPrimariesKey: _match(prim, _AV_PRIMS, av.AVVideoColorPrimaries_ITU_R_709_2),
        av.AVVideoTransferFunctionKey: _match(trans, _AV_TRANS, av.AVVideoTransferFunction_ITU_R_709_2),
        av.AVVideoYCbCrMatrixKey: _match(mat, _AV_MATS, av.AVVideoYCbCrMatrix_ITU_R_709_2),
    }


def describe(resolved: tuple) -> str:
    prim, _, mat, fr = resolved
    p = str(prim).replace("ITU_R_", "").replace("_2", "").replace("SMPTE_C", "601-C")
    m = str(mat).replace("ITU_R_", "").replace("_4", "").replace("_2", "")
    return f"primaries={p} matrix={m} range={'full' if fr else 'video'}"
