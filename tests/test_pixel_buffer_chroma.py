"""Chroma-fidelity net for the CVPixelBuffer upload path (pixel_buffers.py).

upload_frame_to_buffer is where a decoded frame crosses into CoreVideo: a direct
fp16 memcpy for RGBAHalf, or a CoreImage sRGB->BT.709 render for NV12 (4:2:0,
chroma-subsampled). The numpy -> MLX-native rewrite must not perturb a single
channel value, so this pins the upload -> read-back round trip (sha256 of the
recovered RGB) for every (frame dtype, destination format) pair.

Each case runs with BOTH a numpy frame and the equivalent mlx frame: the hash
must match the golden for both, which proves (a) chroma is unchanged vs the
numpy implementation and (b) the mlx path is byte-faithful to the numpy path.
Captured from the numpy implementation. Skipped where pyobjc/VideoToolbox is
unavailable.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.videotoolbox import pixel_buffers as pb

_H = _W = 8


def _have_pyobjc() -> bool:
    try:
        from LTX_2_MLX.videotoolbox import _compat
        _compat.require_pyobjc()
        return True
    except Exception:
        return False


def _np_frame(kind: str) -> np.ndarray:
    if kind == "u8":
        return (np.arange(_H * _W * 3, dtype=np.int32) % 256).astype(np.uint8).reshape(_H, _W, 3)
    return (np.arange(_H * _W * 4, dtype=np.float32) / (_H * _W * 4)).astype(np.float16).reshape(_H, _W, 4)


def _make_buffer(fmt: int):
    from LTX_2_MLX.videotoolbox._compat import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: fmt,
        Quartz.kCVPixelBufferWidthKey: _W,
        Quartz.kCVPixelBufferHeightKey: _H,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    return pb.make_pixel_buffer_from_attrs(_W, _H, attrs)


# (case id, frame kind, destination format name, golden sha256[:24])
_CASES = [
    ("u8_to_nv12", "u8", "PIX_NV12", "54343cfac1b73415c6894809"),
    ("u8_to_rgbahalf", "u8", "PIX_RGBAHALF", "8b4a544837a1a0280fa8a7c8"),
    ("f16_to_nv12", "f16", "PIX_NV12", "4036f9b99ded79ff15df0cb9"),
    ("f16_to_rgbahalf", "f16", "PIX_RGBAHALF", "b0aa18ac9534756d56021442"),
]


@pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc / VideoToolbox unavailable")
@pytest.mark.parametrize("container", ["numpy", "mlx"])
@pytest.mark.parametrize("case_id,kind,fmt_name,golden", _CASES, ids=[c[0] for c in _CASES])
def test_upload_chroma_roundtrip(container, case_id, kind, fmt_name, golden):
    frame = _np_frame(kind)
    if container == "mlx":
        frame = mx.array(frame)
    buf = _make_buffer(getattr(pb, fmt_name))
    pb.upload_frame_to_buffer(frame, buf)
    rgb = pb.read_pixel_buffer_rgb(buf)
    assert hashlib.sha256(bytes(memoryview(mx.contiguous(rgb)))).hexdigest()[:24] == golden
