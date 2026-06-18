"""Render-golden net for the videotoolbox comparison composite (numpy -> MLX).

render_comparison NEAREST-upscales the pre frame and composites it beside the VSR
output via CoreImage. The numpy -> MLX rewrite must not change a pixel, so this
pins the upload -> render -> read-back hash for a uint8 and an fp16 pre frame,
with both a numpy and an mlx input. Captured from the numpy implementation.
Skipped where pyobjc / VideoToolbox is unavailable.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from LTX_2_MLX.videotoolbox import comparison
from LTX_2_MLX.videotoolbox import pixel_buffers as pb


def _have_pyobjc() -> bool:
    try:
        from LTX_2_MLX.videotoolbox import _compat
        _compat.require_pyobjc()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc / VideoToolbox unavailable")


def _make_buf(fmt, w, h):
    from LTX_2_MLX.videotoolbox._compat import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: fmt,
        Quartz.kCVPixelBufferWidthKey: w,
        Quartz.kCVPixelBufferHeightKey: h,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    return pb.make_pixel_buffer_from_attrs(w, h, attrs)


def _render_sha(pre):
    scale = 2
    out_w, out_h = 4 * scale, 4 * scale  # pre is 4x4 -> 8x8
    post = (np.arange(out_h * out_w * 4, dtype=np.float32) / (out_h * out_w * 4)).astype(
        np.float16
    ).reshape(out_h, out_w, 4)
    post_pb = _make_buf(pb.PIX_RGBAHALF, out_w, out_h)
    pb.upload_frame_to_buffer(post, post_pb)
    dest_pb = _make_buf(pb.PIX_BGRA, 2 * out_w, out_h)
    comparison.render_comparison(pre, post_pb, scale, dest_pb)
    rgb = pb.read_pixel_buffer_rgb(dest_pb)
    return hashlib.sha256(rgb.tobytes()).hexdigest()[:24]


_PRE_U8 = (np.arange(4 * 4 * 3, dtype=np.int32) % 256).astype(np.uint8).reshape(4, 4, 3)
_PRE_F16 = (np.arange(4 * 4 * 4, dtype=np.float32) / 64).astype(np.float16).reshape(4, 4, 4)


@pytest.mark.parametrize("container", ["numpy", "mlx"])
@pytest.mark.parametrize(
    "pre,golden",
    [(_PRE_U8, "021cffe4065015b63d29b0c3"), (_PRE_F16, "970a9670456d4aa4877441e2")],
    ids=["u8", "f16"],
)
def test_render_comparison_golden(container, pre, golden):
    frame = mx.array(pre) if container == "mlx" else pre
    assert _render_sha(frame) == golden
