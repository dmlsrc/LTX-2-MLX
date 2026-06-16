"""Side-by-side composite for `comparison.mp4`: NEAREST-upscaled pre vs VSR post.

The pre half is the input frame scaled up with `np.repeat` (true pixel
nearest-neighbor), the post half is the VSR output CVPixelBuffer. Both
halves are composited via CoreImage and rendered into a single BGRA buffer
in one GPU pass.

Accepts pre as either (H,W,3) uint8 RGB or (H,W,4) fp16/fp32 RGBA so it
works with both --video (uint8) and --latent HQ (fp16) chunk formats.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import pixel_buffers as _pb
from ._compat import Foundation, Quartz, require_pyobjc


def render_comparison(
    pre_frame: np.ndarray,
    post_pb: Any,
    scale: int,
    dest_pb: Any,
) -> None:
    """Render the side-by-side comparison into `dest_pb` (BGRA).

    Layout: NEAREST-upscaled pre on the left, VSR post (post_pb) on the
    right. Output dimensions are (2 * pre_W * scale, pre_H * scale).
    """
    require_pyobjc()
    in_h, in_w = pre_frame.shape[:2]
    out_w, out_h = in_w * scale, in_h * scale

    # NEAREST upscale via np.repeat - exact integer-replicated pixels.
    pre_up = np.repeat(np.repeat(pre_frame, scale, axis=0), scale, axis=1)

    # Coerce to uint8 RGBA8 for CIImage(kCIFormatRGBA8). The pre half is a
    # deliberately-blocky visualization, so 8-bit is fine.
    if pre_up.dtype == np.float16 or pre_up.dtype == np.float32:
        rgba = np.clip(pre_up * 255.0, 0, 255).astype(np.uint8)
        if rgba.shape[-1] == 3:
            full = np.empty((out_h, out_w, 4), dtype=np.uint8)
            full[..., 0:3] = rgba
            full[..., 3] = 255
            rgba = full
    else:
        rgba = np.empty((out_h, out_w, 4), dtype=np.uint8)
        if pre_up.shape[-1] == 4:
            rgba[:] = pre_up
        else:
            rgba[..., 0:3] = pre_up
            rgba[..., 3] = 255

    data = Foundation.NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
    pre_ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
        data, out_w * 4, (out_w, out_h),
        Quartz.kCIFormatRGBA8, _pb.srgb_colorspace(),
    )

    # Post comes in via its CVPixelBuffer - CoreImage reads it at whatever
    # bit depth the format provides (fp16 for RGBAHalf, 8-bit for NV12).
    post_ci = Quartz.CIImage.alloc().initWithCVPixelBuffer_(post_pb)
    post_ci_translated = post_ci.imageByApplyingTransform_(
        Quartz.CGAffineTransformMakeTranslation(out_w, 0),
    )
    composite = post_ci_translated.imageByCompositingOverImage_(pre_ci)

    _pb.ci_context().render_toCVPixelBuffer_bounds_colorSpace_(
        composite, dest_pb, ((0, 0), (2 * out_w, out_h)), _pb.srgb_colorspace(),
    )
