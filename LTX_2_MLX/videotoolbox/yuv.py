"""MLX RGB -> 10-bit 4:2:2 YUV conversion for the encoder feed.

AVAssetWriter's internal RGB->YUV is colorspace-metadata-dependent: it keys off
the input buffer's IOSurface-level colorspace, which VideoToolbox's own producers
(decoder, VSR scaler) bake in but uploaded buffers (fastdvd / realesrgan / learned
upscalers / latent) lack. `CVBufferSetAttachment` only sets the CVBuffer attachment,
not the IOSurface one, so feeding RGB gives a path-dependent color shift that can't
be fully tagged away (measured: a uniform green-biased darkening on uploaded paths,
~0.0145 mean, ~5% on green).

Converting RGB->YUV ourselves with the standard ITU-R coefficients removes ALL
metadata dependency: identical RGB yields identical YUV, the matrix is exactly the
one requested, and every pipeline encodes consistently. The encoder is then handed
YUV directly (no RGB->YUV step), so the discrete YCbCr matrix/primaries/transfer
tags ARE honored.

Verified: RGB->YUV->RGB round-trips at 7e-4 (10-bit + 4:2:2 floor); end-to-end it
removes the green-darkening (bare output tracks the source per-channel like native).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ._compat import Quartz, require_pyobjc

# 10-bit 4:2:2 biplanar, the format the HEVC 4:2:2 profile consumes directly.
PIX_422YCBCR10_VIDEO = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
PIX_422YCBCR10_FULL = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarFullRange


def _coef_for_matrix(matrix: Any) -> tuple[float, float]:
    """ITU-R (Kr, Kb) luma coefficients for a CV YCbCrMatrix constant (601 default)."""
    if matrix == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020:
        return 0.2627, 0.0593
    if matrix == Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2:
        return 0.2126, 0.0722
    return 0.299, 0.114  # ITU-R BT.601


def pixel_format(full_range: bool) -> int:
    return PIX_422YCBCR10_FULL if full_range else PIX_422YCBCR10_VIDEO


# Pure RGB -> (luma, chroma) compute, compiled once per (Kr, Kb, full_range) and
# reused for every frame of a run (the frame shape is stable, so no recompile
# thrash). Kept separate from the impure plane memcpy in rgb_to_yuv422_10 so the
# matrix + quantize math is a single fused MLX graph.
_COMPUTE_CACHE: dict = {}


def _compiled_planes(Kr: float, Kb: float, full_range: bool):
    Kg = 1.0 - Kr - Kb

    def _compute(rgb: Any):
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        Y = Kr * R + Kg * G + Kb * B
        Cb = (B - Y) / (2.0 * (1.0 - Kb))            # [-0.5, 0.5]
        Cr = (R - Y) / (2.0 * (1.0 - Kr))
        H, W = rgb.shape[0], rgb.shape[1]
        if full_range:
            Y10 = Y * 1023.0
            Cb10c, Cr10c = Cb * 1023.0 + 512.0, Cr * 1023.0 + 512.0
        else:  # video range 10-bit: Y 64..940, chroma 64..960 (mid 512)
            Y10 = Y * 876.0 + 64.0
            Cb10c, Cr10c = Cb * 896.0 + 512.0, Cr * 896.0 + 512.0
        Cb_s = (Cb10c[:, 0::2] + Cb10c[:, 1::2]) * 0.5   # box-average to 4:2:2
        Cr_s = (Cr10c[:, 0::2] + Cr10c[:, 1::2]) * 0.5
        luma = (mx.clip(mx.round(Y10), 0, 1023).astype(mx.uint16)) << 6
        cb = mx.clip(mx.round(Cb_s), 0, 1023).astype(mx.uint16)
        cr = mx.clip(mx.round(Cr_s), 0, 1023).astype(mx.uint16)
        chroma = (mx.stack([cb, cr], axis=-1).reshape(H, W)) << 6   # Cb,Cr interleaved
        return luma, chroma

    key = (Kr, Kb, full_range)
    fn = _COMPUTE_CACHE.get(key)
    if fn is None:
        fn = mx.compile(_compute)
        _COMPUTE_CACHE[key] = fn
    return fn


def rgb_to_yuv422_10(rgb: Any, dst_buffer: Any, matrix: Any, full_range: bool = False) -> None:
    """Convert `rgb` (H,W,3 float, gamma-encoded, in [0,1]) to 10-bit 4:2:2 YUV and
    write it into `dst_buffer` (must be PIX_422YCBCR10_*). The matrix + quantize math
    runs as a compiled, pure MLX graph (`_compiled_planes`); only the plane memcpy
    below touches CoreVideo. No CoreImage, no colorspace metadata.

    The 10-bit samples are left-justified (<< 6) in their 16-bit words, which is what
    the BiPlanar10 formats expect. Chroma is box-averaged to 4:2:2.
    """
    require_pyobjc()
    Kr, Kb = _coef_for_matrix(matrix)
    luma, chroma = _compiled_planes(Kr, Kb, full_range)(rgb)
    _write_planes(dst_buffer, (luma, chroma))


def _write_planes(buf: Any, planes: tuple) -> None:
    """memcpy each (rows, cols) uint16 MLX plane into the buffer's biplanar storage,
    honoring per-plane bytesPerRow (rows may be padded)."""
    Quartz.CVPixelBufferLockBaseAddress(buf, 0)
    try:
        for plane, arr in enumerate(planes):
            arr = mx.contiguous(arr)
            mx.eval(arr)
            rows, cols = int(arr.shape[0]), int(arr.shape[1])
            base = Quartz.CVPixelBufferGetBaseAddressOfPlane(buf, plane)
            bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(buf, plane)
            mv = base.as_buffer(rows * bpr)
            src = memoryview(arr).cast("B")
            row_bytes = cols * 2
            if bpr == row_bytes:
                mv[: rows * row_bytes] = src
            else:
                for r in range(rows):
                    mv[r * bpr : r * bpr + row_bytes] = src[r * row_bytes : (r + 1) * row_bytes]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buf, 0)
