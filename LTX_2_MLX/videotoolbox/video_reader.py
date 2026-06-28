"""Native AVFoundation video decode for the `--video` VSR path.

`AVURLAsset` + `AVAssetReader` pull decoded frames straight out of the
container in VSR's own source pixel format (NV12 for the LowLatency `fast`
scaler, RGBAHalf for the HighQuality `balanced`/`image` scalers). The decoded
`CVPixelBuffer` is fed directly into VSR via `upscale_buffer_to_buffer` - no
intermediate RGB array, no re-quantization, no per-frame copy through MLX.

This preserves the source's bit depth and chroma:
  - fast: source YUV -> NV12 is a memory-layout change only, no color or
    chroma conversion (8-bit is the LowLatency scaler's ceiling regardless).
  - balanced/image: source YUV (including 10-bit 4:2:2 / 4:2:0) -> RGBAHalf is
    a single decode-time conversion at half-float precision and 4:4:4, so
    10-bit sources keep their precision instead of being clamped through an
    8-bit RGB intermediate.

A track's preferredTransform (rotation / flip) is returned by `probe_video`
and propagated to the output as container metadata by the writer, so rotated
inputs display correctly without ever rotating pixels - lossless.

No ffmpeg, no numpy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import pixel_buffers as _pb
from ._compat import CoreMedia, Foundation, Quartz, av, require_pyobjc, vt


def _first_video_track(asset: Any) -> Any:
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    if tracks is None or len(tracks) == 0:
        raise RuntimeError("no video track in asset")
    return tracks[0]


def _video_codec_fourcc(track: Any) -> str:
    """The track's codec as a 4-char tag ('hvc1', 'hev1', 'avc1', ...), or ''."""
    fmts = track.formatDescriptions()
    if not fmts or len(fmts) == 0:
        return ""
    code = CoreMedia.CMFormatDescriptionGetMediaSubType(fmts[0])
    return bytes((code >> s) & 0xFF for s in (24, 16, 8, 0)).decode("latin-1")


def _assert_decodable(track: Any, path: Path) -> None:
    """Reject 'hev1'-tagged HEVC up front with an actionable message.

    ffmpeg muxes HEVC into MP4 as 'hev1' by default, but AVFoundation can only
    decode 'hvc1' (parameter sets carried out of band in the hvcC box). An hev1
    track otherwise fails deep in the reader with a cryptic -11833 'Cannot
    Decode'; the fix is a lossless container re-tag, no re-encode."""
    if _video_codec_fourcc(track) == "hev1":
        out = path.with_name(f"{path.stem}_hvc1.mp4")
        raise RuntimeError(
            f"{path.name}: HEVC video is tagged 'hev1', which AVFoundation cannot "
            f"decode - it requires 'hvc1' (parameter sets out of band). Re-tag it "
            f"losslessly (no re-encode), then use the result:\n"
            f"    ffmpeg -i '{path}' -c copy -tag:v hvc1 '{out}'"
        )


def probe_video(path: Path) -> tuple[int, int, float, int, Any]:
    """(width, height, fps, n_frames, transform) for the first video track.

    Dimensions are the track's stored naturalSize. `transform` is the track's
    preferredTransform (a CGAffineTransform) - identity for upright content,
    a rotation/flip for camera footage; the writer applies it as output
    metadata so pixels never need rotating. n_frames is round(duration * fps),
    exact for constant-frame-rate content (everything VSR consumes).
    """
    require_pyobjc()
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _assert_decodable(track, path)
    size = track.naturalSize()
    w, h = int(round(size.width)), int(round(size.height))
    fps = float(track.nominalFrameRate())
    duration = CoreMedia.CMTimeGetSeconds(asset.duration())
    n = int(round(duration * fps)) if fps > 0 else 0
    transform = track.preferredTransform()
    return w, h, fps, n, transform


def probe_color(path: Path) -> dict:
    """Source color tags from the container, for output color propagation.

    Returns explicit primaries/transfer/matrix + full-range flag (None where
    untagged) so the encoder can tag the output to match the source instead of a
    hard-coded BT.709. See videotoolbox/color.py.
    """
    from . import color
    require_pyobjc()
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    src = color.read_source_color(track.formatDescriptions()[0])
    if not src["tagged"]:
        # Untagged: read VideoToolbox's decode-time guess (its undocumented,
        # width-keyed choice) off a decoded frame, so 'auto' reports/tags what was
        # actually read instead of assuming BT.709.
        try:
            buf = next(iter(iter_video_buffer_chunks(path, _pb.PIX_RGBAHALF, chunk_size=1)))[0]
            att = Quartz.CVBufferCopyAttachments(buf, Quartz.kCVAttachmentMode_ShouldPropagate) or {}
            by = {str(k): att[k] for k in att}
            src["primaries"] = by.get("CVImageBufferColorPrimaries") or src["primaries"]
            src["transfer"] = by.get("CVImageBufferTransferFunction") or src["transfer"]
            src["matrix"] = by.get("CVImageBufferYCbCrMatrix") or src["matrix"]
            src["guessed"] = src["matrix"] is not None
        except Exception:
            pass
    return src


# 10-bit 4:2:2 YUV for the forced-decode path: a precision-preserving superset
# (4:2:0 / 8-bit sources upsample into it losslessly).
_YUV10_VIDEO = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
_YUV10_FULL = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarFullRange


def iter_forced_color_chunks(
    path: Path, out_format: int, matrix_cv: Any, full_range: bool,
    chunk_size: int = 8, *, start_frame: int = 0, end_frame: int | None = None,
) -> Iterator[list]:
    """Decode the source as raw 10-bit YUV, FORCE the YCbCr matrix (overriding the
    container tag / VideoToolbox's resolution-based guess), then convert to
    out_format ourselves.

    This makes --source-color control how the source is READ -- the fix for the
    untagged SD clips VideoToolbox mis-guesses as BT.601. The 4:2:2/10-bit YUV is a
    precision-preserving superset of any SDR source; range follows the detected
    full/video flag (the YUV->RGB scaling the requested format implies).
    """
    require_pyobjc()
    yuv_fmt = _YUV10_FULL if full_range else _YUV10_VIDEO
    err, xfer = vt.VTPixelTransferSessionCreate(None, None)
    if err != 0 or xfer is None:
        raise RuntimeError(f"VTPixelTransferSessionCreate failed: {err}")
    for chunk in iter_video_buffer_chunks(path, yuv_fmt, chunk_size,
                                          start_frame=start_frame, end_frame=end_frame):
        out: list = []
        for yuv in chunk:
            Quartz.CVBufferSetAttachment(
                yuv, Quartz.kCVImageBufferYCbCrMatrixKey, matrix_cv,
                Quartz.kCVAttachmentMode_ShouldPropagate)
            w = Quartz.CVPixelBufferGetWidth(yuv)
            h = Quartz.CVPixelBufferGetHeight(yuv)
            dst = _pb.make_pixel_buffer_from_attrs(w, h, {
                Quartz.kCVPixelBufferPixelFormatTypeKey: out_format,
                Quartz.kCVPixelBufferWidthKey: w, Quartz.kCVPixelBufferHeightKey: h,
                Quartz.kCVPixelBufferIOSurfacePropertiesKey: {}})
            e = vt.VTPixelTransferSessionTransferImage(xfer, yuv, dst)
            if e != 0:
                raise RuntimeError(f"forced-color YUV->{out_format:#x} transfer failed: {e}")
            out.append(dst)
        yield out


def iter_video_buffer_chunks(
    path: Path, src_format: int, chunk_size: int = 8,
    *, start_frame: int = 0, end_frame: int | None = None,
) -> Iterator[list]:
    """Yield lists of up to `chunk_size` decoded CVPixelBuffers in `src_format`.

    Each buffer is IOSurface-backed and ready to feed straight into
    `VsrSession.upscale_buffer_to_buffer`. Decode is pull-based - the reader
    produces one frame at a time - so peak resident memory is bounded by
    `chunk_size` decoded frames (the harness sizes this to a memory budget and
    frees each frame as it is consumed).

    `start_frame`/`end_frame` trim the input to the half-open frame window
    [start_frame, end_frame) (end_frame=None means to the end). The reader's
    timeRange is seeked just before start_frame so the bulk of a long clip
    before the window is never decoded; the exact window boundary is then
    enforced per frame by presentation timestamp, so trimming is frame-exact
    even though the seek is approximate.

    The decoded CVPixelBuffer is retained independently of its owning
    CMSampleBuffer (pyobjc holds it for the wrapper's lifetime), so the sample
    buffer is released immediately after the image buffer is extracted; the
    image buffer stays valid until the consumer drops its reference.
    """
    require_pyobjc()
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _assert_decodable(track, path)
    fps = float(track.nominalFrameRate())

    reader, err = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"AVAssetReader init failed: {err}")

    trimming = start_frame > 0 or end_frame is not None
    if trimming and fps > 0:
        # Seek the reader's timeRange to just before the window so the head of a
        # long clip isn't decoded. Back off one frame; the per-frame PTS check
        # below enforces the exact start. Compute the end from the asset
        # duration when the window is open-ended.
        ts = 24000
        start_seconds = max(0.0, (start_frame - 1) / fps)
        start_t = CoreMedia.CMTimeMake(int(round(start_seconds * ts)), ts)
        if end_frame is not None:
            dur_seconds = (end_frame - start_frame + 2) / fps
        else:
            dur_seconds = max(0.0, CoreMedia.CMTimeGetSeconds(asset.duration()) - start_seconds)
        dur_t = CoreMedia.CMTimeMake(int(round(dur_seconds * ts)), ts)
        reader.setTimeRange_(CoreMedia.CMTimeRangeMake(start_t, dur_t))

    # Request IOSurface-backed, Metal-compatible buffers. Feeding the decoded
    # buffer straight to VSR bypasses the VSR source pool's attributes, so we
    # must ask for GPU-usable backing here instead - otherwise the Metal-based
    # super-resolution processor can reject the source frame (notably the
    # LowLatency 'fast' path: VTFrameProcessor error -19730).
    output = av.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(
        track, {
            Quartz.kCVPixelBufferPixelFormatTypeKey: src_format,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
            Quartz.kCVPixelBufferMetalCompatibilityKey: True,
        },
    )
    # Keep alwaysCopiesSampleData=YES (the default): we hold decoded buffers
    # past the copyNextSampleBuffer call - across a chunk, and across one
    # iteration for balanced mode's prev-frame chain - and feed them straight
    # into VSR. With NO, AVAssetReader can hand back references to volatile
    # decoder memory that gets recycled while we still reference it, which
    # corrupts the prev frame (flicker / dark output) or yields an invalid
    # source buffer (VTFrameProcessor -19730). The copy is one memcpy of the
    # already-decoded frame, in VSR's source format - no RGB conversion, cheap
    # next to the decode - so the fidelity and no-MLX-round-trip wins stand.
    output.setAlwaysCopiesSampleData_(True)
    if not reader.canAddOutput_(output):
        raise RuntimeError(
            f"AVAssetReader cannot output pixel format {src_format:#x}"
        )
    reader.addOutput_(output)
    if not reader.startReading():
        raise RuntimeError(f"AVAssetReader.startReading failed: {reader.error()}")

    chunk: list = []
    while True:
        sample_buf = output.copyNextSampleBuffer()
        if sample_buf is None:
            break
        image_buf = CoreMedia.CMSampleBufferGetImageBuffer(sample_buf)
        keep = image_buf is not None
        if keep and trimming and fps > 0:
            # Frame-exact window enforcement by presentation timestamp.
            pts_s = CoreMedia.CMTimeGetSeconds(
                CoreMedia.CMSampleBufferGetPresentationTimeStamp(sample_buf),
            )
            idx = int(round(pts_s * fps))
            if idx < start_frame:
                keep = False
            elif end_frame is not None and idx >= end_frame:
                del sample_buf
                break
        # Release the owning sample buffer now; the image buffer outlives it.
        del sample_buf
        if keep:
            chunk.append(image_buf)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if reader.status() == av.AVAssetReaderStatusFailed:
        raise RuntimeError(f"AVAssetReader failed: {reader.error()}")
    if chunk:
        yield chunk
