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

from ._compat import CoreMedia, Foundation, Quartz, av, require_pyobjc


def _first_video_track(asset: Any) -> Any:
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    if tracks is None or len(tracks) == 0:
        raise RuntimeError("no video track in asset")
    return tracks[0]


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
    size = track.naturalSize()
    w, h = int(round(size.width)), int(round(size.height))
    fps = float(track.nominalFrameRate())
    duration = CoreMedia.CMTimeGetSeconds(asset.duration())
    n = int(round(duration * fps)) if fps > 0 else 0
    transform = track.preferredTransform()
    return w, h, fps, n, transform


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
