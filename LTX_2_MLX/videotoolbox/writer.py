"""AVAssetWriter wrapper: HEVC video + optional ALAC/AAC audio, no ffmpeg.

The writer takes a stream of CVPixelBuffers (typically straight from
VsrSession's adaptor pool — zero-copy from VSR output to encoder) and
encodes them as HEVC Main10 4:2:0 or Main42210 4:2:2 10-bit BT.709 at the
target fps. Audio (if attached) is pulled by AVAssetWriter on a dedicated
dispatch queue via requestMediaDataWhenReadyOnQueue:, so the audio encode
doesn't stall the video append loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ._compat import CoreMedia, Foundation, Quartz, av, libdispatch, require_pyobjc
from .audio import AudioTrack, audio_writer_settings
from . import pixel_buffers as _pb


# HEVC profile identifiers (Apple-stable strings; not exposed as PyObjC consts)
HEVC_PROFILE_MAIN10 = "HEVC_Main10_AutoLevel"          # 4:2:0 10-bit
HEVC_PROFILE_MAIN422_10 = "HEVC_Main42210_AutoLevel"   # 4:2:2 10-bit (Range Extensions)


def hevc_video_settings(
    width: int, height: int, quality: float, profile: str,
) -> dict:
    """AVAssetWriterInput output settings for HEVC at the given size + profile."""
    require_pyobjc()
    return {
        av.AVVideoCodecKey: av.AVVideoCodecTypeHEVC,
        av.AVVideoWidthKey: width,
        av.AVVideoHeightKey: height,
        av.AVVideoColorPropertiesKey: {
            av.AVVideoColorPrimariesKey: av.AVVideoColorPrimaries_ITU_R_709_2,
            av.AVVideoTransferFunctionKey: av.AVVideoTransferFunction_ITU_R_709_2,
            av.AVVideoYCbCrMatrixKey: av.AVVideoYCbCrMatrix_ITU_R_709_2,
        },
        av.AVVideoCompressionPropertiesKey: {
            av.AVVideoProfileLevelKey: profile,
            av.AVVideoQualityKey: quality,
        },
    }


class AVWriter:
    """AVAssetWriter wrapping a HEVC video input + optional audio input.

    Construction kicks off `startWriting` + `startSessionAtSourceTime`. If
    `audio_track` is supplied, an audio AVAssetWriterInput is added and a
    GCD callback is scheduled to pull samples from the track as the encoder
    consumes them.

    Per-frame API:
        writer.append(pb)           # pixel buffer in the configured source format
    Finalize:
        writer.finish()             # waits for audio drain + finishWriting
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        source_pixel_format: int,
        profile: str = HEVC_PROFILE_MAIN10,
        quality: float = 0.65,
        label: str = "video",
        audio_track: AudioTrack | None = None,
        audio_codec: str = "alac",
    ):
        require_pyobjc()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        url = Foundation.NSURL.fileURLWithPath_(str(output_path))
        writer, err = av.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url, av.AVFileTypeMPEG4, None,
        )
        if writer is None:
            raise RuntimeError(f"AVAssetWriter init failed for {output_path}: {err}")

        # Video input + pixel buffer adaptor ---------------------------------
        video_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
            av.AVMediaTypeVideo, hevc_video_settings(width, height, quality, profile),
        )
        video_input.setExpectsMediaDataInRealTime_(False)

        src_attrs = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: source_pixel_format,
            Quartz.kCVPixelBufferWidthKey: width,
            Quartz.kCVPixelBufferHeightKey: height,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        adaptor = av.AVAssetWriterInputPixelBufferAdaptor.assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            video_input, src_attrs,
        )
        if not writer.canAddInput_(video_input):
            raise RuntimeError(f"AVAssetWriter cannot add video input for {output_path}")
        writer.addInput_(video_input)

        # Optional audio input -----------------------------------------------
        audio_input = None
        if audio_track is not None:
            audio_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
                av.AVMediaTypeAudio,
                audio_writer_settings(audio_codec, audio_track.sample_rate, audio_track.channels),
            )
            audio_input.setExpectsMediaDataInRealTime_(False)
            if not writer.canAddInput_(audio_input):
                raise RuntimeError(
                    f"AVAssetWriter cannot add audio input ({audio_codec}) for {output_path}"
                )
            writer.addInput_(audio_input)

        # Start the writer ---------------------------------------------------
        if not writer.startWriting():
            raise RuntimeError(f"AVAssetWriter.startWriting failed: {writer.error()}")
        writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, _pb.VIDEO_TIME_SCALE))

        self.writer = writer
        self.video_input = video_input
        self.audio_input = audio_input
        self.adaptor = adaptor
        self.fps = float(fps)
        self.label = label
        self.path = output_path
        self.frame_count = 0
        self.audio_track = audio_track
        self._audio_codec = audio_codec

        audio_desc = f", audio={audio_codec}" if audio_input is not None else ""
        print(
            f"[{label}] AVAssetWriter -> {output_path} "
            f"(HEVC {profile} BT.709 q={quality}{audio_desc})"
        )

        # Audio pump (GCD pull pattern) --------------------------------------
        self._audio_done = threading.Event()
        self._audio_progress = [0]
        if audio_track is not None:
            self._audio_queue = libdispatch.dispatch_queue_create(
                f"vsr_harness.audio.{label}".encode(), None,
            )
            n_samples = audio_track.n_samples
            chunk_frames = max(4096, audio_track.sample_rate // 4)  # ~250 ms

            def pump():
                try:
                    while self.audio_input.isReadyForMoreMediaData():
                        pos = self._audio_progress[0]
                        if pos >= n_samples:
                            self.audio_input.markAsFinished()
                            self._audio_done.set()
                            return
                        end = min(pos + chunk_frames, n_samples)
                        sb = audio_track.make_sample_buffer(pos, end)
                        if sb is None or not self.audio_input.appendSampleBuffer_(sb):
                            self._audio_done.set()
                            raise RuntimeError(
                                f"[{label}] audio appendSampleBuffer failed at "
                                f"{pos}: {self.writer.error()}"
                            )
                        self._audio_progress[0] = end
                except Exception:
                    self._audio_done.set()
                    raise

            self.audio_input.requestMediaDataWhenReadyOnQueue_usingBlock_(
                self._audio_queue, pump,
            )
        else:
            self._audio_done.set()
            self._audio_queue = None

    # ------------------------------------------------------------------------
    # Internal: wait-with-status-check
    # ------------------------------------------------------------------------

    def _wait_for_ready(self, input_obj: Any, what: str) -> None:
        """Block until input_obj.isReadyForMoreMediaData(). Bail with a clean
        error if the writer enters Failed/Cancelled, or after 30 s of no
        progress (so a stuck writer surfaces as a visible failure, not a hang).
        """
        waited = 0.0
        while not input_obj.isReadyForMoreMediaData():
            status = self.writer.status()
            if status in (3, 4):  # Failed, Cancelled
                raise RuntimeError(
                    f"[{self.label}] writer entered status={status} while waiting on "
                    f"{what}: {self.writer.error()}"
                )
            time.sleep(0.001)
            waited += 0.001
            if waited > 30.0:
                raise RuntimeError(
                    f"[{self.label}] {what} input never became ready "
                    f"(waited 30s, status={status})"
                )

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def append(self, pb: Any) -> None:
        """Append one video frame at the next PTS (frame_count/fps)."""
        self._wait_for_ready(self.video_input, "video")
        pts = _pb.frame_pts(self.frame_count, self.fps)
        if not self.adaptor.appendPixelBuffer_withPresentationTime_(pb, pts):
            raise RuntimeError(
                f"[{self.label}] appendPixelBuffer failed at frame {self.frame_count}: "
                f"status={self.writer.status()} error={self.writer.error()}"
            )
        self.frame_count += 1

    def finish(self) -> None:
        """Mark inputs finished, drain audio, end session, finishWriting."""
        self.video_input.markAsFinished()
        if self.audio_input is not None:
            if not self._audio_done.wait(timeout=120.0):
                raise RuntimeError(
                    f"[{self.label}] audio pump didn't finish (progress="
                    f"{self._audio_progress[0]}/{self.audio_track.n_samples})"
                )
        self.writer.endSessionAtSourceTime_(_pb.frame_pts(self.frame_count, self.fps))
        done = threading.Event()
        self.writer.finishWritingWithCompletionHandler_(lambda: done.set())
        done.wait()
        if self.writer.status() != 2:  # AVAssetWriterStatusCompleted = 2
            raise RuntimeError(
                f"[{self.label}] AVAssetWriter finished with status "
                f"{self.writer.status()}: {self.writer.error()}"
            )
