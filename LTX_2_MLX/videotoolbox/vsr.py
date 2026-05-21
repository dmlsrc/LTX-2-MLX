"""VideoToolbox Super Resolution (spatial upscale) session wrapper.

`VsrSession` wraps VTSuperResolutionScalerConfiguration (HQ, scale=4) or
VTLowLatencySuperResolutionScalerConfiguration (LL, scale=2) plus its
VTFrameProcessor and the source/dst CVPixelBufferPools. The caller hands
in a frame (uint8 RGB or fp16 RGBA) and gets back a destination buffer
ready to feed straight into AVAssetWriter.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ._compat import CoreMedia, vt, require_pyobjc
from . import pixel_buffers as _pb


def _validate_combination(width: int, height: int, scale: int, quality: str) -> None:
    """Check the (input size, scale, quality) combo is something VT supports.

    VSR's HQ and LL classes each only support specific scale factors (and LL
    additionally restricts input size to <= 960x960). Failing fast here gives
    a clear error message instead of an opaque init/startSession failure.
    """
    require_pyobjc()
    if quality == "fast":
        cls = vt.VTLowLatencySuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("LowLatency VSR not supported on this device.")
        ok = list(cls.supportedScaleFactorsForFrameWidth_frameHeight_(width, height))
        if not ok:
            mn = cls.minimumDimensions()
            mx = cls.maximumDimensions()
            raise SystemExit(
                f"--quality fast does not support {width}x{height} input. "
                f"Allowed: {mn.width}x{mn.height} to {mx.width}x{mx.height}."
            )
        if float(scale) not in [float(s) for s in ok]:
            raise SystemExit(
                f"--quality fast at {width}x{height} supports scale={ok}, "
                f"got --scale {scale}."
            )
    else:
        cls = vt.VTSuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("High-quality VSR not supported on this device.")
        ok = [int(s) for s in cls.supportedScaleFactors()]
        if scale not in ok:
            raise SystemExit(
                f"--quality {quality} supports scale={ok}, got --scale {scale}. "
                f"Use --quality fast for scale=2."
            )


def _wait_for_model_download(config: Any) -> None:
    """Block until HQ VSR's downloadable model is ready, printing progress."""
    require_pyobjc()
    status = config.configurationModelStatus()
    if status == vt.VTSuperResolutionScalerConfigurationModelStatusReady:
        return
    print(f"VSR model not ready (status={status}); requesting download...")
    done = threading.Event()
    err_box: list[Any] = [None]

    def completion(error):
        err_box[0] = error
        done.set()

    config.downloadConfigurationModelWithCompletionHandler_(completion)
    last_reported = -1
    while not done.is_set():
        pct = int(config.configurationModelPercentageAvailable() * 100)
        if pct // 5 != last_reported // 5:
            print(f"  model download: {pct}%")
            last_reported = pct
        done.wait(timeout=0.5)
    if err_box[0] is not None:
        raise RuntimeError(f"VSR model download failed: {err_box[0]}")
    print("  model download: done")


class VsrSession:
    """Per-frame VSR processor with prev-frame chain for temporal coherence.

    Quality modes:
      "fast"      VTLowLatencySuperResolutionScalerConfiguration. scale=2,
                  input <= 960x960. NV12 source. Per-frame, no temporal context.
      "balanced"  VTSuperResolutionScalerConfiguration InputType=Video.
                  scale=4. RGBAHalf source. Uses prev source + prev output for
                  temporal coherence.
      "high"      VTSuperResolutionScalerConfiguration InputType=Image. scale=4.
                  RGBAHalf source. Per-frame, intended for stills.

    The previous-frame state can be reset at hard cuts via
    `reset_temporal_context()` — useful for `--video` input that may contain
    edits. LTX latents are single-shot generations so this never matters for
    `--latent`.
    """

    def __init__(self, in_w: int, in_h: int, scale: int, quality: str, fps: float = 24.0):
        require_pyobjc()
        _validate_combination(in_w, in_h, scale, quality)
        self.in_w, self.in_h = in_w, in_h
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.quality = quality
        self.fps = float(fps)

        if quality == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc(
            ).initWithFrameWidth_frameHeight_scaleFactor_(in_w, in_h, float(scale))
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if quality == "balanced"
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w, in_h, scale, input_type, False,
                vt.VTSuperResolutionScalerConfigurationQualityPrioritizationNormal,
                cls.defaultRevision(),
            )
            if self.config is None:
                raise RuntimeError("High-quality VSR config init returned nil")
            _wait_for_model_download(self.config)

        self.processor = vt.VTFrameProcessor.alloc().init()
        ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor.startSessionWithConfiguration_error_ failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        print(
            f"VSR session ready ({quality}, {in_w}x{in_h} -> {self.out_w}x{self.out_h}, "
            f"src fmt {_pb.resolve_pixel_format(self.src_attrs):#x}, "
            f"dst fmt {_pb.resolve_pixel_format(self.dst_attrs):#x})"
        )

        self._prev_src_frame: Any = None
        self._prev_dst_frame: Any = None

        # Src pool: two buffers in flight at any time (current + prev_src).
        self._src_pool = _pb.make_pool_from_attrs(self.src_attrs)
        if self._src_pool is None:
            print("  warning: src pool creation failed; falling back to per-frame allocation")
        # Dst pool: typically set by the caller to the AVAssetWriter adaptor's
        # pool for zero-copy from VSR output to encoder.
        self._dst_pool: Any = None

    def use_dst_pool(self, pool: Any) -> None:
        """Wire the writer's adaptor pixelBufferPool() as VSR's dst source —
        zero-copy from VSR output straight into the encoder's queue.
        """
        self._dst_pool = pool

    def reset_temporal_context(self) -> None:
        """Drop the previous-frame chain. Call at scene cuts on --video input."""
        self._prev_src_frame = None
        self._prev_dst_frame = None

    def flush_pools(self) -> None:
        """Release excess cached buffers in the src pool (and dst pool if we
        own it — we usually don't; the writer's adaptor owns the dst pool).

        Pool caching is what makes hot-path buffer allocation fast, but at
        steady state the cache should be ~3 buffers. Periodic flushing
        reclaims peak-watermark allocations that the workload no longer
        needs (e.g. an early VAE chunk that briefly inflated buffer demand).
        """
        _pb.flush_pool(self._src_pool)

    def close(self) -> None:
        if self.processor is not None:
            self.processor.endSession()
            self.processor = None

    # ------------------------------------------------------------------------
    # Internal: buffer factories
    # ------------------------------------------------------------------------

    def _make_src_buffer(self) -> Any:
        if self._src_pool is not None:
            pb = _pb.pool_create_buffer(self._src_pool)
            if pb is not None:
                return pb
        return _pb.make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.src_attrs)

    def _make_dst_buffer(self) -> Any:
        if self._dst_pool is not None:
            pb = _pb.pool_create_buffer(self._dst_pool)
            if pb is not None:
                return pb
        return _pb.make_pixel_buffer_from_attrs(self.out_w, self.out_h, self.dst_attrs)

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def upscale_to_buffer(self, frame: np.ndarray, frame_index: int) -> Any:
        """Upscale one frame. Returns the dst CVPixelBuffer (RGBAHalf for HQ,
        NV12 for LL) ready to append to AVWriter.
        """
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        dst_pb = self._make_dst_buffer()
        pts = _pb.frame_pts(frame_index, self.fps)
        src_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            src_pb, pts,
        )
        dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            dst_pb, pts,
        )

        if self.quality == "fast":
            params = vt.VTLowLatencySuperResolutionScalerParameters.alloc(
            ).initWithSourceFrame_destinationFrame_(src_frame, dst_frame)
        else:
            use_temporal = self.quality == "balanced"
            params = vt.VTSuperResolutionScalerParameters.alloc(
            ).initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
                src_frame,
                self._prev_src_frame if use_temporal else None,
                self._prev_dst_frame if use_temporal else None,
                None,
                vt.VTSuperResolutionScalerParametersSubmissionModeSequential,
                dst_frame,
            )

        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(
                f"VSR processWithParameters failed at frame {frame_index}: {err}"
            )
        self._prev_src_frame = src_frame
        self._prev_dst_frame = dst_frame
        return dst_pb
