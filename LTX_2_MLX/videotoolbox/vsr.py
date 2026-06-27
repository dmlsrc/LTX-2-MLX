"""VideoToolbox Super Resolution (spatial upscale) session wrapper.

`VsrSession` wraps VTSuperResolutionScalerConfiguration (HQ, scale=4) or
VTLowLatencySuperResolutionScalerConfiguration (LL, scale=2) plus its
VTFrameProcessor and the source/dst CVPixelBufferPools. The caller hands
in a frame (uint8 RGB or fp16 RGBA) and gets back a destination buffer
ready to feed straight into AVAssetWriter.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Any

from . import pixel_buffers as _pb
from ._compat import Quartz, require_pyobjc, vt


@contextmanager
def _suppress_native_stderr():
    """Swallow OS-level stderr (fd 2) for the duration of the block.

    VideoToolbox compiles the super-resolution Metal pipeline when the frame
    processor session starts and logs 'Resolved compile flags ...
    SpatialSplitGenericDAG' straight to fd 2 via NSLog - bypassing Python's
    sys.stderr, so contextlib.redirect_stderr can't catch it. This redirects the
    file descriptor itself for the brief compile. VideoToolbox reports real
    failures through API return values (the ok/err tuple), not stderr, so
    nothing important is hidden. Set LTX_VSR_VERBOSE=1 to keep the native logs.
    """
    if os.environ.get("LTX_VSR_VERBOSE"):
        yield
        return
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def scale_for_mode(mode: str) -> int:
    """Map a VSR spatial mode to its forced scale factor.

    VideoToolbox couples the spatial-mode choice to the scale: LowLatency
    is 2x-only, the HQ classes are 4x-only.  Centralized here so call sites
    don't reinvent the mapping.
    """
    if mode == "fast":
        return 2
    if mode in ("balanced", "image", "basicvsrpp"):
        return 4
    raise ValueError(f"unknown VSR spatial-mode: {mode!r}")


# The HighQuality (balanced/image) scaler exposes NO dimension-query API -
# unlike LowLatency's minimumDimensions/maximumDimensions - so these practical
# input caps are determined empirically (config init fails with "Invalid input
# height/width" above them). The cap is per-dimension, not total pixels, and at
# 4x it bounds output to 7680x4320 (8K). Re-probe if a future OS raises it.
HQ_MAX_INPUT_W = 1920
HQ_MAX_INPUT_H = 1080


def source_format_for_mode(mode: str) -> int:
    """The CVPixelBuffer format VSR consumes as its source for this mode.

    fast (LowLatency) takes NV12 ('420v'); balanced/image (HighQuality) take
    RGBAHalf ('RGhA'). Exposed so an external decoder can produce frames
    already in VSR's source format - avoiding an intermediate RGB array and
    its re-quantization - and feed them via `upscale_buffer_to_buffer`.
    """
    if mode == "fast":
        return _pb.PIX_NV12
    if mode in ("balanced", "image", "basicvsrpp"):
        # basicvsrpp upscales in MLX, not VideoToolbox, but it also wants RGB in
        return _pb.PIX_RGBAHALF
    raise ValueError(f"unknown VSR spatial-mode: {mode!r}")


def _validate_combination(width: int, height: int, scale: int, mode: str) -> None:
    """Check the (input size, scale, mode) combo is something VT supports.

    VSR's HQ and LL classes each only support specific scale factors (and LL
    additionally restricts input size to <= 960x960). Failing fast here gives
    a clear error message instead of an opaque init/startSession failure.
    """
    require_pyobjc()
    if mode == "fast":
        cls = vt.VTLowLatencySuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("LowLatency VSR not supported on this device.")
        ok = list(cls.supportedScaleFactorsForFrameWidth_frameHeight_(width, height))
        if not ok:
            mn = cls.minimumDimensions()
            mx = cls.maximumDimensions()
            raise SystemExit(
                f"--spatial-mode fast does not support {width}x{height} input. "
                f"Allowed: {mn.width}x{mn.height} to {mx.width}x{mx.height}."
            )
        if float(scale) not in [float(s) for s in ok]:
            raise SystemExit(
                f"--spatial-mode fast at {width}x{height} supports scale={ok}, "
                f"requested scale={scale}."
            )
    else:
        cls = vt.VTSuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("High-quality VSR not supported on this device.")
        ok = [int(s) for s in cls.supportedScaleFactors()]
        if scale not in ok:
            raise SystemExit(
                f"--spatial-mode {mode} supports scale={ok}, requested scale={scale}. "
                f"Use --spatial-mode fast for 2x."
            )
        # The HQ scaler has no dimension-query API; check the empirical caps so
        # an oversized input fails with a clear message (and before the model
        # download wait) instead of an opaque "config init returned nil".
        if width > HQ_MAX_INPUT_W or height > HQ_MAX_INPUT_H:
            fits_fast = width <= 960 and height <= 960
            hint = (
                "Use --spatial-mode fast for a 2x upscale (input must be <= 960x960)."
                if fits_fast else
                f"This input is larger than any VSR mode supports; downscale it to "
                f"<= {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H} (balanced/image) or <= 960x960 (fast) first."
            )
            raise SystemExit(
                f"--spatial-mode {mode} (4x) does not support {width}x{height} input "
                f"(max {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; a 4x output would exceed 8K). {hint}"
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

    Spatial modes:
      "fast"      VTLowLatencySuperResolutionScalerConfiguration. scale=2,
                  input <= 960x960. NV12 source. Per-frame, no temporal context.
      "balanced"  VTSuperResolutionScalerConfiguration InputType=Video.
                  scale=4. RGBAHalf source. Uses prev source + prev output to
                  inform the per-frame upscale.  Default for video; slightly
                  crisper motion edges at the cost of slightly more
                  frame-to-frame variation than image mode.
      "image"     VTSuperResolutionScalerConfiguration InputType=Image. scale=4.
                  RGBAHalf source. Per-frame deterministic upscale, no
                  prev-frame feedback.  Apple documents this as for stills,
                  but on real video it produces measurably lower temporal
                  second-difference than balanced - a legitimate alternative
                  if you prefer the smoother / less-edge-boosted trade-off.

    The previous-frame state can be reset at hard cuts via
    `reset_temporal_context()` - useful for `--video` input that may contain
    edits. LTX latents are single-shot generations so this never matters for
    `--latent`.
    """

    def __init__(self, in_w: int, in_h: int, mode: str, fps: float = 24.0):
        require_pyobjc()
        scale = scale_for_mode(mode)
        _validate_combination(in_w, in_h, scale, mode)
        self.in_w, self.in_h = in_w, in_h
        self.scale = scale
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.mode = mode
        self.fps = float(fps)

        if mode == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc(
            ).initWithFrameWidth_frameHeight_scaleFactor_(in_w, in_h, float(scale))
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if mode == "balanced"
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w, in_h, scale, input_type, False,
                vt.VTSuperResolutionScalerConfigurationQualityPrioritizationNormal,
                cls.defaultRevision(),
            )
            if self.config is None:
                raise RuntimeError(
                    f"High-quality VSR config init returned nil for {in_w}x{in_h} "
                    f"input at {scale}x. The HQ scaler accepts up to "
                    f"{HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; check the input dimensions."
                )
            _wait_for_model_download(self.config)

        self.processor = vt.VTFrameProcessor.alloc().init()
        # startSession compiles the VSR Metal pipeline, which NSLogs compile
        # chatter to fd 2; suppress just that call (errors come back via `err`).
        with _suppress_native_stderr():
            ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor.startSessionWithConfiguration_error_ failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        print(
            f"VSR session ready (mode={mode}, {in_w}x{in_h} -> {self.out_w}x{self.out_h}, "
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
        # Lazily-created pixel-transfer session, used by upscale_buffer_to_buffer
        # to normalize externally-decoded buffers (see there).
        self._xfer: Any = None

    def use_dst_pool(self, pool: Any) -> None:
        """Wire the writer's adaptor pixelBufferPool() as VSR's dst source -
        zero-copy from VSR output straight into the encoder's queue.
        """
        self._dst_pool = pool

    def reset_temporal_context(self) -> None:
        """Drop the previous-frame chain. Call at scene cuts on --video input."""
        self._prev_src_frame = None
        self._prev_dst_frame = None

    def flush_pools(self) -> None:
        """Release excess cached buffers in the src pool (and dst pool if we
        own it - we usually don't; the writer's adaptor owns the dst pool).

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
        if self._xfer is not None:
            vt.VTPixelTransferSessionInvalidate(self._xfer)
            self._xfer = None

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

    def upscale_to_buffer(self, frame: Any, frame_index: int) -> Any:
        """Upscale one frame from an MLX/numpy array. Returns the dst
        CVPixelBuffer (RGBAHalf for HQ, NV12 for LL) ready to append to AVWriter.

        The array is uploaded into a pooled source buffer in VSR's source
        format (uint8 RGB and fp16 RGBA inputs are both accepted; see
        `pixel_buffers.upload_frame_to_buffer`). For frames that already exist
        as a CVPixelBuffer in the source format - e.g. straight from a native
        decoder - use `upscale_buffer_to_buffer` to skip the upload entirely.
        """
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        return self._process(src_pb, frame_index)

    def upscale_buffer_to_buffer(self, src_pb: Any, frame_index: int) -> Any:
        """Upscale one frame whose source CVPixelBuffer comes from an external
        decoder (already in VSR's source format; see `source_format_for_mode`).

        The buffer is normalized into a clean VSR pool buffer via
        VTPixelTransferSession before processing, rather than fed raw. A raw
        decoder buffer can carry IOSurface/attribute quirks - e.g. for input
        whose coded size is padded to a macroblock multiple (a 544x408 clip is
        coded at 544x416) - that the VSR processor rejects with -19730, even
        though the identical pixels in a VSR pool buffer upscale fine. The
        transfer also normalizes color to BT.709, so untagged / BT.601 /
        full-range sources match the BT.709 encoder output.
        """
        clean = self._normalize_src_buffer(src_pb)
        return self._process(clean, frame_index)

    def _normalize_src_buffer(self, src_pb: Any) -> Any:
        """Copy `src_pb` into a clean VSR-source pool buffer (BT.709), via a
        lazily-created VTPixelTransferSession. See upscale_buffer_to_buffer.
        """
        if self._xfer is None:
            err, xfer = vt.VTPixelTransferSessionCreate(None, None)
            if err != 0 or xfer is None:
                raise RuntimeError(f"VTPixelTransferSessionCreate failed: {err}")
            # Normalize destination color to BT.709 (matches the encoder), so an
            # untagged / BT.601 / full-range source is converted consistently
            # instead of carrying its native attributes into the BT.709 output.
            for key, val in (
                (vt.kVTPixelTransferPropertyKey_DestinationColorPrimaries,
                 Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2),
                (vt.kVTPixelTransferPropertyKey_DestinationTransferFunction,
                 Quartz.kCVImageBufferTransferFunction_ITU_R_709_2),
                (vt.kVTPixelTransferPropertyKey_DestinationYCbCrMatrix,
                 Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2),
            ):
                vt.VTSessionSetProperty(xfer, key, val)
            self._xfer = xfer
        clean = self._make_src_buffer()
        err = vt.VTPixelTransferSessionTransferImage(self._xfer, src_pb, clean)
        if err != 0:
            raise RuntimeError(f"VTPixelTransferSessionTransferImage failed: {err}")
        return clean

    def _process(self, src_pb: Any, frame_index: int) -> Any:
        """Run VSR on a ready source CVPixelBuffer; return the dst buffer.

        Shared tail of both upscale entry points: allocates the dst buffer,
        wraps src/dst as VTFrameProcessorFrames, builds the mode-appropriate
        parameters (threading the prev-frame chain for balanced), and advances
        that chain. The prev VTFrameProcessorFrame retains its CVPixelBuffer,
        so an externally-supplied src buffer stays valid across the one
        iteration balanced mode references it.
        """
        dst_pb = self._make_dst_buffer()
        pts = _pb.frame_pts(frame_index, self.fps)
        src_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            src_pb, pts,
        )
        dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            dst_pb, pts,
        )

        if self.mode == "fast":
            params = vt.VTLowLatencySuperResolutionScalerParameters.alloc(
            ).initWithSourceFrame_destinationFrame_(src_frame, dst_frame)
        else:
            use_temporal = self.mode == "balanced"
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


class NativePassthrough:
    """No-op stand-in for VsrSession used by --spatial-mode none.

    Packs a frame into a native-resolution RGBAHalf CVPixelBuffer with no
    super-resolution, exposing the same surface the harness loop drives on
    VsrSession (upscale_to_buffer / upscale_buffer_to_buffer / use_dst_pool /
    reset_temporal_context / flush_pools / close, plus out_w/out_h/dst_attrs).
    Lets the pipeline run denoise -> encode (or a plain transcode) at native
    resolution without special-casing the loop.
    """

    def __init__(self, in_w: int, in_h: int, fps: float = 24.0):
        require_pyobjc()
        self.in_w, self.in_h = int(in_w), int(in_h)
        self.scale = 1
        self.out_w, self.out_h = self.in_w, self.in_h
        self.mode = "none"
        self.fps = float(fps)
        self.dst_attrs = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: _pb.PIX_RGBAHALF,
            Quartz.kCVPixelBufferWidthKey: self.in_w,
            Quartz.kCVPixelBufferHeightKey: self.in_h,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        self.src_attrs = dict(self.dst_attrs)
        self._pool = _pb.make_pool_from_attrs(self.dst_attrs)
        print(f"Native passthrough (no upscale) ready ({self.in_w}x{self.in_h}, RGBAHalf)")

    def use_dst_pool(self, pool: Any) -> None:
        self._pool = pool

    def reset_temporal_context(self) -> None:
        pass

    def flush_pools(self) -> None:
        _pb.flush_pool(self._pool)

    def close(self) -> None:
        pass

    def _make_buffer(self) -> Any:
        if self._pool is not None:
            pb = _pb.pool_create_buffer(self._pool)
            if pb is not None:
                return pb
        return _pb.make_pixel_buffer_from_attrs(self.out_w, self.out_h, self.dst_attrs)

    def upscale_to_buffer(self, frame: Any, frame_index: int) -> Any:
        """Pack an MLX/numpy frame into a native-res RGBAHalf buffer (no scale)."""
        pb = self._make_buffer()
        _pb.upload_frame_to_buffer(frame, pb)
        return pb

    def upscale_buffer_to_buffer(self, src_pb: Any, frame_index: int) -> Any:
        """Pass an already-decoded RGBAHalf buffer straight through (transcode
        with no denoise). The harness decodes --spatial-mode none to RGBAHalf,
        so the buffer already matches the output format."""
        return src_pb
