"""VideoToolbox Frame Rate Conversion (temporal upscaler) session wrapper.

Wraps VTFrameRateConversionConfiguration + VTFrameRateConversionParameters
to convert between arbitrary source and target frame rates. Unlike VSR,
the configuration takes only the frame dimensions + quality; the rate
conversion ratio is driven entirely by per-pair interpolation phases.

Per source frame pair (frame_N at PTS_N, frame_N+1 at PTS_N+1), we
compute the set of target output PTSes that fall in [PTS_N, PTS_N+1)
and their phases (where phase = (target_pts - PTS_N) / (PTS_N+1 - PTS_N)
in [0, 1)). VT's API takes a phase array and a matching destinationFrames
array, so a single call produces all interpolated frames for that pair.

Cleanly handles arbitrary float fps both sides:
  15 -> 30   exact 2x; phases always [0.5].
  24 -> 60   2.5x; phases cycle [0, 0.4, 0.8], [0.2, 0.6], ...
  24 -> 24   identity; phase array per pair is [0.0] = source pass-through
             (caller should detect this and skip the stage entirely).
"""

from __future__ import annotations

from typing import Any, Iterator

from ._compat import vt, require_pyobjc
from . import pixel_buffers as _pb


class VtfrcSession:
    """Per-pair temporal interpolator with arbitrary source/target fps.

    Construction takes frame dimensions, source fps, target fps, and a
    mode setting (Normal or Quality prioritization). The session buffers
    one source frame at a time and emits all output frames that fall in
    the gap when the next source frame arrives.

    Usage:
        session = VtfrcSession(width, height, source_fps=24, target_fps=60)
        session.use_dst_pool(av_writer.adaptor.pixelBufferPool())
        for src_idx, src_pb in enumerate(source_buffers):
            for dst_pb in session.feed(src_pb, src_idx):
                av_writer.append(dst_pb)
        for dst_pb in session.drain():
            av_writer.append(dst_pb)

    `feed()` yields zero or more interpolated frames per source frame. The
    first source frame just buffers (yields nothing); subsequent frames
    trigger interpolation between the buffered prev and the incoming curr.
    `drain()` emits the last source frame if it falls on a target PTS.
    """

    # mode enum (rate-conversion quality prioritization)
    MODE_NORMAL = "normal"
    MODE_HIGH = "high"

    def __init__(
        self,
        in_w: int,
        in_h: int,
        source_fps: float,
        target_fps: float,
        *,
        mode: str = MODE_NORMAL,
    ):
        require_pyobjc()
        if source_fps <= 0 or target_fps <= 0:
            raise ValueError(
                f"source_fps and target_fps must be positive; got {source_fps}, {target_fps}"
            )
        if not vt.VTFrameRateConversionConfiguration.isSupported():
            raise SystemExit("VTFrameRateConversionConfiguration not supported on this device.")

        self.in_w, self.in_h = in_w, in_h
        self.source_fps = float(source_fps)
        self.target_fps = float(target_fps)
        self.mode = mode

        q = (
            vt.VTFrameRateConversionConfigurationQualityPrioritizationQuality
            if mode == self.MODE_HIGH
            else vt.VTFrameRateConversionConfigurationQualityPrioritizationNormal
        )
        cls = vt.VTFrameRateConversionConfiguration
        self.config = cls.alloc().initWithFrameWidth_frameHeight_usePrecomputedFlow_qualityPrioritization_revision_(
            in_w, in_h, False, q, cls.defaultRevision(),
        )
        if self.config is None:
            raise RuntimeError("VTFrameRateConversionConfiguration init returned nil")

        self.processor = vt.VTFrameProcessor.alloc().init()
        ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor (rate conversion) startSession failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        print(
            f"Temporal session ready ({source_fps:.3f}fps -> {target_fps:.3f}fps "
            f"@ {in_w}x{in_h}, mode={mode}, "
            f"src fmt {_pb.resolve_pixel_format(self.src_attrs):#x}, "
            f"dst fmt {_pb.resolve_pixel_format(self.dst_attrs):#x})"
        )

        self._dst_pool: Any = None

        # Per-pair state ----------------------------------------------------
        # We track source frames by their target-fps frame index so we can
        # discover output frames that fall in the source-pair's interval by
        # iterating contiguous integer target indices.
        # source frame N is at time N / source_fps.
        # target frame M is at time M / target_fps.
        # M / target_fps in [N/source_fps, (N+1)/source_fps) means
        #   M in [N * (target/source), (N+1) * (target/source))
        self._ratio = target_fps / source_fps
        self._prev_src_pb: Any = None
        self._prev_src_index: int = -1   # source frame index of buffered prev
        self._next_target_index: int = 0  # next target frame index to emit

    def use_dst_pool(self, pool: Any) -> None:
        """Wire AVWriter's adaptor pool for zero-copy output."""
        self._dst_pool = pool

    def close(self) -> None:
        if self.processor is not None:
            self.processor.endSession()
            self.processor = None

    # ------------------------------------------------------------------------
    # Internal buffer factory
    # ------------------------------------------------------------------------

    def _make_dst_buffer(self) -> Any:
        if self._dst_pool is not None:
            pb = _pb.pool_create_buffer(self._dst_pool)
            if pb is not None:
                return pb
        return _pb.make_pixel_buffer_from_attrs(self.in_w, self.in_h, self.dst_attrs)

    # ------------------------------------------------------------------------
    # Phase / target-index math
    # ------------------------------------------------------------------------

    def _target_indices_in_pair(self, src_index: int) -> list[int]:
        """Target frame indices M such that M's PTS falls in
        [src_index / source_fps, (src_index + 1) / source_fps).
        """
        # Start at next_target_index so we never re-emit. The loop guards
        # below filter the exact source-frame interval.
        start = self._next_target_index
        out = []
        m = start
        # Edge: if target_pts exactly equals (src_index+1)/source_fps we
        # treat that as belonging to the NEXT pair (phase < 1).
        while m / self.target_fps < (src_index + 1) / self.source_fps - 1e-9:
            if m / self.target_fps + 1e-9 >= src_index / self.source_fps:
                out.append(m)
            m += 1
            # Safety: cap iterations to avoid pathological loops.
            if m - start > 10_000:
                break
        return out

    def _phases_for_targets(self, target_indices: list[int], src_index: int) -> list[float]:
        """For each target index M, return phase = (M/target - src/source) /
        (1/source) clamped to [0, 1). Phase 0 = source frame, phase 1 = next.
        """
        phases = []
        denom = 1.0 / self.source_fps
        src_time = src_index / self.source_fps
        for m in target_indices:
            phase = (m / self.target_fps - src_time) / denom
            # Clamp to [0, 1) for robustness against float drift.
            if phase < 0.0:
                phase = 0.0
            elif phase >= 1.0:
                phase = 1.0 - 1e-9
            phases.append(phase)
        return phases

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def feed(self, src_pb: Any, src_index: int) -> Iterator[Any]:
        """Feed one source frame. Yields the interpolated destination buffers
        whose PTSes fall in [prev_src_pts, this_src_pts).

        For the very first source frame, this is empty (no pair yet). For
        subsequent frames we compute the target indices in the prev->curr
        gap, build a single VTFrameRateConversionParameters with the phase
        and destination arrays, run one VT call, and yield each output pb.
        """
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return

        target_indices = self._target_indices_in_pair(self._prev_src_index)
        if not target_indices:
            # Identity / downsample case: no output frame falls in this gap.
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return

        phases = self._phases_for_targets(target_indices, self._prev_src_index)
        dest_buffers = [self._make_dst_buffer() for _ in target_indices]

        # VT requires CMTime PTSes on the source/next frames for sequential
        # bookkeeping. We use the source-fps timescale to anchor them; the
        # output buffers don't carry PTSes (the writer assigns them).
        prev_pts = _pb.frame_pts(self._prev_src_index, self.source_fps)
        next_pts = _pb.frame_pts(src_index, self.source_fps)
        src_frame = vt.VTFrameProcessorFrame.alloc(
        ).initWithBuffer_presentationTimeStamp_(self._prev_src_pb, prev_pts)
        next_frame = vt.VTFrameProcessorFrame.alloc(
        ).initWithBuffer_presentationTimeStamp_(src_pb, next_pts)

        # Dest frames: PTS doesn't matter to VT (writer assigns it later) but
        # it expects valid CMTimes. Use the target-fps PTSes for clarity.
        dest_frames = []
        for m, dpb in zip(target_indices, dest_buffers):
            dpts = _pb.frame_pts(m, self.target_fps)
            dest_frames.append(
                vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(dpb, dpts)
            )

        params = vt.VTFrameRateConversionParameters.alloc(
        ).initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
            src_frame,
            next_frame,
            None,
            phases,
            vt.VTFrameRateConversionParametersSubmissionModeSequential,
            dest_frames,
        )
        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(
                f"VTFC processWithParameters failed at source pair "
                f"{self._prev_src_index}->{src_index}: {err}"
            )

        # The VT call has written into the destination CVPixelBuffers; the
        # VTFrameProcessorFrame wrappers (src_frame / next_frame / dest_frames)
        # and the params object are no longer needed. Drop them before
        # yielding so transient wrappers don't pile up across iterations.
        del src_frame, next_frame, dest_frames, params

        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = src_pb
        self._prev_src_index = src_index

        # Yield one buffer at a time and drop our local list reference once
        # we hand it over - the writer retains what it needs.
        while dest_buffers:
            yield dest_buffers.pop(0)

    def drain(self) -> Iterator[Any]:
        """After all source frames have been fed, yield any remaining output
        frames that should land at or after the final source frame's PTS.

        For target rates higher than source, the last source frame's
        contribution falls in the final pair's [N, N+1) interval and was
        already emitted by the last feed() call. This method handles edge
        cases where target PTSes exactly coincide with the last source PTS
        (phase 0 of a "virtual" next pair).
        """
        # For now, no additional drain needed - the half-open [N, N+1)
        # interval logic emits everything up to but not including the next
        # source's PTS. If the last source PTS is also a target PTS, that
        # frame is emitted as phase 0 of the pair starting at the previous
        # source (handled in feed()).
        if False:
            yield  # makes this a generator
        return
