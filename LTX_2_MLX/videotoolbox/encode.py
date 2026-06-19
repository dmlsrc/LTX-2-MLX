"""High-level VideoToolbox encode helper for generate.py.

`encode_video_videotoolbox()` is the AVAssetWriter-backed sister of
`LTX_2_MLX.ffmpeg_encoder.encode_video_ffmpeg()`.  It accepts the same frame
list / audio waveform shape generate.py already builds and emits an
HEVC mp4 - no ffmpeg, no on-disk WAV unless `save_audio_sidecar=True`.

Two optional post-VAE stages can be inserted between the frame source
and the writer:

  vsr_spatial_mode={fast,balanced,image}   VideoToolbox Super Resolution.
                                            Scale is forced by the mode
                                            (fast=2x, balanced/image=4x).
  target_fps=FLOAT                          VideoToolbox Frame Rate
                                            Conversion to the requested
                                            output rate.

Both default off; with neither engaged this is a pure ffmpeg->AVWriter
swap for HEVC output.  Dest pools are wired for zero-copy: VSR writes
into VTFRC's source pool (or AVWriter's when no VTFRC), and VTFRC writes
into AVWriter's pool - frames never round-trip through main memory once
they enter the chain.

Frames may be supplied as:
  - list[(H,W,3) uint8]   - generate.py's normalized output
  - list[(H,W,4) float16] - future streaming-VAE path (kept in bf16
                            precision through to VSR's RGBAHalf source)
  - (T,H,W,3) uint8 ndarray
  - Iterator yielding any of the above per-frame shapes

The first frame is peeked to discover (H, W) and dtype; the rest are
consumed lazily so a streaming source doesn't have to materialize.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..progress import StackedPhaseBars
from . import pixel_buffers as _pb
from ._compat import autorelease_pool, require_pyobjc
from .audio import AudioTrack
from .temporal import VtfrcSession
from .vsr import VsrSession, scale_for_mode
from .writer import HEVC_PROFILE_MAIN10, HEVC_PROFILE_MAIN422_10, AVWriter


def _human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _peek_frames(
    frames: Any,
) -> tuple[Any, Iterator[Any], int | None]:
    """Return (first_frame, full_iterator, total_or_none).

    Accepts list / tuple / ndarray (3D peek-first dim) / iterator. For a
    4D ndarray we iterate along axis 0. The returned iterator includes
    the first frame. `total` is the source frame count when known from
    the input shape; None for pure iterators where we'd have to consume
    the stream to count.
    """
    if getattr(frames, "ndim", None) == 4:  # 4D mlx or numpy array
        if frames.shape[0] == 0:
            raise ValueError("encode_video_videotoolbox: empty frames array")
        return (
            frames[0],
            (frames[i] for i in range(frames.shape[0])),
            int(frames.shape[0]),
        )
    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValueError("encode_video_videotoolbox: empty frames list")
        return frames[0], iter(frames), len(frames)
    # Generic iterator: consume one frame, then chain it back.
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("encode_video_videotoolbox: empty frames iterator") from None
    return first, itertools.chain([first], it), None


def _pick_hevc_profile(vsr_spatial_mode: str | None, encode_chroma: str) -> str:
    """4:2:2 (Main42210) when VSR carries fp16 RGBA (balanced/image);
    4:2:0 (Main10) otherwise (no VSR or fast = NV12 source).
    """
    if encode_chroma == "420":
        return HEVC_PROFILE_MAIN10
    if encode_chroma == "422":
        return HEVC_PROFILE_MAIN422_10
    return (
        HEVC_PROFILE_MAIN422_10
        if vsr_spatial_mode in ("balanced", "image")
        else HEVC_PROFILE_MAIN10
    )


def _normalize_audio_for_track(audio_waveform: Any) -> Any:
    """Final encoder boundary: (B,C,T) or (C,T) MLX/ndarray -> MLX (C,T) f32.

    AudioTrack expects (channels, samples). Pipeline outputs are (B,C,T).
    """
    arr = mx.array(audio_waveform, dtype=mx.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"audio_waveform must be (B,C,T) or (C,T); got shape {arr.shape}"
        )
    return arr


def _allocate_writer_src_buffer(adaptor: Any, width: int, height: int, fmt: int) -> Any:
    """Pull a writer-source buffer (zero-copy when the pool is ready)."""
    pool = adaptor.pixelBufferPool() if adaptor is not None else None
    if pool is not None:
        pb = _pb.pool_create_buffer(pool)
        if pb is not None:
            return pb
    attrs = {
        "PixelFormatType": fmt,
        "Width": width,
        "Height": height,
        "IOSurfaceProperties": {},
    }
    return _pb.make_pixel_buffer_from_attrs(width, height, attrs)


def encode_video_videotoolbox(
    frames: Sequence[Any] | Iterable[Any],
    output_path: str | Path,
    *,
    fps: float,
    audio_waveform: Any = None,
    audio_sample_rate: int | None = None,
    audio_bit_depth: str = "float32",
    save_audio_sidecar: bool = False,
    audio_onset_trim_mode: str = "auto",
    audio_onset_trim_ms: float | None = None,
    vsr_spatial_mode: str | None = None,
    target_fps: float | None = None,
    vsr_temporal_mode: str = "normal",
    vsr_save_original: bool = False,
    encode_quality: float = 0.65,
    encode_chroma: str = "auto",
    n_source_frames: int | None = None,
    progress_stack: StackedPhaseBars | None = None,
    audio_codec: str = "alac",
    verbose: bool = True,
) -> Path:
    """Encode frames into an HEVC mp4 via AVAssetWriter (no ffmpeg).

    Returns the actual output path; rewrites the extension to .mp4 if the
    caller supplied something else (matches encode_video_ffmpeg() behavior for
    the ffmpeg `default` tier - both produce .mp4).

    `audio_waveform` is anything castable to (B,C,T) or (C,T) MLX/NumPy
    float32. Pass None for video-only.

    `vsr_spatial_mode`:
      None        no spatial upscale (writer source = NV12).
      "fast"      VTLowLatency VSR, scale 2x, input <= 960x960.
      "balanced"  VT HQ VSR Video mode, scale 4x; prev-frame chain.
      "image"     VT HQ VSR Image mode, scale 4x; per-frame deterministic.

    `target_fps`:
      None or equal to fps   no temporal interpolation.
      otherwise              route through VTFrameRateConversion.

    `audio_bit_depth` is accepted for API parity with encode_video_ffmpeg();
    AVAssetWriter always consumes float32 PCM internally regardless.

    `audio_onset_trim_mode` / `audio_onset_trim_ms` route through to
    `LTX_2_MLX.audio.onset.mitigate_onset()` before the AudioTrack is
    built.  The same cleaned waveform feeds both the muxed audio track
    and the optional sidecar (`save_audio_sidecar=True`).

    `vsr_save_original`: when True AND any VT post-processing is engaged
    (VSR spatial or VTFRC temporal), also write the un-processed source-
    resolution source-fps mp4 alongside the requested output, as
    `<stem>_orig.mp4`.  Both files share the same AudioTrack so each is
    playable standalone.  No-op when neither VSR nor VTFRC is active
    (there's nothing for "original" to differ from).  The companion
    writer mirrors the primary's HEVC profile (RGBAHalf + Main42210 for
    VSR HQ; NV12 + Main10 for VSR fast / VTFRC-only) so the A/B
    comparison isn't precision-floor mismatched.  Implementation cost
    per source frame is one additional source-buffer upload + one HEVC
    HW append; the AVAssetWriter's pump runs on its own GCD queue, so
    the second encode is largely parallel to the primary.
    """
    require_pyobjc()
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Peek first frame to learn dimensions + dtype ---------------------
    first, frame_iter, peeked_total = _peek_frames(frames)
    # Explicit caller-supplied total wins over the peeked one (lets
    # streaming-decode callers pass the known frame count even though
    # they hand the encoder an unsized iterator).
    if n_source_frames is None:
        n_source_frames = peeked_total
    if first.ndim != 3:
        raise ValueError(
            f"frame must be (H,W,C); got shape {first.shape}, dtype {first.dtype}"
        )
    in_h, in_w, in_c = first.shape
    if in_c not in (3, 4):
        raise ValueError(f"frame channel dim must be 3 or 4; got {in_c}")

    do_vsr = vsr_spatial_mode is not None
    if do_vsr:
        scale = scale_for_mode(vsr_spatial_mode)
        out_w, out_h = in_w * scale, in_h * scale
    else:
        scale = 1
        out_w, out_h = in_w, in_h

    do_temporal = target_fps is not None and abs(target_fps - fps) > 1e-6
    if not do_temporal:
        target_fps = fps  # writer fps is the effective output rate

    profile = _pick_hevc_profile(vsr_spatial_mode, encode_chroma)

    # ---- Setup phase ------------------------------------------------------
    # VsrSession / VtfrcSession / AVWriter constructors each print to stdout
    # unconditionally ("VSR session ready ...", "Temporal session ready ...",
    # "[encode] AVAssetWriter -> ..."), plus we add the chain description
    # and the optional audio sidecar line.  When `progress_stack` is alive
    # (the caller is showing a "VAE chunks" bar above us), those raw prints
    # would stomp on the bar row.  Redirect them through a StringIO and
    # emit the captured block via `progress_stack.write()` so the lines
    # land cleanly above the bars (scroll-message-above-bars style); when no stack is
    # supplied, the prints flow normally to stdout.
    _setup_buf: io.StringIO | None = None
    if progress_stack is not None:
        _setup_buf = io.StringIO()
        _setup_ctx = contextlib.redirect_stdout(_setup_buf)
    else:
        _setup_ctx = contextlib.nullcontext()

    with _setup_ctx:
        # VSR session
        vsr: VsrSession | None = None
        if do_vsr:
            vsr = VsrSession(in_w, in_h, mode=vsr_spatial_mode, fps=fps)

        # VTFRC session
        vtfrc: VtfrcSession | None = None
        if do_temporal:
            vtfrc = VtfrcSession(
                out_w, out_h,
                source_fps=fps, target_fps=target_fps,
                mode=vsr_temporal_mode,
            )

        # Audio track
        audio_track: AudioTrack | None = None
        if audio_waveform is not None:
            if audio_sample_rate is None:
                raise ValueError(
                    "audio_sample_rate is required when audio_waveform is provided"
                )
            # Sequence-start onset mitigation.  Runs before the AudioTrack
            # build while the generated waveform can still stay MLX-native;
            # `_normalize_audio_for_track` is the final NumPy byte-boundary
            # adapter for CoreMedia.  See
            # LTX_2_MLX.audio.onset and docs/AUDIO_ISSUES.md ->
            # "Sequence-Start Audio Spike".
            from ..audio import DEFAULT_TRIM_MS, mitigate_onset

            onset_trim_ms = (
                audio_onset_trim_ms
                if audio_onset_trim_ms is not None
                else DEFAULT_TRIM_MS
            )
            onset_result = mitigate_onset(
                audio_waveform, int(audio_sample_rate),
                mode=audio_onset_trim_mode, trim_ms=onset_trim_ms,
            )
            arr = _normalize_audio_for_track(onset_result.samples)
            if verbose and onset_result.applied:
                print(f"  audio onset: {onset_result.detail}")
            audio_track = AudioTrack(arr, sample_rate=int(audio_sample_rate))

        # Pick writer source format.  When VSR or VTFRC is active, the
        # writer source = the last stage's dst.  When neither is active,
        # the writer source = NV12 and we upload through CoreImage (keeps
        # the encoder's RGB->YUV cost in one place).
        if vtfrc is not None:
            writer_src_fmt = _pb.resolve_pixel_format(vtfrc.dst_attrs)
        elif vsr is not None:
            writer_src_fmt = _pb.resolve_pixel_format(vsr.dst_attrs)
        else:
            writer_src_fmt = _pb.PIX_NV12

        # Writer + pool wiring.  Zero-copy hookups: VTFRC writes into the
        # writer's adaptor pool when active; VSR writes into its own dst
        # pool when VTFRC is between (a copy at the VT call boundary), or
        # directly into the writer's adaptor pool when there is no VTFRC.
        writer = AVWriter(
            output_path,
            width=out_w, height=out_h, fps=target_fps,
            source_pixel_format=writer_src_fmt,
            profile=profile,
            quality=encode_quality,
            label="encode",
            audio_track=audio_track,
            audio_codec=audio_codec,
        )
        if vtfrc is not None:
            vtfrc.use_dst_pool(writer.adaptor.pixelBufferPool())
        elif vsr is not None:
            vsr.use_dst_pool(writer.adaptor.pixelBufferPool())

        # Optional "save the un-processed original alongside the VSR/VTFRC
        # result" companion writer.  Only meaningful when some VT post-
        # processing is engaged; otherwise the primary writer IS the
        # original and a duplicate adds zero value.  Shares the same
        # AudioTrack - CMSampleBuffer is fresh per make_sample_buffer()
        # call so two GCD pumps on the same track are safe.
        #
        # Source format + HEVC profile mirror the primary writer's
        # precision envelope (`_pick_hevc_profile`): when the user opted
        # into VSR HQ (balanced / image), the primary is RGBAHalf source
        # -> HEVC Main42210 (4:2:2 10-bit) and the original should match
        # so the A/B comparison isn't a precision-floor mismatch.  For
        # VSR fast and VTFRC-only the primary is NV12 -> Main10 (4:2:0
        # 10-bit); the original matches that too - upgrading the
        # original's source format past what its companion uses adds
        # bits the encoder would just throw away.  When the input frames
        # are fp16 RGBA from the streaming-VAE path, RGBAHalf preserves
        # them all the way to the encoder's internal 4:2:2 conversion;
        # uint8 RGB input goes through upload_frame_to_buffer's RGBA
        # promotion path with no loss vs. the NV12 alternative.
        writer_orig: AVWriter | None = None
        orig_path: Path | None = None
        do_save_original = vsr_save_original and (vsr is not None or vtfrc is not None)
        if do_save_original:
            orig_path = output_path.with_name(
                f"{output_path.stem}_orig{output_path.suffix}"
            )
            if vsr_spatial_mode in ("balanced", "image"):
                orig_src_fmt = _pb.PIX_RGBAHALF
                orig_profile = HEVC_PROFILE_MAIN422_10
            else:
                orig_src_fmt = _pb.PIX_NV12
                orig_profile = HEVC_PROFILE_MAIN10
            writer_orig = AVWriter(
                orig_path,
                width=in_w, height=in_h, fps=fps,
                source_pixel_format=orig_src_fmt,
                profile=orig_profile,
                quality=encode_quality,
                label="encode_orig",
                audio_track=audio_track,
                audio_codec=audio_codec,
            )

        # Optional audio sidecar WAV.
        sidecar_path: Path | None = None
        if audio_track is not None and save_audio_sidecar:
            sidecar_path = output_path.with_suffix(".wav")
            audio_track.save_wav(sidecar_path)
            if verbose:
                print(
                    f"  audio sidecar: {sidecar_path}  "
                    f"({audio_bit_depth}, {audio_track.sample_rate} Hz)"
                )

        # Chain description (above the encode bar so users see what's running).
        stages: list[str] = []
        if vsr is not None:
            stages.append(f"VSR={vsr_spatial_mode}({scale}x)")
        if vtfrc is not None:
            stages.append(f"VTFRC={fps:g}->{target_fps:g}fps")
        chain = " + ".join(stages) if stages else "passthrough"
        if verbose:
            print(f"  encode (videotoolbox): {chain} -> HEVC {profile}")
            print(f"  -> {output_path}")
            if writer_orig is not None:
                print(
                    f"  + original passthrough -> HEVC {orig_profile} "
                    f"-> {orig_path}"
                )

    # If we captured the setup output, route it above the caller's bar
    # stack now - single bars.write() call so the bars stay coherent.
    if _setup_buf is not None:
        _setup_msg = _setup_buf.getvalue().rstrip("\n")
        if _setup_msg:
            progress_stack.write(_setup_msg)

    # PhaseBar gives a stable, fixed-column progress display.  Total is
    # known for list / ndarray inputs; iterators get an indeterminate bar
    # (count-only, no ETA).  Suppress entirely when verbose=False.
    #
    # When `progress_stack` is provided, the encoder shares the caller's
    # stack - useful when generate.py wants a "VAE chunks" bar above the
    # encoder's "VT encode" bar, both rendered in one cohesive display.
    # The caller owns close() in that mode; we just add our row.
    bars: StackedPhaseBars | None = None
    owns_bars = False
    pbar = None
    if verbose:
        if progress_stack is not None:
            bars = progress_stack
        else:
            bars = StackedPhaseBars()
            owns_bars = True
        pbar = bars.add(
            total=n_source_frames,
            desc="VT encode",
            unit="frame",
        )

    started = time.perf_counter()
    n_in = 0
    n_out = 0
    n_orig = 0
    try:
        for src_frame in frame_iter:
            with autorelease_pool():
                # Companion "original" writer: source frame -> orig_src_fmt
                # (NV12 or RGBAHalf, matching the primary's precision
                # envelope) -> append.  Independent of VSR/VTFRC chain;
                # uses its own source buffer pool.  AVAssetWriter's audio
                # + video pumps run on their own GCD queues so this second
                # append is largely parallel to the primary chain's encode
                # pass.  upload_frame_to_buffer dispatches on the buffer's
                # pixel format, so the same call works for both src
                # formats and both source dtypes (uint8 RGB / fp16 RGBA).
                if writer_orig is not None:
                    orig_pb = _allocate_writer_src_buffer(
                        writer_orig.adaptor, in_w, in_h, orig_src_fmt,
                    )
                    _pb.upload_frame_to_buffer(src_frame, orig_pb)
                    writer_orig.append(orig_pb)
                    n_orig += 1
                    del orig_pb

                if vsr is not None:
                    src_pb = vsr.upscale_to_buffer(src_frame, n_in)
                else:
                    src_pb = _allocate_writer_src_buffer(
                        writer.adaptor, in_w, in_h, writer_src_fmt,
                    )
                    _pb.upload_frame_to_buffer(src_frame, src_pb)

                if vtfrc is not None:
                    for out_pb in vtfrc.feed(src_pb, n_in):
                        writer.append(out_pb)
                        n_out += 1
                        del out_pb
                else:
                    writer.append(src_pb)
                    n_out += 1
                del src_pb
            n_in += 1
            if pbar is not None:
                pbar.update(1)
            # Periodic janitorial work: CIContext caches + src pool drain.
            if n_in % 64 == 0:
                _pb.clear_ci_caches()
                if vsr is not None:
                    vsr.flush_pools()
        # VTFRC drain (no-op today; kept for symmetry with vsr_harness).
        if vtfrc is not None:
            for out_pb in vtfrc.drain():
                writer.append(out_pb)
                n_out += 1
                del out_pb
    finally:
        if bars is not None and owns_bars:
            bars.close()
        writer.finish()
        if writer_orig is not None:
            writer_orig.finish()
        if vtfrc is not None:
            vtfrc.close()
        if vsr is not None:
            vsr.close()

    if verbose:
        elapsed = time.perf_counter() - started
        size = output_path.stat().st_size
        orig_part = ""
        if writer_orig is not None and orig_path is not None:
            orig_size = orig_path.stat().st_size
            orig_part = (
                f" + original {_human_size(orig_size)} "
                f"({n_orig} src frame{'s' if n_orig != 1 else ''})"
            )
        done_msg = (
            f"  done: {_human_size(size)} in {elapsed:.1f}s "
            f"({n_in} src frame{'s' if n_in != 1 else ''}, "
            f"{n_out} written){orig_part}"
        )
        # In caller-managed-stack mode the bars are still alive at this
        # point (the caller closes them after we return), so a raw print
        # would stomp on the bottom bar's row.  Route through bars.write()
        # with position="below" so the visual order in the persisted
        # scrollback is "bars (frozen at 100%) -> done summary".  The
        # stack is reset by that call; the caller's `with` cleanup is
        # a safe no-op afterwards (StackedPhaseBars.close early-returns
        # when _bars is empty).
        if progress_stack is not None and not owns_bars:
            progress_stack.write(done_msg, position="below")
        else:
            print(done_msg)

    return output_path
