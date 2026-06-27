#!/usr/bin/env python3
"""VAE-decode (or read MP4) and pump frames through VideoToolbox VSR +
optional temporal frame-rate conversion. Writes the upscaled MP4 directly
via AVAssetWriter - no ffmpeg, no PNG round-trip, no disk WAV by default.

Usage
-----
    # Latent path: VAE-decode the NPZ sidecar, then VSR it.
    scripts/vsr_harness.py --latent run.npz --weights $LTX_DEFAULT_WEIGHTS_PATH \
        --output-dir outputs/vsr/run1

    # Same, with audio muxed and frame rate doubled to 48 fps.
    scripts/vsr_harness.py --latent run.npz --weights ... \
        --output-dir outputs/vsr/run1 --audio --target-fps 48

    # Video path: skip VAE; VSR an existing clip. Add --audio to carry the
    # source file's audio track through to the upscaled MP4.
    scripts/vsr_harness.py --video clip.mp4 \
        --output-dir outputs/vsr/run2 --spatial-mode balanced --audio

    # Process only the middle: upscale the [5s, 8s) window of a long clip.
    # --start/--end (and --max-frames) accept frames (120), seconds (5s / 1.5),
    # or a clock string (0:05, 1:02:03). --video seeks natively to the window.
    scripts/vsr_harness.py --video clip.mp4 \
        --output-dir outputs/vsr/run3 --start 0:05 --end 0:08 --audio

Spatial modes (VideoToolbox-imposed; scale is implied by the mode)
------------------------------------------------------------------
    fast      VTLowLatencySuperResolutionScalerConfiguration.  Scale 2x.
              Input must fit between 96x96 and 960x960.  Per-frame,
              no temporal context.
    balanced  VTSuperResolutionScalerConfiguration, InputType=Video.
              Scale 4x.  Downloadable model (auto-fetched on first use).
              Uses previous source + previous output frames to inform the
              upscale.  Default for video.  Tends to be slightly crisper
              on motion at the cost of slightly higher frame-to-frame
              variation than image mode.
    image     VTSuperResolutionScalerConfiguration, InputType=Image.  Scale 4x.
              Per-frame deterministic upscale, no prev-frame feedback.
              Apple documents this as for stills, but on real video it's a
              legitimate alternative - slightly softer per-frame detail
              than balanced but measurably smoother frame-to-frame (lower
              temporal second-difference).  Use scripts/compare_video_shimmer.py
              to A/B the two modes on your own content.

Temporal modes (only relevant when --target-fps is set)
-------------------------------------------------------
    normal    Default.  Fast and adequate for ~2x rate-up.
    high      VTFrameRateConversion's QualityPrioritizationQuality - more
              compute per interpolated frame, cleaner motion.

The VAE decoder defaults track LTX_2_MLX/generate.py's happy path
(native backend + zero spatial padding) via the encode_modes_harness
helpers. Chunks are cast to fp16 RGBA inside MLX so the full bf16
precision is preserved through to VSR's RGBAHalf source format -
quantization happens at the destination (either CIContext rendering
to NV12 for LL, or AVAssetWriter encoding to HEVC for HQ).

Known limitation for `--video` on edited footage
------------------------------------------------
`--spatial-mode balanced` chains previous-frame state through VSR for
temporal coherence. Across a hard cut that's the wrong context and can
produce ghosting around the cut frame. LTX latents are single-shot
generations so this is moot for `--latent`. For edited MP4s, enable
`--cut-detect` to reset the chain at hard cuts.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.progress import StackedPhaseBars
from LTX_2_MLX.videotoolbox import (
    AudioTrack,
    AVWriter,
    CutDetector,
    VsrSession,
    VtfrcSession,
    autorelease_pool,
    require_pyobjc,
)
from LTX_2_MLX.videotoolbox import pixel_buffers as _pb
from LTX_2_MLX.videotoolbox import video_reader as _vr
from LTX_2_MLX.videotoolbox.comparison import render_comparison
from LTX_2_MLX.videotoolbox.images import save_image
from LTX_2_MLX.videotoolbox.writer import (
    HEVC_PROFILE_MAIN10,
    HEVC_PROFILE_MAIN422_10,
)

NATIVE_FPS = 24.0


def parse_time_or_frames(spec: str, fps: float) -> int:
    """Convert a position/duration spec to a frame count at `fps`.

    Accepted forms (case-insensitive):
        "120"         bare integer  -> 120 frames
        "120f"        explicit f    -> 120 frames
        "5s", "2.5s"  seconds       -> round(seconds * fps) frames
        "1.5"         bare decimal  -> seconds (a fractional frame is meaningless)
        "1:30"        mm:ss         -> seconds -> frames
        "1:02:03"     hh:mm:ss      -> seconds -> frames
        "0:04.5"      mm:ss.frac    -> seconds -> frames

    The bare-integer-is-frames / bare-decimal-is-seconds split keeps existing
    integer `--max-frames N` invocations meaning frames, while letting any
    time be given as seconds or a clock string. Returns a non-negative int.
    """
    s = str(spec).strip().lower()
    if not s:
        raise ValueError("empty time/frame spec")
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            hh, (mm, ss) = "0", parts
        elif len(parts) == 3:
            hh, mm, ss = parts
        else:
            raise ValueError(f"bad time spec {spec!r} (use mm:ss or hh:mm:ss)")
        seconds = int(hh) * 3600 + int(mm) * 60 + float(ss)
        frames = round(seconds * fps)
    elif s.endswith("f"):
        frames = int(s[:-1])
    elif s.endswith("s"):
        frames = round(float(s[:-1]) * fps)
    elif "." in s:
        frames = round(float(s) * fps)
    else:
        frames = int(s)
    if frames < 0:
        raise ValueError(f"time/frame spec {spec!r} is negative")
    return int(frames)


def resolve_trim(
    start_spec: str | None, end_spec: str | None, fps: float, total_frames: int,
) -> tuple[int, int | None]:
    """Resolve --start/--end specs to a half-open frame window [start, end).

    `end` is None for an open-ended window. Clamps end to the input length and
    rejects an empty or out-of-range window with a clean SystemExit.
    """
    try:
        start_frame = parse_time_or_frames(start_spec, fps) if start_spec else 0
        end_frame = parse_time_or_frames(end_spec, fps) if end_spec else None
    except ValueError as e:
        raise SystemExit(f"bad --start/--end value: {e}") from None
    if end_frame is not None and end_frame <= start_frame:
        raise SystemExit(
            f"--end ({end_frame}f) must be greater than --start ({start_frame}f)"
        )
    if total_frames and start_frame >= total_frames:
        raise SystemExit(
            f"--start ({start_frame}f) is at or past the input length "
            f"({total_frames} frames)"
        )
    if total_frames and end_frame is not None:
        end_frame = min(end_frame, total_frames)
    return start_frame, end_frame


# ---------------------------------------------------------------------------
# MLX-side chunk conversion. Lives in the harness (not the videotoolbox
# package) because it depends on LTX_2_MLX.model.video_vae internals.
# ---------------------------------------------------------------------------

# A/B toggle (env): VSR_CHUNK_AS_ARRAY=1 returns one big ndarray per chunk
# (300 MiB resident until chunk-end). Default 0 returns a list of per-frame
# ndarrays so each frame's ~1.2 MiB can be freed as the inner loop progresses.
#
# Measured A/B (721-frame latent, bare `time`, no instrumentation):
#   --vae-tiling auto  list   wall 142.2s  VAE 109.3s  5.07 fps
#   --vae-tiling auto  array  wall 142.3s  VAE 109.3s  5.07 fps
#   --vae-tiling single list  wall 151.1s  VAE  58.3s  4.77 fps
#   --vae-tiling single array wall 165.5s  VAE  70.2s  4.36 fps
# Tiled mode: list/array indistinguishable - chunks are small enough that
# list-vs-array allocation overhead is in the noise. Single-shot: list is
# ~10% faster wall and ~17% faster through the VAE itself. So this env var
# is NOT a no-op - it controls a real perf difference for `--vae-tiling off`.
# List stays the default because it's the faster path everywhere AND lets
# the inner loop drop per-frame memory as it goes.
import os as _os

_CHUNK_AS_ARRAY = _os.environ.get("VSR_CHUNK_AS_ARRAY", "0") == "1"


def chunk_to_rgba_fp16(chunk: Any, mx_mod: Any):
    """(B,3,T,H,W) bf16 in [-1,1] -> list[(H,W,4) fp16] per-frame arrays.

    Direct path for VSR's RGBAHalf source format and for CIImage's
    kCIFormatRGBAh upload to NV12. Skips uint8 quantization that
    chunk_to_uint8 would impose, so the VAE's full bf16 precision
    survives into VSR.

    Returns a list of independently-allocated per-frame ndarrays rather than
    one big (T,H,W,4) array. The downstream inner loop can then null out
    `chunk[i]` once a frame is consumed, freeing that frame's ~1.2 MB back
    to the OS - so the resident chunk memory tapers as we work through it
    instead of sitting at full size until chunk-end. Allocator overhead is
    one mmap per frame (cheap; macOS mmaps allocations >= 16 KiB directly).
    """
    B, C, T, H, W = chunk.shape
    rescaled = mx_mod.clip((chunk + 1.0) * 0.5, 0.0, 1.0).astype(mx_mod.float16)
    alpha = mx_mod.ones((B, 1, T, H, W), dtype=mx_mod.float16)
    rgba = mx_mod.concatenate([rescaled, alpha], axis=1)
    transposed = mx_mod.transpose(rgba, (0, 2, 3, 4, 1))  # (B, T, H, W, 4)
    mx_mod.eval(transposed)
    if _CHUNK_AS_ARRAY:
        arr = mx_mod.contiguous(transposed)
        result: Any = arr[0] if arr.ndim == 5 else arr
    else:
        # List of per-frame mx arrays so each frame's memory can be freed
        # independently by the main loop. mx.contiguous gives each frame its own
        # buffer, so dropping a frame lets MLX release it without pinning the
        # whole chunk's Metal state across the loop.
        result = [mx_mod.contiguous(transposed[0, t]) for t in range(T)]
    # Drop refs to all MLX intermediates AND force the cache to release.
    # Without clear_cache here, the rescaled/alpha/rgba/transposed Metal
    # buffers (which can be GiB-scale for single-shot decodes) sit in MLX's
    # cache for the entire downstream inner loop - only released when the
    # generator resumes after the loop drains. The numpy result is already
    # an independent Python-owned copy, so MLX state is safe to drop now.
    del rescaled, alpha, rgba, transposed
    try:
        mx_mod.clear_cache()
    except Exception:
        pass
    return result


def make_video_decoder_default(
    weights_path: str, compute_dtype: Any,
):
    """generate.py's happy-path defaults via encode_modes_harness."""
    from scripts.encode_modes_harness import make_video_decoder
    return make_video_decoder(weights_path, compute_dtype)


def latent_dims(latent: Any) -> tuple[int, int, int]:
    _, _, latent_frames, latent_height, latent_width = latent.shape
    n_frames = 1 + (latent_frames - 1) * 8
    height = latent_height * 32
    width = latent_width * 32
    return n_frames, height, width


def plan_vae_tiling(latent: Any) -> tuple[Any, int, str]:
    """Decide the tiling cfg + chunk count up front.

    Returns (cfg, n_chunks, human_description). `cfg` is the TilingConfig
    (or None for single-shot decode). Pure CPU/dim arithmetic - no GPU
    work - so it's cheap to call before any tqdm bar starts (which is
    what avoids clobbering the bar with VAE tiling status mid-stream).
    """
    from LTX_2_MLX.model.video_vae.tiling import TilingConfig

    n_frames, height, width = latent_dims(latent)
    cfg = TilingConfig.auto(
        height=height, width=width, num_frames=n_frames,
    )
    if cfg is None:
        return cfg, 1, f"off (single-shot decode of {n_frames} frames)"

    sp = cfg.spatial_config
    tp = cfg.temporal_config
    spatial_desc = (
        f"spatial tile={sp.tile_size_in_pixels} overlap={sp.tile_overlap_in_pixels}"
        if sp else "no spatial"
    )
    temporal_desc = (
        f"temporal tile={tp.chunk_size_in_frames} overlap={tp.chunk_overlap_in_frames}"
        if tp else "no temporal"
    )
    if tp is not None:
        tile = tp.chunk_size_in_frames
        overlap = tp.chunk_overlap_in_frames
        step = max(1, tile - overlap)
        n_chunks = max(1, -(-(n_frames - overlap) // step))
    else:
        n_chunks = 1
    return cfg, n_chunks, f"{spatial_desc}, {temporal_desc}"


def iter_latent_chunks(
    latent: Any,
    decoder: Any,
    *,
    cfg: Any,
    mx_mod: Any,
    output_format: str = "uint8_rgb",
    single_pass: bool = False,
) -> Iterator[mx.array]:
    """Yield decoded chunks. output_format selects the conversion:
       "uint8_rgb"  -> (T,H,W,3) uint8  (for LowLatency VSR / NV12 source)
       "fp16_rgba"  -> (T,H,W,4) fp16   (for HighQuality VSR / RGBAHalf source)
    """
    from LTX_2_MLX.model.video_vae.tiling import decode_single_pass, decode_streaming
    from scripts.encode_modes_harness import chunk_to_uint8

    convert = chunk_to_rgba_fp16 if output_format == "fp16_rgba" else chunk_to_uint8

    if single_pass:
        # --vae-tiling single: one whole-clip decode. decode_single_pass logs whether
        # the frame count crosses the int32 boundary (frames past it decode white).
        out = convert(decode_single_pass(latent, decoder), mx_mod)
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        return

    # decode_streaming handles cfg=None (no spatial tiling + default temporal
    # chunking), so this streams chunk-by-chunk for every case -- no whole-video
    # accumulate. convert() quantizes each chunk at the destination format.
    for chunk in decode_streaming(latent, decoder, cfg, show_progress=False):
        out = convert(chunk, mx_mod)
        # convert() clears the cache; `chunk` is the only MLX tensor still
        # live, so drop it + clear before yielding to the downstream loop.
        del chunk
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        del out


# ---------------------------------------------------------------------------
# Audio decode (latent only)
# ---------------------------------------------------------------------------

def _decode_audio_track(audio_latent: Any, weights: str, compute_dtype: Any) -> AudioTrack:
    """Decode the audio latent through the audio VAE + vocoder into an
    in-memory AudioTrack. No disk WAV unless the caller asks for a sidecar.
    """
    import mlx.core as mx

    from scripts.decode_latent_debug import decode_audio_latent, make_audio_decoder_and_vocoder

    print("Decoding audio latent (audio VAE + vocoder)...")
    audio_decoder, vocoder, sample_rate = make_audio_decoder_and_vocoder(weights, compute_dtype)
    waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx, onset_mode="auto")
    arr = waveform
    if arr.ndim == 3:
        arr = arr[0]
    track = AudioTrack(arr, sample_rate=int(sample_rate))
    print(f"  audio: {track.channels}ch, {track.sample_rate} Hz, {track.n_samples} samples")
    del waveform, audio_decoder, vocoder
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return track


def _read_audio_track_from_video(mp4_path: Path) -> AudioTrack | None:
    """Read the audio track of an MP4/MOV into an in-memory AudioTrack.

    Uses AVFoundation's AVAudioFile (via videotoolbox.audio.read_wav), which
    decodes the container's audio stream straight into a (channels, frames)
    float32 MLX array - no ffmpeg, no disk WAV. Returns None if the file has
    no audio track.
    """
    from LTX_2_MLX.videotoolbox.audio import read_wav

    print(f"[setup] reading audio track from {mp4_path}")
    try:
        sample_rate, samples = read_wav(mp4_path)
    except Exception as e:
        # No audio track (or an unsupported audio format) - carry on silent.
        print(f"[setup] no usable audio track ({type(e).__name__}: {e}); output will be silent")
        return None
    track = AudioTrack(samples, sample_rate=int(sample_rate))
    print(f"  audio: {track.channels}ch, {track.sample_rate} Hz, {track.n_samples} samples")
    return track


# ---------------------------------------------------------------------------
# HEVC profile selection
# ---------------------------------------------------------------------------

def _pick_hevc_profile(spatial_mode: str, encode_chroma: str) -> str:
    """auto picks 4:2:2 for HQ modes (RGBAHalf preserves chroma), 4:2:0 for fast (LL/NV12)."""
    if encode_chroma == "420":
        return HEVC_PROFILE_MAIN10
    if encode_chroma == "422":
        return HEVC_PROFILE_MAIN422_10
    return HEVC_PROFILE_MAIN422_10 if spatial_mode in ("balanced", "image") else HEVC_PROFILE_MAIN10


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    from LTX_2_MLX.generate import sanitize_output_prefix
    stem = f"{sanitize_output_prefix(args.output_prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pre_dir = out_root / f"{stem}_pre"
    post_dir = out_root / f"{stem}_post"
    if args.save_pre_frames:
        pre_dir.mkdir(parents=True, exist_ok=True)
    if args.save_post_frames:
        post_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output stem: {stem}")

    audio_track: AudioTrack | None = None
    src_transform: Any = None  # source rotation/flip (set on the --video path)
    # Input frame window [loop_win_start, loop_win_end). The --video reader
    # trims at decode time (efficient seek), so for that path these stay
    # (0, None) and the trim happens upstream; the --latent path enforces the
    # window in the main loop instead.
    loop_win_start, loop_win_end = 0, None
    win_start, win_end = 0, None  # resolved input window (both paths set these)

    # ---- Input source ------------------------------------------------------
    if args.latent:
        from scripts.decode_latent_debug import load_latents, parse_dtype

        print(f"[setup] VAE-decoding latent: {args.latent}")
        t = time.perf_counter()
        latent, audio_latent = load_latents(args.latent, mx, "auto", stage=args.latent_stage)
        compute_dtype = parse_dtype(mx, args.vae_dtype)
        print(
            f"[setup] load_latents done in {time.perf_counter() - t:.2f}s "
            f"(video_latent={tuple(latent.shape)}, "
            f"audio_latent={'yes' if audio_latent is not None else 'no'})"
        )

        # Audio decode runs serially - threading it against VAE chunk 1 was
        # tried and made total setup slower (MLX serializes work across
        # threads on the single Metal scheduler).
        if audio_latent is not None and args.audio:
            t = time.perf_counter()
            audio_track = _decode_audio_track(audio_latent, args.weights, compute_dtype)
            print(f"[setup] audio decode in {time.perf_counter() - t:.2f}s")
        # The audio latent is consumed at this point; free it (and any other
        # post-audio state) so the Metal heap is clean before VAE decode.
        del audio_latent
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        t = time.perf_counter()
        decoder = make_video_decoder_default(
            args.weights, compute_dtype,
        )
        print(f"[setup] video VAE loaded in {time.perf_counter() - t:.2f}s")
        total_frames, in_h, in_w = latent_dims(latent)
        source_fps = args.source_fps

        # --start/--end trim the decoded frames. The VAE still decodes the whole
        # latent (it is temporally tiled, not seekable), so the window is
        # enforced in the main loop rather than at the reader.
        win_start, win_end = resolve_trim(args.start, args.end, source_fps, total_frames)
        if win_start or win_end is not None:
            loop_win_start, loop_win_end = win_start, win_end
            total_frames = (win_end if win_end is not None else total_frames) - win_start
            print(f"[setup] input trim: frames [{win_start}, "
                  f"{win_end if win_end is not None else 'end'}) -> {total_frames} frames")

        if args.vae_tiling == "single":
            vae_cfg, n_vae_chunks, vae_tiling_desc = None, 1, "single (one decode)"
        else:
            vae_cfg, n_vae_chunks, vae_tiling_desc = plan_vae_tiling(latent)
        print(
            f"VAE tiling: {vae_tiling_desc} "
            f"({n_vae_chunks} chunk{'s' if n_vae_chunks != 1 else ''})"
        )
        # Always carry fp16 RGBA from MLX through to VSR - quantization
        # happens at the destination format, not earlier. For LL this means
        # CIContext quantizes once at NV12 render time (in YUV space) rather
        # than twice (in RGB then YUV). For HQ this preserves full bf16
        # precision into RGBAHalf.
        chunks = iter_latent_chunks(
            latent, decoder,
            cfg=vae_cfg, mx_mod=mx,
            output_format="fp16_rgba",
            single_pass=args.vae_tiling == "single",
        )
    else:
        from LTX_2_MLX.videotoolbox.vsr import source_format_for_mode

        print(f"Reading video: {args.video}")
        in_w, in_h, source_fps, total_frames, src_transform = _vr.probe_video(
            Path(args.video),
        )
        # Decode straight into VSR's source format (NV12 for fast, RGBAHalf for
        # balanced/image) and feed the buffers directly to VSR - no RGB
        # intermediate, no MLX round-trip, no re-quantization. Size the decode
        # chunk to a ~64 MiB budget so peak resident decoded frames stay bounded
        # regardless of resolution (1 frame for 4K RGBAHalf, more for small SD).
        vsr_src_fmt = source_format_for_mode(args.spatial_mode)
        bytes_per_px = 8 if vsr_src_fmt == _pb.PIX_RGBAHALF else 2
        frame_bytes = max(1, in_w * in_h * bytes_per_px)
        buf_chunk = max(1, min(args.video_chunk_size, (64 * 1024 * 1024) // frame_bytes))
        # --start/--end trim the input. The reader seeks to the window so the
        # head of a long clip is never decoded (frame-exact, see video_reader).
        win_start, win_end = resolve_trim(args.start, args.end, source_fps, total_frames)
        if win_start or win_end is not None:
            total_frames = (win_end if win_end is not None else total_frames) - win_start
            print(f"[setup] input trim: frames [{win_start}, "
                  f"{win_end if win_end is not None else 'end'}) -> {total_frames} frames")
        chunks = _vr.iter_video_buffer_chunks(
            Path(args.video), vsr_src_fmt, chunk_size=buf_chunk,
            start_frame=win_start, end_frame=win_end,
        )
        n_vae_chunks = None  # no VAE on --video path

        # Carry the source file's audio through to the output MP4 (native
        # AVFoundation read; no ffmpeg). Latents decode audio from a latent
        # instead - see _decode_audio_track above. Trim + sidecar happen below,
        # uniformly for both paths.
        if args.audio:
            audio_track = _read_audio_track_from_video(Path(args.video))

    # ---- Audio trim + sidecar (uniform for both paths) ---------------------
    # When --start/--end trim the video, trim the audio to the same window so
    # the muxed track stays in sync (otherwise a short clip carries full-length
    # audio). The sidecar, if requested, reflects the trimmed audio.
    if audio_track is not None and (win_start or win_end is not None):
        a_start = win_start / source_fps
        a_end = (win_end / source_fps) if win_end is not None else None
        audio_track = audio_track.trimmed(a_start, a_end)
        print(
            f"[setup] audio trimmed to [{a_start:.3f}s, "
            f"{'end' if a_end is None else f'{a_end:.3f}s'})"
        )
    if audio_track is not None and args.save_audio_sidecar:
        sidecar = out_root / f"{stem}_audio.wav"
        audio_track.save_wav(sidecar)
        print(f"[setup] audio sidecar: {sidecar}")

    # ---- Output geometry + encoder settings --------------------------------
    from LTX_2_MLX.videotoolbox.vsr import scale_for_mode
    spatial_scale = scale_for_mode(args.spatial_mode)
    out_w, out_h = in_w * spatial_scale, in_h * spatial_scale
    profile = _pick_hevc_profile(args.spatial_mode, args.encode_chroma)
    target_fps = args.target_fps if args.target_fps is not None else source_fps
    do_temporal = abs(target_fps - source_fps) > 1e-6
    # --max-frames caps OUTPUT frames; a time spec here is output duration at
    # the target fps. Parsed now that target_fps is known.
    try:
        max_frames = (
            parse_time_or_frames(args.max_frames, target_fps)
            if args.max_frames is not None else None
        )
    except ValueError as e:
        raise SystemExit(f"bad --max-frames value: {e}") from None

    print(
        f"Source: {in_w}x{in_h}, "
        f"total frames: {total_frames or 'unknown'}, "
        f"fps: {source_fps:.3f}"
    )
    print(
        f"Target: {out_w}x{out_h} (spatial {spatial_scale}x), "
        f"fps: {target_fps:.3f}"
        f"{' (temporal upscale)' if do_temporal else ''}, "
        f"spatial-mode={args.spatial_mode}"
    )
    print(
        f"Encoder: HEVC profile={profile} q={args.encode_quality} "
        f"audio={args.audio_codec if audio_track else 'none'}"
    )

    # ---- Sessions + writers ------------------------------------------------
    # Defer constructing the VSR session, VtfrcSession, and AVWriters until
    # the *first* VAE chunk has materialized.  These hold Metal resources
    # - the HQ VSR model in particular pins ~100MB of Metal heap - so
    # creating them up front would compete with chunk-1 VAE decode for
    # the same unified-memory pool.  Lazy init via _build_post_pipeline.
    session: VsrSession | None = None
    vtfrc: VtfrcSession | None = None
    post_writer: AVWriter | None = None
    comparison_writer: AVWriter | None = None

    def _build_post_pipeline() -> tuple[
        VsrSession, VtfrcSession | None, AVWriter | None, AVWriter | None
    ]:
        """Materialize VSR + temporal + writer sessions just-in-time.

        Called on the first chunk so chunk-1 VAE has the Metal heap to
        itself.  Returns (session, vtfrc, post_writer, comparison_writer);
        the dst-pool wiring for zero-copy is set up before returning.
        """
        s = VsrSession(in_w, in_h, mode=args.spatial_mode, fps=source_fps)
        v: VtfrcSession | None = None
        if do_temporal:
            v = VtfrcSession(
                out_w, out_h,
                source_fps=source_fps, target_fps=target_fps,
                mode=args.temporal_mode,
            )
        audio_kwargs: dict[str, Any] = {}
        if audio_track is not None:
            audio_kwargs = {"audio_track": audio_track, "audio_codec": args.audio_codec}

        pw: AVWriter | None = None
        if not args.skip_post_mp4:
            # The producer feeding this writer's pool is the temporal session
            # if present, else VSR. Pass its dst attrs so the pool's buffers
            # carry the extended-pixel padding that producer requires (else
            # VTFrameProcessor rejects them with -19730 for some geometries).
            producer_attrs = v.dst_attrs if v is not None else s.dst_attrs
            pw = AVWriter(
                out_root / f"{stem}_post.mp4",
                width=out_w, height=out_h, fps=target_fps,
                source_pixel_format=_pb.resolve_pixel_format(producer_attrs),
                profile=profile,
                quality=args.encode_quality,
                label="post",
                transform=src_transform,
                source_attrs=producer_attrs,
                **audio_kwargs,
            )
            # Zero-copy from VSR (or VtfrcSession's output) into encoder.
            if v is not None:
                v.use_dst_pool(pw.adaptor.pixelBufferPool())
            else:
                s.use_dst_pool(pw.adaptor.pixelBufferPool())

        cw: AVWriter | None = None
        if args.comparison:
            cw = AVWriter(
                out_root / f"{stem}_comparison.mp4",
                width=2 * out_w, height=out_h, fps=target_fps,
                source_pixel_format=_pb.PIX_BGRA,
                profile=profile,
                quality=args.encode_quality,
                label="comparison",
                **audio_kwargs,
            )
        return s, v, pw, cw

    # ---- Cut detector ------------------------------------------------------
    cut_detector: CutDetector | None = None
    cut_log = None
    if args.cut_detect != "off":
        cut_detector = CutDetector(args.cut_detect, args.cut_threshold)
        if args.cut_log:
            cut_log = open(args.cut_log, "w")
        print(
            f"Cut detector: mode={args.cut_detect} threshold={args.cut_threshold}"
            + (f", log={args.cut_log}" if args.cut_log else "")
        )

    # ---- Progress bars (stacked, deferred-start, median-window rate) -------
    # PhaseBar's clock starts at the first update() and the displayed pace
    # is the median of the last-N inter-tick intervals; see videotoolbox/
    # progress.py for the rationale. Plain tqdm gave both stacked bars the
    # same wall-clock elapsed (= total run time) and inflated the VSR rate
    # as the VAE-chunk-1 warmup amortized over a growing frame count.
    target_frame_total = total_frames
    if target_frame_total and do_temporal:
        target_frame_total = int(round(total_frames * (target_fps / source_fps)))
    pbar_total = (
        min(target_frame_total, max_frames)
        if (target_frame_total and max_frames is not None) else
        (max_frames or target_frame_total or None)
    )
    bars = StackedPhaseBars()
    vae_pbar = (
        bars.add(total=n_vae_chunks, desc="VAE chunks", unit="chunk")
        if n_vae_chunks is not None else None
    )
    out_pbar = bars.add(
        total=pbar_total,
        desc="OUT frames" if do_temporal else "VSR frames",
        unit="frame",
    )

    # ---- Pipeline loop -----------------------------------------------------
    processed = 0          # source frames upscaled
    appended = 0           # output frames written (= processed when no temporal)
    in_idx = 0             # input frame index (counts skipped pre-window frames)
    window_done = False    # set once the input window [loop_win_*) is exhausted
    t_total = time.perf_counter()
    try:
        for chunk in chunks:
            # Both paths yield a list, freed per-frame as consumed: the latent
            # path a list of MLX arrays (fp16 RGBA), the --video path a list of
            # decoded CVPixelBuffers in VSR's source format (fed to VSR direct).
            chunk_len = len(chunk)
            frames_are_buffers = not isinstance(chunk[0], mx.array)
            if frames_are_buffers:
                t_w, t_h = _pb.buffer_dims(chunk[0])
            else:
                t_h, t_w = int(chunk[0].shape[0]), int(chunk[0].shape[1])
            if (t_w, t_h) != (in_w, in_h):
                raise RuntimeError(
                    f"chunk dims {t_w}x{t_h} don't match VSR config {in_w}x{in_h}"
                )
            if vae_pbar is not None:
                vae_pbar.update(1)
            # Lazy init of the post-VAE pipeline.  Doing this *after* chunk 1
            # materialized keeps the VSR HQ model + AVWriter pixel pool out of
            # the Metal heap during chunk-1 VAE decode (which is the most
            # memory-contended step of the run).
            #
            # _build_post_pipeline() prints status from VsrSession / AVWriter
            # constructors; route those through bars.write() so they appear
            # above the live progress bars instead of stomping mid-line.
            if session is None:
                import contextlib
                import io as _io
                _buf = _io.StringIO()
                with contextlib.redirect_stdout(_buf):
                    session, vtfrc, post_writer, comparison_writer = _build_post_pipeline()
                msg = _buf.getvalue().rstrip("\n")
                if msg:
                    bars.write(msg)
            for i in range(chunk_len):
                if max_frames is not None and appended >= max_frames:
                    break
                # Input-frame window (latent path; --video already trimmed at
                # the reader, where these bounds are 0/None). Skip frames before
                # the window, and stop once past it.
                if in_idx < loop_win_start:
                    chunk[i] = None
                    in_idx += 1
                    continue
                if loop_win_end is not None and in_idx >= loop_win_end:
                    window_done = True
                    break
                # Wrap the per-frame body in a fresh ObjC autorelease pool so
                # transient autoreleased objects (NSData, CIImage, CIImage
                # affine-translated, CIImage composited, CIContext render
                # intermediates, ...) drain at the end of each iteration
                # instead of piling up on the process top-level pool until
                # the interpreter exits. Without this the RSS climbs
                # unboundedly during long runs even though Python refcounts
                # are tracking correctly - PyObjC just doesn't drain
                # autoreleased ObjC objects on Python GC.
                with autorelease_pool():
                    src_frame = chunk[i]

                    # The upscale itself never needs an MLX array on the buffer
                    # path - the decoded buffer feeds VSR directly. Only
                    # materialize a uint8 RGB array when a feature consumes the
                    # source pixels: cut detection, --save-pre-frames, or the
                    # --comparison composite. For the latent path src_frame is
                    # already the array.
                    src_arr = None
                    if (
                        cut_detector is not None
                        or args.save_pre_frames
                        or comparison_writer is not None
                    ):
                        src_arr = (
                            _pb.read_pixel_buffer_rgb(src_frame)
                            if frames_are_buffers else src_frame
                        )

                    if cut_detector is not None and cut_detector.is_cut(src_arr):
                        session.reset_temporal_context()
                        if cut_log is not None:
                            cut_log.write(f"{processed}\n")
                            cut_log.flush()

                    if frames_are_buffers:
                        vsr_pb = session.upscale_buffer_to_buffer(src_frame, processed)
                    else:
                        vsr_pb = session.upscale_to_buffer(src_frame, processed)

                    # PNG sidecars (opt-in via --save-*-frames). Done BEFORE
                    # the writer-append loop so vsr_pb is still in scope; the
                    # readback uses CIContext.
                    if args.save_pre_frames:
                        if src_arr.dtype != mx.uint8:
                            pre_rgb_u8 = mx.clip(src_arr[..., :3] * 255.0, 0, 255).astype(mx.uint8)
                        else:
                            pre_rgb_u8 = (
                                src_arr if src_arr.shape[-1] == 3
                                else src_arr[..., :3]
                            )
                        save_image(pre_rgb_u8, pre_dir / f"frame_{processed:05d}.png")
                    if args.save_post_frames:
                        save_image(
                            _pb.read_pixel_buffer_rgb(vsr_pb),
                            post_dir / f"frame_{processed:05d}.png",
                        )

                    # Iterate VSR/temporal output buffers directly - don't
                    # materialize into a list - so each buffer's local ref
                    # drops the moment the writer takes it.
                    out_iter = (
                        iter([vsr_pb]) if vtfrc is None
                        else vtfrc.feed(vsr_pb, processed)
                    )
                    for out_pb in out_iter:
                        if max_frames is not None and appended >= max_frames:
                            break
                        if post_writer is not None:
                            post_writer.append(out_pb)
                        if comparison_writer is not None:
                            comp_pb = _pb.make_bgra_buffer(
                                comparison_writer.adaptor, 2 * out_w, out_h,
                            )
                            # Pre half uses the source frame (not temporal-
                            # interpolated post) so before/after is honest.
                            render_comparison(src_arr, out_pb, spatial_scale, comp_pb)
                            comparison_writer.append(comp_pb)
                            del comp_pb
                        del out_pb
                        appended += 1
                        out_pbar.update(1)
                    del vsr_pb, out_iter

                    processed += 1
                    in_idx += 1
                    # Drop this frame's reference so its memory (the MLX array,
                    # or the decoded CVPixelBuffer on the --video path) can be
                    # freed now instead of staying resident until the outer
                    # `del chunk` at chunk-end.
                    chunk[i] = None
                    del src_frame, src_arr
                # autorelease pool drains here; PyObjC objects created in
                # this iteration are released back to the system.

                # Periodic janitorial work: CIContext caches grow with
                # render calls, and CVPixelBufferPools accumulate cached
                # buffers that the workload no longer needs.
                if processed % 64 == 0:
                    _pb.clear_ci_caches()
                    session.flush_pools()

            if window_done or (max_frames is not None and appended >= max_frames):
                break
            del chunk
            gc.collect()
    finally:
        bars.close()
        if vtfrc is not None:
            vtfrc.close()
        if session is not None:
            session.close()
        for writer in (post_writer, comparison_writer):
            if writer is not None:
                writer.finish()
        if cut_log is not None:
            cut_log.close()

    elapsed = time.perf_counter() - t_total
    rate = appended / elapsed if elapsed > 0 else 0
    print(f"Processed {processed} source frames, wrote {appended} output frames "
          f"in {elapsed:.2f}s ({rate:.2f} fps out)")
    if post_writer is not None:
        print(f"Post: {post_writer.path}")
    if comparison_writer is not None:
        print(f"Comparison: {comparison_writer.path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--latent", help="--save-latents NPZ sidecar (VAE-decoded first).")
    src.add_argument("--video", help="Already-decoded video file (mp4/mov/...).")

    parser.add_argument(
        "--latent-stage",
        choices=["final", "stage1", "stage2"],
        default="final",
        help=(
            "Which latent to decode from a distilled-two-stage sidecar. "
            "'final' (default) = final_video_latent = stage 2 (the upscaled+refined result). "
            "'stage1' = pre-upscaler half-resolution latent. "
            "'stage2' = explicit stage 2 (same content as 'final' on distilled two-stage)."
        ),
    )
    parser.add_argument("--weights", help="LTX-2 .safetensors path (required with --latent).")
    parser.add_argument(
        "--vae-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
    )
    parser.add_argument(
        "--vae-tiling", choices=["auto", "single"], default="auto",
        help=(
            "auto (default) lets TilingConfig.auto_native_conv3d size to RAM + the "
            "int32 conv3d boundary (one decode if it fits, else bounded tiles). single "
            "forces one decode; frames past the int32 boundary decode white."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-prefix", default="vsr",
        help="Filename prefix for the timestamped outputs (matches generate.py).",
    )
    parser.add_argument(
        "--source-fps", type=float, default=NATIVE_FPS,
        help=(
            f"Source frame rate for --latent (latents don't carry an fps; "
            f"default {NATIVE_FPS} matches generate.py). Ignored for --video - "
            f"the input file's r_frame_rate is honored instead. Pair with "
            f"--target-fps to drive temporal frame-rate conversion."
        ),
    )
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help=(
            "Target output fps. Defaults to the source fps (no temporal upscale). "
            "Setting a different value routes VSR output through "
            "VTFrameRateConversionConfiguration, motion-interpolating to the "
            "target rate. Arbitrary float values supported; 24->60, 15->30, "
            "30->24 (downsample), etc. The CMTime base is 24000 so common rates "
            "land bit-exact."
        ),
    )
    parser.add_argument(
        "--temporal-mode", choices=["normal", "high"], default="normal",
        help=(
            "VTFrameRateConversion mode. Only active when --target-fps is set. "
            "normal (default) = fast and adequate for 2x rate-up; "
            "high = QualityPrioritizationQuality, more compute for cleaner motion."
        ),
    )
    parser.add_argument(
        "--spatial-mode", choices=["fast", "balanced", "image"], default="balanced",
        help=(
            "VSR spatial mode.  Scale factor is implied by the mode (fast=2x, "
            "balanced=4x, image=4x). "
            "balanced (default) = HQ Video mode; uses prev source + prev output "
            "frames to inform the upscale.  Tends to produce crisper motion edges "
            "at the cost of slightly more frame-to-frame variation in detail. "
            "fast = VTLowLatency 2x scaler.  Per-frame, no temporal context.  "
            "Input must be 96x96 to 960x960. "
            "image = HQ Image mode.  Per-frame deterministic upscale with no "
            "prev-frame feedback.  Slightly softer per-frame detail than balanced, "
            "but measurably smoother frame-to-frame (lower temporal second-difference). "
            "Apple documents this as for stills; on video it's a legitimate "
            "alternative to balanced if you prefer the smoother trade-off."
        ),
    )
    parser.add_argument(
        "--encode-quality", type=float, default=0.65,
        help="AVVideoQualityKey (0..1) for the HEVC encoder. 0.65 matches the default tier.",
    )
    parser.add_argument(
        "--encode-chroma", choices=["auto", "420", "422"], default="auto",
        help=(
            "HEVC profile chroma subsampling. auto = 4:2:2 (Main42210) for "
            "balanced/image modes, 4:2:0 (Main10) for fast. 420 forces Main10 for "
            "generate.py-tier parity."
        ),
    )
    parser.add_argument(
        "--audio", action="store_true",
        help=(
            "Mux audio into both MP4s. For --latent the audio is decoded from "
            "final_audio_latent (audio VAE + vocoder); for --video the source "
            "file's audio track is read natively (AVFoundation) and carried "
            "through. A --video input with no audio track stays silent."
        ),
    )
    parser.add_argument(
        "--audio-codec", choices=["alac", "aac"], default="alac",
        help="Audio codec for muxed audio (alac=lossless, aac=256kbps).",
    )
    parser.add_argument(
        "--save-audio-sidecar", action="store_true",
        help="Also write the muxed audio as <stem>_audio.wav next to the MP4s.",
    )
    parser.add_argument(
        "--cut-detect", choices=["off", "simple", "hist"], default="off",
        help=(
            "Reset VSR's prev-frame chain at hard cuts. off = never reset "
            "(correct for single-shot LTX latents). Only meaningful for "
            "edited --video input under --spatial-mode balanced (which "
            "chains prev-frame state); a no-op under fast/image modes."
        ),
    )
    parser.add_argument("--cut-threshold", type=float, default=0.25)
    parser.add_argument(
        "--cut-log", default=None,
        help="Write detected cut frame indices to this file (one per line).",
    )
    parser.add_argument(
        "--video-chunk-size", type=int, default=32,
        help=(
            "Upper bound on decoded frames held in flight for --video input. "
            "The actual chunk is further capped to a ~64 MiB memory budget "
            "based on resolution, so peak resident decode memory stays bounded "
            "(often 1 frame at 4K, this many at SD)."
        ),
    )
    parser.add_argument(
        "--start", default=None,
        help=(
            "Trim the input to start at this position (process the middle of a "
            "clip). Accepts frames or time: bare integer = frames (e.g. 120), "
            "Nf = frames (120f), Ns / decimal = seconds (5s, 1.5), or a clock "
            "string mm:ss / hh:mm:ss (0:05, 1:02:03). --video seeks here "
            "natively (the head is not decoded); --latent windows the decode."
        ),
    )
    parser.add_argument(
        "--end", default=None,
        help=(
            "Trim the input to stop before this position (exclusive), same "
            "frames-or-time forms as --start. Output is a fresh clip starting "
            "at PTS 0 spanning [--start, --end)."
        ),
    )
    parser.add_argument(
        "--max-frames", default=None,
        help=(
            "Cap the number of OUTPUT frames. Same frames-or-time forms as "
            "--start (a time here is output duration, measured at the target "
            "fps). Composes with --start/--end, which trim the input."
        ),
    )
    parser.add_argument("--save-pre-frames", action="store_true")
    parser.add_argument("--save-post-frames", action="store_true")
    parser.add_argument(
        "--skip-post-mp4", action="store_true",
        help="Skip writing the upscaled _post.mp4 (e.g. when you only want frame dumps).",
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="Also write a side-by-side <stem>_comparison.mp4 "
             "(NEAREST-upscaled pre vs VSR post).",
    )
    args = parser.parse_args()

    if args.latent and not args.weights:
        parser.error("--latent requires --weights")

    require_pyobjc()
    run(args)


if __name__ == "__main__":
    main()
