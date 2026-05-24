#!/usr/bin/env python3
"""Encode-mode harness for VAE-decoded video + audio.

Loads a final latent NPZ (from `scripts/generate.py --save-latents`), decodes
video + audio ONCE through the VAE / vocoder, then re-encodes the SAME decoded
result through a list of named ffmpeg presets so codec / bit-depth / chroma /
audio-codec choices can be A/B compared without re-running diffusion.

Frames are piped straight to ffmpeg via stdin (rawvideo) — no PNG round-trip.
Use `--frame-bit-depth 16` when running 10-bit or RGB-lossless presets so the
source feeding the encoder is rgb48le, not rgb24.

Examples
--------
  # List built-in presets
  scripts/encode_modes_harness.py --list-presets

  # Compare a curated subset
  scripts/encode_modes_harness.py \\
      --latent  outputs/<run_name>_latent.npz \\
      --weights $LTX_DEFAULT_WEIGHTS_PATH \\
      --modes baseline_h264_aac h265_10bit_444 hevc_vt_10bit \\
      --output-dir outputs/encode_compare

  # Test only the lossless RGB path (requires 16-bit frame source)
  scripts/encode_modes_harness.py --latent ... --weights ... \\
      --modes h265_rgb_lossless --frame-bit-depth 16
"""

from __future__ import annotations

import argparse
import gc
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.decode_latent_debug import (  # noqa: E402
    RunTimings,
    Timer,
    decode_audio_latent,
    load_latents,
    make_audio_decoder_and_vocoder,
    parse_dtype,
)
from LTX_2_MLX.video_encoder import (  # noqa: E402
    COLOR_TAGS_BT709,
    COLOR_TAGS_RGB,
    NATIVE_FPS,
    TIERS,
    EncodePreset,
    build_ffmpeg_cmd,
    h264_baseline_video,
    scale_filter,
    write_wav_float32,
    write_wav_int16,
)


def make_video_decoder(
    weights_path: str,
    compute_dtype: Any,
    *,
    backend: str = "native",
):
    """Build a VAE decoder matching scripts/generate.py's happy-path defaults."""
    from scripts.generate import get_vae_config
    from LTX_2_MLX.model.video_vae.native_decoder import (
        NativeConv3dVideoDecoder, load_native_vae_decoder_weights,
    )

    cfg = get_vae_config(weights_path)
    decoder_blocks = cfg.get("decoder_blocks")
    base_channels = cfg.get("decoder_base_channels", cfg.get("base_channels", 128))
    timestep_conditioning = cfg.get("timestep_conditioning", True)

    print(
        f"VAE backend: {backend}, "
        f"blocks={len(decoder_blocks) if decoder_blocks else 'default'}, "
        f"base_ch={base_channels}, timestep={timestep_conditioning}"
    )

    if backend == "native":
        decoder = NativeConv3dVideoDecoder(
            decoder_blocks=decoder_blocks,
            base_channels=base_channels,
            timestep_conditioning=timestep_conditioning,
            compute_dtype=compute_dtype,
        )
        load_native_vae_decoder_weights(decoder, weights_path)
    else:
        raise ValueError(
            f"Unsupported VAE decoder backend: {backend!r}. Only 'native' is supported."
        )

    import gc as _gc
    _gc.collect()
    return decoder


# Benchmark-only presets that aren't part of the production TIERS ladder.
# These exist for A/B testing — banding fixes, lossless ground truth, hardware
# H.264 (kept as documentation), HEVC VT q-sweep, audio codec A/Bs, etc.
HARNESS_EXTRAS: dict[str, EncodePreset] = {
    "h265_10bit_420": EncodePreset(
        name="h265_10bit_420",
        video=[
            *scale_filter("yuv420p10le"),
            "-c:v", "libx265", "-profile:v", "main10",
            "-crf", "16", "-preset", "slow",
            "-tag:v", "hvc1", *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="HEVC SW 10-bit 4:2:0 CRF 16 + ALAC. Banding fix without chroma upgrade.",
    ),
    "h265_rgb_lossless": EncodePreset(
        name="h265_rgb_lossless",
        video=[
            *scale_filter("gbrp10le"),
            "-c:v", "libx265", "-pix_fmt", "gbrp10le",
            # Color tagging lives inside x265-params: global -colorspace gbr
            # makes ffmpeg's libx265 wrapper forward `colorspace=gbr` to
            # libx265, which only knows `colormatrix` and rejects the value.
            "-x265-params",
            "lossless=1:profile=main-444-10:colormatrix=gbr:colorprim=bt709:transfer=bt709:range=full",
            "-tag:v", "hvc1",
        ],
        audio=["-c:a", "alac"],
        frame_bit_depth=16,
        notes="Lossless RGB H.265 (no YUV conversion). Needs --frame-bit-depth 16.",
    ),
    "h264_vt_8bit": EncodePreset(
        name="h264_vt_8bit",
        video=[
            *scale_filter("nv12"),
            "-c:v", "h264_videotoolbox", "-profile:v", "high",
            # q=80 lands at "visually transparent" for H.264 VT, but the
            # SW libx264 path (web tier) still wins on size/quality.
            "-q:v", "80", "-tag:v", "avc1",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="Hardware H.264 8-bit 4:2:0. Kept as documentation: SW libx264 wins for H.264.",
    ),
    "hevc_vt_8bit": EncodePreset(
        name="hevc_vt_8bit",
        video=[
            *scale_filter("nv12"),
            "-c:v", "hevc_videotoolbox", "-q:v", "55",
            "-realtime", "0",
            "-tag:v", "hvc1", *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="Hardware HEVC 8-bit 4:2:0 q=55 (fast path).",
    ),
    "hevc_vt_10bit_q55": EncodePreset(
        name="hevc_vt_10bit_q55",
        video=[
            *scale_filter("p010le"),
            "-c:v", "hevc_videotoolbox", "-profile:v", "main10",
            "-q:v", "55", "-realtime", "0",
            "-tag:v", "hvc1", *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="HEVC HW 10-bit 4:2:0 q=55. Lower-quality variant for q-sweep comparison.",
    ),
    "hevc_vt_10bit_q75": EncodePreset(
        name="hevc_vt_10bit_q75",
        video=[
            *scale_filter("p010le"),
            "-c:v", "hevc_videotoolbox", "-profile:v", "main10",
            "-q:v", "75", "-realtime", "0",
            "-tag:v", "hvc1", *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "alac"],
        notes="HEVC HW 10-bit 4:2:0 q=75. Higher-quality variant for q-sweep comparison.",
    ),
    "prores_4444_xq": EncodePreset(
        name="prores_4444_xq",
        container="mov",
        video=[
            *scale_filter("yuva444p10le"),
            "-c:v", "prores_ks", "-profile:v", "4444xq",
            "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
            *COLOR_TAGS_BT709,
        ],
        audio=["-c:a", "pcm_s24le"],
        frame_bit_depth=16,
        notes="ProRes 4444 XQ. Identical bytes to reference tier at 10-bit (XQ only differs at 12-bit).",
    ),
    "ffv1_lossless": EncodePreset(
        name="ffv1_lossless",
        container="mkv",
        video=[
            "-c:v", "ffv1", "-level", "3", "-coder", "1",
            "-context", "1", "-g", "1", "-slicecrc", "1",
            "-pix_fmt", "gbrp10le", *COLOR_TAGS_RGB,
        ],
        audio=["-c:a", "flac"],
        frame_bit_depth=16,
        notes="Bit-exact FFV1 in MKV. Ground-truth reference (not playable in browsers).",
    ),
    # audio-only A/Bs: video held at web tier, only audio codec changes.
    "audio_alac": EncodePreset(
        name="audio_alac",
        video=[*h264_baseline_video(), *COLOR_TAGS_BT709],
        audio=["-c:a", "alac"],
        notes="Web-tier video + ALAC audio (lossless in MP4).",
    ),
    "audio_flac": EncodePreset(
        name="audio_flac",
        video=[*h264_baseline_video(), *COLOR_TAGS_BT709],
        audio=["-c:a", "flac"],
        extra=["-strict", "experimental"],
        notes="Web-tier video + FLAC audio (player support spotty in MP4).",
    ),
    "audio_pcm": EncodePreset(
        name="audio_pcm",
        video=[*h264_baseline_video(), *COLOR_TAGS_BT709],
        audio=["-c:a", "pcm_s16le"],
        notes="Web-tier video + uncompressed PCM audio.",
    ),
}

# Production tiers first (web/default/hq/export/reference), then benchmark-only.
PRESETS: dict[str, EncodePreset] = {**TIERS, **HARNESS_EXTRAS}


def list_presets() -> None:
    width = max(len(k) for k in PRESETS)
    print("Available presets:")
    for name, preset in PRESETS.items():
        ext = f".{preset.container}"
        depth = f"src={preset.frame_bit_depth}bpc"
        print(f"  {name:<{width}}  {ext:<5}  {depth:<10}  {preset.notes}")


def chunk_to_uint8(chunk: Any, mx_mod: Any) -> np.ndarray:
    """Cast one (B,C,T,H,W) decoded chunk to (T,H,W,3) uint8 numpy.

    The rescale + clip + cast happens in MLX so we never materialize a
    chunk-sized float32 numpy intermediate. That's what made the naive
    "decode then convert" path peak at ~60 GiB.
    """
    rescaled = mx_mod.clip((chunk + 1.0) * 127.5, 0, 255).astype(mx_mod.uint8)
    transposed = mx_mod.transpose(rescaled, (0, 2, 3, 4, 1))  # B,T,H,W,C
    mx_mod.eval(transposed)
    arr = np.array(transposed)
    return arr[0] if arr.ndim == 5 else arr


def chunk_to_uint16(chunk: Any, mx_mod: Any) -> np.ndarray:
    """Same as chunk_to_uint8 but emits uint16 (for rgb48le source feed)."""
    rescaled = mx_mod.clip((chunk + 1.0) * 32767.5, 0, 65535).astype(mx_mod.uint16)
    transposed = mx_mod.transpose(rescaled, (0, 2, 3, 4, 1))
    mx_mod.eval(transposed)
    arr = np.array(transposed)
    return arr[0] if arr.ndim == 5 else arr


def open_encoder(
    preset: EncodePreset,
    *,
    width: int,
    height: int,
    fps: float,
    audio_path: Path | None,
    output_path: Path,
) -> tuple[subprocess.Popen, str]:
    raw_pix_fmt = "rgb48le" if preset.frame_bit_depth == 16 else "rgb24"
    cmd = build_ffmpeg_cmd(
        preset,
        raw_pix_fmt=raw_pix_fmt,
        width=width, height=height, fps=fps,
        audio_path=audio_path, output_path=output_path,
    )
    print(f"\n[{preset.name}] open ffmpeg ({raw_pix_fmt} -> {preset.container})")
    print("  cmd:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc, raw_pix_fmt


def humanize_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Encode-mode harness for VAE decode output.")
    parser.add_argument("--latent", help="NPZ sidecar from generate.py --save-latents.")
    parser.add_argument("--weights", help="VAE / audio weights path.")
    parser.add_argument("--output-dir", default="outputs/encode_tests")
    parser.add_argument("--fps", type=float, default=NATIVE_FPS)
    parser.add_argument("--vae-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--latent-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--vae-decoder-backend",
        choices=["native", "legacy"],
        default="native",
        help=(
            "VAE decoder backend. Both do 3D convolution. "
            "native (default, matches scripts/generate.py's happy path) uses "
            "MLX-native nn.Conv3d. legacy is the older slice-based Conv3d "
            "emulation, kept as the A/B baseline."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Preset names to run. Default: all presets.",
    )
    parser.add_argument("--list-presets", action="store_true", help="Print presets and exit.")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio decode; video-only outputs.")
    parser.add_argument(
        "--audio-bit-depth",
        choices=["int16", "float32"],
        default="int16",
        help="Precision of the WAV intermediate fed to ffmpeg.",
    )
    parser.add_argument(
        "--frame-bit-depth",
        choices=["8", "16"],
        default="8",
        help="Bit depth of raw frames piped to ffmpeg. 16 unlocks lossless / 10-bit presets.",
    )
    args = parser.parse_args(argv)

    if args.list_presets:
        list_presets()
        return

    if not args.latent or not args.weights:
        parser.error("--latent and --weights are required unless --list-presets is set")

    selected: list[EncodePreset] = []
    for name in args.modes or list(PRESETS):
        if name not in PRESETS:
            parser.error(f"unknown preset {name!r}. Run --list-presets to see options.")
        selected.append(PRESETS[name])

    need_u8 = any(p.frame_bit_depth == 8 for p in selected)
    need_u16 = any(p.frame_bit_depth == 16 for p in selected)

    import mlx.core as mx
    timings = RunTimings()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Timer(timings, "load latent"):
        latent, audio_latent = load_latents(args.latent, mx, args.latent_dtype)
    compute_dtype = parse_dtype(mx, args.vae_dtype)

    with Timer(timings, "load vae decoder"):
        decoder = make_video_decoder(
            args.weights,
            compute_dtype,
            backend=args.vae_decoder_backend,
        )

    # Output dims derived from latent shape (same formula as decode_latent_debug).
    _, _, latent_frames, latent_height, latent_width = latent.shape
    n_frames = 1 + (latent_frames - 1) * 8
    height = latent_height * 32
    width = latent_width * 32
    print(f"\nDecoded video target: {n_frames} frames {width}x{height}")

    # Audio decode first (small, finite memory) so encoders can mux it.
    audio_path: Path | None = None
    audio_sample_rate: int | None = None
    if not args.skip_audio and audio_latent is not None:
        with Timer(timings, "load audio decoder"):
            audio_decoder, vocoder, audio_sample_rate = make_audio_decoder_and_vocoder(
                args.weights, compute_dtype,
            )
        with Timer(timings, "audio decode"):
            audio_waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx)
        audio_path = output_dir / (
            "decoded_audio_float32.wav" if args.audio_bit_depth == "float32"
            else "decoded_audio_int16.wav"
        )
        with Timer(timings, "audio wav write"):
            if args.audio_bit_depth == "float32":
                write_wav_float32(audio_waveform, audio_path, audio_sample_rate)
            else:
                write_wav_int16(audio_waveform, audio_path, audio_sample_rate)
        print(
            f"Wrote audio WAV: {audio_path}  "
            f"({args.audio_bit_depth}, {audio_sample_rate} Hz)"
        )
        del audio_waveform, audio_decoder, vocoder
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
    elif not args.skip_audio:
        print("NPZ has no audio latent; encoding video-only.")

    # Resolve tiling the same way scripts/generate.py does. With the
    # native backend `TilingConfig.auto` may return None (no tiling
    # needed) and the decode runs as a single pass through decode_latent —
    # NO spatial seams. The legacy simple-decoder branch always splits
    # spatially for width > 512, which is what was creating edge artifacts.
    from LTX_2_MLX.model.video_vae.tiling import TilingConfig, decode_tiled
    from LTX_2_MLX.model.video_vae.decode_utils import decode_latent
    tiling_cfg = TilingConfig.auto(
        height=height, width=width, num_frames=n_frames,
        decoder_backend=args.vae_decoder_backend,
    )
    if tiling_cfg is None:
        print("VAE tiling: off (auto picked single-shot decode for this backend/shape)")
        def chunk_iter():
            video = decode_latent(latent, decoder)
            mx.eval(video)
            yield video
    else:
        sp = tiling_cfg.spatial_config
        tp = tiling_cfg.temporal_config
        spatial_desc = (
            f"spatial tile={sp.tile_size_in_pixels} overlap={sp.tile_overlap_in_pixels}"
            if sp else "no spatial tiling"
        )
        temporal_desc = (
            f"temporal tile={tp.tile_size_in_frames} overlap={tp.tile_overlap_in_frames}"
            if tp else "no temporal tiling"
        )
        print(f"VAE tiling: {spatial_desc}, {temporal_desc}")
        def chunk_iter():
            yield from decode_tiled(latent, decoder, tiling_cfg, show_progress=True)

    # Pipelined decode -> per-encoder writer queues -> ffmpeg.
    # Subprocesses launch BEFORE the decode loop so libx265 can start chewing
    # on chunk 1 while the GPU is producing chunk 2. Each queue is bounded
    # so the decode side never gets more than QUEUE_DEPTH chunks ahead;
    # in practice decode is slower than encode here so the queue stays
    # mostly empty, but the bound caps peak memory if a slow preset stalls.
    QUEUE_DEPTH = 4
    encoders: list[
        tuple[EncodePreset, subprocess.Popen, str, Path, float, queue.Queue]
    ] = []
    for preset in selected:
        output_path = output_dir / f"encode_{preset.name}.{preset.container}"
        started = time.perf_counter()
        proc, raw_pix_fmt = open_encoder(
            preset,
            width=width, height=height, fps=args.fps,
            audio_path=audio_path, output_path=output_path,
        )
        q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        encoders.append((preset, proc, raw_pix_fmt, output_path, started, q))

    def _writer(proc: subprocess.Popen, q: queue.Queue, name: str) -> None:
        while True:
            item = q.get()
            if item is None:
                break
            try:
                proc.stdin.write(item.tobytes())
            except BrokenPipeError:
                # ffmpeg already exited; drain remaining queue so the producer
                # doesn't deadlock on a full queue.
                while q.get() is not None:
                    pass
                break
        try:
            proc.stdin.close()
        except Exception:
            pass

    writers: list[threading.Thread] = []
    for preset, proc, raw_pix_fmt, _, _, q in encoders:
        t = threading.Thread(target=_writer, args=(proc, q, preset.name), daemon=False)
        t.start()
        writers.append(t)

    with Timer(timings, "decode + encode (pipelined)"):
        decoded_frames = 0
        for chunk in chunk_iter():
            u8 = chunk_to_uint8(chunk, mx) if need_u8 else None
            u16 = chunk_to_uint16(chunk, mx) if need_u16 else None
            chunk_frames = (u8 if u8 is not None else u16).shape[0]
            decoded_frames += chunk_frames
            for preset, _, raw_pix_fmt, _, _, q in encoders:
                data = u8 if raw_pix_fmt == "rgb24" else u16
                if data is None:
                    raise RuntimeError(
                        f"preset {preset.name!r} wants {raw_pix_fmt} but no matching buffer"
                    )
                q.put(data)
            del chunk, u8, u16
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

        # Signal EOF to each writer.
        for *_, q in encoders:
            q.put(None)
        for t in writers:
            t.join()

        results: list[tuple[str, int, float]] = []
        for preset, proc, _, output_path, started, _ in encoders:
            rc = proc.wait()
            elapsed = time.perf_counter() - started
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed for preset {preset.name} (rc={rc})")
            size = output_path.stat().st_size
            results.append((preset.name, size, elapsed))
            print(f"  -> {output_path}  {humanize_bytes(size)}  ({elapsed:.1f}s)")

    print("\nEncode results:")
    name_w = max(len(n) for n, _, _ in results)
    for name, size, elapsed in results:
        print(f"  {name:<{name_w}}  {humanize_bytes(size):>10}  {elapsed:>6.1f}s")

    timings.print_summary()


if __name__ == "__main__":
    main()
