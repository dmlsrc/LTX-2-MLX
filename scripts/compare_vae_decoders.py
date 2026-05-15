#!/usr/bin/env python3
"""Compare simple and native Conv3d video VAE decoders on the same latent."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LTX_2_MLX.model.video_vae.native_decoder import (
    NativeConv3dVideoDecoder,
    load_native_vae_decoder_weights,
)
from LTX_2_MLX.model.video_vae.simple_decoder import (
    SimpleVideoDecoder,
    load_vae_decoder_weights,
)
from LTX_2_MLX.model.video_vae.tiling import TilingConfig, decode_tiled
from scripts.generate import (
    build_vae_tiling_config,
    describe_vae_tiling_config,
    get_vae_config,
    parse_compute_dtype,
    parse_non_negative_int,
)
from LTX_2_MLX.loader import ensure_weight_family_caches


DEFAULT_WEIGHTS = (
    "/Users/Shared/huggingface/hub/models--Lightricks--LTX-2.3/"
    "snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/"
    "ltx-2.3-22b-distilled-1.1.safetensors"
)
def _clear() -> None:
    gc.collect()
    mx.clear_cache()


def _memory_snapshot() -> dict[str, float]:
    return {
        "active_gb": mx.get_active_memory() / (1024**3),
        "cache_gb": mx.get_cache_memory() / (1024**3),
        "peak_gb": mx.get_peak_memory() / (1024**3),
    }


def _load_latent(path: Path, key: str, dtype: mx.Dtype) -> mx.array:
    data = np.load(path)
    if key not in data.files:
        raise ValueError(f"{path} does not contain {key!r}; found {data.files}")
    latent = mx.array(data[key]).astype(dtype)
    if latent.ndim == 4:
        latent = latent[None]
    mx.eval(latent)
    return latent


def _infer_video_shape(latent: mx.array) -> tuple[int, int, int]:
    _b, _c, latent_t, latent_h, latent_w = latent.shape
    frames = 1 + (latent_t - 1) * 8
    height = latent_h * 32
    width = latent_w * 32
    return height, width, frames


def _decode_with_timing(label: str, decoder, latent: mx.array, tiling: TilingConfig | None) -> tuple[mx.array, dict]:
    _clear()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    if tiling:
        chunks = list(decode_tiled(latent, decoder, tiling, show_progress=True))
        video = mx.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]
    else:
        video = decoder(latent, show_progress=False)
    mx.eval(video)
    decode_s = time.perf_counter() - t0
    stats = {
        "label": label,
        "decode_s": decode_s,
        "shape": tuple(int(v) for v in video.shape),
        "dtype": str(video.dtype),
        "memory": _memory_snapshot(),
    }
    return video, stats


def _sample_frames(frame_count: int, requested: str | None) -> list[int]:
    if requested:
        frames = [int(item) for item in requested.split(",") if item.strip()]
    else:
        frames = [0, frame_count // 8, frame_count // 4, frame_count // 2, (3 * frame_count) // 4, frame_count - 1]
    return sorted({min(max(f, 0), frame_count - 1) for f in frames})


def _sample_stats(video: mx.array, frames: list[int]) -> tuple[dict, list[np.ndarray]]:
    samples: list[np.ndarray] = []
    lumas: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    for frame_idx in frames:
        arr = np.array(video[0, :, frame_idx, :, :].transpose(1, 2, 0).astype(mx.float32))
        raw_values.append(arr.reshape(-1, 3))
        rgb01 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        luma = 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]
        lumas.append(luma.reshape(-1))
        samples.append((rgb01 * 255.0 + 0.5).astype(np.uint8))

    lum = np.concatenate(lumas)
    raw = np.concatenate(raw_values)
    return (
        {
            "luma_mean": float(lum.mean()),
            "luma_std": float(lum.std()),
            "luma_p01": float(np.percentile(lum, 1)),
            "luma_p50": float(np.percentile(lum, 50)),
            "luma_p99": float(np.percentile(lum, 99)),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "raw_clip_low_frac": float((raw <= -1.0).mean()),
            "raw_clip_high_frac": float((raw >= 1.0).mean()),
        },
        samples,
    )


def _video_to_uint8_frames(label: str, video: mx.array) -> np.ndarray:
    """Materialize decoded BCFHW video as compact uint8 FHWC frames."""
    _b, _c, frame_count, height, width = video.shape
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    try:
        from tqdm import tqdm

        iterator = tqdm(range(frame_count), desc=f"{label} frames", ncols=80, ascii=True, mininterval=1.0, miniters=10)
    except ImportError:
        iterator = range(frame_count)

    for idx in iterator:
        arr = np.array(video[0, :, idx, :, :].transpose(1, 2, 0).astype(mx.float32))
        rgb01 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        frames[idx] = (rgb01 * 255.0 + 0.5).astype(np.uint8)
    return frames


def _diff_stats(a_samples: list[np.ndarray], b_samples: list[np.ndarray]) -> dict:
    diffs = [
        np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(-1, 3)
        for a, b in zip(a_samples, b_samples)
    ]
    diff = np.concatenate(diffs)
    return {
        "mean": float(diff.mean()),
        "p50": float(np.percentile(diff, 50)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
        "max": int(diff.max()),
    }


def _uint8_absdiff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).reshape(-1, 3)
    return {
        "mean": float(diff.mean()),
        "p50": float(np.percentile(diff, 50)),
        "p95": float(np.percentile(diff, 95)),
        "p99": float(np.percentile(diff, 99)),
        "max": int(diff.max()),
    }


def _tail_brightness_stats(frames: np.ndarray, fps: float, seconds: float = 2.0) -> dict:
    tail_count = max(1, min(frames.shape[0], int(round(float(fps) * seconds))))
    tail = frames[-tail_count:]
    flat = tail.reshape(tail.shape[0], -1)
    means = flat.mean(axis=1)
    bright_frac = (tail > 245).mean(axis=(1, 2, 3))
    first_bright = np.where((means > 240.0) | (bright_frac > 0.95))[0]
    first_frame = None
    if first_bright.size:
        first_frame = int(frames.shape[0] - tail_count + first_bright[0])
    return {
        "frames_checked": int(tail_count),
        "mean": float(means.mean()),
        "max_frame_mean": float(means.max()),
        "max_bright_frac": float(bright_frac.max()),
        "first_bright_frame": first_frame,
    }


def _label_row(width: int, labels: tuple[str, ...], height: int = 24) -> np.ndarray:
    from PIL import Image, ImageDraw

    panel_width = width // len(labels)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for idx, label in enumerate(labels):
        draw.text((idx * panel_width + 6, 6), label, fill=(0, 0, 0))
    return np.array(image, dtype=np.uint8)


def _comparison_frame_iter(
    simple_frames: np.ndarray,
    native_frames: np.ndarray,
    diff_scale: int,
) -> Iterable[np.ndarray]:
    frame_count, height, width, _channels = simple_frames.shape
    label = _label_row(
        width * 3,
        ("simple", "native Conv3d", f"abs diff x{diff_scale}"),
    )
    for idx in range(frame_count):
        simple = simple_frames[idx]
        native = native_frames[idx]
        diff = np.clip(
            np.abs(simple.astype(np.int16) - native.astype(np.int16)) * diff_scale,
            0,
            255,
        ).astype(np.uint8)
        frame = np.concatenate([simple, native, diff], axis=1)
        yield np.concatenate([label, frame], axis=0)


def _write_mp4_from_frames(
    path: Path,
    frames: Iterable[np.ndarray],
    width: int,
    height: int,
    frame_count: int,
    fps: float,
    audio_path: Path | None,
    crf: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0?"])
    else:
        cmd.append("-an")
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
        ]
    )
    if audio_path is not None:
        cmd.extend(["-c:a", "aac", "-b:a", "320k", "-shortest"])
    cmd.extend(["-loglevel", "error", str(path)])

    print(f"writing video: {path}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        try:
            from tqdm import tqdm

            iterator = tqdm(frames, total=frame_count, desc="video frames", ncols=80, ascii=True, mininterval=1.0, miniters=10)
        except ImportError:
            iterator = frames
        for frame in iterator:
            if frame.shape != (height, width, 3):
                raise ValueError(f"Frame has shape {frame.shape}; expected {(height, width, 3)}")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}: {stderr}")


def _write_comparison_video(
    path: Path,
    simple_frames: np.ndarray,
    native_frames: np.ndarray,
    fps: float,
    audio_path: Path | None,
    crf: int,
    diff_scale: int,
) -> None:
    frame_count, height, width, _channels = simple_frames.shape
    _write_mp4_from_frames(
        path,
        _comparison_frame_iter(simple_frames, native_frames, diff_scale),
        width * 3,
        height + 24,
        frame_count,
        fps,
        audio_path,
        crf,
    )


def _write_individual_video(
    path: Path,
    frames: np.ndarray,
    fps: float,
    audio_path: Path | None,
    crf: int,
) -> None:
    frame_count, height, width, _channels = frames.shape
    _write_mp4_from_frames(path, frames, width, height, frame_count, fps, audio_path, crf)


def _write_contact_sheet(path: Path, frames: list[int], simple_samples: list[np.ndarray], native_samples: list[np.ndarray]) -> None:
    from PIL import Image, ImageDraw

    thumb_w, thumb_h = 256, 144
    label_h = 22
    sheet = Image.new("RGB", (3 * thumb_w, len(frames) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    headers = ("simple", "native Conv3d", "abs diff x3")

    for row, frame_idx in enumerate(frames):
        y0 = row * (thumb_h + label_h)
        simple = simple_samples[row]
        native = native_samples[row]
        diff = np.clip(np.abs(simple.astype(np.int16) - native.astype(np.int16)) * 3, 0, 255).astype(np.uint8)
        for col, arr in enumerate((simple, native, diff)):
            img = Image.fromarray(arr).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(img, (col * thumb_w, y0 + label_h))
            text = f"{headers[col]} f={frame_idx}" if col == 0 else headers[col]
            draw.text((col * thumb_w + 4, y0 + 4), text, fill=(0, 0, 0))

    sheet.save(path)


def _write_single_contact_sheet(path: Path, frames: list[int], samples: list[np.ndarray], label: str) -> None:
    from PIL import Image, ImageDraw

    thumb_w, thumb_h = 256, 144
    label_h = 22
    sheet = Image.new("RGB", (thumb_w, len(frames) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for row, frame_idx in enumerate(frames):
        y0 = row * (thumb_h + label_h)
        img = Image.fromarray(samples[row]).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(img, (0, y0 + label_h))
        draw.text((4, y0 + 4), f"{label} f={frame_idx}", fill=(0, 0, 0))

    sheet.save(path)


def _build_simple_decoder(config: dict, dtype: mx.Dtype, padding: str, vae_weights: Path) -> SimpleVideoDecoder:
    decoder = SimpleVideoDecoder(
        decoder_blocks=config.get("decoder_blocks"),
        base_channels=config.get("decoder_base_channels", 128),
        timestep_conditioning=config.get("timestep_conditioning", False),
        compute_dtype=dtype,
        spatial_padding_mode=padding,
    )
    load_vae_decoder_weights(decoder, str(vae_weights))
    return decoder


def _build_native_decoder(
    config: dict,
    dtype: mx.Dtype,
    padding: str,
    vae_weights: Path,
    force_causal: bool,
) -> NativeConv3dVideoDecoder:
    decoder = NativeConv3dVideoDecoder(
        decoder_blocks=config.get("decoder_blocks"),
        base_channels=config.get("decoder_base_channels", 128),
        timestep_conditioning=config.get("timestep_conditioning", False),
        compute_dtype=dtype,
        spatial_padding_mode=padding,
        causal=force_causal,
    )
    load_native_vae_decoder_weights(decoder, str(vae_weights))
    return decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", required=True, type=Path, help="NPZ containing final_video_latent")
    parser.add_argument("--latent-key", default="final_video_latent")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, type=Path, help="Stock checkpoint for VAE config metadata")
    parser.add_argument(
        "--vae-weights",
        "--component-weights",
        dest="vae_weights",
        default=None,
        type=Path,
        help=(
            "Safetensors containing video VAE weights. Defaults to a split "
            "video VAE cache derived from --weights."
        ),
    )
    parser.add_argument("--weights-cache", choices=["off", "auto", "rebuild"], default="auto")
    parser.add_argument("--weights-cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae-spatial-padding", default="zero", choices=["reflect", "zero"])
    parser.add_argument("--height", type=int, default=None, help="Decoded height; inferred from latent if omitted")
    parser.add_argument("--width", type=int, default=None, help="Decoded width; inferred from latent if omitted")
    parser.add_argument("--frames", type=int, default=None, help="Decoded frame count; inferred from latent if omitted")
    parser.add_argument(
        "--decoder",
        choices=["both", "simple", "native-conv3d"],
        default="both",
        help="Which decoder to run. Use native-conv3d alone when simple would exceed memory.",
    )
    parser.add_argument("--no-tiling", action="store_true", help="Alias for --vae-tiling off")
    parser.add_argument(
        "--vae-tiling",
        choices=["auto", "off", "custom"],
        default="auto",
        help="Tiling policy for both decoders. Use custom to try larger temporal chunks.",
    )
    parser.add_argument(
        "--vae-temporal-tile-frames",
        type=parse_non_negative_int,
        default=None,
        help="Custom temporal tile size in decoded frames; must be divisible by 8, or 0 to disable temporal tiling",
    )
    parser.add_argument(
        "--vae-temporal-overlap-frames",
        type=parse_non_negative_int,
        default=24,
        help="Custom temporal overlap in decoded frames; must be divisible by 8",
    )
    parser.add_argument(
        "--vae-spatial-tile-pixels",
        type=parse_non_negative_int,
        default=None,
        help="Custom spatial tile size in decoded pixels; must be divisible by 32, or 0 to disable spatial tiling",
    )
    parser.add_argument(
        "--vae-spatial-overlap-pixels",
        type=parse_non_negative_int,
        default=64,
        help="Custom spatial overlap in decoded pixels; must be divisible by 32",
    )
    parser.add_argument("--sample-frames", default=None, help="Comma-separated decoded frame indices to sample")
    parser.add_argument(
        "--native-causal",
        action="store_true",
        help="Diagnostic: run the native Conv3d decoder with causal temporal padding.",
    )
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=None)
    parser.add_argument("--fps", type=float, default=24.0, help="FPS for generated comparison videos")
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help=(
            "Optional WAV/MP4/etc. whose first audio stream is muxed into comparison "
            "videos. Defaults to a sibling .wav next to --latent when present."
        ),
    )
    parser.add_argument("--no-audio", action="store_true", help="Do not auto-mux a sibling WAV sidecar")
    parser.add_argument(
        "--no-comparison-video",
        action="store_true",
        help="Skip writing the side-by-side MP4; JSON/contact sheet are still written.",
    )
    parser.add_argument(
        "--save-individual-videos",
        action="store_true",
        help="Also write separate simple/native MP4 files, using the same optional audio track.",
    )
    parser.add_argument("--video-crf", type=int, default=18, help="x264 CRF for comparison videos")
    parser.add_argument("--diff-video-scale", type=int, default=3, help="Amplification for the diff panel")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output prefix for JSON/contact sheet. Defaults next to the latent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = parse_compute_dtype(args.dtype)
    if args.mlx_cache_limit_gb is not None:
        mx.set_cache_limit(int(args.mlx_cache_limit_gb * (1000**3)))
        mx.clear_cache()
    vae_weights = args.vae_weights
    if vae_weights is None:
        if args.weights_cache == "off":
            vae_weights = args.weights
        else:
            cache_result = ensure_weight_family_caches(
                str(args.weights),
                families=("video_vae",),
                cache_mode=args.weights_cache,
                cache_root=args.weights_cache_dir,
            )
            vae_weights = cache_result.cache_paths["video_vae"]
    audio_path = args.audio
    if audio_path is None and not args.no_audio:
        sibling_wav = args.latent.with_suffix(".wav")
        if sibling_wav.exists():
            audio_path = sibling_wav
    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"Audio source does not exist: {audio_path}")

    latent = _load_latent(args.latent, args.latent_key, dtype)
    inferred_h, inferred_w, inferred_frames = _infer_video_shape(latent)
    height = args.height or inferred_h
    width = args.width or inferred_w
    frame_count = args.frames or inferred_frames
    sample_frames = _sample_frames(frame_count, args.sample_frames)
    vae_tiling_mode = "off" if args.no_tiling else args.vae_tiling
    # Native-only runs can use the RAM-aware native planner. A/B runs should
    # stay conservative because the simple decoder has a larger decode peak.
    auto_decoder_backend = (
        "native-conv3d" if args.decoder == "native-conv3d" else "simple"
    )
    tiling, auto_tiling = build_vae_tiling_config(
        vae_tiling_mode,
        height=height,
        width=width,
        num_frames=frame_count,
        decoder_backend=auto_decoder_backend,
        temporal_tile_frames=args.vae_temporal_tile_frames,
        temporal_overlap_frames=args.vae_temporal_overlap_frames,
        spatial_tile_pixels=args.vae_spatial_tile_pixels,
        spatial_overlap_pixels=args.vae_spatial_overlap_pixels,
    )
    config = get_vae_config(str(args.weights))

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.latent.with_name(f"{args.latent.stem}_vae_ab_{time.strftime('%Y%m%d_%H%M%S')}")
    json_path = prefix.with_suffix(".json")
    sheet_path = prefix.with_suffix(".png")

    print("VAE decoder comparison")
    print(f"  latent: {args.latent}")
    print(f"  VAE weights: {vae_weights}")
    print(f"  shape: {width}x{height}, {frame_count} frames")
    print(f"  decoder: {args.decoder}")
    print(f"  tiling: {describe_vae_tiling_config(tiling, auto_tiling)}")
    if args.native_causal:
        print("  native temporal padding: causal")
    print(f"  sample frames: {sample_frames}")
    print(f"  audio: {audio_path if audio_path is not None else 'none'}")

    results: dict = {
        "latent": str(args.latent),
        "weights": str(args.weights),
        "vae_weights": str(vae_weights),
        "height": height,
        "width": width,
        "frames": frame_count,
        "decoder": args.decoder,
        "sample_frames": sample_frames,
        "vae_tiling_mode": vae_tiling_mode,
        "tiling": describe_vae_tiling_config(tiling, auto_tiling),
        "vae_temporal_tile_frames": args.vae_temporal_tile_frames,
        "vae_temporal_overlap_frames": args.vae_temporal_overlap_frames,
        "vae_spatial_tile_pixels": args.vae_spatial_tile_pixels,
        "vae_spatial_overlap_pixels": args.vae_spatial_overlap_pixels,
        "dtype": args.dtype,
        "vae_spatial_padding": args.vae_spatial_padding,
        "native_causal": args.native_causal,
        "fps": args.fps,
        "audio": None if audio_path is None else str(audio_path),
    }

    frames_by_decoder: dict[str, np.ndarray] = {}
    samples_by_decoder: dict[str, list[np.ndarray]] = {}

    if args.decoder in ("both", "simple"):
        print("\n[simple] loading...")
        t0 = time.perf_counter()
        simple_decoder = _build_simple_decoder(config, dtype, args.vae_spatial_padding, vae_weights)
        results["simple_load_s"] = time.perf_counter() - t0
        simple_video, simple_decode = _decode_with_timing("simple", simple_decoder, latent, tiling)
        simple_stats, simple_samples = _sample_stats(simple_video, sample_frames)
        frames_by_decoder["simple"] = _video_to_uint8_frames("simple", simple_video)
        samples_by_decoder["simple"] = simple_samples
        results["simple"] = {
            **simple_decode,
            "samples": simple_stats,
            "tail_uint8": _tail_brightness_stats(frames_by_decoder["simple"], args.fps),
        }
        del simple_video, simple_decoder
        _clear()

    if args.decoder in ("both", "native-conv3d"):
        print("\n[native-conv3d] loading...")
        t0 = time.perf_counter()
        native_decoder = _build_native_decoder(
            config,
            dtype,
            args.vae_spatial_padding,
            vae_weights,
            args.native_causal,
        )
        results["native_load_s"] = time.perf_counter() - t0
        native_video, native_decode = _decode_with_timing("native-conv3d", native_decoder, latent, tiling)
        native_stats, native_samples = _sample_stats(native_video, sample_frames)
        frames_by_decoder["native_conv3d"] = _video_to_uint8_frames("native Conv3d", native_video)
        samples_by_decoder["native_conv3d"] = native_samples
        results["native_conv3d"] = {
            **native_decode,
            "samples": native_stats,
            "tail_uint8": _tail_brightness_stats(frames_by_decoder["native_conv3d"], args.fps),
        }
        del native_video, native_decoder
        _clear()

    if args.decoder == "both":
        simple_frames = frames_by_decoder["simple"]
        native_frames = frames_by_decoder["native_conv3d"]
        simple_samples = samples_by_decoder["simple"]
        native_samples = samples_by_decoder["native_conv3d"]
        results["sample_uint8_absdiff"] = _diff_stats(simple_samples, native_samples)
        results["full_video_uint8_absdiff"] = _uint8_absdiff_stats(simple_frames, native_frames)

    if args.decoder == "both" and not args.no_comparison_video:
        comparison_path = prefix.with_name(f"{prefix.name}_side_by_side.mp4")
        try:
            _write_comparison_video(
                comparison_path,
                simple_frames,
                native_frames,
                args.fps,
                audio_path,
                args.video_crf,
                args.diff_video_scale,
            )
            results["comparison_video"] = str(comparison_path)
            print(f"comparison video: {comparison_path}")
        except Exception as exc:
            results["comparison_video_error"] = repr(exc)
            print(f"comparison video failed: {exc!r}")
    elif args.decoder != "both" and not args.no_comparison_video:
        key = "simple" if args.decoder == "simple" else "native_conv3d"
        video_path = prefix.with_name(f"{prefix.name}_{key}.mp4")
        try:
            _write_individual_video(video_path, frames_by_decoder[key], args.fps, audio_path, args.video_crf)
            results[f"{key}_video"] = str(video_path)
            print(f"{key} video: {video_path}")
        except Exception as exc:
            results[f"{key}_video_error"] = repr(exc)
            print(f"{key} video failed: {exc!r}")

    if args.save_individual_videos:
        for label, frames in frames_by_decoder.items():
            video_path = prefix.with_name(f"{prefix.name}_{label}.mp4")
            try:
                _write_individual_video(video_path, frames, args.fps, audio_path, args.video_crf)
                results[f"{label}_video"] = str(video_path)
                print(f"{label} video: {video_path}")
            except Exception as exc:
                results[f"{label}_video_error"] = repr(exc)
                print(f"{label} video failed: {exc!r}")

    try:
        if args.decoder == "both":
            _write_contact_sheet(
                sheet_path,
                sample_frames,
                samples_by_decoder["simple"],
                samples_by_decoder["native_conv3d"],
            )
        else:
            key = "simple" if args.decoder == "simple" else "native_conv3d"
            label = "simple" if key == "simple" else "native Conv3d"
            _write_single_contact_sheet(sheet_path, sample_frames, samples_by_decoder[key], label)
        results["contact_sheet"] = str(sheet_path)
        print(f"contact sheet: {sheet_path}")
    except Exception as exc:
        results["contact_sheet_error"] = repr(exc)
        print(f"contact sheet failed: {exc!r}")

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results json: {json_path}")
    print("\nSummary:")
    summary = {}
    if "simple" in results:
        summary.update(
            {
                "simple_decode_s": results["simple"]["decode_s"],
                "simple_peak_gb": results["simple"]["memory"]["peak_gb"],
                "simple_luma_mean": results["simple"]["samples"]["luma_mean"],
                "simple_tail_uint8": results["simple"]["tail_uint8"],
            }
        )
    if "native_conv3d" in results:
        summary.update(
            {
                "native_conv3d_decode_s": results["native_conv3d"]["decode_s"],
                "native_conv3d_peak_gb": results["native_conv3d"]["memory"]["peak_gb"],
                "native_conv3d_luma_mean": results["native_conv3d"]["samples"]["luma_mean"],
                "native_conv3d_tail_uint8": results["native_conv3d"]["tail_uint8"],
            }
        )
    if "sample_uint8_absdiff" in results:
        summary["sample_uint8_absdiff"] = results["sample_uint8_absdiff"]
        summary["full_video_uint8_absdiff"] = results["full_video_uint8_absdiff"]
        summary["comparison_video"] = results.get("comparison_video")
    summary["video"] = results.get("simple_video") or results.get("native_conv3d_video")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
