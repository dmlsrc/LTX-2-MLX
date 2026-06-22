#!/usr/bin/env python3
"""Probe VAE spatial boundary behavior from a saved latent sidecar.

The script decodes the same saved final_video_latent with small boundary
variants, then writes comparison sheets and simple difference metrics. It is
intended for diagnosing frame-edge artifacts, not for production generation.
"""

from __future__ import annotations

import argparse
import gc
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.videotoolbox.images import draw_labels, load_image_rgb, resize_lanczos, save_image

np = None
RunTimings = None
Timer = None
encode_frames_dir = None
frames_from_video_tensor = None
load_latents = None
make_decoder = None
make_audio_decoder_and_vocoder = None
mlx_mem_summary = None
parse_dtype = None
tiling_config_for_mode = None
write_frames = None
decode_audio_latent = None
write_wav = None


def load_runtime_deps() -> None:
    global np
    global RunTimings, Timer, encode_frames_dir, frames_from_video_tensor
    global load_latents, make_decoder, make_audio_decoder_and_vocoder, mlx_mem_summary
    global parse_dtype, tiling_config_for_mode, write_frames, decode_audio_latent, write_wav

    if np is None:
        import numpy as _np

        np = _np

    if RunTimings is None:
        from scripts.decode_latent_debug import (
            RunTimings as _RunTimings,
        )
        from scripts.decode_latent_debug import (
            Timer as _Timer,
        )
        from scripts.decode_latent_debug import (
            decode_audio_latent as _decode_audio_latent,
        )
        from scripts.decode_latent_debug import (
            encode_frames_dir as _encode_frames_dir,
        )
        from scripts.decode_latent_debug import (
            frames_from_video_tensor as _frames_from_video_tensor,
        )
        from scripts.decode_latent_debug import (
            load_latents as _load_latents,
        )
        from scripts.decode_latent_debug import (
            make_audio_decoder_and_vocoder as _make_audio_decoder_and_vocoder,
        )
        from scripts.decode_latent_debug import (
            make_decoder as _make_decoder,
        )
        from scripts.decode_latent_debug import (
            mlx_mem_summary as _mlx_mem_summary,
        )
        from scripts.decode_latent_debug import (
            parse_dtype as _parse_dtype,
        )
        from scripts.decode_latent_debug import (
            tiling_config_for_mode as _tiling_config_for_mode,
        )
        from scripts.decode_latent_debug import (
            write_frames as _write_frames,
        )
        from scripts.decode_latent_debug import (
            write_wav as _write_wav,
        )

        RunTimings = _RunTimings
        Timer = _Timer
        encode_frames_dir = _encode_frames_dir
        frames_from_video_tensor = _frames_from_video_tensor
        load_latents = _load_latents
        make_decoder = _make_decoder
        make_audio_decoder_and_vocoder = _make_audio_decoder_and_vocoder
        mlx_mem_summary = _mlx_mem_summary
        parse_dtype = _parse_dtype
        tiling_config_for_mode = _tiling_config_for_mode
        write_frames = _write_frames
        decode_audio_latent = _decode_audio_latent
        write_wav = _write_wav


@dataclass(frozen=True)
class BoundaryVariant:
    name: str
    latent_pad_cells: int = 0
    latent_pad_mode: str | None = None
    zero_spatial_conv_padding: bool = False

    @property
    def crop_pixels(self) -> int:
        return self.latent_pad_cells * 32


def parse_variant(name: str) -> BoundaryVariant:
    if name == "orig":
        return BoundaryVariant(name)
    if name == "orig_zero_convpad":
        return BoundaryVariant(name, zero_spatial_conv_padding=True)

    match = re.fullmatch(r"pad(\d+)_(replicate|reflect|zero)_crop", name)
    if not match:
        raise ValueError(
            f"Unknown variant {name!r}. Expected orig, orig_zero_convpad, "
            "or padN_replicate_crop / padN_reflect_crop / padN_zero_crop."
        )

    amount = int(match.group(1))
    if amount < 1:
        raise ValueError("Padding variants require N >= 1")
    return BoundaryVariant(name, latent_pad_cells=amount, latent_pad_mode=match.group(2))


def np_pad_mode(mode: str) -> str:
    if mode == "replicate":
        return "edge"
    if mode == "reflect":
        return "reflect"
    if mode == "zero":
        return "constant"
    raise ValueError(f"Unsupported pad mode: {mode}")


def pad_latent_spatial(latent: Any, variant: BoundaryVariant, mx_mod: Any) -> Any:
    if variant.latent_pad_cells == 0:
        return latent

    amount = variant.latent_pad_cells
    # NumPy cannot consume MLX BF16 buffers directly, so do the small diagnostic
    # pad copy through FP32 and cast back to the original decode dtype afterward.
    arr = np.array(latent.astype(mx_mod.float32))
    if amount >= arr.shape[3] or amount >= arr.shape[4]:
        raise ValueError(
            f"{variant.name} cannot pad latent spatial shape {arr.shape[3:]} by {amount} cells"
        )

    pad_width = ((0, 0), (0, 0), (0, 0), (amount, amount), (amount, amount))
    mode = np_pad_mode(variant.latent_pad_mode or "replicate")
    if mode == "constant":
        padded = np.pad(arr, pad_width, mode=mode, constant_values=0)
    else:
        padded = np.pad(arr, pad_width, mode=mode)

    result = mx_mod.array(padded).astype(latent.dtype)
    mx_mod.eval(result)
    return result


def crop_frames(frames: np.ndarray, crop_px: int) -> np.ndarray:
    if crop_px <= 0:
        return frames
    if frames.shape[1] <= crop_px * 2 or frames.shape[2] <= crop_px * 2:
        raise ValueError(f"Cannot crop {crop_px}px from frame shape {frames.shape}")
    return frames[:, crop_px:-crop_px, crop_px:-crop_px, :]


class DecoderSpatialPadding:
    def __init__(self, decoder: Any, mode: str):
        self.decoder = decoder
        self.mode = mode
        self.previous_mode = None

    def __enter__(self):
        self.previous_mode = getattr(self.decoder, "spatial_padding_mode", "reflect")
        self.decoder.set_spatial_padding_mode(self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.decoder.set_spatial_padding_mode(self.previous_mode)


def maybe_clear_mlx(mx_mod: Any) -> None:
    gc.collect()
    try:
        mx_mod.clear_cache()
    except Exception:
        pass


def decode_variant(
    *,
    variant: BoundaryVariant,
    latent: Any,
    decoder: Any,
    output_dir: Path,
    fps: float,
    mx_mod: Any,
    mode: str,
    full_decode_chunk_size: int,
    show_memory: bool,
    timings: RunTimings,
    audio_wav_path: Path | None = None,
    audio_sample_rate: int | None = None,
) -> int:
    from LTX_2_MLX.model.video_vae.decode_utils import decode_latent
    from LTX_2_MLX.model.video_vae.tiling import decode_streaming

    print("\n" + "=" * 80)
    print(f"Boundary variant: {variant.name}")
    if variant.latent_pad_cells:
        print(
            f"  latent spatial pad: {variant.latent_pad_cells} cells "
            f"({variant.crop_pixels}px crop after decode), mode={variant.latent_pad_mode}"
        )
    if variant.zero_spatial_conv_padding:
        print("  decoder spatial padding: zero for this variant")
    if show_memory:
        print("before decode:", mlx_mem_summary(mx_mod))

    variant_latent = pad_latent_spatial(latent, variant, mx_mod)
    cfg = tiling_config_for_mode(mode, variant_latent)
    frames_dir = output_dir / f"frames_{variant.name}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    decode_seconds = 0.0
    write_seconds = 0.0

    patch_context = DecoderSpatialPadding(decoder, "zero") if variant.zero_spatial_conv_padding else nullcontext()
    with patch_context:
        if cfg is None:
            started = time.perf_counter()
            video = decode_latent(
                variant_latent,
                decoder,
                temporal_chunk_size=full_decode_chunk_size,
            )
            mx_mod.eval(video)
            decode_seconds += time.perf_counter() - started

            started = time.perf_counter()
            frames = crop_frames(frames_from_video_tensor(video, mx_mod), variant.crop_pixels)
            frame_index = write_frames(frames, frames_dir, 0)
            write_seconds += time.perf_counter() - started
            del video, frames
        else:
            chunk_index = 0
            chunk_iter = iter(decode_streaming(variant_latent, decoder, cfg, show_progress=True))
            while True:
                started = time.perf_counter()
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                mx_mod.eval(chunk)
                decode_seconds += time.perf_counter() - started
                if show_memory:
                    print(f"chunk {chunk_index}:", mlx_mem_summary(mx_mod))

                started = time.perf_counter()
                frames = crop_frames(frames_from_video_tensor(chunk, mx_mod), variant.crop_pixels)
                frame_index = write_frames(frames, frames_dir, frame_index)
                write_seconds += time.perf_counter() - started
                del chunk, frames
                maybe_clear_mlx(mx_mod)
                if show_memory:
                    print(f"after writing chunk {chunk_index}:", mlx_mem_summary(mx_mod))
                chunk_index += 1

    timings.add(f"{variant.name} decode", decode_seconds)
    timings.add(f"{variant.name} write frames", write_seconds)

    output_path = output_dir / f"{variant.name}.mp4"
    with Timer(timings, f"{variant.name} ffmpeg encode"):
        encode_frames_dir(frames_dir, output_path, fps, audio_wav_path, audio_sample_rate)
    print(f"Saved: {output_path}")

    if variant_latent is not latent:
        del variant_latent
    maybe_clear_mlx(mx_mod)
    if show_memory:
        print("after cleanup:", mlx_mem_summary(mx_mod))
    return frame_index


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def parse_comparison_frames(value: str | None, frame_count: int) -> list[int]:
    if value:
        frames = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
        return [idx for idx in frames if 0 <= idx < frame_count]
    if frame_count <= 1:
        return [0]
    return sorted(
        {
            0,
            frame_count // 4,
            frame_count // 2,
            (frame_count * 3) // 4,
            frame_count - 1,
        }
    )


def load_frame(frames_dir: Path, index: int) -> np.ndarray:
    return np.asarray(load_image_rgb(frames_dir / f"frame_{index:05d}.png"))


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    new_h = max(1, round(image.shape[0] * width / image.shape[1]))
    return np.asarray(resize_lanczos(image, width, new_h))


def draw_sheet(cells: list[tuple[str, np.ndarray]], output_path: Path, columns: int, cell_width: int) -> None:
    if not cells:
        return

    label_h = 18
    pad = 8
    images = [(label, resize_to_width(image, cell_width)) for label, image in cells]
    cell_h = max(image.shape[0] for _, image in images) + label_h
    rows = (len(images) + columns - 1) // columns
    sheet_w = columns * cell_width + (columns + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet = np.full((sheet_h, sheet_w, 3), (24, 24, 24), dtype=np.uint8)

    labels: list[tuple[int, int, str]] = []
    for idx, (label, image) in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = pad + col * (cell_width + pad)
        y = pad + row * (cell_h + pad)
        labels.append((x, y, label))
        h, w = image.shape[0], image.shape[1]
        sheet[y + label_h:y + label_h + h, x:x + w] = image

    sheet = np.asarray(draw_labels(sheet, labels, color=(235, 235, 235)))
    save_image(sheet, output_path)


def band_crop(frame: np.ndarray, band_px: int, side: str) -> np.ndarray:
    if side == "top":
        arr = frame[:band_px]
    elif side == "bottom":
        arr = frame[-band_px:]
    elif side == "left":
        arr = frame[:, :band_px]
    elif side == "right":
        arr = frame[:, -band_px:]
    else:
        raise ValueError(side)
    return arr


def write_comparisons(
    *,
    output_dir: Path,
    variants: list[BoundaryVariant],
    comparison_frames: list[int],
    band_px: int,
    sheet_width: int,
) -> None:
    names = [variant.name for variant in variants]
    cells_full: list[tuple[str, np.ndarray]] = []
    cells_top: list[tuple[str, np.ndarray]] = []
    cells_bottom: list[tuple[str, np.ndarray]] = []
    cells_left: list[tuple[str, np.ndarray]] = []
    cells_right: list[tuple[str, np.ndarray]] = []

    for frame_index in comparison_frames:
        for name in names:
            frame = load_frame(output_dir / f"frames_{name}", frame_index)
            label = f"{name} f{frame_index}"
            cells_full.append((label, frame))
            cells_top.append((label, band_crop(frame, band_px, "top")))
            cells_bottom.append((label, band_crop(frame, band_px, "bottom")))
            cells_left.append((label, band_crop(frame, band_px, "left")))
            cells_right.append((label, band_crop(frame, band_px, "right")))

    columns = len(names)
    draw_sheet(cells_full, output_dir / "compare_full_frames.png", columns=columns, cell_width=sheet_width)
    draw_sheet(cells_top, output_dir / f"compare_top_{band_px}px.png", columns=columns, cell_width=sheet_width)
    draw_sheet(cells_bottom, output_dir / f"compare_bottom_{band_px}px.png", columns=columns, cell_width=sheet_width)
    draw_sheet(cells_left, output_dir / f"compare_left_{band_px}px.png", columns=columns, cell_width=max(96, sheet_width // 4))
    draw_sheet(cells_right, output_dir / f"compare_right_{band_px}px.png", columns=columns, cell_width=max(96, sheet_width // 4))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def write_metrics(
    *,
    output_dir: Path,
    variants: list[BoundaryVariant],
    comparison_frames: list[int],
    band_px: int,
) -> None:
    if not variants or variants[0].name != "orig":
        return

    lines = [
        "Mean absolute uint8 difference against orig on sampled frames.",
        f"sampled_frames={','.join(str(idx) for idx in comparison_frames)}",
        f"band_px={band_px}",
        "",
        f"{'variant':<24} {'full':>10} {'top':>10} {'bottom':>10} {'left':>10} {'right':>10} {'center':>10}",
    ]

    for variant in variants[1:]:
        full = []
        top = []
        bottom = []
        left = []
        right = []
        center = []
        for frame_index in comparison_frames:
            orig = load_frame(output_dir / "frames_orig", frame_index)
            other = load_frame(output_dir / f"frames_{variant.name}", frame_index)
            h, w = orig.shape[:2]
            cy0 = h // 2 - band_px // 2
            cy1 = cy0 + band_px
            cx0 = w // 2 - band_px // 2
            cx1 = cx0 + band_px
            full.append(mean_abs_diff(orig, other))
            top.append(mean_abs_diff(orig[:band_px], other[:band_px]))
            bottom.append(mean_abs_diff(orig[-band_px:], other[-band_px:]))
            left.append(mean_abs_diff(orig[:, :band_px], other[:, :band_px]))
            right.append(mean_abs_diff(orig[:, -band_px:], other[:, -band_px:]))
            center.append(mean_abs_diff(orig[cy0:cy1, cx0:cx1], other[cy0:cy1, cx0:cx1]))

        lines.append(
            f"{variant.name:<24} "
            f"{np.mean(full):10.4f} {np.mean(top):10.4f} {np.mean(bottom):10.4f} "
            f"{np.mean(left):10.4f} {np.mean(right):10.4f} {np.mean(center):10.4f}"
        )

    (output_dir / "boundary_metrics.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", required=True, help="NPZ sidecar produced by LTX_2_MLX/generate.py --save-latents.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--vae-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--latent-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--mode",
        default="auto",
        help="Decode mode from decode_latent_debug.py: auto, none, default, both384_24, etc.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "orig",
            "pad1_replicate_crop",
            "pad1_reflect_crop",
            "pad1_zero_crop",
            "pad2_replicate_crop",
            "orig_zero_convpad",
        ],
    )
    parser.add_argument("--comparison-frames", help="Comma-separated decoded frame indices for sheets/metrics.")
    parser.add_argument("--band-px", type=int, default=64)
    parser.add_argument("--sheet-width", type=int, default=320)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--show-memory", action="store_true")
    parser.add_argument(
        "--decode-audio",
        action="store_true",
        help="Decode final_audio_latent once, write boundary_probe_audio.wav, and mux it into every variant MP4.",
    )
    parser.add_argument("--full-decode-chunk-size", type=int, default=9999)
    args = parser.parse_args()
    load_runtime_deps()

    import mlx.core as mx

    variants = [parse_variant(name) for name in args.variants]
    if not variants or variants[0].name != "orig":
        raise ValueError("--variants must include orig first so comparisons have a baseline")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timings = RunTimings()

    with Timer(timings, "load latent"):
        latent, audio_latent = load_latents(args.latent, mx, args.latent_dtype)
    compute_dtype = parse_dtype(mx, args.vae_dtype)
    print(f"VAE compute dtype: {args.vae_dtype}")
    with Timer(timings, "load vae decoder"):
        decoder = make_decoder(args.weights, compute_dtype)

    audio_wav_path = None
    audio_sample_rate = None
    if args.decode_audio:
        if audio_latent is None:
            raise ValueError("--decode-audio was requested, but this NPZ does not contain final_audio_latent")
        with Timer(timings, "load audio decoder"):
            audio_decoder, vocoder, audio_sample_rate = make_audio_decoder_and_vocoder(args.weights, compute_dtype)
        with Timer(timings, "audio decode"):
            audio_waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx)
        audio_wav_path = args.output_dir / "boundary_probe_audio.wav"
        with Timer(timings, "audio wav write"):
            write_wav(audio_waveform, audio_wav_path, audio_sample_rate)
        del audio_waveform, audio_decoder, vocoder
        maybe_clear_mlx(mx)

    frame_counts: dict[str, int] = {}
    for variant in variants:
        frame_counts[variant.name] = decode_variant(
            variant=variant,
            latent=latent,
            decoder=decoder,
            output_dir=args.output_dir,
            fps=args.fps,
            mx_mod=mx,
            mode=args.mode,
            full_decode_chunk_size=args.full_decode_chunk_size,
            show_memory=args.show_memory,
            timings=timings,
            audio_wav_path=audio_wav_path,
            audio_sample_rate=audio_sample_rate,
        )

    frame_count = frame_counts["orig"]
    if any(count != frame_count for count in frame_counts.values()):
        raise RuntimeError(f"Variant frame counts differ: {frame_counts}")

    comparison_frames = parse_comparison_frames(args.comparison_frames, frame_count)
    if not comparison_frames:
        raise ValueError("No valid comparison frames")

    with Timer(timings, "comparison sheets"):
        write_comparisons(
            output_dir=args.output_dir,
            variants=variants,
            comparison_frames=comparison_frames,
            band_px=args.band_px,
            sheet_width=args.sheet_width,
        )
        write_metrics(
            output_dir=args.output_dir,
            variants=variants,
            comparison_frames=comparison_frames,
            band_px=args.band_px,
        )

    if not args.keep_frames:
        with Timer(timings, "cleanup frames"):
            for variant in variants:
                shutil.rmtree(args.output_dir / f"frames_{variant.name}", ignore_errors=True)

    timings.print_summary()
    print(f"Wrote VAE boundary probe to {args.output_dir}")


if __name__ == "__main__":
    main()
