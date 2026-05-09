#!/usr/bin/env python3
"""Decode saved LTX video latents with different VAE tiling modes.

Supports the --save-latents NPZ sidecar:
  - final_video_latent
  - final_video_latent_mlx_dtype
  - final_audio_latent
  - final_audio_latent_mlx_dtype
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# Add repo root to import path when run as scripts/decode_latent_debug.py.
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - minutes * 60:04.1f}s"
    return f"{seconds:.2f}s"


class RunTimings:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.sections: list[tuple[str, float]] = []

    def add(self, label: str, seconds: float) -> None:
        self.sections.append((label, seconds))

    def print_summary(self) -> None:
        total = time.perf_counter() - self.started_at
        if not self.sections:
            return

        width = max(len(label) for label, _ in self.sections + [("total", total)])
        print("\nTiming summary:")
        for label, seconds in self.sections:
            print(f"  {label:<{width}}  {format_duration(seconds)}")
        print(f"  {'total':<{width}}  {format_duration(total)}")


class Timer:
    def __init__(self, timings: RunTimings, label: str):
        self.timings = timings
        self.label = label
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.timings.add(self.label, time.perf_counter() - self.started_at)



def rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 ** 3)
    return value / (1024 ** 2)


def mlx_mem_summary(mx_mod: Any) -> str:
    parts = [f"rss_peak={rss_gib():.2f}GiB"]
    for label, names in (
        ("active", ("get_active_memory", "metal.get_active_memory")),
        ("peak", ("get_peak_memory", "metal.get_peak_memory")),
        ("cache", ("get_cache_memory", "metal.get_cache_memory")),
    ):
        fn = None
        for name in names:
            obj = mx_mod
            for part in name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            else:
                fn = obj
                break
        if fn is not None:
            try:
                parts.append(f"{label}={fn() / (1024 ** 3):.2f}GiB")
            except Exception:
                pass
    return "memory[" + ", ".join(parts) + "]"


def parse_dtype(mx_mod: Any, name: str):
    if name == "auto":
        return name
    if name in {"mlx.core.bfloat16", "bfloat16", "bf16"}:
        return mx_mod.bfloat16
    if name in {"mlx.core.float16", "float16", "fp16"}:
        return mx_mod.float16
    if name in {"mlx.core.float32", "float32", "fp32"}:
        return mx_mod.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_latents(path: str, mx_mod: Any, latent_dtype: str = "auto"):
    with np.load(path) as data:
        keys = sorted(data.files)
        if "final_video_latent" not in data:
            raise KeyError(f"{path} does not contain final_video_latent; keys={keys}")
        if "final_video_latent_mlx_dtype" not in data:
            raise KeyError(
                f"{path} is not a --save-latents sidecar: missing final_video_latent_mlx_dtype; "
                f"keys={keys}"
            )

        latent_np = data["final_video_latent"]
        stored_dtype = str(data["final_video_latent_mlx_dtype"])
        has_audio = "final_audio_latent" in data
        audio_np = data["final_audio_latent"] if has_audio else None
        audio_shape = audio_np.shape if audio_np is not None else None
        audio_dtype = str(data["final_audio_latent_mlx_dtype"]) if "final_audio_latent_mlx_dtype" in data else None

    print(f"NPZ keys: {', '.join(keys)}")
    print(f"Loaded video latent: shape={latent_np.shape}, numpy_dtype={latent_np.dtype}, mlx_dtype={stored_dtype or 'unknown'}")
    if has_audio:
        print(f"Found audio latent: shape={audio_shape}, mlx_dtype={audio_dtype or 'unknown'}")

    resolved = parse_dtype(mx_mod, latent_dtype)
    if resolved == "auto":
        resolved = parse_dtype(mx_mod, stored_dtype)

    latent = mx_mod.array(latent_np)
    if resolved is not None:
        latent = latent.astype(resolved)
    mx_mod.eval(latent)
    print(f"Decode latent dtype: {latent.dtype}")

    audio_latent = None
    if audio_np is not None:
        if audio_dtype is None:
            raise KeyError(f"{path} contains final_audio_latent but is missing final_audio_latent_mlx_dtype")
        audio_resolved = parse_dtype(mx_mod, audio_dtype)
        audio_latent = mx_mod.array(audio_np).astype(audio_resolved)
        mx_mod.eval(audio_latent)
        print(f"Decode audio latent dtype: {audio_latent.dtype}")

    return latent, audio_latent


def make_decoder(weights_path: str, compute_dtype: Any):
    from scripts.generate import get_vae_config
    from LTX_2_MLX.model.video_vae.simple_decoder import SimpleVideoDecoder, load_vae_decoder_weights

    vae_cfg = get_vae_config(weights_path)
    decoder_blocks = vae_cfg.get("decoder_blocks")
    base_channels = vae_cfg.get("base_channels", 128)
    timestep_conditioning = vae_cfg.get("timestep_conditioning", False)

    print(
        f"VAE config: blocks={len(decoder_blocks) if decoder_blocks else 'default'}, "
        f"base_ch={base_channels}, timestep={timestep_conditioning}"
    )

    decoder = SimpleVideoDecoder(
        decoder_blocks=decoder_blocks,
        base_channels=base_channels,
        timestep_conditioning=timestep_conditioning,
        compute_dtype=compute_dtype,
    )
    load_vae_decoder_weights(decoder, weights_path)
    gc.collect()
    return decoder


def make_audio_decoder_and_vocoder(weights_path: str, compute_dtype: Any):
    from scripts.generate import create_vocoder_for_checkpoint, print_audio_dtype_summary
    from LTX_2_MLX.model.audio_vae import (
        AudioDecoder,
        load_audio_decoder_weights,
        load_vocoder_weights,
        load_vocoder_with_bwe_weights,
    )

    print("Loading Audio VAE decoder...")
    audio_decoder = AudioDecoder(compute_dtype=compute_dtype)
    load_audio_decoder_weights(audio_decoder, weights_path)

    print("Loading Vocoder...")
    vocoder, is_bwe = create_vocoder_for_checkpoint(weights_path, compute_dtype)
    if is_bwe:
        print("  Detected BWE vocoder")
        load_vocoder_with_bwe_weights(vocoder, weights_path)
    else:
        load_vocoder_weights(vocoder, weights_path)
    print_audio_dtype_summary(compute_dtype, is_bwe)

    sample_rate = vocoder.output_sample_rate
    gc.collect()
    return audio_decoder, vocoder, sample_rate


def decode_audio_latent(audio_latent: Any, audio_decoder: Any, vocoder: Any, mx_mod: Any):
    mel_spectrogram = audio_decoder(audio_latent)
    mx_mod.eval(mel_spectrogram)
    waveform = vocoder(mel_spectrogram)
    waveform = waveform.astype(mx_mod.float32)
    mx_mod.eval(waveform)
    return waveform


def write_wav(audio_waveform: Any, output_path: Path, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_np = np.array(audio_waveform[0])
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
    audio_flat = audio_int16.T.flatten()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(audio_int16.shape[0])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_flat.tobytes())

    print(f"Wrote WAV: {output_path}")
    print(f"Audio: {audio_int16.shape[1]} samples, {audio_int16.shape[1] / sample_rate:.2f}s at {sample_rate}Hz")


def tiling_config_for_mode(mode: str, latent: Any | None = None):
    from LTX_2_MLX.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig

    if mode == "none":
        return None
    if mode == "auto":
        if latent is None:
            raise ValueError("mode=auto requires a latent to infer decoded dimensions")
        _, _, latent_frames, latent_height, latent_width = latent.shape
        return TilingConfig.auto(
            height=latent_height * 32,
            width=latent_width * 32,
            num_frames=1 + (latent_frames - 1) * 8,
        )
    if mode == "default":
        return TilingConfig.default()
    if mode == "spatial64":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(512, 64),
            temporal_config=None,
        )
    if mode == "spatial128":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(512, 128),
            temporal_config=None,
        )
    if mode == "temporal24":
        return TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(64, 24),
        )
    if mode == "temporal32":
        return TilingConfig(
            spatial_config=None,
            temporal_config=TemporalTilingConfig(64, 32),
        )
    if mode == "both64_24":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(512, 64),
            temporal_config=TemporalTilingConfig(64, 24),
        )
    if mode == "both384_24":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(384, 64),
            temporal_config=TemporalTilingConfig(64, 24),
        )
    if mode == "both256_24":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(256, 64),
            temporal_config=TemporalTilingConfig(64, 24),
        )
    if mode == "both128_24":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(512, 128),
            temporal_config=TemporalTilingConfig(64, 24),
        )
    if mode == "both128_32":
        return TilingConfig(
            spatial_config=SpatialTilingConfig(512, 128),
            temporal_config=TemporalTilingConfig(64, 32),
        )
    if mode == "test_small_both":
        return TilingConfig.test_small_both()
    raise ValueError(f"Unknown mode: {mode}")


def frames_from_video_tensor(video: Any, mx_mod: Any) -> np.ndarray:
    mx_mod.eval(video)
    arr = np.array(video)

    if arr.ndim == 5:
        arr = arr[0].transpose(1, 2, 3, 0)  # B,C,T,H,W -> T,H,W,C
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected decoded video shape: {arr.shape}")

    print(
        f"Decoded chunk/array: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}"
    )

    if arr.dtype == np.uint8:
        return arr
    if arr.min() >= -2.0 and arr.max() <= 2.0:
        arr = (arr + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8)


def encode_frames_dir(frames_dir: Path, output: Path, fps: float, audio_path: Path | None = None, audio_sample_rate: int | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
    ]
    if audio_path is not None:
        cmd.extend(["-i", str(audio_path)])
    cmd.extend(["-c:v", "libx264"])
    if audio_path is not None:
        cmd.extend(["-c:a", "aac", "-b:a", "320k"])
        if audio_sample_rate is not None:
            cmd.extend(["-ar", str(audio_sample_rate)])
    cmd.extend(["-pix_fmt", "yuv420p", "-crf", "18"])
    if audio_path is not None:
        cmd.append("-shortest")
    cmd.extend(["-loglevel", "error"])
    cmd.append(str(output))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_frames(frames: np.ndarray, frames_dir: Path, start: int = 0) -> int:
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{start + idx:05d}.png")
    return start + len(frames)


def decode_mode(
    *,
    mode: str,
    latent: Any,
    decoder: Any,
    output_dir: Path,
    fps: float,
    mx_mod: Any,
    keep_frames: bool,
    full_decode_chunk_size: int,
    show_memory: bool,
    timings: RunTimings,
    audio_wav_path: Path | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    from LTX_2_MLX.model.video_vae.simple_decoder import decode_latent
    from LTX_2_MLX.model.video_vae.tiling import decode_tiled

    print("\n" + "=" * 80)
    print(f"Decode mode: {mode}")
    if show_memory:
        print("before decode:", mlx_mem_summary(mx_mod))

    output_path = output_dir / f"decode_{mode}.mp4"
    frame_root = output_dir / f"frames_{mode}"
    if frame_root.exists():
        import shutil

        shutil.rmtree(frame_root)
    frame_root.mkdir(parents=True, exist_ok=True)

    cfg = tiling_config_for_mode(mode, latent)
    frame_index = 0
    decode_seconds = 0.0
    write_seconds = 0.0

    if cfg is None:
        started = time.perf_counter()
        video = decode_latent(
            latent,
            decoder,
            temporal_chunk_size=full_decode_chunk_size,
        )
        mx_mod.eval(video)
        decode_seconds += time.perf_counter() - started

        started = time.perf_counter()
        frames = frames_from_video_tensor(video, mx_mod)
        frame_index = write_frames(frames, frame_root, 0)
        write_seconds += time.perf_counter() - started
        del video, frames
    else:
        chunk_index = 0
        chunk_iter = iter(decode_tiled(latent, decoder, cfg, show_progress=True))
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
            frames = frames_from_video_tensor(chunk, mx_mod)
            frame_index = write_frames(frames, frame_root, frame_index)
            write_seconds += time.perf_counter() - started
            del chunk, frames
            gc.collect()
            try:
                mx_mod.clear_cache()
            except Exception:
                pass
            if show_memory:
                print(f"after writing chunk {chunk_index}:", mlx_mem_summary(mx_mod))
            chunk_index += 1

    timings.add(f"{mode} decode", decode_seconds)
    timings.add(f"{mode} write frames", write_seconds)

    print(f"Wrote {frame_index} frames to {frame_root}")
    with Timer(timings, f"{mode} ffmpeg encode"):
        encode_frames_dir(frame_root, output_path, fps, audio_wav_path, audio_sample_rate)
    print(f"Saved: {output_path}")

    with Timer(timings, f"{mode} cleanup"):
        if not keep_frames:
            import shutil

            shutil.rmtree(frame_root, ignore_errors=True)

        gc.collect()
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
    if show_memory:
        print("after mode cleanup:", mlx_mem_summary(mx_mod))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latent",
        required=True,
        help="NPZ sidecar produced by scripts/generate.py --save-latents.",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", default="/Users/Shared/huggingface/output/decode_tests")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--vae-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="VAE activation dtype for decode tests.",
    )
    parser.add_argument(
        "--latent-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Dtype to cast loaded video latent to. auto uses final_video_latent_mlx_dtype.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["auto"],
        help=(
            "Modes: none, auto, default, spatial64, spatial128, temporal24, temporal32, "
            "both64_24, both384_24, both256_24, both128_24, both128_32, test_small_both"
        ),
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--show-memory", action="store_true", help="Print RSS/MLX memory summaries around decode chunks.")
    parser.add_argument("--decode-audio", action="store_true", help="Decode final_audio_latent, write a WAV sidecar, and mux it into decoded MP4s.")
    parser.add_argument(
        "--full-decode-chunk-size",
        type=int,
        default=9999,
        help="Temporal chunk size for mode=none. 9999 means true full decode for typical tests.",
    )
    args = parser.parse_args(argv)

    import mlx.core as mx

    timings = RunTimings()

    with Timer(timings, "load latent"):
        latent, audio_latent = load_latents(args.latent, mx, args.latent_dtype)
    compute_dtype = parse_dtype(mx, args.vae_dtype)
    print(f"VAE compute dtype: {args.vae_dtype}")
    with Timer(timings, "load vae decoder"):
        decoder = make_decoder(args.weights, compute_dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_wav_path = None
    audio_sample_rate = None
    if args.decode_audio:
        if audio_latent is None:
            raise ValueError("--decode-audio was requested, but this NPZ does not contain final_audio_latent")
        with Timer(timings, "load audio decoder"):
            audio_decoder, vocoder, audio_sample_rate = make_audio_decoder_and_vocoder(args.weights, compute_dtype)
        with Timer(timings, "audio decode"):
            audio_waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx)
        audio_wav_path = output_dir / "decode_audio.wav"
        with Timer(timings, "audio wav write"):
            write_wav(audio_waveform, audio_wav_path, audio_sample_rate)
        del audio_waveform, audio_decoder, vocoder
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    for mode in args.modes:
        decode_mode(
            mode=mode,
            latent=latent,
            decoder=decoder,
            output_dir=output_dir,
            fps=args.fps,
            mx_mod=mx,
            keep_frames=args.keep_frames,
            full_decode_chunk_size=args.full_decode_chunk_size,
            show_memory=args.show_memory,
            timings=timings,
            audio_wav_path=audio_wav_path,
            audio_sample_rate=audio_sample_rate,
        )

    timings.print_summary()


if __name__ == "__main__":
    main()
