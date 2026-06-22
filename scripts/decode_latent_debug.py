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
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to import path when run as scripts/decode_latent_debug.py.
sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.sidecars import load_sidecar  # noqa: E402
from LTX_2_MLX.videotoolbox.audio import write_wav_int16  # noqa: E402
from LTX_2_MLX.videotoolbox.images import save_image  # noqa: E402


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


def load_latents(
    path: str,
    mx_mod: Any,
    latent_dtype: str = "auto",
    stage: str = "final",
):
    """Load video (and optionally audio) latent from a --save-latents sidecar.

    Args:
        path: NPZ sidecar path.
        mx_mod: imported mlx.core module.
        latent_dtype: cast target ("auto" reads the stored mlx_dtype; or one of
            "bfloat16"/"float16"/"float32"/"none").
        stage: which latent to grab. "final" (default) reads ``final_video_latent``
            (= stage 2 on distilled two-stage sidecars). "stage1" grabs the
            pre-upscaler latent (half-resolution); "stage2" explicitly grabs the
            post-refinement latent (same content as "final" on distilled
            two-stage, but errors out on non-two-stage sidecars).
    """
    if stage == "final":
        video_key, dtype_key = "final_video_latent", "final_video_latent_mlx_dtype"
        audio_key, audio_dtype_key = "final_audio_latent", "final_audio_latent_mlx_dtype"
        missing_hint = "not a --save-latents sidecar"
    elif stage == "stage1":
        video_key, dtype_key = "stage_1_video_latent", "stage_1_video_latent_mlx_dtype"
        audio_key, audio_dtype_key = "stage_1_audio_latent", "stage_1_audio_latent_mlx_dtype"
        missing_hint = "not a distilled-two-stage sidecar (need --save-latents on a distilled two-stage run)"
    elif stage == "stage2":
        video_key, dtype_key = "stage_2_video_latent", "stage_2_video_latent_mlx_dtype"
        audio_key, audio_dtype_key = "stage_2_audio_latent", "stage_2_audio_latent_mlx_dtype"
        missing_hint = "not a distilled-two-stage sidecar"
    else:
        raise ValueError(f"unknown stage {stage!r}; expected 'final', 'stage1', or 'stage2'")

    arrays, metadata = load_sidecar(path)
    keys = sorted([*arrays.keys(), *metadata.keys()])
    if video_key not in arrays:
        raise KeyError(
            f"{path} does not contain {video_key} (stage={stage!r}, {missing_hint}); keys={keys}"
        )
    # The *_mlx_dtype tag exists only in the legacy npz sidecar (numpy has no
    # bf16, so the writer widens to float32 and records the original dtype here).
    # Safetensors preserves the dtype natively, so the tag is absent and the
    # loaded tensor already carries the correct dtype -- no re-cast needed.
    latent_array = arrays[video_key]
    stored_dtype = metadata.get(dtype_key)
    has_audio = audio_key in arrays
    audio_array = arrays[audio_key] if has_audio else None
    audio_shape = audio_array.shape if audio_array is not None else None
    audio_dtype = metadata.get(audio_dtype_key)

    print(f"NPZ keys: {', '.join(keys)}")
    print(f"Stage:    {stage} (video_key={video_key!r}, audio_key={audio_key!r})")
    print(f"Loaded video latent: shape={latent_array.shape}, mlx_dtype={stored_dtype or 'unknown'}")
    if has_audio:
        print(f"Found audio latent (key={audio_key!r}): shape={audio_shape}, mlx_dtype={audio_dtype or 'unknown'}")
    elif audio_key in keys:
        # Defensive: shouldn't happen because we checked `audio_key in arrays`, but
        # if a future code path strips the key after the load, surface it.
        print(f"Audio key {audio_key!r} present but not loaded - check load_latents logic.")
    else:
        # Explicitly state we looked for the matching-stage audio and found none.
        print(f"No audio latent at key {audio_key!r} - video-only output.")

    resolved = parse_dtype(mx_mod, latent_dtype)
    if resolved == "auto":
        resolved = parse_dtype(mx_mod, stored_dtype) if stored_dtype else None

    latent = latent_array
    if resolved is not None:
        latent = latent.astype(resolved)
    mx_mod.eval(latent)
    print(f"Decode latent dtype: {latent.dtype}")

    audio_latent = None
    if audio_array is not None:
        # npz widens bf16 -> float32 and tags the dtype, so re-cast when tagged;
        # safetensors keeps the native dtype (no tag, no re-cast).
        if audio_dtype is not None:
            audio_array = audio_array.astype(parse_dtype(mx_mod, audio_dtype))
        audio_latent = audio_array
        mx_mod.eval(audio_latent)
        print(f"Decode audio latent dtype: {audio_latent.dtype}")

    return latent, audio_latent


def make_decoder(
    weights_path: str,
    compute_dtype: Any,
    spatial_padding_mode: str = "conv",
    config_weights_path: str | None = None,
):
    from LTX_2_MLX.generate import get_vae_config
    from LTX_2_MLX.model.video_vae.native_decoder import (
        NativeConv3dVideoDecoder,
        load_native_vae_decoder_weights,
    )

    config_candidates = [config_weights_path, weights_path]
    vae_cfg = {}
    config_source = None
    for candidate in config_candidates:
        if not candidate:
            continue
        candidate_cfg = get_vae_config(candidate)
        if candidate_cfg.get("decoder_blocks"):
            vae_cfg = candidate_cfg
            config_source = candidate
            break
        if not vae_cfg:
            vae_cfg = candidate_cfg
            config_source = candidate

    decoder_blocks = vae_cfg.get("decoder_blocks")
    base_channels = vae_cfg.get("decoder_base_channels", vae_cfg.get("base_channels", 128))
    timestep_conditioning = vae_cfg.get("timestep_conditioning", False)

    print(
        f"VAE config: blocks={len(decoder_blocks) if decoder_blocks else 'default'}, "
        f"base_ch={base_channels}, timestep={timestep_conditioning}"
    )
    if config_source:
        print(f"VAE config source: {config_source}")

    decoder = NativeConv3dVideoDecoder(
        decoder_blocks=decoder_blocks,
        base_channels=base_channels,
        timestep_conditioning=timestep_conditioning,
        compute_dtype=compute_dtype,
        spatial_padding_mode=spatial_padding_mode,
    )
    load_native_vae_decoder_weights(decoder, weights_path)
    gc.collect()
    return decoder


def compare_tensors(label: str, lhs: Any, rhs: Any, mx_mod: Any) -> tuple[float, float, float, float]:
    lhs = lhs.astype(mx_mod.float32)
    rhs = rhs.astype(mx_mod.float32)
    diff = lhs - rhs
    abs_diff = mx_mod.abs(diff)
    dot = mx_mod.sum(lhs * rhs)
    denom = mx_mod.sqrt(mx_mod.sum(lhs * lhs) * mx_mod.sum(rhs * rhs))
    max_abs = mx_mod.max(abs_diff)
    mean_abs = mx_mod.mean(abs_diff)
    rms = mx_mod.sqrt(mx_mod.mean(diff * diff))
    cos = dot / denom
    mx_mod.eval(max_abs, mean_abs, rms, cos)
    result = (
        float(max_abs.item()),
        float(mean_abs.item()),
        float(rms.item()),
        float(cos.item()),
    )
    print(
        f"  {label:<24} max_abs={result[0]:.8f} mean_abs={result[1]:.8f} "
        f"rms={result[2]:.8f} cos={result[3]:.10f}"
    )
    return result


def compare_spatial_padding_modes(
    *,
    mode: str,
    latent: Any,
    manual_decoder: Any,
    conv_decoder: Any,
    mx_mod: Any,
    full_decode_chunk_size: int,
    timings: RunTimings,
) -> None:
    from LTX_2_MLX.model.video_vae.decode_utils import decode_latent
    from LTX_2_MLX.model.video_vae.tiling import decode_tiled

    print("\n" + "=" * 80)
    print(f"Spatial padding parity mode: {mode}")

    cfg = tiling_config_for_mode(mode, latent)
    worst_max = 0.0
    worst_mean = 0.0
    worst_rms = 0.0
    worst_cos = 1.0
    chunks = 0
    started = time.perf_counter()

    if cfg is None:
        lhs = decode_latent(
            latent,
            manual_decoder,
            temporal_chunk_size=full_decode_chunk_size,
        )
        rhs = decode_latent(
            latent,
            conv_decoder,
            temporal_chunk_size=full_decode_chunk_size,
        )
        mx_mod.eval(lhs, rhs)
        max_abs, mean_abs, rms, cos = compare_tensors("full", lhs, rhs, mx_mod)
        worst_max = max(worst_max, max_abs)
        worst_mean = max(worst_mean, mean_abs)
        worst_rms = max(worst_rms, rms)
        worst_cos = min(worst_cos, cos)
        chunks = 1
        del lhs, rhs
    else:
        lhs_iter = iter(decode_tiled(latent, manual_decoder, cfg, show_progress=False))
        rhs_iter = iter(decode_tiled(latent, conv_decoder, cfg, show_progress=False))
        while True:
            try:
                lhs = next(lhs_iter)
            except StopIteration:
                try:
                    next(rhs_iter)
                except StopIteration:
                    break
                raise RuntimeError(
                    "Conv-padding decode yielded more chunks than manual-padding decode"
                ) from None

            try:
                rhs = next(rhs_iter)
            except StopIteration as exc:
                raise RuntimeError("Manual-padding decode yielded more chunks than conv-padding decode") from exc

            mx_mod.eval(lhs, rhs)
            max_abs, mean_abs, rms, cos = compare_tensors(
                f"chunk {chunks}",
                lhs,
                rhs,
                mx_mod,
            )
            worst_max = max(worst_max, max_abs)
            worst_mean = max(worst_mean, mean_abs)
            worst_rms = max(worst_rms, rms)
            worst_cos = min(worst_cos, cos)
            chunks += 1
            del lhs, rhs
            gc.collect()
            try:
                mx_mod.clear_cache()
            except Exception:
                pass

    timings.add(f"{mode} spatial padding compare", time.perf_counter() - started)
    print(
        f"worst over {chunks} chunk(s): max_abs={worst_max:.8f} "
        f"mean_abs={worst_mean:.8f} rms={worst_rms:.8f} cos={worst_cos:.10f}"
    )


def make_audio_decoder_and_vocoder(weights_path: str, compute_dtype: Any):
    from LTX_2_MLX.generate import create_vocoder_for_checkpoint, print_audio_dtype_summary
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


def decode_audio_latent(
    audio_latent: Any, audio_decoder: Any, vocoder: Any, mx_mod: Any, onset_mode: str = "off"
):
    # Optional sequence-start onset mitigation. Default "off" = faithful/raw
    # decode (for debug/analysis of a saved latent). Output-producing harnesses
    # pass onset_mode="auto" to match AVPipeline._decode_audio. See
    # docs/AUDIO_ISSUES.md.
    onset_fired = False
    if onset_mode != "off":
        from LTX_2_MLX.audio import detect_onset_latent_spike

        onset_fired = onset_mode == "force" or detect_onset_latent_spike(audio_latent)
    mel_spectrogram = audio_decoder(audio_latent)
    mx_mod.eval(mel_spectrogram)
    waveform = vocoder(mel_spectrogram)
    waveform = waveform.astype(mx_mod.float32)
    mx_mod.eval(waveform)
    if onset_fired:
        # The causal-VAE first-frame spike decodes to a transient confined to the
        # leading ~120 ms (verified); zero-fill it (content starts ~175 ms).
        from LTX_2_MLX.audio import DEFAULT_TRIM_MS, trim_onset

        sr = int(getattr(vocoder, "output_sample_rate", 48000))
        waveform = trim_onset(waveform, sr, trim_ms=DEFAULT_TRIM_MS)
        print(f"  audio onset (latent-detected): zero-filled leading {DEFAULT_TRIM_MS:g} ms (sequence-start spike)")
    return waveform


def write_wav(audio_waveform: Any, output_path: Path, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Native int16 PCM WAV via AVFoundation; accepts (B,C,T)/(C,T) float and
    # quantizes to int16 internally.
    write_wav_int16(audio_waveform, output_path, sample_rate)

    # Channel/sample counts for the log come straight off the waveform shape:
    # (B,C,T) drops the batch axis, (C,T) is already channels-first.
    shape = audio_waveform.shape
    num_samples = int(shape[-1])
    print(f"Wrote WAV: {output_path}")
    print(f"Audio: {num_samples} samples, {num_samples / sample_rate:.2f}s at {sample_rate}Hz")


def tiling_config_for_mode(mode: str, latent: Any | None = None):
    from LTX_2_MLX.model.video_vae.tiling import (
        SpatialTilingConfig,
        TemporalTilingConfig,
        TilingConfig,
    )

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
            decoder_backend="native",
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


def frames_from_video_tensor(video: Any, mx_mod: Any) -> Any:
    mx_mod.eval(video)
    arr = video

    if arr.ndim == 5:
        arr = mx_mod.transpose(arr[0], (1, 2, 3, 0))  # B,C,T,H,W -> T,H,W,C
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected decoded video shape: {arr.shape}")

    arr_min = float(mx_mod.min(arr).item())
    arr_max = float(mx_mod.max(arr).item())
    print(
        f"Decoded chunk/array: shape={tuple(arr.shape)}, dtype={arr.dtype}, "
        f"min={arr_min:.4f}, max={arr_max:.4f}"
    )

    if arr.dtype == mx_mod.uint8:
        return arr
    if arr_min >= -2.0 and arr_max <= 2.0:
        arr = (arr + 1.0) * 127.5
    return mx_mod.clip(arr, 0, 255).astype(mx_mod.uint8)


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


def write_frames(frames: Any, frames_dir: Path, start: int = 0) -> int:
    for idx, frame in enumerate(frames):
        save_image(frame, frames_dir / f"frame_{start + idx:05d}.png")
    return start + len(frames)


def decoded_frame_count(video: Any) -> int:
    if len(video.shape) == 5:
        return int(video.shape[2])  # B,C,T,H,W
    if len(video.shape) == 4:
        return int(video.shape[0])  # T,H,W,C
    raise ValueError(f"Unexpected decoded video shape: {video.shape}")


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
    video_output_mode: str,
    output_backend: str,
    encode_tier: str,
    audio_wav_path: Path | None = None,
    audio_waveform: Any | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    from LTX_2_MLX.model.video_vae.decode_utils import decode_latent
    from LTX_2_MLX.model.video_vae.tiling import decode_tiled

    print("\n" + "=" * 80)
    print(f"Decode mode: {mode}")
    if show_memory:
        print("before decode:", mlx_mem_summary(mx_mod))

    output_path = output_dir / f"decode_{mode}.mp4"
    cfg = tiling_config_for_mode(mode, latent)

    if video_output_mode == "production":
        from LTX_2_MLX.generate import encode_video_dispatch
        from LTX_2_MLX.pipelines.streaming import (
            iter_decoded_chunks,
            latent_dims,
            plan_vae_tiling,
        )
        from LTX_2_MLX.progress import StackedPhaseBars

        n_source_frames, height, width = latent_dims(latent)
        n_chunks, tiling_desc = plan_vae_tiling(latent, cfg)
        print(
            "Production output: "
            f"{n_source_frames} frames, {width}x{height}, "
            f"{n_chunks} chunk(s), tiling={tiling_desc}"
        )

        chunk_count = 0
        frame_count = 0
        decode_convert_seconds = 0.0
        first_chunk_seconds = 0.0

        production_started = time.perf_counter()
        with StackedPhaseBars() as stream_bars:
            vae_pbar = stream_bars.add(
                total=n_chunks,
                desc="VAE chunks",
                unit="chunk",
            )

            def frame_iter():
                nonlocal chunk_count, frame_count, decode_convert_seconds, first_chunk_seconds
                chunk_iter = iter(
                    iter_decoded_chunks(
                        latent,
                        decoder,
                        tiling=cfg,
                        output_format="fp16_rgba",
                    )
                )
                while True:
                    started = time.perf_counter()
                    try:
                        chunk_frames = next(chunk_iter)
                    except StopIteration:
                        break
                    chunk_seconds = time.perf_counter() - started
                    decode_convert_seconds += chunk_seconds
                    if chunk_count == 0:
                        first_chunk_seconds = chunk_seconds
                    chunk_count += 1
                    frame_count += len(chunk_frames)
                    vae_pbar.update(1)
                    for i, frame in enumerate(chunk_frames):
                        # Null the slot at hand-off so the decoded frame frees
                        # as consumed; plain iteration would pin the whole chunk.
                        chunk_frames[i] = None
                        yield frame

            saved_path = encode_video_dispatch(
                frame_iter(),
                output_path,
                tier=encode_tier,
                fps=fps,
                audio_waveform=audio_waveform,
                audio_sample_rate=audio_sample_rate,
                output_backend=output_backend,
                n_source_frames=n_source_frames,
                progress_stack=stream_bars,
            )
        production_seconds = time.perf_counter() - production_started
        encoder_wall_seconds = max(0.0, production_seconds - first_chunk_seconds)
        output_tail_seconds = max(0.0, production_seconds - decode_convert_seconds)
        timings.add(f"{mode} vae decode+convert", decode_convert_seconds)
        timings.add(f"{mode} non-vae output tail", output_tail_seconds)
        print("\nStreaming timing split:")
        print(f"  first chunk ready       {format_duration(first_chunk_seconds)}")
        print(f"  vae decode+convert      {format_duration(decode_convert_seconds)}")
        print(f"  encoder wall after c1   {format_duration(encoder_wall_seconds)}")
        print(f"  non-vae output tail     {format_duration(output_tail_seconds)}")
        print(f"Encoded {frame_count} frames from {chunk_count} chunk(s)")
        print(f"Saved: {saved_path}")

        with Timer(timings, f"{mode} cleanup"):
            gc.collect()
            try:
                mx_mod.clear_cache()
            except Exception:
                pass
        if show_memory:
            print("after mode cleanup:", mlx_mem_summary(mx_mod))
        return

    frame_index = 0
    decode_seconds = 0.0
    write_seconds = 0.0

    frame_root = output_dir / f"frames_{mode}"
    if video_output_mode == "frames":
        if frame_root.exists():
            import shutil

            shutil.rmtree(frame_root)
        frame_root.mkdir(parents=True, exist_ok=True)

    if cfg is None:
        started = time.perf_counter()
        video = decode_latent(
            latent,
            decoder,
            temporal_chunk_size=full_decode_chunk_size,
        )
        mx_mod.eval(video)
        decode_seconds += time.perf_counter() - started

        if video_output_mode == "none":
            frame_index = decoded_frame_count(video)
            del video
        else:
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
            if video_output_mode == "none":
                frame_index += decoded_frame_count(chunk)
                del chunk
            else:
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
            if show_memory and video_output_mode == "frames":
                print(f"after writing chunk {chunk_index}:", mlx_mem_summary(mx_mod))
            chunk_index += 1

    timings.add(f"{mode} decode", decode_seconds)
    if video_output_mode == "frames":
        timings.add(f"{mode} write frames", write_seconds)

        print(f"Wrote {frame_index} frames to {frame_root}")
        with Timer(timings, f"{mode} ffmpeg encode"):
            encode_frames_dir(frame_root, output_path, fps, audio_wav_path, audio_sample_rate)
        print(f"Saved: {output_path}")
    else:
        print(f"Decoded {frame_index} frames; no video artifact requested.")

    with Timer(timings, f"{mode} cleanup"):
        if video_output_mode == "frames" and not keep_frames:
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
        help="NPZ sidecar produced by LTX_2_MLX/generate.py --save-latents.",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--config-weights",
        default=None,
        help=(
            "Checkpoint containing VAE architecture metadata. Defaults to --weights; "
            "use the main model safetensors when --weights is a split video_vae cache."
        ),
    )
    parser.add_argument(
        "--audio-weights",
        default=None,
        help=(
            "Checkpoint containing audio VAE/vocoder weights. Defaults to "
            "--config-weights when set, otherwise --weights."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/decode_tests")
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
    parser.add_argument(
        "--spatial-padding-mode",
        choices=["manual", "conv"],
        default="conv",
        help="VAE Conv3d spatial padding implementation for decode output.",
    )
    parser.add_argument(
        "--compare-spatial-padding",
        action="store_true",
        help=(
            "Decode each mode once with manual spatial mx.pad and once with "
            "Conv3d spatial padding, then compare float outputs and exit."
        ),
    )
    parser.add_argument(
        "--video-output-mode",
        choices=["frames", "production", "none"],
        default="frames",
        help=(
            "frames: write PNG frames and ffmpeg/libx264 MP4; "
            "production: stream decoded chunks through generate.py's "
            "VideoToolbox/ffmpeg dispatch; none: decode/eval only."
        ),
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Alias for --video-output-mode none.",
    )
    parser.add_argument(
        "--output-backend",
        choices=["auto", "ffmpeg", "videotoolbox"],
        default="auto",
        help="Production output backend used with --video-output-mode production.",
    )
    parser.add_argument(
        "--encode-tier",
        default="default",
        help="Production encode tier used with --video-output-mode production.",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--show-memory", action="store_true", help="Print RSS/MLX memory summaries around decode chunks.")
    parser.add_argument("--decode-audio", action="store_true", help="Decode final_audio_latent and mux it into decoded MP4s.")
    parser.add_argument(
        "--full-decode-chunk-size",
        type=int,
        default=9999,
        help="Temporal chunk size for mode=none. 9999 means true full decode for typical tests.",
    )
    args = parser.parse_args(argv)
    if args.decode_only:
        args.video_output_mode = "none"

    import mlx.core as mx

    timings = RunTimings()

    with Timer(timings, "load latent"):
        latent, audio_latent = load_latents(args.latent, mx, args.latent_dtype)
    compute_dtype = parse_dtype(mx, args.vae_dtype)
    print(f"VAE compute dtype: {args.vae_dtype}")

    if args.compare_spatial_padding:
        with Timer(timings, "load manual-pad vae decoder"):
            manual_decoder = make_decoder(
                args.weights,
                compute_dtype,
                spatial_padding_mode="manual",
                config_weights_path=args.config_weights,
            )
        with Timer(timings, "load conv-pad vae decoder"):
            conv_decoder = make_decoder(
                args.weights,
                compute_dtype,
                spatial_padding_mode="conv",
                config_weights_path=args.config_weights,
            )
        for mode in args.modes:
            compare_spatial_padding_modes(
                mode=mode,
                latent=latent,
                manual_decoder=manual_decoder,
                conv_decoder=conv_decoder,
                mx_mod=mx,
                full_decode_chunk_size=args.full_decode_chunk_size,
                timings=timings,
            )
        timings.print_summary()
        return

    print(f"VAE spatial padding mode: {args.spatial_padding_mode}")
    with Timer(timings, "load vae decoder"):
        decoder = make_decoder(
            args.weights,
            compute_dtype,
            spatial_padding_mode=args.spatial_padding_mode,
            config_weights_path=args.config_weights,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_waveform = None
    audio_wav_path = None
    audio_sample_rate = None
    if args.decode_audio:
        if audio_latent is None:
            raise ValueError("--decode-audio was requested, but this NPZ does not contain final_audio_latent")
        audio_weights = args.audio_weights or args.config_weights or args.weights
        with Timer(timings, "load audio decoder"):
            audio_decoder, vocoder, audio_sample_rate = make_audio_decoder_and_vocoder(audio_weights, compute_dtype)
        with Timer(timings, "audio decode"):
            audio_waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx)
        if args.video_output_mode == "frames":
            audio_wav_path = output_dir / "decode_audio.wav"
            with Timer(timings, "audio wav write"):
                write_wav(audio_waveform, audio_wav_path, audio_sample_rate)
        del audio_decoder, vocoder
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
            video_output_mode=args.video_output_mode,
            output_backend=args.output_backend,
            encode_tier=args.encode_tier,
            audio_wav_path=audio_wav_path,
            audio_waveform=audio_waveform,
            audio_sample_rate=audio_sample_rate,
        )

    timings.print_summary()


if __name__ == "__main__":
    main()
