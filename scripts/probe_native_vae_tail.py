#!/usr/bin/env python3
"""Probe where native Conv3d VAE full-volume decode corrupts tail frames."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LTX_2_MLX.model.video_vae.native_decoder import (
    NativeConv3dVideoDecoder,
    _pixel_norm_bfhwc,
    _unpatchify_spatial_bfhwc,
    load_native_vae_decoder_weights,
)
from LTX_2_MLX.loader import ensure_weight_family_caches
from scripts.generate import get_vae_config, parse_compute_dtype


# Default weights path is taken from $LTX_DEFAULT_WEIGHTS_PATH (a personal
# convenience for repeated invocations).  When unset, --weights is required.
DEFAULT_WEIGHTS = os.environ.get("LTX_DEFAULT_WEIGHTS_PATH") or None
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


def _as_float_list(values: mx.array) -> list[float]:
    return [float(v) for v in np.array(values.astype(mx.float32))]


def _scalar(value: mx.array) -> float:
    return float(np.array(value.astype(mx.float32)).item())


def _reduce_tail_bfhwc(
    stage: str,
    x: mx.array,
    tail_frames: int,
    final_rgb: bool = False,
) -> dict:
    """Record compact tail stats for a BFHWC tensor."""
    mx.eval(x)
    time_dim = int(x.shape[1])
    tail_count = max(1, min(int(tail_frames), time_dim))
    tail = x[:, -tail_count:, :, :, :].astype(mx.float32)

    tail_by_frame = tail.transpose(1, 0, 2, 3, 4).reshape(tail_count, -1)
    frame_mean = mx.mean(tail_by_frame, axis=1)
    frame_abs_max = mx.max(mx.abs(tail_by_frame), axis=1)
    frame_min = mx.min(tail_by_frame, axis=1)
    frame_max = mx.max(tail_by_frame, axis=1)
    mx.eval(frame_mean, frame_abs_max, frame_min, frame_max)

    stats = {
        "stage": stage,
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype),
        "tail_frames_checked": int(tail_count),
        "tail_frame_start": int(time_dim - tail_count),
        "tail_mean": _scalar(mx.mean(tail_by_frame)),
        "tail_std": _scalar(mx.std(tail_by_frame)),
        "tail_min": _scalar(mx.min(tail_by_frame)),
        "tail_max": _scalar(mx.max(tail_by_frame)),
        "tail_abs_max": _scalar(mx.max(mx.abs(tail_by_frame))),
        "tail_frac_ge_1": _scalar(mx.mean((tail_by_frame >= 1.0).astype(mx.float32))),
        "tail_frac_le_neg_1": _scalar(mx.mean((tail_by_frame <= -1.0).astype(mx.float32))),
        "frame_mean": _as_float_list(frame_mean),
        "frame_min": _as_float_list(frame_min),
        "frame_max": _as_float_list(frame_max),
        "frame_abs_max": _as_float_list(frame_abs_max),
        "memory": _memory_snapshot(),
    }

    if final_rgb:
        rgb01 = mx.clip((tail + 1.0) / 2.0, 0.0, 1.0)
        rgb_by_frame = rgb01.transpose(1, 0, 2, 3, 4).reshape(tail_count, -1)
        frame_uint8_mean = mx.mean(rgb_by_frame, axis=1) * 255.0
        frame_bright_frac = mx.mean((rgb_by_frame > (245.0 / 255.0)).astype(mx.float32), axis=1)
        mx.eval(frame_uint8_mean, frame_bright_frac)
        bright = np.where(np.array(frame_bright_frac) > 0.95)[0]
        first_bright_frame = None
        if bright.size:
            first_bright_frame = int(time_dim - tail_count + bright[0])
        stats["final_uint8"] = {
            "frame_mean": _as_float_list(frame_uint8_mean),
            "frame_bright_frac": _as_float_list(frame_bright_frac),
            "max_frame_mean": _scalar(mx.max(frame_uint8_mean)),
            "max_bright_frac": _scalar(mx.max(frame_bright_frac)),
            "first_bright_frame": first_bright_frame,
        }

    return stats


def _print_stage(stats: dict) -> None:
    final = stats.get("final_uint8")
    extra = ""
    if final is not None:
        extra = (
            f" | final max mean={final['max_frame_mean']:.1f} "
            f"bright={final['max_bright_frac']:.3f} "
            f"first={final['first_bright_frame']}"
        )
    print(
        f"{stats['stage']:24s} shape={tuple(stats['shape'])} "
        f"tail_max={stats['tail_max']:.3f} abs={stats['tail_abs_max']:.3f} "
        f"ge1={stats['tail_frac_ge_1']:.3f}{extra}"
    )


def _range_stats_bfhwc(x: mx.array, start: int, end: int) -> dict:
    """Compact per-frame stats for a BFHWC frame range."""
    span = max(1, int(end) - int(start))
    sliced = x[:, start:end, :, :, :].astype(mx.float32)
    by_frame = sliced.transpose(1, 0, 2, 3, 4).reshape(span, -1)
    frame_mean = mx.mean(by_frame, axis=1)
    frame_min = mx.min(by_frame, axis=1)
    frame_max = mx.max(by_frame, axis=1)
    frame_abs_max = mx.max(mx.abs(by_frame), axis=1)
    mx.eval(frame_mean, frame_min, frame_max, frame_abs_max)
    return {
        "frame_mean": _as_float_list(frame_mean),
        "frame_min": _as_float_list(frame_min),
        "frame_max": _as_float_list(frame_max),
        "frame_abs_max": _as_float_list(frame_abs_max),
    }


def _compare_conv_window(
    *,
    stage: str,
    block,
    x: mx.array,
    full_conv: mx.array,
    causal: bool,
    tail_frames: int,
    window_start: int | None,
    window_end: int | None,
    compare_start: int | None,
    compare_end: int | None,
) -> dict:
    """Compare a full-volume Conv3d result with the same conv on a tail window."""
    time_dim = int(x.shape[1])
    default_window_span = max(int(tail_frames) + 16, 64)
    ws = max(0, time_dim - default_window_span) if window_start is None else int(window_start)
    we = time_dim if window_end is None else int(window_end)
    ws = max(0, min(ws, time_dim - 1))
    we = max(ws + 1, min(we, time_dim))

    # Skip the first frame of a non-causal window by default so symmetric
    # temporal padding has real left context.  Causal padding needs two frames
    # of left context for the default 3-frame kernel.
    context = 2 if causal else 1
    cs_default = max(ws + context, time_dim - int(tail_frames))
    cs = cs_default if compare_start is None else int(compare_start)
    ce = time_dim if compare_end is None else int(compare_end)
    cs = max(ws, min(cs, we - 1))
    ce = max(cs + 1, min(ce, we))

    window_x = x[:, ws:we, :, :, :]
    window_conv = block.conv(window_x, causal=causal)
    mx.eval(window_conv)

    full_slice = full_conv[:, cs:ce, :, :, :].astype(mx.float32)
    window_slice = window_conv[:, cs - ws : ce - ws, :, :, :].astype(mx.float32)
    absdiff = mx.abs(full_slice - window_slice)

    span = ce - cs
    diff_by_frame = absdiff.transpose(1, 0, 2, 3, 4).reshape(span, -1)
    diff_frame_mean = mx.mean(diff_by_frame, axis=1)
    diff_frame_max = mx.max(diff_by_frame, axis=1)
    mx.eval(diff_frame_mean, diff_frame_max)

    result = {
        "stage": stage,
        "window_start": int(ws),
        "window_end": int(we),
        "compare_start": int(cs),
        "compare_end": int(ce),
        "full_shape": [int(v) for v in full_conv.shape],
        "window_shape": [int(v) for v in window_conv.shape],
        "full": _range_stats_bfhwc(full_conv, cs, ce),
        "window": _range_stats_bfhwc(window_conv, cs - ws, ce - ws),
        "absdiff_mean": _scalar(mx.mean(absdiff)),
        "absdiff_max": _scalar(mx.max(absdiff)),
        "absdiff_frame_mean": _as_float_list(diff_frame_mean),
        "absdiff_frame_max": _as_float_list(diff_frame_max),
        "memory": _memory_snapshot(),
    }
    print(
        f"{stage:24s} window={ws}:{we} compare={cs}:{ce} "
        f"diff_mean={result['absdiff_mean']:.6f} "
        f"diff_max={result['absdiff_max']:.6f}"
    )

    del window_x, window_conv, full_slice, window_slice, absdiff
    return result


def _run_upsample_with_optional_probe(
    *,
    idx: int,
    block,
    x: mx.array,
    causal: bool,
    stages: list[dict],
    tail_frames: int,
    probe_internals: bool,
    window_probe_index: int | None,
    window_start: int | None,
    window_end: int | None,
    window_compare_start: int | None,
    window_compare_end: int | None,
) -> mx.array:
    if not probe_internals:
        x = block(x, causal=causal)
        stages.append(_reduce_tail_bfhwc(f"{idx:02d}_upsample", x, tail_frames))
        _print_stage(stages[-1])
        return x

    ft, _fh, _fw = block.stride
    stride_product = ft * block.stride[1] * block.stride[2]
    residual = None
    if block.residual:
        residual_channels = x.shape[-1] // stride_product
        residual = block._depth_to_space(x, residual_channels)
        if ft > 1:
            residual = residual[:, 1:, :, :, :]
        num_repeat = stride_product // block.out_channels_reduction_factor
        residual = mx.tile(residual, (1, 1, 1, 1, num_repeat))
        stages.append(_reduce_tail_bfhwc(f"{idx:02d}_upsample_residual", residual, tail_frames))
        _print_stage(stages[-1])

    conv = block.conv(x, causal=causal)
    stages.append(_reduce_tail_bfhwc(f"{idx:02d}_upsample_conv", conv, tail_frames))
    _print_stage(stages[-1])
    if window_probe_index == idx:
        stages.append(
            _compare_conv_window(
                stage=f"{idx:02d}_upsample_conv_window",
                block=block,
                x=x,
                full_conv=conv,
                causal=causal,
                tail_frames=tail_frames,
                window_start=window_start,
                window_end=window_end,
                compare_start=window_compare_start,
                compare_end=window_compare_end,
            )
        )

    x = block._depth_to_space(conv, block.final_out_channels)
    if ft > 1:
        x = x[:, 1:, :, :, :]
    stages.append(_reduce_tail_bfhwc(f"{idx:02d}_upsample_d2s", x, tail_frames))
    _print_stage(stages[-1])

    if residual is not None:
        x = x + residual
        stages.append(_reduce_tail_bfhwc(f"{idx:02d}_upsample", x, tail_frames))
        _print_stage(stages[-1])
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", required=True, type=Path)
    parser.add_argument("--latent-key", default="final_video_latent")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        required=DEFAULT_WEIGHTS is None,
        type=Path,
        help="LTX weights checkpoint. Required unless $LTX_DEFAULT_WEIGHTS_PATH is set.",
    )
    parser.add_argument(
        "--vae-weights",
        "--component-weights",
        dest="vae_weights",
        default=None,
        type=Path,
        help="Video VAE weights. Defaults to a split video VAE cache derived from --weights.",
    )
    parser.add_argument("--weights-cache", choices=["off", "auto", "rebuild"], default="auto")
    parser.add_argument("--weights-cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--native-causal", action="store_true")
    parser.add_argument(
        "--probe-upsample-internals",
        action="store_true",
        help="Materialize each upsample block's residual, Conv3d output, and depth-to-space output separately.",
    )
    parser.add_argument(
        "--probe-upsample-window-index",
        type=int,
        default=None,
        help=(
            "Diagnostic: for this upsample block index, compare the full Conv3d "
            "output against running the same Conv3d on a tail temporal window."
        ),
    )
    parser.add_argument("--probe-window-start-frame", type=int, default=None)
    parser.add_argument("--probe-window-end-frame", type=int, default=None)
    parser.add_argument("--probe-window-compare-start-frame", type=int, default=None)
    parser.add_argument("--probe-window-compare-end-frame", type=int, default=None)
    parser.add_argument("--tail-frames", type=int, default=48)
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mlx_cache_limit_gb is not None:
        mx.set_cache_limit(int(args.mlx_cache_limit_gb * 1024**3))

    dtype = parse_compute_dtype(args.dtype)
    latent = _load_latent(args.latent, args.latent_key, dtype)
    config = get_vae_config(str(args.weights))
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

    decoder = NativeConv3dVideoDecoder(
        decoder_blocks=config.get("decoder_blocks"),
        base_channels=config.get("decoder_base_channels", 128),
        timestep_conditioning=config.get("timestep_conditioning", False),
        compute_dtype=dtype,
        causal=args.native_causal,
    )
    load_native_vae_decoder_weights(decoder, str(vae_weights))

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    stages: list[dict] = []
    probe_upsample_internals = args.probe_upsample_internals or args.probe_upsample_window_index is not None

    print("Native Conv3d VAE tail probe")
    print(f"  latent: {args.latent}")
    print(f"  VAE weights: {vae_weights}")
    print(f"  shape: {tuple(int(v) for v in latent.shape)}")
    print(f"  causal: {args.native_causal}")
    print(f"  upsample internals: {probe_upsample_internals}")
    if args.probe_upsample_window_index is not None:
        print(f"  upsample window probe: block {args.probe_upsample_window_index}")
    print(f"  tail frames: {args.tail_frames}")

    x = latent
    if x.ndim == 4:
        x = x[None]
    if x.dtype != decoder.compute_dtype:
        x = x.astype(decoder.compute_dtype)

    x = x.transpose(0, 2, 3, 4, 1)
    x = x * decoder.std_of_means.reshape(1, 1, 1, 1, -1)
    x = x + decoder.mean_of_means.reshape(1, 1, 1, 1, -1)
    if x.dtype != decoder.compute_dtype:
        x = x.astype(decoder.compute_dtype)
    stages.append(_reduce_tail_bfhwc("denorm", x, args.tail_frames))
    _print_stage(stages[-1])

    causal = decoder.causal
    x = decoder.conv_in(x, causal=causal)
    stages.append(_reduce_tail_bfhwc("conv_in", x, args.tail_frames))
    _print_stage(stages[-1])

    for idx, (block, btype) in enumerate(zip(decoder.up_blocks, decoder.block_types)):
        if btype == "upsample":
            x = _run_upsample_with_optional_probe(
                idx=idx,
                block=block,
                x=x,
                causal=causal,
                stages=stages,
                tail_frames=args.tail_frames,
                probe_internals=probe_upsample_internals,
                window_probe_index=args.probe_upsample_window_index,
                window_start=args.probe_window_start_frame,
                window_end=args.probe_window_end_frame,
                window_compare_start=args.probe_window_compare_start_frame,
                window_compare_end=args.probe_window_compare_end_frame,
            )
        else:
            x = block(x, causal=causal)
            stage = f"{idx:02d}_{btype}"
            stages.append(_reduce_tail_bfhwc(stage, x, args.tail_frames))
            _print_stage(stages[-1])

    x = decoder.conv_out(nn.silu(_pixel_norm_bfhwc(x)), causal=causal)
    stages.append(_reduce_tail_bfhwc("conv_out", x, args.tail_frames))
    _print_stage(stages[-1])

    x = _unpatchify_spatial_bfhwc(x, patch_size=4)
    stages.append(_reduce_tail_bfhwc("unpatchify", x, args.tail_frames, final_rgb=True))
    _print_stage(stages[-1])

    elapsed = time.perf_counter() - t0
    result = {
        "latent": str(args.latent),
        "weights": str(args.weights),
        "vae_weights": str(vae_weights),
        "dtype": args.dtype,
        "native_causal": args.native_causal,
        "probe_upsample_internals": probe_upsample_internals,
        "probe_upsample_window_index": args.probe_upsample_window_index,
        "tail_frames": args.tail_frames,
        "elapsed_s": elapsed,
        "memory": _memory_snapshot(),
        "stages": stages,
    }

    output_json = args.output_json
    if output_json is None:
        suffix = "causal" if args.native_causal else "noncausal"
        output_json = args.latent.with_name(
            f"{args.latent.stem}_native_vae_tail_probe_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {output_json}")

    del x, decoder, latent
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
