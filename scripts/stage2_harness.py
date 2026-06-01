#!/usr/bin/env python3
"""Run distilled stage 2 from saved stage-1 latents.

This is a focused A/B harness for the expensive full-resolution distilled
refinement path. It loads a `_text.npz` conditioning sidecar plus the
`stage_1_*_latent` arrays from a two-stage latent sidecar, then runs:

    spatial upscaler -> stage-2 denoise -> VAE/audio decode -> MP4 export

By default it burns the stage-1 RNG draws before stage-2 noising so a same-seed
run can match the stage-2 noise stream from a full two-stage generation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import generate as gen  # noqa: E402

from LTX_2_MLX.progress import StackedPhaseBars  # noqa: E402
from LTX_2_MLX.pipelines.common import audio_modality_from_state, modality_from_state  # noqa: E402


def adjacent_run_log_path(latents_path: str) -> str:
    stem = os.path.splitext(latents_path)[0]
    return stem + "_run.json"


def load_adjacent_run_params(latents_path: str) -> dict:
    path = adjacent_run_log_path(latents_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    params = data.get("parameters", {})
    return params if isinstance(params, dict) else {}


def load_stage1_latents(path: str) -> tuple[mx.array, mx.array | None, dict]:
    data = np.load(path)
    if "stage_1_video_latent" not in data.files:
        raise ValueError(
            f"{path} does not contain stage_1_video_latent. "
            "Use a distilled two-stage latent sidecar saved with --save-latents."
        )

    video = gen._load_mlx_npz_array(
        data,
        "stage_1_video_latent",
        "stage_1_video_latent_mlx_dtype",
    )
    audio = None
    if "stage_1_audio_latent" in data.files:
        audio = gen._load_mlx_npz_array(
            data,
            "stage_1_audio_latent",
            "stage_1_audio_latent_mlx_dtype",
        )

    metadata = {
        "pipeline": gen._npz_scalar_str(data, "pipeline"),
        "video_shape": tuple(video.shape),
        "audio_shape": tuple(audio.shape) if audio is not None else None,
    }
    return video, audio, metadata


def infer_dimensions_from_stage1(video_latent: mx.array) -> tuple[int, int, int]:
    if video_latent.ndim != 5:
        raise ValueError(
            "stage_1_video_latent must be 5D (B,C,F,H,W), "
            f"got shape {video_latent.shape}"
        )
    frames = (video_latent.shape[2] - 1) * 8 + 1
    height = video_latent.shape[3] * 64
    width = video_latent.shape[4] * 64
    return height, width, frames


def video_tensor_to_frames(video: mx.array) -> list[np.ndarray]:
    video_np = np.array(video)
    print(f"  Raw video shape: {video_np.shape}, dtype: {video_np.dtype}")
    video_np = np.squeeze(video_np)
    if video_np.ndim == 4 and video_np.shape[0] == 3:
        video_np = np.transpose(video_np, (1, 2, 3, 0))
    if video_np.dtype != np.uint8:
        video_np = np.clip((video_np + 1) / 2 * 255.0, 0, 255).astype(np.uint8)
    frames = [video_np[t] for t in range(video_np.shape[0])]
    print(f"  Generated {len(frames)} frames at {frames[0].shape[:2]}")
    return frames


def parse_spatial_pool(value: str) -> tuple[int, int] | None:
    lowered = value.strip().lower()
    if lowered in {"", "0", "off", "none", "false", "no"}:
        return None
    if "x" in lowered:
        left, right = lowered.split("x", 1)
        pool = (int(left), int(right))
    else:
        n = int(lowered)
        pool = (n, n)
    if pool[0] < 1 or pool[1] < 1:
        raise argparse.ArgumentTypeError("pool dimensions must be positive")
    return pool


def print_kv_downsample_summary(attention_module, *, stream=None) -> None:
    if attention_module is None:
        return
    summary = attention_module.kv_downsample_summary()
    if not summary:
        return
    stream = stream or sys.stdout
    config = summary["config"]
    counts = summary["counts"]
    reasons = summary["fallback_reasons"]
    total = counts.get("applied", 0) + counts.get("fallback", 0)
    print("", file=stream)
    print("[stage2-video-attn-kv-pool] summary:", file=stream)
    print(
        "  pool: "
        f"{config['pool_h']}x{config['pool_w']}  "
        f"mode={config.get('mode', 'mean')}  "
        f"grid={config['frames']}x{config['height']}x{config['width']}  "
        f"tokens={config['tokens']}",
        file=stream,
    )
    if config.get("max_applied") is not None:
        print(f"  budget: {config['max_applied']} applied calls", file=stream)
    print(
        f"  applied: {counts.get('applied', 0)} / {total} no-mask attention calls",
        file=stream,
    )
    if reasons:
        print("  fallback reasons:", file=stream)
        for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {reason:<14} {count}", file=stream)


def _mx_float(value: mx.array) -> float:
    return float(np.array(value))


def _delta_stats(a: mx.array, b: mx.array) -> dict:
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    diff = a32 - b32
    abs_diff = mx.abs(diff)
    dot = mx.sum(a32 * b32)
    norm_a = mx.sqrt(mx.sum(a32 * a32))
    norm_b = mx.sqrt(mx.sum(b32 * b32))
    denom = norm_a * norm_b
    max_abs = mx.max(abs_diff)
    mean_abs = mx.mean(abs_diff)
    rms_abs = mx.sqrt(mx.mean(diff * diff))
    mx.eval(max_abs, mean_abs, rms_abs, dot, denom)
    denom_f = _mx_float(denom)
    cos = _mx_float(dot) / denom_f if denom_f else float("nan")
    if not np.isnan(cos):
        cos = max(-1.0, min(1.0, cos))
    return {
        "official_shape": tuple(a.shape),
        "legacy_shape": tuple(b.shape),
        "max_abs": _mx_float(max_abs),
        "mean_abs": _mx_float(mean_abs),
        "rms_abs": _mx_float(rms_abs),
        "cos": cos,
    }


def _cross_sigma(modality):
    sigma = modality.sigma if modality.sigma is not None else modality.timesteps
    if sigma.ndim > 1:
        sigma = sigma[:, 0]
    return sigma


def _cross_scale_shift_pair(preprocessor, modality, cross_modality) -> tuple[mx.array, mx.array]:
    batch_size = modality.latent.shape[0]
    cross_sigma = _cross_sigma(cross_modality)
    official, _ = preprocessor._prepare_cross_attention_timestep(
        modality.timesteps,
        cross_sigma,
        batch_size,
    )
    legacy, _ = preprocessor._prepare_cross_attention_timestep(
        cross_sigma,
        cross_sigma,
        batch_size,
    )
    return official, legacy


def print_av_cross_timestep_probe(
    model,
    video_state,
    audio_state,
    video_context: mx.array,
    audio_context: mx.array | None,
    sigma: float,
    *,
    stream=None,
) -> None:
    stream = stream or sys.stdout
    if audio_state is None or audio_context is None:
        print("[av-cross-timestep-probe] skipped: audio branch inactive", file=stream)
        return
    video_modality = modality_from_state(video_state, video_context, sigma)
    audio_modality = audio_modality_from_state(audio_state, audio_context, sigma)

    video_preprocessor = getattr(model, "_video_args_preprocessor", None)
    audio_preprocessor = getattr(model, "_audio_args_preprocessor", None)
    if video_preprocessor is None or audio_preprocessor is None:
        print("[av-cross-timestep-probe] skipped: AV preprocessors unavailable", file=stream)
        return

    video_official, video_legacy = _cross_scale_shift_pair(
        video_preprocessor,
        video_modality,
        audio_modality,
    )
    audio_official, audio_legacy = _cross_scale_shift_pair(
        audio_preprocessor,
        audio_modality,
        video_modality,
    )
    video_stats = _delta_stats(video_official, video_legacy)
    audio_stats = _delta_stats(audio_official, audio_legacy)

    print("", file=stream)
    print("[av-cross-timestep-probe] official scale/shift vs legacy scale/shift", file=stream)
    for label, stats in (("video<-audio", video_stats), ("audio<-video", audio_stats)):
        print(
            f"  {label:<13} official_shape={stats['official_shape']} "
            f"legacy_shape={stats['legacy_shape']} "
            f"max_abs={stats['max_abs']:.6g} mean_abs={stats['mean_abs']:.6g} "
            f"rms_abs={stats['rms_abs']:.6g} cos={stats['cos']:.9f}",
            file=stream,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume distilled LTX-2.3 two-stage generation from saved stage-1 latents.",
    )
    parser.add_argument("--stage1-latents", required=True, help="Two-stage latent NPZ sidecar.")
    parser.add_argument("--embedding", required=True, help="Text conditioning `_text.npz` sidecar.")
    parser.add_argument("--output", default=None, help="Output MP4 path.")
    parser.add_argument("--output-dir", default=None, help="Directory for timestamped output.")
    parser.add_argument("--output-prefix", default="ltx_stage2", help="Timestamped output prefix.")
    parser.add_argument("--height", type=int, default=None, help="Final output height.")
    parser.add_argument("--width", type=int, default=None, help="Final output width.")
    parser.add_argument("--fps", type=float, default=None, help="Generation/output FPS.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used by the full run.")
    parser.add_argument(
        "--generate-audio",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode and mux audio if stage-1 audio latents are present.",
    )
    parser.add_argument(
        "--no-internal-audio-branch",
        action="store_false",
        dest="use_internal_audio_branch",
        help="Disable the AV audio branch when running video-only diagnostics.",
    )
    parser.set_defaults(use_internal_audio_branch=True)
    parser.add_argument(
        "--independent-stage2-noise",
        action="store_true",
        help="Do not burn stage-1 RNG draws before adding stage-2 noise.",
    )

    parser.add_argument("--weights", default=None, help="Full LTX-2.3 checkpoint bundle.")
    parser.add_argument("--transformer-weights", default=None, help="Optional transformer override.")
    parser.add_argument("--vae-weights", default=None, help="Optional video VAE override.")
    parser.add_argument("--audio-vae-weights", default=None, help="Optional audio VAE override.")
    parser.add_argument("--vocoder-weights", default=None, help="Optional vocoder override.")
    parser.add_argument("--config-weights", default=None, help="Optional config/metadata override.")
    parser.add_argument(
        "--spatial-upscaler-weights",
        default=None,
        help="Optional spatial upscaler override.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(gen.SUPPORTED_COMPUTE_DTYPES),
        default="bfloat16",
        help="Compute dtype.",
    )
    parser.add_argument(
        "--vae-decoder",
        choices=["native"],
        default="native",
        help="Video VAE decoder backend (MLX-native Conv3d).",
    )
    parser.add_argument(
        "--vae-tiling",
        choices=["auto", "off", "custom"],
        default="auto",
        help="VAE decode tiling policy.",
    )
    parser.add_argument("--tiled-vae", action="store_true", help="Force tiled VAE decode.")
    parser.add_argument("--vae-temporal-tile-frames", type=int, default=None)
    parser.add_argument("--vae-temporal-overlap-frames", type=int, default=24)
    parser.add_argument("--vae-spatial-tile-pixels", type=int, default=None)
    parser.add_argument("--vae-spatial-overlap-pixels", type=int, default=64)

    parser.add_argument(
        "--weights-cache",
        choices=["auto", "off", "rebuild"],
        default="auto",
        help="Converted-weight cache mode.",
    )
    parser.add_argument("--weights-cache-dir", default=None, help="Converted-weight cache root.")
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=1.0)
    parser.add_argument(
        "--stream-transformer",
        action="store_true",
        help="Use the r16/compile/group-4 streaming preset.",
    )
    parser.add_argument("--transformer-block-resident-blocks", type=int, default=0)
    parser.add_argument("--transformer-block-compile", action="store_true")
    parser.add_argument("--transformer-block-compile-group-size", type=int, default=0)

    parser.add_argument(
        "--video-ff-quantize",
        type=gen.parse_video_ff_quantize_specs,
        default=(),
        metavar="TARGET:MODE[,TARGET:MODE]",
        help="Experimental video FF quantization for this stage-2-only run.",
    )
    parser.add_argument(
        "--video-ff-quantize-layers",
        type=gen.parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
    )
    parser.add_argument("--video-ff-quantize-group-size", type=int, default=None)
    parser.add_argument("--video-ff-quantize-bits", type=int, default=None)
    parser.add_argument(
        "--transformer-cache-quantize",
        choices=gen.TRANSFORMER_CACHE_QUANTIZE_MODES,
        default=gen.TRANSFORMER_CACHE_QUANTIZE_OFF,
        help=(
            "Build/load a quantized transformer block cache. "
            "'mxfp8-blocks' mirrors the downloaded MXFP8 block32 policy in "
            "MLX-native QuantizedLinear format; 'mxfp8-blocks-pretranspose' "
            "packs weight.T and uses transpose=False quantized matmul. "
            "Can run resident or with block streaming."
        ),
    )
    parser.add_argument(
        "--video-ff-layout",
        type=gen.parse_video_ff_layout_specs,
        default=gen.DEFAULT_VIDEO_FF_LAYOUT_SPECS,
        metavar="TARGET:LAYOUT[,TARGET:LAYOUT]",
        help="Same-math video FF layout transform; use 'off' to disable.",
    )
    parser.add_argument(
        "--video-ff-layout-layers",
        type=gen.parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
    )
    parser.add_argument(
        "--video-attn-layout",
        type=gen.parse_video_attn_layout_specs,
        default=gen.DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
        metavar="TARGET:LAYOUT[,TARGET:LAYOUT]",
        help="Same-math video attention output layout; use 'off' to disable.",
    )
    parser.add_argument(
        "--video-attn-layout-layers",
        type=gen.parse_transformer_layer_selection,
        default=(),
        metavar="LAYERS",
    )

    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--save-run-log", action="store_true")
    parser.add_argument(
        "--save-all-sidecars",
        action="store_true",
        help="Save stage-2 latent sidecar and run log.",
    )
    parser.add_argument(
        "--av-cross-timestep-mode",
        choices=["official", "legacy"],
        default="official",
        help=(
            "A/V cross-attention timestep semantics. official matches current "
            "Lightricks: own-modality timesteps for scale/shift, cross-modality "
            "sigma for gates. legacy uses cross-modality sigma for both."
        ),
    )
    parser.add_argument(
        "--probe-av-cross-timestep",
        action="store_true",
        help=(
            "At the first stage-2 state, print official-vs-legacy cross-attn "
            "scale/shift embedding deltas, then continue the requested run."
        ),
    )
    parser.add_argument(
        "--rope-precision",
        choices=["metadata", "float64", "fp32"],
        default="metadata",
        help=(
            "RoPE frequency-grid precision for transformer positional embeddings. "
            "metadata uses checkpoint config (production default); fp32 recreates "
            "the old MLX behavior for A/B testing."
        ),
    )
    parser.add_argument(
        "--bench-mode",
        type=int,
        default=0,
        metavar="N",
        help=(
            "A/B bench mode: print per-step wall time to stderr and exit "
            "immediately after stage-2 step N completes (skipping VAE decode, "
            "audio decode, and output save).  N=2 is the standard recipe: "
            "step 1 warms MLX caches + kernels, step 2 is the measured step. "
            "0 = disabled (default), run full pipeline."
        ),
    )
    parser.add_argument(
        "--profile-transformer",
        action="store_true",
        help=(
            "Diagnostic shorthand: profile stage-2 step 1 unless "
            "--profile-transformer-steps is supplied.  Each profiled step "
            "inserts forced eval checkpoints and perturbs timing."
        ),
    )
    parser.add_argument(
        "--profile-transformer-steps",
        type=gen.parse_profile_transformer_steps,
        default=(),
        metavar="STEPS",
        help=(
            "Diagnostic: comma-separated 1-based stage-2 denoise steps to "
            "profile, e.g. '2'.  Each profiled step inserts forced eval "
            "checkpoints and perturbs timing."
        ),
    )
    parser.add_argument(
        "--profile-transformer-blocks",
        type=gen.parse_profile_transformer_blocks,
        default=(),
        metavar="BLOCKS",
        help=(
            "Diagnostic: comma-separated 0-based transformer blocks to profile "
            "in detail within selected --profile-transformer-steps, e.g. "
            "'0,40,47'."
        ),
    )
    parser.add_argument(
        "--stage2-video-attn-kv-pool",
        type=parse_spatial_pool,
        default=None,
        metavar="HxW",
        help=(
            "Failed diagnostic approximation: spatially reduce K/V tokens for "
            "stage-2 D128 video self-attention only.  1x2 measured faster but "
            "produced blur/ghosting; keep this for negative-evidence probes, "
            "not production."
        ),
    )
    parser.add_argument(
        "--stage2-video-attn-kv-pool-mode",
        choices=["mean", "stride"],
        default="mean",
        help=(
            "K/V reduction mode for --stage2-video-attn-kv-pool. mean averages "
            "each spatial window; stride keeps the top-left token from each window. "
            "Both are diagnostic-only."
        ),
    )
    parser.add_argument(
        "--stage2-video-attn-kv-pool-steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Apply K/V pooling only to the first N stage-2 steps.  Leave unset "
            "to pool all stage-2 steps.  First-2-step 1x2 mean still produced "
            "visible smear/ghosting in local A/B."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_all_sidecars:
        args.save_latents = True
        args.save_run_log = True

    stage1_video, stage1_audio, latent_metadata = load_stage1_latents(args.stage1_latents)
    inferred_height, inferred_width, inferred_frames = infer_dimensions_from_stage1(stage1_video)
    run_params = load_adjacent_run_params(args.stage1_latents)

    height = args.height or int(run_params.get("height") or inferred_height)
    width = args.width or int(run_params.get("width") or inferred_width)
    fps = args.fps or float(run_params.get("output_fps") or run_params.get("fps") or gen.NATIVE_FPS)
    seed = args.seed if args.seed is not None else int(run_params.get("seed") or 42)
    generate_audio = args.generate_audio
    if generate_audio is None:
        generate_audio = bool(run_params.get("generate_audio", stage1_audio is not None))

    if height != inferred_height or width != inferred_width:
        raise ValueError(
            f"Requested {height}x{width}, but stage-1 latents imply "
            f"{inferred_height}x{inferred_width}."
        )

    output_path = args.output or gen.build_default_output_path(
        args.output_dir,
        args.output_prefix,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    weights_path = gen.resolve_default_ltx_weights(args.weights, "distilled")
    transformer_weights_path = gen.resolve_weight_source(args.transformer_weights, weights_path)
    video_vae_weights_path = gen.resolve_weight_source(args.vae_weights, weights_path)
    audio_vae_weights_path = gen.resolve_weight_source(args.audio_vae_weights, weights_path)
    vocoder_weights_path = gen.resolve_weight_source(args.vocoder_weights, weights_path)
    config_weights_path = gen.resolve_weight_source(args.config_weights, weights_path)
    spatial_upscaler_weights = gen.resolve_default_spatial_upscaler_weights(
        args.spatial_upscaler_weights,
        config_weights_path,
    )
    if not gen.is_v2_model(config_weights_path):
        raise ValueError("Stage-2 harness currently supports LTX-2.3 distilled checkpoints only.")
    if not spatial_upscaler_weights:
        raise ValueError("Stage-2 harness requires spatial upscaler weights.")

    if args.stream_transformer:
        if args.transformer_block_resident_blocks == 0:
            args.transformer_block_resident_blocks = 16
        args.transformer_block_compile = True
        if args.transformer_block_compile_group_size == 0:
            args.transformer_block_compile_group_size = min(
                4,
                args.transformer_block_resident_blocks,
            )
    if args.transformer_block_resident_blocks and args.weights_cache == "off":
        args.weights_cache = "auto"
    if args.transformer_cache_quantize != gen.TRANSFORMER_CACHE_QUANTIZE_OFF:
        if args.weights_cache == "off":
            args.weights_cache = "auto"
        if args.video_ff_quantize:
            raise ValueError("--transformer-cache-quantize and --video-ff-quantize are separate experiments")
    if args.video_ff_quantize and args.video_ff_layout:
        raise ValueError("--video-ff-quantize and --video-ff-layout should be tested separately")
    if args.transformer_block_compile and not args.transformer_block_resident_blocks:
        raise ValueError("--transformer-block-compile requires --transformer-block-resident-blocks")
    if (
        args.transformer_block_compile_group_size
        and args.transformer_block_compile_group_size > args.transformer_block_resident_blocks
    ):
        raise ValueError(
            "--transformer-block-compile-group-size cannot exceed --transformer-block-resident-blocks"
        )

    ff_layout_layers = gen.normalize_layout_layers(
        args.video_ff_layout,
        args.video_ff_layout_layers,
    )
    attn_layout_layers = gen.normalize_layout_layers(
        args.video_attn_layout,
        args.video_attn_layout_layers,
    )
    args.video_ff_quantize_layers = gen.normalize_layout_layers(
        args.video_ff_quantize,
        args.video_ff_quantize_layers,
    )
    transformer_cache_quantize_layouts_disabled = False
    if args.transformer_cache_quantize != gen.TRANSFORMER_CACHE_QUANTIZE_OFF:
        transformer_cache_quantize_layouts_disabled = bool(
            args.video_ff_layout or args.video_attn_layout
        )
        args.video_ff_layout = ()
        ff_layout_layers = ()
        args.video_attn_layout = ()
        attn_layout_layers = ()
    compute_dtype = gen.SUPPORTED_COMPUTE_DTYPES[args.dtype]
    if args.mlx_cache_limit_gb is not None:
        mx.set_cache_limit(int(args.mlx_cache_limit_gb * (1000**3)))
        mx.clear_cache()

    vae_tiling_config, vae_auto_tiling = gen.build_vae_tiling_config(
        args.vae_tiling,
        height=height,
        width=width,
        num_frames=inferred_frames,
        decoder_backend=args.vae_decoder,
        force_tiled=args.tiled_vae,
        temporal_tile_frames=args.vae_temporal_tile_frames,
        temporal_overlap_frames=args.vae_temporal_overlap_frames,
        spatial_tile_pixels=args.vae_spatial_tile_pixels,
        spatial_overlap_pixels=args.vae_spatial_overlap_pixels,
    )

    family_sources = {
        "video_vae": video_vae_weights_path,
    }
    if generate_audio:
        family_sources["audio_vae"] = audio_vae_weights_path
        family_sources["vocoder"] = vocoder_weights_path
    family_load_paths = gen.maybe_cache_weight_families(
        family_sources,
        cache_mode=args.weights_cache,
        cache_root=args.weights_cache_dir,
    )
    video_vae_load_path = family_load_paths.get("video_vae", video_vae_weights_path)
    audio_vae_load_path = family_load_paths.get("audio_vae", audio_vae_weights_path)
    vocoder_load_path = family_load_paths.get("vocoder", vocoder_weights_path)

    timings = gen.RunTimings()
    print(f"\n{'=' * 50}")
    print("LTX-2 MLX Stage-2 Harness")
    print(f"{'=' * 50}")
    print(f"Stage-1 latents: {args.stage1_latents}")
    print(f"Text conditioning: {args.embedding}")
    print(f"Resolution: {width}x{height}, {inferred_frames} frames")
    print(f"Stage-2 steps: {len(gen.STAGE_2_DISTILLED_SIGMA_VALUES) - 1}, Seed: {seed}")
    print(f"Compute dtype: {gen.compute_dtype_name(compute_dtype)}")
    print(f"VAE tiling: {gen.describe_vae_tiling_config(vae_tiling_config, vae_auto_tiling)}")
    print(f"Stage-1 latent shape: {latent_metadata['video_shape']}")
    if latent_metadata["audio_shape"] is not None:
        print(f"Stage-1 audio latent shape: {latent_metadata['audio_shape']}")
    if args.use_internal_audio_branch:
        print("Internal AV audio branch: ENABLED")
    print(f"Audio generation: {'ENABLED' if generate_audio else 'disabled'}")
    if args.stream_transformer:
        partial_quant = (
            args.video_ff_quantize
            and args.video_ff_quantize_layers
            and tuple(args.video_ff_quantize_layers) != gen.DEFAULT_TRANSFORMER_LAYOUT_LAYERS
        )
        compile_desc = (
            "compile disabled for partial FF quant"
            if partial_quant
            else f"compile, group {args.transformer_block_compile_group_size}"
        )
        print(
            "Transformer streaming preset: ENABLED "
            f"(r{args.transformer_block_resident_blocks}, "
            f"{compile_desc})"
        )
    if args.weights_cache != "off":
        print(f"Weights cache: ENABLED (mode={args.weights_cache}, dir={args.weights_cache_dir or 'default'})")
    if args.transformer_cache_quantize != gen.TRANSFORMER_CACHE_QUANTIZE_OFF:
        print(f"Transformer cache quantization: ENABLED (mode={args.transformer_cache_quantize})")
        if transformer_cache_quantize_layouts_disabled:
            print("  Same-math transformer layouts disabled for quantized cache")
    if args.video_ff_quantize:
        spec = ",".join(f"{target}:{mode}" for target, mode in args.video_ff_quantize)
        layers = gen.describe_transformer_layers(args.video_ff_quantize_layers)
        print(f"Video FF quantization: ENABLED (specs={spec}, layers={layers})")
    if args.video_ff_layout:
        spec = ",".join(f"{target}:{layout}" for target, layout in args.video_ff_layout)
        layers = gen.describe_transformer_layers(ff_layout_layers)
        print(f"Video FF layout: ENABLED (specs={spec}, layers={layers})")
    if args.video_attn_layout:
        spec = ",".join(f"{target}:{layout}" for target, layout in args.video_attn_layout)
        layers = gen.describe_transformer_layers(attn_layout_layers)
        print(f"Video attention layout: ENABLED (specs={spec}, layers={layers})")
    print(f"AV cross-attn timestep mode: {args.av_cross_timestep_mode}")
    rope_precision_override = {
        "metadata": None,
        "float64": True,
        "fp32": False,
    }[args.rope_precision]
    print(f"RoPE precision: {args.rope_precision}")
    if args.probe_av_cross_timestep:
        print("AV cross-attn timestep probe: ENABLED")
    print(f"Stage-1 RNG burn: {'disabled' if args.independent_stage2_noise else 'ENABLED'}")

    stage2_kv_pool_attention = None
    if args.stage2_video_attn_kv_pool is not None:
        from LTX_2_MLX.model.transformer import attention as stage2_kv_pool_attention

        pool_h, pool_w = args.stage2_video_attn_kv_pool
        stage2_frames = int(stage1_video.shape[2])
        stage2_latent_h = int(stage1_video.shape[3]) * 2
        stage2_latent_w = int(stage1_video.shape[4]) * 2
        pool_steps = args.stage2_video_attn_kv_pool_steps
        if pool_steps is not None and pool_steps < 0:
            raise ValueError("--stage2-video-attn-kv-pool-steps must be non-negative")
        max_applied = None if pool_steps is None else pool_steps * 48
        stage2_kv_pool_attention.configure_kv_downsample(
            pool_h=pool_h,
            pool_w=pool_w,
            frames=stage2_frames,
            height=stage2_latent_h,
            width=stage2_latent_w,
            mode=args.stage2_video_attn_kv_pool_mode,
            max_applied=max_applied,
        )
        pooled_tokens = stage2_frames * (stage2_latent_h // pool_h) * (stage2_latent_w // pool_w)
        full_tokens = stage2_frames * stage2_latent_h * stage2_latent_w
        budget_desc = (
            "all stage-2 steps"
            if pool_steps is None
            else f"first {pool_steps} stage-2 step{'s' if pool_steps != 1 else ''}"
        )
        print(
            "Stage-2 video self-attn K/V pooling: ENABLED [FAILED DIAGNOSTIC] "
            f"(pool={pool_h}x{pool_w}, mode={args.stage2_video_attn_kv_pool_mode}, "
            f"K/V tokens {full_tokens}->{pooled_tokens}, {budget_desc}; "
            "not same-math, known blur/ghosting)"
        )

    print("\n[1/5] Loading text conditioning...")
    conditioning = gen.load_text_conditioning(args.embedding, use_av_encoder=True)
    text_encoding = conditioning["positive_video_encoding"]
    text_audio_encoding = conditioning["positive_audio_encoding"]
    if args.use_internal_audio_branch and text_audio_encoding is None:
        raise ValueError("Text conditioning sidecar does not include positive audio encoding.")
    timings.mark("text conditioning load")

    print("\n[2/5] Loading AudioVideo transformer...")
    # Mirror generate.py's default: audio pretranspose tracks video layout
    # for AV models unless LTX_DISABLE_AUDIO_PRETRANSPOSE=1 is set.  Without
    # this, the harness would build a different cache hash than generate.py
    # and re-pretranspose only the video half.
    if os.environ.get("LTX_DISABLE_AUDIO_PRETRANSPOSE"):
        _audio_ff_layout_specs = ()
        _audio_attn_layout_specs = ()
    else:
        _audio_ff_layout_specs = args.video_ff_layout
        _audio_attn_layout_specs = args.video_attn_layout
    _adaln_pretranspose = bool(os.environ.get("LTX_ADALN_PRETRANSPOSE"))
    model = gen.load_av_transformer(
        transformer_weights_path,
        num_layers=48,
        compute_dtype=compute_dtype,
        video_ff_quantize_specs=args.video_ff_quantize,
        video_ff_quantize_group_size=args.video_ff_quantize_group_size,
        video_ff_quantize_bits=args.video_ff_quantize_bits,
        video_ff_quantize_layers=args.video_ff_quantize_layers,
        video_ff_layout_specs=args.video_ff_layout,
        video_ff_layout_layers=ff_layout_layers,
        video_attn_layout_specs=args.video_attn_layout,
        video_attn_layout_layers=attn_layout_layers,
        audio_ff_layout_specs=_audio_ff_layout_specs,
        audio_ff_layout_layers=ff_layout_layers,
        audio_attn_layout_specs=_audio_attn_layout_specs,
        audio_attn_layout_layers=attn_layout_layers,
        adaln_pretranspose=_adaln_pretranspose,
        transformer_cache_quantize=args.transformer_cache_quantize,
        weights_cache_mode=args.weights_cache,
        weights_cache_dir=args.weights_cache_dir,
        transformer_block_resident_blocks=args.transformer_block_resident_blocks,
        transformer_block_compile=args.transformer_block_compile,
        transformer_block_compile_group_size=args.transformer_block_compile_group_size,
        caption_channels=None,
        cross_attention_adaln=True,
        apply_gated_attention=True,
        config_weights_path=config_weights_path,
        double_precision_rope=rope_precision_override,
    )
    if hasattr(model, "set_av_cross_timestep_mode"):
        model.set_av_cross_timestep_mode(args.av_cross_timestep_mode)
    print("  CFG disabled (distilled stage-2 harness)")
    timings.mark("transformer load")

    print("\n[3/5] Loading VAE decoder/encoder and spatial upscaler...")
    vae_config = gen.get_vae_config(config_weights_path)
    decoder_blocks = vae_config.get("decoder_blocks", None)
    base_channels = vae_config.get("decoder_base_channels", 128)
    timestep_cond = vae_config.get("timestep_conditioning", True)
    vae_decoder = gen.NativeConv3dVideoDecoder(
        decoder_blocks=decoder_blocks,
        base_channels=base_channels,
        timestep_conditioning=timestep_cond,
        compute_dtype=compute_dtype,
    )
    gen.load_native_vae_decoder_weights(vae_decoder, video_vae_load_path)

    video_encoder = gen.NativeConv3dVideoEncoder(compute_dtype=compute_dtype)
    gen.load_native_vae_encoder_weights(video_encoder, video_vae_load_path)

    spatial_upscaler = gen.SpatialUpscaler()
    gen.load_spatial_upscaler_weights(spatial_upscaler, spatial_upscaler_weights)

    audio_decoder = None
    vocoder = None
    audio_sample_rate = 24000
    if generate_audio:
        print("  Loading Audio VAE decoder...")
        audio_decoder = gen.AudioDecoder(compute_dtype=compute_dtype)
        gen.load_audio_decoder_weights(audio_decoder, audio_vae_load_path)

        print("  Loading Vocoder...")
        vocoder, is_bwe = gen.create_vocoder_for_checkpoint(config_weights_path, compute_dtype)
        if is_bwe:
            print("  Detected BWE vocoder (LTX-2.3)")
            gen.load_vocoder_with_bwe_weights(vocoder, vocoder_load_path)
        else:
            gen.load_vocoder_weights(vocoder, vocoder_load_path)
        gen.print_audio_dtype_summary(compute_dtype, is_bwe)
        audio_sample_rate = vocoder.output_sample_rate if vocoder else 24000
    timings.mark("decoder/upscaler load")

    print("\n[4/5] Creating stage-2 pipeline...")
    av_pipeline = gen.AVPipeline(
        transformer=model,
        video_encoder=video_encoder,
        video_decoder=vae_decoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
    )
    probe_model = av_pipeline._velocity_transformer() if args.probe_av_cross_timestep else None

    # LTX_MONO_INLINED=1: swap the pipeline's transformer with InlinedAVModel
    # (mono_pipeline.transformer_step + flat pretransposed weights).  Same
    # math as the modular path, no nn.Module dispatch in the per-step forward.
    # Used to A/B "does collapsing the 48-block stack into one inlined
    # function change MLX's graph and per-step time?"
    if os.environ.get("LTX_MONO_INLINED"):
        import mono_pipeline
        base = av_pipeline.transformer.velocity_model
        av_pipeline.transformer = mono_pipeline.InlinedAVModel(base)
        print("  Transformer: InlinedAVModel [LTX_MONO_INLINED=1]")

    del model

    profile_transformer_steps = tuple(args.profile_transformer_steps or ())
    profile_transformer_once = bool(args.profile_transformer and not profile_transformer_steps)

    config = gen.AVCFGConfig(
        height=height,
        width=width,
        num_frames=inferred_frames,
        seed=seed,
        fps=fps,
        num_inference_steps=len(gen.STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
        use_distilled_sigmas=True,
        cfg_scale=1.0,
        audio_cfg_scale=1.0,
        rescale_scale=0.0,
        dtype=compute_dtype,
        tiling_config=vae_tiling_config,
        auto_tiling=False,
        audio_enabled=generate_audio,
        use_internal_audio_branch=args.use_internal_audio_branch,
        profile_transformer_once=profile_transformer_once,
        profile_transformer_steps=profile_transformer_steps,
        profile_transformer_blocks=tuple(args.profile_transformer_blocks or ()),
    )
    timings.mark("pipeline object setup")

    print("\n[5/5] Running stage 2 from saved latents...")

    def stage2_state_probe(video_state, audio_state, stage_2_sigmas):
        if probe_model is None:
            return
        print_av_cross_timestep_probe(
            probe_model,
            video_state,
            audio_state,
            text_encoding,
            text_audio_encoding,
            float(np.array(stage_2_sigmas[0])),
            stream=sys.stderr if bench_mode_active else sys.stdout,
        )

    # Bench-mode state: track per-step wall time + trigger early exit.
    # ``_BenchModeStop`` is caught by the bench-mode wrapper just below the
    # generate_distilled_stage2_from_latents call so we get a clean exit
    # (with timing summary) instead of a generic traceback.
    class _BenchModeStop(Exception):
        pass

    bench_mode_step = max(0, int(args.bench_mode))
    bench_mode_active = bench_mode_step > 0
    bench_step_times: list[float] = []
    bench_step_t_prev: list[float] = [0.0]

    if bench_mode_active:
        # Skip the StackedPhaseBars UI entirely in bench mode: progress
        # messages from generate_distilled_stage2_from_latents (the
        # "Upsampling..." / "Distilled stage 2: ..." inter-stage notes)
        # use position="below" which freezes the bar stack and detaches
        # subsequent bar updates from the cursor accounting — so any
        # per-step bench print collides with the in-place bar redraws.
        # We don't need a polished UI for a timing-only run, and plain
        # stderr prints (no bars, no \r updates) have no clobbering risk.

        def stage_callback(stage_name: str, step: int, total: int) -> None:
            if stage_name != "stage_2":
                return
            now = time.perf_counter()
            if step == 0:
                bench_step_t_prev[0] = now
                print(
                    f"[bench-mode] stage_2 begin (will exit after step {bench_mode_step})",
                    file=sys.stderr,
                    flush=True,
                )
                return
            elapsed = now - bench_step_t_prev[0]
            bench_step_t_prev[0] = now
            bench_step_times.append(elapsed)
            label = "WARMUP" if step == 1 else ("MEASURE" if step == bench_mode_step else "extra")
            print(
                f"[bench-mode] stage_2 step {step}/{total} done: "
                f"{elapsed:.3f}s  ({label})",
                file=sys.stderr,
                flush=True,
            )
            if step >= bench_mode_step:
                raise _BenchModeStop()

        def progress_message(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

        try:
            av_pipeline.generate_distilled_stage2_from_latents(
                stage_1_video_latent=stage1_video,
                stage_1_audio_latent=stage1_audio,
                positive_encoding=text_encoding,
                positive_audio_encoding=text_audio_encoding,
                config=config,
                spatial_upscaler=spatial_upscaler,
                stage_callback=stage_callback,
                progress_message=progress_message,
                latent_save_path=None,
                burn_stage1_rng=not args.independent_stage2_noise,
                stage2_state_probe=stage2_state_probe if args.probe_av_cross_timestep else None,
            )
        except _BenchModeStop:
            pass

        # Print the per-step summary and stop here (skip VAE decode,
        # audio decode, output save, run log).
        print("", file=sys.stderr, flush=True)
        print("== stage2_harness bench-mode summary ==", file=sys.stderr)
        for i, dt in enumerate(bench_step_times, start=1):
            tag = "WARMUP" if i == 1 else ("MEASURE" if i == bench_mode_step else "extra")
            print(f"  step {i}: {dt:.3f}s  ({tag})", file=sys.stderr)
        if len(bench_step_times) >= bench_mode_step:
            print(
                f"  measured (step {bench_mode_step}): "
                f"{bench_step_times[bench_mode_step - 1]:.3f}s",
                file=sys.stderr,
            )
        print_kv_downsample_summary(stage2_kv_pool_attention, stream=sys.stderr)
        print("== end bench-mode ==", file=sys.stderr, flush=True)
        return

    # Normal (non-bench) path: mirror generate.py's default output route.
    # The default HEVC tier uses VideoToolbox, so ask the pipeline to return
    # final latents and stream VAE chunks directly into the encoder instead of
    # materializing a full decoded video tensor first.
    output_tier = "default"
    output_backend = "auto"
    resolved_output_backend = gen.resolve_output_backend(output_backend, output_tier)
    stream_decode = resolved_output_backend == "videotoolbox"

    # Normal (non-bench) path: use the StackedPhaseBars UI.
    with StackedPhaseBars() as denoise_bars:
        stage_bar = denoise_bars.add(
            total=len(gen.STAGE_2_DISTILLED_SIGMA_VALUES) - 1,
            desc="Stage 2 denoising",
            unit="step",
            show_step1=True,
        )

        def stage_callback(stage_name: str, step: int, total: int) -> None:
            stage_bar.set_n(step)

        def progress_message(message: str) -> None:
            denoise_bars.write(message, position="below")

        video, audio_waveform = av_pipeline.generate_distilled_stage2_from_latents(
            stage_1_video_latent=stage1_video,
            stage_1_audio_latent=stage1_audio,
            positive_encoding=text_encoding,
            positive_audio_encoding=text_audio_encoding,
            config=config,
            spatial_upscaler=spatial_upscaler,
            stage_callback=stage_callback,
            progress_message=progress_message,
            latent_save_path=gen.latent_sidecar_path(output_path) if args.save_latents else None,
            burn_stage1_rng=not args.independent_stage2_noise,
            decode_video=not stream_decode,
            stage2_state_probe=stage2_state_probe if args.probe_av_cross_timestep else None,
        )

    pipeline_timings = getattr(av_pipeline, "last_timing_sections", None)
    if pipeline_timings:
        timings.extend(pipeline_timings)
    else:
        timings.mark("stage2 + decode")
    print_kv_downsample_summary(stage2_kv_pool_attention)

    final_path = output_path
    if stream_decode:
        from LTX_2_MLX.pipelines.streaming import (
            iter_decoded_chunks,
            latent_dims,
            plan_vae_tiling,
        )

        final_video_latent = video
        effective_tiling = config._get_tiling_config()
        n_total_frames, latent_h, latent_w = latent_dims(final_video_latent)
        n_vae_chunks, tiling_desc = plan_vae_tiling(final_video_latent, effective_tiling)
        decoder_name = av_pipeline.video_decoder.__class__.__name__
        print(
            f"  Streaming VAE decode -> encoder: latent shape "
            f"{tuple(final_video_latent.shape)} -> {latent_w}x{latent_h}, "
            f"{n_total_frames} frames"
        )
        print(
            f"  VAE decoder: {decoder_name} ({tiling_desc}, "
            f"{n_vae_chunks} chunk{'s' if n_vae_chunks != 1 else ''})"
        )
        if audio_waveform is not None:
            print(f"  Generated audio: {audio_waveform.shape}")
        timings.mark("post-pipeline prep")

        print(f"\nSaving video to {output_path}...")
        with StackedPhaseBars() as stream_bars:
            vae_pbar = stream_bars.add(
                total=n_vae_chunks,
                desc="VAE chunks",
                unit="chunk",
            )

            def chunk_aware_frames():
                for chunk_frames in iter_decoded_chunks(
                    final_video_latent,
                    av_pipeline.video_decoder,
                    tiling=effective_tiling,
                    output_format="fp16_rgba",
                ):
                    vae_pbar.update(1)
                    while chunk_frames:
                        yield chunk_frames.pop(0)

            final_path = gen.encode_video_dispatch(
                chunk_aware_frames(),
                output_path,
                tier=output_tier,
                fps=fps,
                audio_waveform=audio_waveform if audio_waveform is not None else None,
                audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
                output_backend=output_backend,
                n_source_frames=n_total_frames,
                progress_stack=stream_bars,
            )
    else:
        frames = video_tensor_to_frames(video)
        if audio_waveform is not None:
            print(f"  Generated audio: {audio_waveform.shape}")
        timings.mark("frame conversion")

        print(f"\nSaving video to {output_path}...")
        final_path = gen.encode_video_dispatch(
            frames,
            output_path,
            tier=output_tier,
            fps=fps,
            audio_waveform=audio_waveform if audio_waveform is not None else None,
            audio_sample_rate=audio_sample_rate if audio_waveform is not None else None,
            output_backend=output_backend,
        )
    print(f"Done! Video saved to {final_path}")
    timings.mark("output save")

    if args.save_run_log:
        run_log = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv[:],
            "cwd": os.getcwd(),
            "output_path": str(final_path),
            "stage1_latents": args.stage1_latents,
            "embedding": args.embedding,
            "sidecars": {
                "run_log": gen.run_log_sidecar_path(output_path),
                "latents": gen.latent_sidecar_path(output_path) if args.save_latents else None,
            },
            "parameters": {
                "height": height,
                "width": width,
                "num_frames": inferred_frames,
                "fps": fps,
                "seed": seed,
                "generate_audio": generate_audio,
                "use_internal_audio_branch": args.use_internal_audio_branch,
                "burn_stage1_rng": not args.independent_stage2_noise,
                "weights_path": weights_path,
                "transformer_weights_path": transformer_weights_path,
                "video_vae_weights_path": video_vae_weights_path,
                "audio_vae_weights_path": audio_vae_weights_path,
                "vocoder_weights_path": vocoder_weights_path,
                "config_weights_path": config_weights_path,
                "spatial_upscaler_weights": spatial_upscaler_weights,
                "weights_cache_mode": args.weights_cache,
                "weights_cache_dir": args.weights_cache_dir,
                "stream_transformer": args.stream_transformer,
                "transformer_block_resident_blocks": args.transformer_block_resident_blocks,
                "transformer_block_compile": args.transformer_block_compile,
                "transformer_block_compile_group_size": args.transformer_block_compile_group_size,
                "video_ff_quantize_specs": [
                    {"target": target, "mode": mode}
                    for target, mode in args.video_ff_quantize
                ],
                "video_ff_quantize_layers": list(args.video_ff_quantize_layers),
                "video_ff_layout_specs": [
                    {"target": target, "layout": layout}
                    for target, layout in args.video_ff_layout
                ],
                "video_ff_layout_layers": list(ff_layout_layers),
                "video_attn_layout_specs": [
                    {"target": target, "layout": layout}
                    for target, layout in args.video_attn_layout
                ],
                "video_attn_layout_layers": list(attn_layout_layers),
                "av_cross_timestep_mode": args.av_cross_timestep_mode,
                "rope_precision": args.rope_precision,
                "probe_av_cross_timestep": args.probe_av_cross_timestep,
                "stage2_video_attn_kv_pool": (
                    {
                        "pool_h": args.stage2_video_attn_kv_pool[0],
                        "pool_w": args.stage2_video_attn_kv_pool[1],
                        "mode": args.stage2_video_attn_kv_pool_mode,
                        "steps": args.stage2_video_attn_kv_pool_steps,
                    }
                    if args.stage2_video_attn_kv_pool is not None
                    else None
                ),
                "stage2_video_attn_kv_pool_summary": (
                    stage2_kv_pool_attention.kv_downsample_summary()
                    if stage2_kv_pool_attention is not None
                    else None
                ),
                "transformer_cache_quantize": args.transformer_cache_quantize,
                "transformer_cache_quantize_layouts_disabled": transformer_cache_quantize_layouts_disabled,
            },
            "timings": timings.to_dict(),
        }
        path = gen.run_log_sidecar_path(output_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=2)
        print(f"  Saved run log: {path}")

    timings.print_summary()


if __name__ == "__main__":
    main()
