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
        choices=["native-conv3d"],
        default="native-conv3d",
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
    print(f"Stage-1 RNG burn: {'disabled' if args.independent_stage2_noise else 'ENABLED'}")

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
    )
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
    )
    timings.mark("pipeline object setup")

    print("\n[5/5] Running stage 2 from saved latents...")
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
        )
    pipeline_timings = getattr(av_pipeline, "last_timing_sections", None)
    if pipeline_timings:
        timings.extend(pipeline_timings)
    else:
        timings.mark("stage2 + decode")

    frames = video_tensor_to_frames(video)
    if audio_waveform is not None:
        print(f"  Generated audio: {audio_waveform.shape}")
    timings.mark("frame conversion")

    print(f"\nSaving video to {output_path}...")
    if audio_waveform is not None:
        gen.save_video_with_audio(
            frames,
            audio_waveform,
            output_path,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
        )
    else:
        gen.save_video(frames, output_path, fps=fps)
    print(f"Done! Video saved to {output_path}")
    timings.mark("output save")

    if args.save_run_log:
        run_log = {
            "schema_version": 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv[:],
            "cwd": os.getcwd(),
            "output_path": output_path,
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
