"""Micro-benchmark: time a single AV transformer forward pass.

Loads the same V2 distilled AV transformer the real generate.py uses, builds
fake video+audio modalities at a configurable shape, and times N forward
passes.  Default shape is stage-1-ish (288x512x481, T_v ~ 4.4k tokens) so each
iteration is ~50 s instead of ~390 s.

Used to A/B candidate optimizations by toggling env vars before invocation:

    # baseline
    python scripts/bench_eval_cadence.py --weights ...

    # disable @mx.compile around reshape+SDPA+reshape
    LTX_DISABLE_COMPILED_ATTN=1 python scripts/bench_eval_cadence.py --weights ...

Each invocation reports per-iteration wall time and a trimmed mean.  The
first iteration includes any one-time Metal kernel/compile setup; the script
discards N warmup iterations before reporting the mean.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Make package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from LTX_2_MLX.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
)
from LTX_2_MLX.conditioning.tools import AudioLatentTools, VideoLatentTools

# Reuse the production loader so the bench has the exact same layout/quant
# pipeline (pretranspose, V2 cross_attention_adaln, gated attention, etc).
from LTX_2_MLX.generate import (
    DEFAULT_VIDEO_ATTN_LAYOUT_SPECS,
    DEFAULT_VIDEO_FF_LAYOUT_SPECS,
    load_av_transformer,
)
from LTX_2_MLX.pipelines.common import (
    audio_modality_from_state,
    modality_from_state,
)
from LTX_2_MLX.types import (
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--weights-cache-dir", default=None)
    p.add_argument("--weights-cache-mode", default="auto")
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=481)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--text-tokens", type=int, default=205,
                   help="Text context length (matches typical real prompt).")
    p.add_argument("--iters", type=int, default=3,
                   help="Timed iterations (after warmups).")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlx-cache-limit-gb", type=float, default=1.0,
                   help="MLX allocator cache cap in GB.  Use 0 to release "
                        "freed buffers immediately, or a large value (e.g. 64) "
                        "to effectively disable the cap.")
    p.add_argument("--fast-mode", action="store_true",
                   help="Disable the per-block forced-eval cadence (the model "
                        "default forces mx.eval every 8 blocks = 6x per "
                        "48-block forward).  mlx-video does no intermediate "
                        "evals, so this matches their pattern.")
    p.add_argument("--no-pretranspose", action="store_true",
                   help="Disable the default pretranspose layout for video FF "
                        "(project_in/out) and video attention (to_out).  When "
                        "enabled, our path uses addmm with cached weight.T; "
                        "without, we use plain nn.Linear like mlx-video.")
    return p.parse_args()


def build_modalities(args, video_shape, audio_shape, dtype, rng_key):
    """Construct fake but correctly-shaped Modality inputs.

    Uses the production VideoLatentTools / AudioLatentTools paths to get
    real positional encodings (RoPE expects shape (B, n_dims, T, 2)), then
    swaps in random latents/context so we don't pay encode time.
    """
    b = video_shape.batch

    video_tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=video_shape,
        fps=args.fps,
    )
    audio_tools = AudioLatentTools(
        patchifier=AudioPatchifier(patch_size=1),
        target_shape=audio_shape,
    )

    video_state = video_tools.create_initial_state(dtype=dtype)
    audio_state = audio_tools.create_initial_state(dtype=dtype)

    # Replace the all-zero latent with random, same shape, so the matmuls
    # see realistic input magnitudes (not strictly required for timing,
    # but avoids any sparse-zero shortcuts).
    video_state = video_state.replace(
        latent=mx.random.normal(video_state.latent.shape, dtype=dtype, key=rng_key)
    )
    audio_state = audio_state.replace(
        latent=mx.random.normal(audio_state.latent.shape, dtype=dtype, key=rng_key)
    )

    # Text contexts: V2 uses 4096 for video, 2048 for audio.
    s = args.text_tokens
    video_context = mx.random.normal((b, s, 4096), dtype=dtype, key=rng_key)
    audio_context = mx.random.normal((b, s, 2048), dtype=dtype, key=rng_key)

    sigma_value = 0.5
    video_modality = modality_from_state(video_state, video_context, sigma_value)
    audio_modality = audio_modality_from_state(audio_state, audio_context, sigma_value)

    mx.eval(video_modality.latent, video_modality.context, video_modality.positions)
    mx.eval(audio_modality.latent, audio_modality.context, audio_modality.positions)
    mx.eval(video_modality.timesteps, video_modality.sigma)
    mx.eval(audio_modality.timesteps, audio_modality.sigma)
    return video_modality, audio_modality


def describe_variant(fast_mode: bool = False, no_pretranspose: bool = False) -> str:
    """Summarize which knobs the env/CLI have flipped."""
    flags = []
    if fast_mode:
        flags.append("--fast-mode (eval_frequency=0, no per-forward block evals)")
    if no_pretranspose:
        flags.append("--no-pretranspose (plain nn.Linear, no weight.T cache)")
    if os.environ.get("LTX_DISABLE_COMPILED_ATTN"):
        flags.append("LTX_DISABLE_COMPILED_ATTN=1 (uncompiled attn reshape)")
    if os.environ.get("LTX_DISABLE_COMPILED_HELPERS"):
        flags.append("LTX_DISABLE_COMPILED_HELPERS=1 (inline adaln+residual_gate)")
    return ", ".join(flags) if flags else "baseline (defaults)"


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    key = mx.random.key(args.seed)

    # Default mirrors generate.py's `--mlx-cache-limit-gb 1`; CLI overrides.
    cache_bytes = int(args.mlx_cache_limit_gb * (1 << 30))
    mx.set_cache_limit(cache_bytes)

    print("=" * 64)
    print(f"bench_eval_cadence: {args.height}x{args.width} x {args.num_frames}f")
    print(f"variant: {describe_variant(args.fast_mode, args.no_pretranspose)} | mlx_cache_limit={args.mlx_cache_limit_gb} GB")
    print("=" * 64)

    pixel_shape = VideoPixelShape(
        batch=1, frames=args.num_frames, height=args.height, width=args.width, fps=args.fps,
    )
    video_shape = VideoLatentShape.from_pixel_shape(pixel_shape, latent_channels=128)
    audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    print(f"  video latent T = {video_shape.frames * video_shape.height * video_shape.width:,} tokens")
    print(f"  audio latent T = {audio_shape.frames * audio_shape.mel_bins:,} tokens")

    print("Loading transformer (matches generate.py defaults: V2, BF16, pretranspose layouts)...")
    t_load = time.perf_counter()
    ff_layout = () if args.no_pretranspose else DEFAULT_VIDEO_FF_LAYOUT_SPECS
    attn_layout = () if args.no_pretranspose else DEFAULT_VIDEO_ATTN_LAYOUT_SPECS
    model = load_av_transformer(
        weights_path=args.weights,
        compute_dtype=mx.bfloat16,
        caption_channels=None,           # V2
        cross_attention_adaln=True,      # V2
        apply_gated_attention=True,      # V2
        fast_mode=args.fast_mode,        # eval_frequency = 0 when True
        weights_cache_mode=args.weights_cache_mode,
        weights_cache_dir=args.weights_cache_dir,
        video_ff_layout_specs=ff_layout,
        video_attn_layout_specs=attn_layout,
    )
    mx.eval(model.parameters())
    print(f"  loaded in {time.perf_counter() - t_load:.1f} s")

    print("Building fake modalities...")
    video, audio = build_modalities(args, video_shape, audio_shape, mx.bfloat16, key)

    n_total = args.warmups + args.iters
    print()
    times = []
    for i in range(n_total):
        is_warmup = i < args.warmups
        tag = "warmup" if is_warmup else "timed "
        t0 = time.perf_counter()
        v_out, a_out = model(video=video, audio=audio)
        # Mimic the post-process + Euler step (tiny math) and final eval, so
        # the bench captures the full per-step shape, not just the forward.
        v_step = v_out * mx.array(0.9, dtype=v_out.dtype)
        a_step = a_out * mx.array(0.9, dtype=a_out.dtype)
        mx.eval(v_step, a_step)
        dt = time.perf_counter() - t0
        if not is_warmup:
            times.append(dt)
        print(f"  iter {i+1}/{n_total} ({tag}): {dt:6.2f} s", flush=True)

    mean = sum(times) / len(times)
    print()
    print("=" * 64)
    print(f"variant: {describe_variant()}")
    print(f"mean of {len(times)} timed iters: {mean:6.2f} s/it")
    print(f"min / max:                       {min(times):6.2f} / {max(times):6.2f} s")
    print("=" * 64)


if __name__ == "__main__":
    main()
