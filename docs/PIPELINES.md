# LTX-2 MLX Pipelines

LTX-2 MLX provides **6 specialized pipelines** for different use cases. Use the `--pipeline` flag to select.

## Available Pipelines

### 1. `text-to-video` (Default)

Standard text-to-video generation with simple CFG denoising.

```bash
ltx2mlx "A cat walking" \
    --pipeline text-to-video \
    --frames 25
```

**Best for**: Basic video generation, quick testing

### 2. `distilled` - Fast

Two-stage distilled model optimized for speed (no CFG, 11 total denoise steps).

```bash
ltx2mlx "A cat walking" \
    --pipeline distilled \
    --frames 25
```

- **Stage 1**: 8 steps at half resolution
- **Stage 2**: 3 steps refinement
- **Speed**: ~2x faster than standard
- **Quality**: Good for most use cases
- **Best for**: Fast iteration, batch generation

### 3. `one-stage` - Quality

Single-stage CFG with full control and adaptive sigma scheduling.

```bash
ltx2mlx "A cat walking" \
    --pipeline one-stage \
    --cfg 5.0 --steps 20 \
    --frames 25
```

- Uses LTX2Scheduler for token-count-dependent sigma schedule
- Optional image conditioning via latent replacement
- Full CFG control with positive/negative prompts
- **Best for**: High-quality single-resolution generation

### 4. `two-stage` - HQ + Upscaling (Recommended for High-Res)

Two-stage pipeline with spatial upscaling for high-resolution output.

```bash
ltx2mlx "A cat walking" \
    --pipeline two-stage \
    --height 512 --width 704 \
    --cfg 5.0 --steps-stage1 15 \
    --spatial-upscaler-weights /path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --distilled-lora weights/ltx-2/ltx-2-19b-distilled-lora-384.safetensors \
    --dtype bfloat16
```

> **Note**: Resolution is automatically adjusted to be divisible by 64 (required for the two-stage pipeline).

- **Stage 1**: Generate at half resolution (256x352) with CFG
- **Stage 2**: 2x spatial upscale + 3 distilled refinement steps
- **Final output**: 512x704 (or higher)
- Combines CFG quality with distilled speed
- **Best for**: High-resolution video generation (512x704+)

**Quality**: Matches one-stage baseline with 2x resolution increase

## Additional Pipelines (Code-Only)

### 5. Image Conditioning LoRA (`ic_lora`)

Video-to-video generation with control signals (depth, pose, edges).

```python
from LTX_2_MLX.pipelines import ICLoraPipeline, create_ic_lora_pipeline

pipeline = create_ic_lora_pipeline(
    transformer=model,
    video_decoder=decoder,
    # ... requires IC-LoRA weights
)
```

- Two-stage with IC-LoRA in Stage 1 only
- Control signals: depth maps, human pose, edge detection
- **Best for**: Controlled video-to-video generation

### 6. Keyframe Interpolation (`keyframe_interpolation`)

Generate videos by interpolating between keyframe images.

```python
from LTX_2_MLX.pipelines import KeyframeInterpolationPipeline, Keyframe

keyframes = [
    Keyframe(image=img1, frame_index=0),
    Keyframe(image=img2, frame_index=24),
]

video = pipeline(keyframes=keyframes, ...)
```

- Two-stage with keyframe conditioning
- Smooth interpolation between images
- **Best for**: Image sequence animation

## Pipeline Comparison

| Pipeline | Speed | Quality | Resolution | CFG | Best Use Case |
|----------|-------|---------|------------|-----|---------------|
| `text-to-video` | Medium | Good | Any | Yes | Basic generation |
| `distilled` | **Fast** (8+3 steps) | Good | 512p+ | No | Quick iteration |
| `one-stage` | Slow (20+ steps) | **High** | Any | Yes | Quality priority |
| `two-stage` | Medium (18 steps) | **High** | **512p+** | Yes | High-resolution |
| `ic_lora` | Medium | High | 512p+ | Yes | Controlled gen |
| `keyframe_interpolation` | Medium | High | 512p+ | Yes | Image animation |

`--pipeline distilled` now uses the AudioVideo two-stage distilled path for
LTX-2.3 checkpoints. Use `--pipeline one-stage --model-variant distilled` for
the existing single-pass distilled path.

## Recommended Settings by Use Case

### Fast Previews

```bash
ltx2mlx "Your prompt" \
    --pipeline distilled --height 512 --width 768 --frames 65
```

### High Quality

```bash
ltx2mlx "Your prompt" \
    --pipeline one-stage --frames 65 \
    --steps 15 --cfg 4.0 --dtype bfloat16
```

### High Resolution

```bash
ltx2mlx "Your prompt" \
    --pipeline two-stage --height 768 --width 1024 --frames 65 \
    --steps-stage1 15 --cfg 5.0 --dtype bfloat16
```

## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--pipeline` | Pipeline: `text-to-video`, `distilled`, `one-stage`, `two-stage` | text-to-video |
| `--height` | Video height (divisible by 32; distilled/two-stage modes round to 64) | 288 |
| `--width` | Video width (divisible by 32; distilled/two-stage modes round to 64) | 512 |
| `--frames` | Number of frames (N*8+1) | 97 |
| `--duration` | Duration in seconds; overrides `--frames` and rounds up to a valid frame count | None |
| `--fps` | Generation and output frame rate | 24 |
| `--steps` | Single-pass denoising steps for dev; distilled one-stage uses fixed 8 sigmas and distilled two-stage uses fixed 8+3 sigmas | model-aware: 8 distilled, 30 dev |
| `--steps-stage1` | Stage 1 steps (two-stage pipeline) | 15 |
| `--steps-stage2` | Stage 2 steps (two-stage pipeline) | 3 |
| `--cfg` | Classifier-free guidance scale | model-aware: 1.0 distilled, 5.0 dev |
| `--seed` | Random seed | 42 |
| `--output` | Exact output video path; overrides timestamped naming | None |
| `--output-dir` | Directory for default timestamped outputs; falls back to `DIFFUSERS_OUTPUT_DIR`, then `OUTPUT_DIR`, then `outputs/` | env or outputs |
| `--output-prefix` | Filename prefix for default timestamped outputs | ltx |
| `--weights` | Path to a full weight bundle; default resolves cached `Lightricks/LTX-2.3` from `HF_HOME` / `HF_HUB_CACHE` | LTX-2.3 distilled/dev |
| `--transformer-weights` | Optional transformer-only override; defaults to `--weights` | None |
| `--connector-weights` | Optional text connector / AV projection override; defaults to `--weights` | None |
| `--vae-weights` | Optional video VAE override; defaults to `--weights` | None |
| `--audio-vae-weights` | Optional audio VAE override; defaults to `--weights` | None |
| `--vocoder-weights` | Optional vocoder override; defaults to `--weights` | None |
| `--config-weights` | Optional metadata/config source for model version, VAE shape, and vocoder type; defaults to `--weights` | None |
| `--weights-cache` | Converted-weight cache: `auto`, `off`, or `rebuild`; stores transformer plus named connector/video-VAE/audio-VAE/vocoder families | auto |
| `--mlx-cache-limit-gb` | MLX in-memory allocator cache limit in decimal GB | 1 |
| `--stream-transformer` | Recommended block-streaming preset: r16, compile, 4-block groups | False |
| `--dtype` | Compute dtype: `bfloat16`, `float16`, or `float32` | bfloat16 |
| `--vae-decoder` | VAE decoder backend.  Only `native` (MLX-native Conv3d) is supported; the historical `legacy` `SimpleVideoDecoder` was removed 2026-05-23 and remains available in git history. | native |
| `--vae-tiling` | VAE decode tiling policy: RAM-aware `auto`, `off`, or `custom` | auto |
| `--video-ff-layout` | Same-math video FF pretranspose layout, or `off` for baseline A/B | project_in/project_out pretranspose |
| `--video-attn-layout` | Same-math video attention output pretranspose layout, or `off` for baseline A/B | to_out pretranspose |
| `--model-variant` | `distilled` (fast) or `dev` (quality) | distilled |
| `--spatial-upscaler-weights` | Path to spatial upscaler weights (for distilled/two-stage) | cached LTX-2.3 x2 upscaler |
| `--temporal-upscaler-weights` | Path to temporal upscaler weights | None |
| `--upscale-spatial` | Apply 2x spatial upscaling (legacy model-based) | False |
| `--upscale-temporal` | Apply 2x temporal upscaling (legacy model-based) | False |
| `--output-backend` | Encode backend: `auto` (VT for HEVC/`default` tier or any `--vsr-*` flag, ffmpeg otherwise), `ffmpeg`, or `videotoolbox` | auto |
| `--vsr-spatial-mode` | VideoToolbox Super Resolution: `off`, `fast` (2x LowLatency), `balanced` (4x HQ Video), `image` (4x HQ Image). Forces `--output-backend videotoolbox`. | off |
| `--vsr-target-fps` | VideoToolbox frame-rate conversion target. When set and different from `--fps`, interpolates via VTFrameRateConversion. Forces `--output-backend videotoolbox`. | None |
| `--vsr-temporal-mode` | VTFRC quality: `normal` or `high` (QualityPrioritizationQuality). Only meaningful with `--vsr-target-fps`. | normal |
| `--vsr-save-original` | When VSR/VTFRC is engaged, also write the un-processed source-resolution source-fps mp4 as `<stem>_orig.mp4`. Mirrors the primary's HEVC profile (Main42210 4:2:2 for VSR HQ, Main10 4:2:0 for fast/VTFRC-only) so the A/B isn't precision-mismatched. Both files share one audio track. No-op when no VT post-processing is engaged. | False |
| `--vsr-encode-quality` | `AVVideoQualityKey` (0..1) for the AVAssetWriter HEVC encoder. Matches the ffmpeg default tier's `-q:v 65`. | 0.65 |
| `--generate-audio` | Generate synchronized audio (experimental) | False |
| `--low-memory` | Legacy emergency eval-cadence knob; usually redundant with distilled streaming runs | False |
| `--save-latents` | Save video/audio latents as an NPZ sidecar next to the output; distilled two-stage runs include stage-1 and stage-2 latents plus final aliases | False |
| `--save-text-embeddings` | Save positive/negative text conditioning as an `_text.npz` sidecar next to the output; reload it with `--embedding` | False |
| `--save-run-log` | Save generation parameters, argv, output paths, and timings as an `_run.json` sidecar, created at run start and finalized on completion | False |
| `--save-all-sidecars` | Enable latent, text conditioning, run metadata, audio WAV, and (when VSR/VTFRC is engaged) pre-VSR original mp4 sidecars together | False |
| `--skip-vae` | Skip VAE decoding (output latent visualization) | False |
| `--no-gemma` | Use dummy embeddings (testing only) | False |
| `--embedding` | Path to pre-computed text embedding (.npz) | None |
| `--gemma-path` | Path to Gemma 3 weights; default resolves cached `google/gemma-3-12b-it` from `HF_HOME` / `HF_HUB_CACHE` | HF cache |

## Precision Policy

- BF16 is the default compute dtype for model execution.
- `--dtype float16` and `--dtype float32` are available for experiments.
- Distilled defaults to CFG 1.0 and 8 steps; dev defaults to CFG 5.0 and 30 steps.
- Scheduler/time/position math and tiled VAE blending keep FP32 where needed for stability.
- Audio VAE decode and the plain vocoder follow the configured compute dtype.
- LTX-2.3 Vocoder+BWE keeps a scoped FP32 island, matching the Lightricks BWE precision caution.
- The VAE is native MLX-Conv3d (encoder and decoder) with zero spatial padding and RAM-aware auto tiling.  Three knobs were retired 2026-05-23: `--vae-decoder legacy` (the `SimpleVideoDecoder` PyTorch-layout slice-conv backend), `--vae-spatial-padding reflect` (lost in every A/B against `zero`), and the encoder-side `SimpleVideoEncoder` per-temporal-slice path.  The retired source remains available in git history.
- Converted-weight caching, a 1GB MLX allocator cache limit, same-math video projection pretranspose layouts, and native Conv3d/zero/auto VAE decode are enabled by default.  Use `--weights-cache off`, `--mlx-cache-limit-gb 0`, or `--video-ff-layout off --video-attn-layout off` for focused baselines.
- The `default` encode tier (HEVC HW Main10 + ALAC) auto-routes through `AVAssetWriter` (no ffmpeg dependency). Other tiers still go through ffmpeg. Force the legacy ffmpeg HEVC path with `--output-backend ffmpeg` for A/B comparisons.

## VideoToolbox post-processing

Two optional stages run between VAE decode and the encoder when
`--output-backend` resolves to `videotoolbox`. Both are off by default
and stream frames straight from the decoder into AVAssetWriter — no
disk round-trip, no ffmpeg.

- **`--vsr-spatial-mode {fast,balanced,image}`** — VideoToolbox Super
  Resolution. `fast` uses `VTLowLatencySuperResolutionScaler` (2x,
  input ≤ 960x960, NV12 source). `balanced` and `image` use
  `VTSuperResolutionScaler` (4x, RGBAHalf source, downloadable model
  fetched once on first use); `balanced` chains prev-frame state for
  crisper motion, `image` is per-frame deterministic with measurably
  smoother frame-to-frame detail (lower temporal second-difference).
- **`--vsr-target-fps FLOAT`** — `VTFrameRateConversion` to the
  requested rate. Source rate is `--fps`; identity rates skip the
  stage. Tested ratios include 24→48, 24→60, 30→24.
- **`--vsr-save-original`** — write a second mp4 at source resolution
  and source fps alongside the VSR/VTFRC output (`<stem>_orig.mp4`).
  The companion writer mirrors the primary's HEVC profile so the A/B
  is precision-floor-matched: VSR HQ (`balanced`/`image`) -> RGBAHalf
  source + Main42210 (4:2:2 10-bit); VSR `fast` / VTFRC-only -> NV12
  + Main10 (4:2:0 10-bit).  Companion AVAssetWriter runs on its own
  GCD queue, sharing the same `AudioTrack`, so the second encode is
  largely parallel to the primary.  Per-frame overhead is one source
  buffer upload + one HEVC HW pass.  Useful for A/B comparisons
  against the upscaled version without re-running the model.

These are independent of `--upscale-spatial` / `--upscale-temporal`
(transformer LoRA upscalers). The model-based path stays available
for cases where its quality profile is preferred; the VT path is
typically 100x+ faster for the same scale factor.

## Frame Count

LTX-2 requires frames to satisfy `frames % 8 == 1`:
- Valid: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121
- Formula: latent_frames = 1 + (frames - 1) / 8
