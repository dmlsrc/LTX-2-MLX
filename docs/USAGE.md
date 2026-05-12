# LTX-2 MLX Usage Guide

Complete guide for generating videos with LTX-2 on Apple Silicon.

## Quick Start

```bash
# Generate a video with default settings
python scripts/generate.py "A golden retriever running through a meadow"

# Generate with specific settings
python scripts/generate.py "A rocket launching into space" \
    --height 512 --width 768 \
    --frames 65 --steps 8 \
    --seed 42 --output rocket.mp4
```

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~25GB available RAM (128GB recommended for high resolution)
- ffmpeg for video encoding

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/LTX-2-MLX.git
cd LTX-2-MLX

# Install dependencies (using uv - recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install mlx safetensors numpy pillow tqdm einops transformers sentencepiece protobuf

# Install ffmpeg
brew install ffmpeg
```

### Download Weights

```bash
# Interactive download (recommended)
uv run scripts/download_weights.py

# Or download specific weights
uv run scripts/download_weights.py --weights distilled gemma

# Or download everything
uv run scripts/download_weights.py --weights all
```

At runtime, `scripts/generate.py` resolves cached LTX-2.3 and Gemma weights from
`HF_HUB_CACHE`, `$HF_HOME/hub`, `/Users/Shared/huggingface/hub`, then the normal
user cache. Use `--weights` or `--gemma-path` only when you want to override that
cache lookup.

`--weights` is treated as a full bundle for the normal path. Advanced runs can
override individual subsystems without changing the rest of the bundle:

```bash
python scripts/generate.py "Your prompt" \
    --weights /path/to/full-ltx-bundle.safetensors \
    --transformer-weights /path/to/transformer-only.safetensors
```

The available split overrides are `--transformer-weights`,
`--connector-weights`, `--vae-weights`, `--audio-vae-weights`,
`--vocoder-weights`, and `--config-weights`. Keep `--weights` pointed at a full
bundle unless you also provide `--config-weights`, since model version, VAE
shape, and vocoder type are read from the config source.

Available weights from [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2):

| Weight | Size | Description |
|--------|------|-------------|
| `ltx-2-19b-distilled.safetensors` | 43GB | Fast generation (8 steps) |
| `ltx-2-19b-dev.safetensors` | 43GB | Higher quality (25-50 steps) |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | 995MB | 2x resolution |
| `ltx-2-temporal-upscaler-x2-1.0.safetensors` | 262MB | 2x framerate |
| `ltx-2-19b-distilled-lora-384.safetensors` | 1.5GB | LoRA for two-stage |

## Generation Options

### Resolution

Height and width must be divisible by 32:

| Resolution | Aspect Ratio | Use Case |
|------------|--------------|----------|
| 256×384 | 2:3 | Fast testing |
| 288×512 | 16:9 | Default fast preview |
| 480×704 | ~2:3 | Taller balanced quality/speed |
| 512×768 | 2:3 | High quality |
| 768×1024 | 3:4 | Maximum quality |

```bash
python scripts/generate.py "Your prompt" --height 512 --width 768
```

### Frame Count

Frames must satisfy `frames % 8 == 1`:

| Frames | Duration (24fps) | Latent Frames |
|--------|------------------|---------------|
| 17 | 0.7s | 3 |
| 33 | 1.4s | 5 |
| 65 | 2.7s | 9 |
| 97 | 4.0s | 13 |
| 121 | 5.0s | 16 |

```bash
python scripts/generate.py "Your prompt" --frames 65
```

### Steps

More steps = higher quality but slower:

| Model | Recommended Steps |
|-------|-------------------|
| Distilled | 8 by default |
| Dev | 30 by default, 25-50 for experiments |

```bash
python scripts/generate.py "Your prompt" --steps 8
```

CFG is also model-aware by default: distilled runs at CFG `1.0`, while dev runs
at CFG `5.0`. Override `--cfg` only when you are intentionally A/B testing
guidance.

### Seed

Control randomness for reproducible results:

```bash
python scripts/generate.py "Your prompt" --seed 42
```

## Pipelines

### Distilled Pipeline (Default)

Optimized for speed with 8-step generation:

```bash
python scripts/generate.py "A cat walking through a garden" \
    --pipeline distilled \
    --height 512 --width 768 \
    --frames 65 --steps 8
```

**Characteristics:**
- No CFG (classifier-free guidance)
- Fast generation (~2 minutes for 65 frames at 512×768)
- Good quality for most use cases

### One-Stage Pipeline

Full CFG control for maximum quality:

```bash
python scripts/generate.py "A majestic eagle soaring over mountains" \
    --pipeline one-stage \
    --frames 65 --steps 20 \
    --cfg 5.0
```

**Characteristics:**
- CFG scale controls prompt adherence (3.0-7.0 typical)
- Higher steps = better quality
- Slower than distilled

### Two-Stage Pipeline

High-resolution output with upscaling:

```bash
python scripts/generate.py "A waterfall in a tropical forest" \
    --pipeline two-stage \
    --height 768 --width 1024 \
    --frames 65 \
    --steps-stage1 15 \
    --cfg 5.0 \
    --dtype bfloat16
```

**Characteristics:**
- Stage 1: Generate at half resolution with CFG
- Stage 2: 2× spatial upscale + refinement
- Best for high-resolution output

## Memory Optimization

### Compute Dtype

BF16 is the default and is usually the best balance of memory and quality:

```bash
python scripts/generate.py "Your prompt" --dtype bfloat16
```

Precision exceptions are intentionally narrow:

- Scheduler/time/position math and tiled VAE blending keep FP32 where needed for stability.
- Audio VAE decode and the plain vocoder follow `--dtype`.
- LTX-2.3 Vocoder+BWE keeps a scoped FP32 island, matching the Lightricks BWE precision caution.

### VAE Decode Defaults

Video decode defaults to the native Conv3d VAE decoder with
`--vae-spatial-padding zero` and `--vae-tiling auto`. That keeps the common
command short while using the RAM-aware native tiling planner. Use
`--vae-decoder simple` or `--vae-spatial-padding reflect` only when you want an
A/B baseline against the older decode path:

```bash
python scripts/generate.py "Your prompt" \
    --vae-decoder simple \
    --vae-spatial-padding reflect
```

Saved-latent A/B tests on motion-heavy bakery and talking-subject clips showed
`zero` substantially reduced edge ghosting, background flicker, and boundary
smearing versus `reflect`, with no meaningful decode-time cost.

### Transformer Streaming

For the common low-RAM transformer path, use the preset instead of spelling out
the resident-block knobs:

```bash
python scripts/generate.py "Your prompt" --stream-transformer
```

This enables 16 resident transformer blocks, resident-group compile, and
4-block compile groups. It pairs with the default converted-weight cache and the
default 1GB MLX allocator cache cap.

The converted-weight cache is split by semantic weight family. Transformer
caches are keyed by transformer source and layout options; connector, video VAE,
audio VAE, and vocoder caches are keyed by their own source files. That makes it
reasonable to test a transformer-only checkpoint against stock VAE/audio/vocoder
weights without duplicating unrelated cache blobs.

### Low Memory Mode

`--low-memory` is still available as an emergency knob. It adds more frequent
eval checkpoints and sequential CFG materialization in older guided paths, but
it is usually slower and mostly redundant for distilled single-pass runs using
`--stream-transformer`:

```bash
python scripts/generate.py "Your prompt" --low-memory
```

### Sequential Loading

Models are loaded/unloaded sequentially by default:
1. Gemma 3 → encode text → unload
2. Transformer → denoise → unload
3. VAE → decode video

## Example Prompts

### Nature Scenes
```bash
python scripts/generate.py "Ocean waves crashing on a sandy beach at sunset, golden hour lighting"
python scripts/generate.py "A serene forest with sunlight filtering through the trees"
python scripts/generate.py "Snow falling gently in a mountain landscape"
```

### Action Sequences
```bash
python scripts/generate.py "A golden retriever running through a sunny meadow with wildflowers"
python scripts/generate.py "A rocket ship launching into space with flames and smoke"
python scripts/generate.py "A sports car racing on a winding mountain road"
```

### Urban/Architectural
```bash
python scripts/generate.py "A bustling city street at night with neon lights and rain reflections"
python scripts/generate.py "Time-lapse of clouds moving over a modern cityscape"
```

### Cinematic Style
```bash
python scripts/generate.py "Cinematic shot of a lone astronaut on Mars, dramatic lighting"
python scripts/generate.py "Epic wide shot of a medieval castle at dawn"
```

## Prompt Tips

1. **Be descriptive**: Include details about lighting, setting, and mood
2. **Specify motion**: Describe what's moving and how
3. **Use cinematic terms**: "cinematic", "dramatic lighting", "wide shot"
4. **Mention style**: "photorealistic", "animation style", "oil painting"

## Output

Videos are saved as MP4 files with H.264 encoding at 24fps:

```bash
# Default timestamped output location
python scripts/generate.py "Your prompt"
# → saves to outputs/ltx_YYYYmmdd_HHMMSS.mp4

# Custom output directory and filename prefix
python scripts/generate.py "Your prompt" \
    --output-dir /Users/Shared/huggingface/output \
    --output-prefix ltx_bakery_r16

# Exact output path override
python scripts/generate.py "Your prompt" --output my_video.mp4
```

When `--output` is omitted, the output directory resolves in this order:
`--output-dir`, `DIFFUSERS_OUTPUT_DIR`, `OUTPUT_DIR`, then `outputs/`.
Sidecars use the same timestamped stem as the MP4.

### Latent Sidecars

Use `--save-latents` to write final video/audio latents as an NPZ sidecar next to
the requested output. The sidecar uses the same basename as the MP4.

```bash
python scripts/generate.py "Your prompt" \
    --generate-audio \
    --save-latents \
    --output outputs/sample.mp4
```

Decode-only validation can read that sidecar without rerunning denoising:

```bash
python scripts/decode_latent_debug.py \
    --latent outputs/sample.npz \
    --weights weights/ltx-2/ltx-2.3-22b-distilled-1.1.safetensors \
    --modes auto \
    --vae-spatial-padding zero \
    --decode-audio \
    --show-memory \
    --output-dir outputs/decode_tests
```

For VAE boundary/padding A/B checks, `probe_vae_boundary.py` can mux the saved
audio into each variant so the results can be judged as complete clips:

```bash
python scripts/probe_vae_boundary.py \
    --latent outputs/sample.npz \
    --weights weights/ltx-2/ltx-2.3-22b-distilled-1.1.safetensors \
    --variants orig orig_zero_convpad \
    --decode-audio \
    --output-dir outputs/boundary_probe
```

The useful comparison is usually `orig` versus `orig_zero_convpad`: those modes
decode the same latent while changing only the decoder's spatial padding policy.

Use `--save-text-embeddings` to write the positive/negative text conditioning as
a separate `_text.npz` sidecar. This captures the video/audio text encoder
outputs and masks after Gemma and the AV text encoder, which is useful when
checking whether text-conditioning precision changes alter denoising inputs.
Pass that `_text.npz` back through `--embedding` to reuse the saved conditioning
without loading Gemma again. Legacy embedding NPZs with `embedding` and
`attention_mask` are still supported, but they do not carry audio conditioning.

```bash
python scripts/generate.py "Your prompt" \
    --generate-audio \
    --save-text-embeddings \
    --output outputs/sample.mp4
```

Use `--save-run-log` to write a human-readable `_run.json` sidecar with the
exact command-line arguments, prompt, generation parameters, sidecar paths, output
paths, and timing summary for the run. The sidecar is created at run start with
`status: "started"` and overwritten with completed timings when the output is
saved, so interrupted long runs still leave their parameters behind.

Use `--save-all-sidecars` to enable final latents, text conditioning, and the
run log together. It is equivalent to passing:

```bash
--save-latents --save-text-embeddings --save-run-log
```

## Troubleshooting

### Out of Memory

- Use the default BF16 compute dtype, or try `--dtype float16` for experiments
- Reduce resolution: `--height 256 --width 384`
- Reduce frames: `--frames 17`
- Use `--stream-transformer` for the r16/g4 compiled streaming preset
- Try `--low-memory` only as an emergency fallback if streaming is still too high

### Slow Generation

- Use `--pipeline distilled` (fastest)
- Reduce steps: `--steps 5`
- Reduce resolution and frames
- Use `--profile-transformer-steps 1,2,8` and optionally `--profile-transformer-blocks 40,47` when you need cold/warm transformer timing breakdowns
- The default `--mlx-cache-limit-gb 1` caps MLX's in-memory allocator cache. Use `--mlx-cache-limit-gb 0` only when testing stricter cache pressure.
- For experimental same-settings denoise A/Bs, try `--video-ff-quantize project_out:mxfp8` to replace video FF projections with MLX weight-only quantized linears after loading stock BF16 weights. Use `project_in:mxfp8` to test the FF input projection by itself, or `project_in:mxfp8,project_out:mxfp8` to test both. Add `--video-ff-quantize-layers 40-47` to test only selected 0-based layers. This is non-canonical and needs visual/audio validation.
- Same-math FF and attention pretranspose layouts are enabled by default. Use `--video-ff-layout off --video-attn-layout off` when you need a baseline A/B against untransposed stock layout.
- For same-settings denoise-speed research, see [Performance Optimization Notes](PERFORMANCE.md)

### Video Quality Issues

- Increase steps: `--steps 8` or higher
- Use `--pipeline one-stage` with `--cfg 5.0`
- Try different seeds: `--seed 123`
- Use more descriptive prompts

### Black/Dark Output

This is typically a timestep conditioning issue. Ensure you're using the latest code with proper VAE timestep handling.

## API Usage

```python
from LTX_2_MLX.pipelines import DistilledPipeline
from LTX_2_MLX.model.text_encoder import create_av_text_encoder
from LTX_2_MLX.model.transformer import LTXModel, X0Model
from LTX_2_MLX.model.video_vae import SimpleVideoDecoder

# Load models
text_encoder = create_av_text_encoder()
transformer = LTXModel(...)
x0_model = X0Model(transformer)
vae_decoder = SimpleVideoDecoder()

# Create pipeline
pipeline = DistilledPipeline(
    text_encoder=text_encoder,
    transformer=x0_model,
    video_decoder=vae_decoder,
)

# Generate
video = pipeline(
    prompt="A cat walking through a garden",
    height=512,
    width=768,
    num_frames=65,
    num_steps=8,
    seed=42,
)
```

## Performance Benchmarks

Measured on M3 Max with 128GB unified memory:

| Resolution | Frames | Steps | Time |
|------------|--------|-------|------|
| 512×768 | 65 | 8 | ~2 min |
| 768×1024 | 65 | 8 | ~4 min |
| 512×768 | 97 | 8 | ~3 min |

VAE decoding adds ~10-15 seconds regardless of resolution.

## Text Encoding

### Automatic (Recommended)

The generation script automatically resolves cached `google/gemma-3-12b-it`
from `HF_HUB_CACHE`, `$HF_HOME/hub`, the shared Hugging Face cache, or the
normal user cache:

```bash
python scripts/generate.py "A cat walking through a garden"
```

### Dummy Embeddings (Testing)

For testing the pipeline without Gemma:

```bash
python scripts/generate.py "A cat walking" --no-gemma --height 128 --width 128
```

### Gemma 3 Requirements

LTX-2 requires **Gemma 3 12B** (~25GB download, ~12GB in memory at 16-bit precision):

```bash
uv run scripts/download_weights.py --weights gemma
```

**Requirements:**
1. Get a HuggingFace token at: https://huggingface.co/settings/tokens
2. Accept the Gemma license at: https://huggingface.co/google/gemma-3-12b-it

## Known Limitations

### Memory Requirements

| Configuration | RAM Required |
|--------------|--------------|
| Text-to-Video (512×288, 97 frames) | Depends on transformer residency; use `--stream-transformer` for constrained-memory runs |
| Two-Stage (960×1408, 97 frames) | ~44GB (sequential), ~59GB (parallel) |
| With Audio Generation | Add ~6GB |

**Recommendation**: Try `--stream-transformer` first on constrained systems. Use
`--low-memory` only when the extra eval checkpoints are worth the speed hit.

### Audio Generation

- `--generate-audio` uses the AudioVideo one-stage path from the default CLI flow.
- `--model-variant distilled` is supported; `--pipeline distilled` is a separate video-only pipeline.
- Two-stage pipeline modes do not support audio generation.
- LTX-2.3 audio quality is usable in current smoke tests, but still benefits from decode-only checks with `--save-latents` when changing precision or tiling code.

## Current Status

### Working

- Full PyTorch parity verified (97%+ correlation)
- Text-to-video generation with semantic content
- 6 specialized pipelines
- Two-stage pipeline with spatial upscaling
- Temporal upscaler (`--upscale-temporal`)
- IC-LoRA conditioning (`--control-video`)
- Generic LoRA support (`--lora`)
- Resolutions up to 768x1024

### In Progress

- Audio-video joint generation with two-stage pipeline

### Pending

- Keyframe interpolation CLI integration
