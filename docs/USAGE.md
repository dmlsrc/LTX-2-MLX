# LTX-2 MLX Usage Guide

Complete guide for generating videos with LTX-2 on Apple Silicon.

## Quick Start

```bash
# Generate a video with default settings
ltx2mlx "A golden retriever running through a meadow"

# Generate with specific settings
ltx2mlx "A rocket launching into space" \
    --height 512 --width 768 \
    --frames 65 --steps 8 \
    --seed 42 --output rocket.mp4
```

## Installation

### Prerequisites

- macOS with Apple Silicon (developed and tested on M1 Max, 64 GB)
- Python 3.14+
- ~25GB available RAM (64GB recommended for high resolution)
- ffmpeg (optional, only needed for non-default/software encode tiers; the default uses native VideoToolbox/AVAssetWriter)

### Setup

```bash
# Clone the repository
git clone https://github.com/dmlsrc/LTX-2-MLX.git
cd LTX-2-MLX

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# Or using pip
pip install -e .

# Optional: install ffmpeg (only needed for non-default/software encode tiers;
# the default encode path uses native VideoToolbox/AVAssetWriter)
brew install ffmpeg
```

### Download Weights

Fetch the weights with the HuggingFace CLI (`hf` ships with `huggingface_hub` -
`uv pip install huggingface_hub` if you don't have it), or download the files by
hand from each model's "Files" tab on huggingface.co. Accept the Gemma license
first at https://huggingface.co/google/gemma-3-12b-it.

```bash
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors
hf download google/gemma-3-12b-it
```

At runtime, `LTX_2_MLX/generate.py` resolves cached LTX-2.3 and Gemma weights from
`HF_HUB_CACHE`, `$HF_HOME/hub`, then the normal user cache
(`~/.cache/huggingface/hub`). Use `--weights` or `--gemma-path` only when you
want to override that cache lookup.

`--weights` is treated as a full bundle for the normal path. Advanced runs can
override individual subsystems without changing the rest of the bundle:

```bash
ltx2mlx "Your prompt" \
    --weights /path/to/full-ltx-bundle.safetensors \
    --transformer-weights /path/to/transformer-only.safetensors
```

The available split overrides are `--transformer-weights`,
`--connector-weights`, `--vae-weights`, `--audio-vae-weights`,
`--vocoder-weights`, and `--config-weights`. Keep `--weights` pointed at a full
bundle unless you also provide `--config-weights`, since model version, VAE
shape, and vocoder type are read from the config source.

Available weights from [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3):

| Weight | Size | Description |
|--------|------|-------------|
| `ltx-2.3-22b-distilled-1.1.safetensors` | ~46GB | 22B distilled, 8 steps (default) |
| `ltx-2.3-22b-dev.safetensors` | ~46GB | 22B dev, higher quality (more steps) |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~950MB | 2x spatial upscaler (two-stage) |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | ~1.5GB | distilled two-stage LoRA |

## Generation Options

### Resolution

Resolution is `WIDTHxHEIGHT` (the form the CLI prints). Both dimensions must be
divisible by **32** for the single-stage pipelines (`text-to-video`, `one-stage`)
and by **64** for `distilled` and `two-stage` -- the spatial upscaler runs stage 1
at half size then doubles it, so the final canvas must be a multiple of 64. For
the usual `--pipeline distilled` workflow, pick a /64 row.

| Resolution (WxH) | Aspect | /64 (distilled / two-stage) | Use case |
|------------------|--------|-----------------------------|----------|
| 352x192  | 11:6 | no (single-stage only) | Tiny / quick test |
| 512x288  | 16:9 | no (single-stage only) | Default; fast preview |
| 576x320  | 9:5  | yes | Fast distilled |
| 768x448  | 12:7 | yes | Balanced distilled |
| 1024x576 | 16:9 | yes | High-quality distilled |

These are the sizes that actually recur in this project's runs (512x288 is the CLI
default). 352x192 and 512x288 are not multiples of 64, so they only work with the
single-stage pipelines, not `distilled` / `two-stage`.

```bash
# distilled requires /64 dimensions:
ltx2mlx "Your prompt" --pipeline distilled --height 448 --width 768
```

### Frame Count

Easiest: pass `--duration SECONDS` and the frame count is computed for you -- it
takes `ceil(seconds * fps)` and rounds up to the next valid `8*k + 1`, so the clip
covers at least the requested time (the run log prints the resolved value, e.g.
`Resolved duration 30.0s at 24.0fps to 721 frames`).

To set the count directly, `--frames` must be a valid `8*k + 1` value (otherwise
generation errors):

| Frames | Duration (24fps) | Latent Frames |
|--------|------------------|---------------|
| 49 | 2.0s | 7 |
| 121 | 5.0s | 16 |
| 241 | 10.0s | 31 |
| 361 | 15.0s | 46 |
| 481 | 20.0s | 61 |
| 721 | 30.0s | 91 |

20s (481) and 30s (721) clips are the most common in practice.

```bash
ltx2mlx "Your prompt" --duration 20          # computes 481 frames; or --frames 481
```

### Steps

More steps = higher quality but slower:

| Model | Recommended Steps |
|-------|-------------------|
| Distilled | 8 by default |
| Dev | 30 by default, 25-50 for experiments |

```bash
ltx2mlx "Your prompt" --steps 8
```

CFG is also model-aware by default: distilled runs at CFG `1.0`, while dev runs
at CFG `5.0`. Override `--cfg` only when you are intentionally A/B testing
guidance.

### Seed

Control randomness for reproducible results:

```bash
ltx2mlx "Your prompt" --seed 42
```

## Pipelines

### Distilled Two-Stage Pipeline

Optimized for speed with no-CFG two-stage generation:

```bash
ltx2mlx "A cat walking through a garden" \
    --pipeline distilled \
    --height 512 --width 768 \
    --frames 65
```

**Characteristics:**
- No CFG (classifier-free guidance)
- 8 distilled steps at half resolution, then 3 refinement steps after spatial upscaling
- Fast generation (~2 minutes for 65 frames at 512x768)
- Good quality for most use cases

### Stage-2 Harness

Use `scripts/stage2_harness.py` to A/B the full-resolution distilled stage-2
path without rerunning prompt encoding or stage 1:

```bash
python scripts/stage2_harness.py \
    --stage1-latents /path/to/exact_run.npz \
    --embedding /path/to/exact_run_text.npz \
    --seed 124 \
    --generate-audio \
    --stream-transformer \
    --save-all-sidecars \
    --output-prefix ltx_stage2_ab
```

The harness infers frame count and final resolution from
`stage_1_video_latent`, loads adjacent run metadata when present, burns the
stage-1 RNG draws by default for same-seed parity, then runs spatial upscale,
stage-2 denoise, VAE/audio decode, and export.

### One-Stage Pipeline

Full CFG control for maximum quality:

```bash
ltx2mlx "A majestic eagle soaring over mountains" \
    --pipeline one-stage \
    --frames 65 --steps 20 \
    --cfg 5.0
```

**Characteristics:**
- CFG scale controls prompt adherence (3.0-7.0 typical)
- Higher steps = better quality for dev checkpoints
- Slower than distilled
- With `--model-variant distilled`, this is the existing single-pass distilled path with the fixed 8-step distilled sigma schedule

### Two-Stage Pipeline

High-resolution output with upscaling:

```bash
ltx2mlx "A waterfall in a tropical forest" \
    --pipeline two-stage \
    --height 768 --width 1024 \
    --frames 65 \
    --steps-stage1 15 \
    --cfg 5.0 \
    --dtype bfloat16
```

**Characteristics:**
- Stage 1: Generate at half resolution with CFG
- Stage 2: 2x spatial upscale + refinement
- Best for high-resolution output

## Memory Optimization

### Compute Dtype

BF16 is the default and is usually the best balance of memory and quality:

```bash
ltx2mlx "Your prompt" --dtype bfloat16
```

Precision exceptions are intentionally narrow:

- Scheduler/time/position math and tiled VAE blending keep FP32 where needed for stability.
- Audio VAE decode and the plain vocoder follow `--dtype`.
- LTX-2.3 Vocoder+BWE keeps a scoped FP32 island, matching the Lightricks BWE precision caution.

### VAE Decode Defaults

Video decode uses the native MLX-Conv3d VAE encoder + decoder with zero
spatial padding and `--vae-tiling auto`'s RAM-aware tile plan.  Pass
nothing extra to use the production default.

Historical retirements (2026-05-23):
- The `SimpleVideoDecoder` (PyTorch-layout slice-conv decoder) was
  removed from the working tree and remains available in git history; the
  `--vae-decoder` CLI flag was later removed entirely (native Conv3d is the
  only decoder).
- The `SimpleVideoEncoder` was removed from the working tree and replaced by
  `NativeConv3dVideoEncoder` (parity verified at cos sim 0.99965 FP32; ~2-3x
  faster).
- The `--vae-spatial-padding` flag was removed entirely.  A/B testing
  showed `reflect` produced worse boundary artifacts than `zero` in
  every workflow tested (motion-heavy bakery, talking-subject clips,
  static images), with no meaningful decode-time difference.  Zero is
  hardcoded.

### Transformer Streaming

For the common low-RAM transformer path, use the preset instead of spelling out
the resident-block knobs:

```bash
ltx2mlx "Your prompt" --stream-transformer
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
ltx2mlx "Your prompt" --low-memory
```

### Sequential Loading

Models are loaded/unloaded sequentially by default:
1. Gemma 3 → encode text → unload
2. Transformer → denoise → unload
3. VAE → decode video

## Example Prompts

### Nature Scenes
```bash
ltx2mlx "Ocean waves crashing on a sandy beach at sunset, golden hour lighting"
ltx2mlx "A serene forest with sunlight filtering through the trees"
ltx2mlx "Snow falling gently in a mountain landscape"
```

### Action Sequences
```bash
ltx2mlx "A golden retriever running through a sunny meadow with wildflowers"
ltx2mlx "A rocket ship launching into space with flames and smoke"
ltx2mlx "A sports car racing on a winding mountain road"
```

### Urban/Architectural
```bash
ltx2mlx "A bustling city street at night with neon lights and rain reflections"
ltx2mlx "Time-lapse of clouds moving over a modern cityscape"
```

### Cinematic Style
```bash
ltx2mlx "Cinematic shot of a lone astronaut on Mars, dramatic lighting"
ltx2mlx "Epic wide shot of a medieval castle at dawn"
```

## Prompt Tips

1. **Be descriptive**: Include details about lighting, setting, and mood
2. **Specify motion**: Describe what's moving and how
3. **Use cinematic terms**: "cinematic", "dramatic lighting", "wide shot"
4. **Mention style**: "photorealistic", "animation style", "oil painting"

## Output

Videos are written at 24fps via one of two backends, controlled by
`--output-backend` (default `auto`):

- `ffmpeg` — `LTX_2_MLX.ffmpeg_encoder.encode_video_ffmpeg()`.  Handles every
  tier and is the default for non-HEVC tiers (`web`, `hq`, `export`,
  `reference`).
- `videotoolbox` — `LTX_2_MLX.videotoolbox.encode.encode_video_videotoolbox()`.
  Pure AVAssetWriter; no ffmpeg dependency.  Supports the `default`
  tier (HEVC Main10 4:2:0 + ALAC) and is also the only backend that
  can run the optional VSR / VTFRC stages.

In `auto` mode the dispatcher picks `videotoolbox` whenever the run
maps onto AVAssetWriter (`--encode-tier default`, or any `--vsr-*`
flag engaged); otherwise it falls back to ffmpeg.

| Tier        | Codec / Container                                | Backend (auto)  | Audience                                |
|-------------|--------------------------------------------------|-----------------|-----------------------------------------|
| `web`       | H.264 SW 8-bit 4:2:0 CRF 18 + AAC 320k (.mp4)    | ffmpeg          | Universal browser/player compat          |
| `default`   | HEVC HW 10-bit 4:2:0 q=65 + ALAC (.mp4)          | **videotoolbox**| Everyday output on Apple / modern browsers |
| `hq`        | HEVC SW 10-bit 4:4:4 CRF 14 + ALAC (.mp4)        | ffmpeg          | Local viewing, full chroma (no browser support) |
| `export`    | ProRes 422 HQ + PCM 24-bit (.mov)                | ffmpeg          | Editor / colorist hand-off              |
| `reference` | ProRes 4444 + PCM 24-bit + alpha (.mov)          | ffmpeg          | Canonical highest-fidelity copy         |

The container extension (`.mp4` / `.mov`) is forced by the tier; if
`--output` specifies a `.mp4` for a ProRes tier, the ffmpeg encoder
rewrites it to `.mov`.

```bash
# Default timestamped output (default tier -> AVAssetWriter HEVC + ALAC,
# no ffmpeg involved).
ltx2mlx "Your prompt"

# Force the ffmpeg backend even for the default tier (useful for A/B).
ltx2mlx "Your prompt" --output-backend ffmpeg

# Web-compat output (H.264 + AAC) for browser embed — ffmpeg only.
ltx2mlx "Your prompt" --encode-tier web

# Editor-grade output (ProRes 422 HQ in .mov) — ffmpeg only.
ltx2mlx "Your prompt" --encode-tier export

# Custom output directory and filename prefix
ltx2mlx "Your prompt" \
    --output-dir /path/to/outputs \
    --output-prefix ltx_bakery_r16

# Exact output path override
ltx2mlx "Your prompt" --output my_video.mp4
```

### VideoToolbox post-processing (VSR, VTFRC)

When the `videotoolbox` backend is active, two optional post-VAE
stages can be inserted between the decoded frames and the encoder.
Both are off by default; engaging either forces the VT backend.

- `--vsr-spatial-mode {off,fast,balanced,image}` — VideoToolbox Super
  Resolution.  `fast` is `VTLowLatencySuperResolutionScaler` (2x,
  input <= 960x960).  `balanced` and `image` are
  `VTSuperResolutionScaler` (4x, downloadable model on first use;
  `balanced` uses prev-frame feedback for crisper motion; `image` is
  per-frame deterministic).
- `--vsr-target-fps FLOAT` - `VTFrameRateConversion` that interpolates to a
  higher output rate (e.g. 24->48 or 24->60) for smoother, high-refresh
  playback at the same clip duration.  It raises the frame rate; it does not
  slow the video down.  Source rate is `--fps`.
- `--vsr-temporal-mode {normal,high}` — VTFRC quality
  (`QualityPrioritizationQuality` when `high`).
- `--vsr-encode-quality FLOAT` — `AVVideoQualityKey` for the
  AVAssetWriter encoder (default `0.65`, matches the ffmpeg `default`
  tier's `-q:v 65`).
- `--vsr-save-original` — when VSR or VTFRC is engaged, ALSO write
  the un-processed source-resolution source-fps mp4 alongside the
  requested output as `<stem>_orig.mp4`.  Both files share the same
  audio track so each is playable standalone.  The companion writer
  mirrors the primary's precision envelope so the A/B comparison
  isn't skewed by encoder settings: VSR `balanced`/`image` (4x HQ)
  -> original gets RGBAHalf source + HEVC Main42210 (4:2:2 10-bit),
  matching the primary; VSR `fast` and VTFRC-only -> original gets
  NV12 + HEVC Main10 (4:2:0 10-bit), again matching.  The companion
  encode runs on its own AVAssetWriter GCD queue parallel to the
  primary one — per-frame cost is one extra source-format upload
  plus one HEVC HW pass, so wall-time overhead is small (typically
  a few extra seconds on a 30-second clip).  Useful for keeping a
  reference copy for A/B comparison without re-running the model.

These are independent of the model-based `--upscale-spatial` /
`--upscale-temporal` flags — the two systems should not be combined.

```bash
# Generate at 384x216 and let VideoToolbox upscale 4x to 1536x864
# with the HQ "Image" model (no prev-frame feedback -> smoother motion).
ltx2mlx "Your prompt" \
    --width 384 --height 216 \
    --vsr-spatial-mode image

# Higher frame rate (smoother motion, same duration): interpolate 24->48fps via VTFRC.
ltx2mlx "Your prompt" \
    --vsr-target-fps 48 --vsr-temporal-mode high

# 4x VSR + 60fps interp, AAC instead of ALAC for size.
ltx2mlx "Your prompt" \
    --width 384 --height 216 \
    --vsr-spatial-mode balanced --vsr-target-fps 60

# Save both the 4x VSR upscale AND the source-resolution original
# in one model run, for side-by-side comparison.  Outputs:
#   sample.mp4      (1536x864, 4x VSR)
#   sample_orig.mp4 (384x216, no VSR)
ltx2mlx "Your prompt" \
    --width 384 --height 216 \
    --vsr-spatial-mode image --vsr-save-original \
    --output outputs/sample.mp4
```

When `--output` is omitted, the output directory resolves in this order:
`--output-dir`, `DIFFUSERS_OUTPUT_DIR`, `OUTPUT_DIR`, then `outputs/`.
Sidecars use the same timestamped stem as the encoded video.

### Latent Sidecars

Use `--save-latents` to write final video/audio latents as an NPZ sidecar next to
the requested output. The sidecar uses the same basename as the MP4.

```bash
ltx2mlx "Your prompt" \
    --generate-audio \
    --save-latents \
    --output outputs/sample.mp4
```

Decode-only validation can read that sidecar without rerunning denoising:

```bash
python scripts/decode_latent_debug.py \
    --latent outputs/sample.npz \
    --weights weights/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --modes auto \
    --decode-audio \
    --show-memory \
    --output-dir outputs/decode_tests
```

For VAE boundary/padding A/B checks, `probe_vae_boundary.py` can mux the saved
audio into each variant so the results can be judged as complete clips:

```bash
python scripts/probe_vae_boundary.py \
    --latent outputs/sample.npz \
    --weights weights/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
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
ltx2mlx "Your prompt" \
    --generate-audio \
    --save-text-embeddings \
    --output outputs/sample.mp4
```

Use `--save-run-log` to write a human-readable `_run.json` sidecar with the
exact command-line arguments, prompt, generation parameters, sidecar paths, output
paths, and timing summary for the run. The sidecar is created at run start with
`status: "started"` and overwritten with completed timings when the output is
saved, so interrupted long runs still leave their parameters behind.

Use `--save-audio-sidecar` to keep the vocoder's WAV output alongside the
encoded video. The encoded audio inside the video container is whatever codec
the tier specifies (ALAC for `default`/`hq`, AAC for `web`, PCM for ProRes
tiers); the sidecar is always uncompressed WAV at the vocoder's native sample
rate. Useful for A/B comparisons against the codec-compressed audio.

Use `--save-all-sidecars` to enable every sidecar together. It is equivalent to
passing:

```bash
--save-latents --save-text-embeddings --save-run-log --save-audio-sidecar \
--vsr-save-original
```

`--vsr-save-original` is a no-op unless `--vsr-spatial-mode` or
`--vsr-target-fps` is also set, so including it in the "all sidecars"
bundle costs nothing when VSR/VTFRC is not engaged.

### Sequence-start audio onset trim

Some AV generations produce a short, near-clipping transient at t=0,
followed by silence before the first spoken word. The default
`--audio-onset-trim auto` runs a two-window detector (first 50 ms > 2x
global RMS AND 100-250 ms < 0.1x global RMS) and zero-fills the leading
120 ms only when the click signature is present; clean clips pass
through untouched. Sample count is preserved, so audio and video stay
in sync to the millisecond.

```bash
# Default — detect, trim leading 120 ms only when the click is present
ltx2mlx "..." --generate-audio

# Disable the check (e.g. when re-deriving the raw vocoder output)
ltx2mlx "..." --generate-audio --audio-onset-trim off

# Force a specific trim duration in milliseconds (regardless of detection)
ltx2mlx "..." --generate-audio --audio-onset-trim 150
```

The cleaned waveform feeds both the muxed track and the optional
`.wav` sidecar. The trim runs *after* any latent save, so
`--save-latents` keeps the raw audio latent for reproduction with
`scripts/analyze_audio_onset.py`. The chosen mode is captured in the
run-log sidecar (`--save-run-log`) for post-hoc forensics.

## Troubleshooting

### Out of Memory

- Use the default BF16 compute dtype, or try `--dtype float16` for experiments
- Reduce resolution: `--height 256 --width 384`
- Reduce frames: `--frames 17`
- Use `--stream-transformer` for the r16/g4 compiled streaming preset
- Try `--low-memory` only as an emergency fallback if streaming is still too high

### Slow Generation

- Use `--pipeline distilled` for no-CFG two-stage distilled generation
- Use `--pipeline one-stage --model-variant distilled` for single-pass distilled generation
- Reduce steps on one-stage runs: `--steps 5`
- Reduce resolution and frames
- Use `--profile-transformer-steps 1,2,8` and optionally `--profile-transformer-blocks 40,47` when you need cold/warm transformer timing breakdowns
- The default `--mlx-cache-limit-gb 1` caps MLX's in-memory allocator cache. Use `--mlx-cache-limit-gb 0` only when testing stricter cache pressure.
- For experimental same-settings denoise A/Bs, try `--video-ff-quantize project_out:mxfp8` to replace video FF projections with MLX weight-only quantized linears. With `--stream-transformer`, selected FF tensors are quantized into the converted transformer cache and streamed as quantized weights; all-layer streaming quant can keep resident-group compile, while partial-layer streaming quant disables compile. Use `project_in:mxfp8` to test the FF input projection by itself, or `project_in:mxfp8,project_out:mxfp8` to test both. Add `--video-ff-quantize-layers 40-47` to test only selected 0-based layers. This is non-canonical and needs visual/audio validation.
- For a broader non-parity quantized-cache experiment, use `--stream-transformer --transformer-cache-quantize mxfp8-blocks`. This mirrors the downloaded MXFP8 block32 transformer policy in MLX-native cache form by quantizing heavy attention/FF block linears while leaving biases, norms, AdaLN tables, connector, patch/output projections, VAE, audio VAE, and vocoder in their normal precision. It disables same-math layout caches because the relevant weights are replaced by quantized linears. To test whether the lost pretranspose layout is the bottleneck, use `--transformer-cache-quantize mxfp8-blocks-pretranspose`; that packs `weight.T` and calls quantized matmul with `transpose=False`.
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

The old `DistilledPipeline` library wrapper has been archived. Programmatic
distilled two-stage generation now uses the same `AVPipeline` /
`AVPipeline.generate_distilled_two_stage` route as `LTX_2_MLX/generate.py`.
Use the CLI as the reference entry point unless you are wiring the model
components directly.

## Performance Benchmarks

See [docs/PERFORMANCE.md](PERFORMANCE.md) for current benchmarks and the
denoise-speed analysis (M1 Max, 64 GB).

## Text Encoding

### Automatic (Recommended)

The generation script automatically resolves cached `google/gemma-3-12b-it`
from `HF_HUB_CACHE`, `$HF_HOME/hub`, the shared Hugging Face cache, or the
normal user cache:

```bash
ltx2mlx "A cat walking through a garden"
```

### Dummy Embeddings (Testing)

For testing the pipeline without Gemma:

```bash
ltx2mlx "A cat walking" --no-gemma --height 128 --width 128
```

### Gemma 3 Requirements

LTX-2 requires **Gemma 3 12B** (~25GB download, ~12GB in memory at 16-bit precision):

```bash
hf download google/gemma-3-12b-it
```

**Requirements:**
1. Get a HuggingFace token at: https://huggingface.co/settings/tokens
2. Accept the Gemma license at: https://huggingface.co/google/gemma-3-12b-it

## Known Limitations

### Memory Requirements

| Configuration | RAM Required |
|--------------|--------------|
| Text-to-Video (512x288, 97 frames) | Depends on transformer residency; use `--stream-transformer` for constrained-memory runs |
| Two-Stage (960x1408, 97 frames) | ~44GB (sequential), ~59GB (parallel) |
| With Audio Generation | Add ~6GB |

**Recommendation**: Try `--stream-transformer` first on constrained systems. Use
`--low-memory` only when the extra eval checkpoints are worth the speed hit.

### Audio Generation

- `--generate-audio` is supported on the AudioVideo one-stage path and the distilled two-stage path.
- `--model-variant distilled` is supported; use `--pipeline one-stage` for single-pass distilled generation.
- The CFG `--pipeline two-stage` path still has its own older audio behavior.
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
