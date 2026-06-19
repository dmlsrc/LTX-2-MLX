# LTX-2-MLX

[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![Original Repo](https://img.shields.io/badge/GitHub-LTX--2-181717?logo=github)](https://github.com/Lightricks/LTX-2)

Runs Lightricks LTX-2.3 (22B distilled two-stage, synchronized audio+video) natively on Apple Silicon via MLX.

<table>
<tr>
<td width="50%" align="center">

**"A golden retriever running through a sunny meadow"**

https://github.com/user-attachments/assets/59290fef-6900-460f-afe9-e24a40de01bb

</td>
<td width="50%" align="center">

**"A city street at night with neon lights and rain"**

https://github.com/user-attachments/assets/245e54ac-aff2-446f-a717-8797d0cc50c0

</td>
</tr>
<tr>
<td width="50%" align="center">

**"A rocket ship launching into space with flames"**

https://github.com/user-attachments/assets/f84284da-ea6d-4ce3-9b09-3ad773d7edc2

</td>
<td width="50%" align="center">

**"Ocean waves crashing on a beach at sunset"**

https://github.com/user-attachments/assets/a508dc05-6d02-453e-9ee9-07878613a137

</td>
</tr>
</table>

*768x512, 65 frames (~2.7s at 24fps), 8 steps on Apple Silicon*

## Quick Start

```bash
# 1. Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# 2. Download weights with the HuggingFace CLI. `hf` ships with huggingface_hub
#    (`uv pip install huggingface_hub` if you don't have it), or grab the files
#    by hand from each model's "Files" tab on huggingface.co. Accept the Gemma
#    license first: https://huggingface.co/google/gemma-3-12b-it
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors
hf download google/gemma-3-12b-it

# 3. Generate (auto-resolves the cached LTX-2.3 + Gemma weights from the HF cache)
ltx2mlx "A golden retriever running through a meadow"
```

## Required Models

All weights live on HuggingFace. Fetch them with the `hf` CLI (it ships with
`huggingface_hub` - `uv pip install huggingface_hub` if you don't have it), or
download any file by hand from the model's "Files" tab. Downloaded files land in
the HuggingFace cache, which `ltx2mlx` resolves automatically.

### LTX-2.3 (recommended) - [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3)

| File | Size | Description |
|------|------|-------------|
| `ltx-2.3-22b-distilled-1.1.safetensors` | ~46GB | 22B distilled, 8 steps (default) |
| `ltx-2.3-22b-dev.safetensors` | ~46GB | 22B dev, higher quality (more steps) |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~950MB | 2x spatial upscaler (two-stage) |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | ~1.5GB | distilled two-stage LoRA |

```bash
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors
hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors
```

**Text Encoder**: [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it)
(~25GB, gated) - accept the license, then `hf download google/gemma-3-12b-it`.

## Available Pipelines

| Pipeline | Speed | Quality | Best For |
|----------|-------|---------|----------|
| `text-to-video` | Medium | Good | Basic generation |
| `distilled` | **Fast** | Good | No-CFG two-stage quick iteration |
| `one-stage` | Slow | **High** | Quality priority or single-pass distilled |
| `two-stage` | Medium | **High** | High resolution (512p+) |

```bash
# Fast two-stage distilled preview
ltx2mlx "Your prompt" --pipeline distilled

# Existing single-pass distilled path
ltx2mlx "Your prompt" --pipeline one-stage --model-variant distilled

# High quality dev-style sampling
ltx2mlx "Your prompt" --pipeline one-stage --steps 20 --cfg 5.0

# High resolution
ltx2mlx "Your prompt" --pipeline two-stage --height 768 --width 1024
```

See [Pipelines Guide](docs/PIPELINES.md) for all 6 pipelines and options.

## Optimization Tips

- **Use default BF16 compute** - override with `--dtype float16` or `--dtype float32` only for experiments
- **Audio follows compute dtype where safe** - LTX-2.3 Vocoder+BWE keeps a scoped FP32 island matching Lightricks' precision caution
- **VAE decode uses native Conv3d + zero padding** - `--vae-tiling auto` picks a RAM-aware native tile plan.  `--vae-decoder` and `--vae-tiling` remain user knobs; `--vae-spatial-padding` was retired 2026-05-23 (only `zero` is supported; `reflect` lost in every A/B and was dropped)
- **Default canvas is 512x288** - pass `--height`/`--width` only when you want to leave the fast 16:9 preview size
- **Default outputs are timestamped** - without `--output`, runs save to `DIFFUSERS_OUTPUT_DIR`, then `OUTPUT_DIR`, then `outputs/` as `ltx_YYYYmmdd_HHMMSS.mp4`; use `--output-prefix` to name a run family
- **Pick the encode tier by destination** - `--encode-tier {web,default,hq,export,reference}` (default: `default`) selects codec/container/audio together. `web` is libx264 + AAC for universal browser compat; `default` is hardware HEVC + ALAC for Apple/modern browsers; `hq` is software HEVC 4:4:4; `export` and `reference` are ProRes (.mov) for NLE / mastering workflows
- **Converted-weight cache defaults to auto** - the first run builds reusable transformer, connector, video VAE, audio VAE, and vocoder cache files; pass `--weights-cache off` only when you specifically want direct stock-weight loading
- **Keep `--weights` as the bundle path** - advanced runs can override individual subsystems with `--transformer-weights`, `--connector-weights`, `--vae-weights`, `--audio-vae-weights`, `--vocoder-weights`, and `--config-weights`
- **MLX allocator cache defaults to 1GB** - this keeps unified-memory pressure lower without needing a routine `--mlx-cache-limit-gb 1`
- **Same-math video layouts default on** - FF `project_in`/`project_out` and attention `to_out` pretranspose are enabled by default; pass `--video-ff-layout off --video-attn-layout off` for baseline A/Bs
- **Use `--stream-transformer` for the block-streaming preset** - it expands to 16 resident blocks, resident-group compile, and 4-block compile groups
- **Save latents for decode-only tests** - add `--save-latents` to write an NPZ sidecar next to the requested output; distilled two-stage runs include both stage-1 and stage-2 latents plus the existing final-latent keys
- **Save text conditioning for denoise A/Bs** - add `--save-text-embeddings` to write the positive/negative AV text encoder outputs as an `_text.npz` sidecar that can be reused with `--embedding`
- **Save run metadata for reproducibility** - add `--save-run-log` to write params, argv, outputs, and timings as an `_run.json` sidecar, starting before the long generation step
- **Save the lossless audio next to the encoded video** - add `--save-audio-sidecar` to write the vocoder's raw WAV alongside the output (useful for A/B against the codec-compressed audio inside the container)
- **Save all reproducibility sidecars** - add `--save-all-sidecars` to turn on latents, text conditioning, run metadata, and the audio WAV sidecar together
- **Use `--pipeline distilled`** - Fast no-CFG two-stage inference (8+3 steps)
- **Use `--stream-transformer` before `--low-memory`** - the streaming preset is the cleaner constrained-memory path for modern distilled runs; `--low-memory` remains an emergency fallback
- **Reduce resolution** - Start with `--height 256 --width 384` for testing
- **Research denoise speed carefully** - `--video-ff-quantize project_out:mxfp8` can A/B weight-only quantized video FF projections, and `--video-ff-quantize-layers 40-47` narrows it to selected layers; this is non-canonical and needs quality checks
- **A/B same-math layout baselines** - use `--video-ff-layout off --video-attn-layout off` when you want to compare against untransposed stock weight layout
- **Track denoise-speed experiments** - see [Performance Optimization Notes](docs/PERFORMANCE.md) for MLX runtime optimization ideas and benchmark rules

See [Usage Guide](docs/USAGE.md) for memory requirements and benchmarks.

## Prompting Tips

Focus on detailed, chronological descriptions. Include movements, appearances, camera angles, and environment details in a flowing paragraph. Keep under 200 words.

**Structure your prompts:**
1. Main action in a single sentence
2. Specific movements and gestures
3. Character/object appearances
4. Background and environment
5. Camera angles and movements
6. Lighting and colors

See [Lightricks prompting guide](https://ltx.video/blog/how-to-prompt-for-ltx-2) for more tips.

## Requirements

- macOS with Apple Silicon (developed and tested on M1 Max, 64 GB)
- ~25GB RAM (64GB recommended for high resolution)
- ffmpeg (optional; only for non-default/software encode tiers): `brew install ffmpeg`

## Documentation

- [Usage Guide](docs/USAGE.md) - Options, examples, troubleshooting
- [Pipelines](docs/PIPELINES.md) - All 6 pipelines explained
- [Architecture](docs/ARCHITECTURE.md) - Model architecture details
- [Performance Optimization Notes](docs/PERFORMANCE.md) - Denoise-speed benchmark ideas and implementation candidates
- [Parity Testing](docs/PARITY.md) - PyTorch/MLX parity verification

## License

Research and educational use. See [LTX-2](https://github.com/Lightricks/LTX-2) for model licensing.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for LTX-2
- [Apple MLX Team](https://github.com/ml-explore/mlx) for MLX
- [Google](https://ai.google.dev/gemma) for Gemma 3
