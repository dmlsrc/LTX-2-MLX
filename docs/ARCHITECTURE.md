# LTX-2 MLX Architecture

This document describes the architecture of the LTX-2 model and its MLX implementation.

## Overview

LTX-2 is a **19-billion parameter Diffusion Transformer (DiT)** that generates synchronized video and audio from text prompts. This MLX port enables native Apple Silicon inference.

```
Text Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemma 3 (12B)                                              │
│  Text encoder with 48 layers, 3840-dim hidden states        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Text Encoder Projection                                    │
│  Projects 3840-dim → 4096-dim for video context             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  48-Layer Transformer (19B parameters)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Video Stream (14B)                                 │    │
│  │  • 32 attention heads × 128 dim = 4096 hidden       │    │
│  │  • 3D RoPE positional encoding (x, y, t)            │    │
│  │  • Cross-attention to text embeddings               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↕                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Audio Stream (5B) - optional                       │    │
│  │  • 16 attention heads × 128 dim = 2048 hidden       │    │
│  │  • 1D RoPE positional encoding (temporal)           │    │
│  │  • Bidirectional cross-attention with video         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Video VAE Decoder                                          │
│  • 128 latent channels → 3 RGB channels                     │
│  • 1:192 compression (32× spatial, 8× temporal)             │
│  • Timestep conditioning for final denoising                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  Video Output (up to 768×1024, 24fps)
```

## MLX Package Structure

```
LTX_2_MLX/
├── model/
│   ├── transformer/
│   │   ├── model.py          # LTXModel (velocity) and X0Model (denoised)
│   │   ├── transformer.py    # LTXTransformer with 48 blocks
│   │   ├── attention.py      # Multi-head attention with RoPE
│   │   └── rope.py           # 3D rotary position embeddings
│   │
│   ├── video_vae/
│   │   ├── native_blocks.py  # Shared Conv3d/ResBlock primitives for encoder + decoder
│   │   ├── native_decoder.py # Production MLX-native Conv3d VAE decoder
│   │   ├── native_encoder.py # Production MLX-native Conv3d VAE encoder
│   │   ├── decode_utils.py   # decode_latent() polymorphic decode helper
│   │   └── ops.py            # Depth-to-space, pixel shuffle operations
│   │
│   ├── audio_vae/
│   │   ├── decoder.py        # Audio latent → mel spectrogram
│   │   └── encoder.py        # Audio encoding (experimental)
│   │
│   ├── text_encoder/
│   │   ├── gemma3.py         # Native Gemma 3 12B implementation
│   │   └── encoder.py        # Text encoder projection layers
│   │
│   └── upscaler/
│       ├── spatial.py        # 2× spatial resolution upscaler
│       └── temporal.py       # 2× framerate interpolation
│
├── components/
│   ├── schedulers.py         # Sigma schedules (distilled, LTX2Scheduler)
│   ├── patchifiers.py        # Latent ↔ sequence conversion
│   ├── noisers.py            # Gaussian noise addition
│   └── guiders.py            # CFG implementation
│
├── conditioning/
│   └── tools.py              # Latent conditioning utilities
│
├── pipelines/
│   ├── one_stage.py          # AV one-stage and distilled two-stage
│   ├── two_stage.py          # Two-stage with upscaling
│   ├── ic_lora.py            # Image conditioning LoRA
│   └── keyframe_interpolation.py
│
├── loader/
│   └── weight_converter.py   # PyTorch → MLX weight conversion
│
├── types.py                  # VideoLatentShape, SpatioTemporalScaleFactors

archive/                      # documentation-only; not imported.  source for
├── distilled.py.bak          # archived legacy distilled pipeline API
├── simple_decoder.py.bak     # per-temporal-slice VAE decoder
├── simple_encoder.py.bak     # per-temporal-slice VAE encoder
├── encoder.py.bak            # configurable-blocks VideoEncoder (PyTorch-style)
├── decoder.py.bak            # configurable-blocks VideoDecoder
├── resnet.py.bak             # ResnetBlock3D, UNetMidBlock3D, PixelNorm
├── sampling.py.bak           # SpaceToDepthDownsample, DepthToSpaceUpsample
├── convolution.py.bak        # DualConv3d, CausalConv3d, PaddingModeType, ...
└── text_to_video.py.bak      # legacy TextToVideoPipeline + GenerationConfig
```

## Key Components

### Transformer (19B parameters)

The transformer uses a **velocity prediction** formulation:
- Raw model output is velocity `v`
- Denoised prediction: `x0 = latent - sigma * velocity`
- The `X0Model` wrapper handles this conversion

Each of the 48 transformer blocks contains:
- Self-attention with 3D RoPE (video stream)
- Cross-attention to text embeddings
- Feed-forward network with GELU activation
- AdaLN (Adaptive Layer Norm) conditioning on timestep

### Video VAE Decoder

The VAE decoder converts 128-channel latents to RGB video:
- **Compression ratio**: 32× spatial, 8× temporal
- **Input**: `(B, 128, T, H, W)` latents
- **Output**: `(B, 3, T×8, H×32, W×32)` pixels

Key feature: **Timestep conditioning** - the decoder performs a final denoising step during decode, using learned scale/shift tables indexed by the current sigma value.

### Text Encoder

Two-stage text encoding:
1. **Gemma 3 (12B)**: Extracts 3840-dim hidden states from all 48 layers
2. **Projection**: Linear projection from 3840 → 4096 dimensions

The projection includes:
- Multi-layer hidden state aggregation
- Layer normalization
- Caption projection to transformer hidden dimension

### Patchifier

Converts between spatial latent format and sequence format:
- **Patchify**: `(B, C, T, H, W)` → `(B, T×H×W, C)` for transformer
- **Unpatchify**: `(B, T×H×W, C)` → `(B, C, T, H, W)` for VAE

### 3D RoPE (Rotary Position Embeddings)

Position encoding for video tokens:
- Separate frequency components for (x, y, t) dimensions
- Uses "SPLIT" rope type: cos/sin applied to separate halves of features
- Precomputed based on latent grid coordinates

## Inference Flow

### Distilled Two-Stage Pipeline (8 + 3 steps)

```python
# 1. Text encoding
text_encoding = text_encoder.encode(prompt)  # (1, 1024, 4096)

# 2. Stage 1 initializes half-resolution video/audio latents, applies
#    first-frame or keyframe conditioning, then noises at sigma=1.0.
stage_1_sigmas = [1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]
video_latent, audio_latent = denoise_av(stage_1_sigmas, width // 2, height // 2)

# 3. Spatial upscaler works in the VAE's un-normalized latent space.
video_latent = stats.normalize(spatial_upscaler(stats.un_normalize(video_latent)))

# 4. Stage 2 re-noises the upscaled video latent and stage-1 audio latent at
#    sigma=0.909375, then runs 3 refinement steps.
stage_2_sigmas = [0.909, 0.725, 0.422, 0.0]
video_latent, audio_latent = denoise_av(stage_2_sigmas, width, height)

# 5. Decode after the transformer is released.
video = vae_decoder.decode(video_latent)
audio = audio_decoder.decode(audio_latent)
```

## Memory Requirements

| Configuration | RAM Usage |
|---------------|-----------|
| Gemma 3 text encoder | ~12 GB |
| Transformer (FP16) | ~20 GB |
| VAE decoder | ~2 GB |
| **Total (sequential)** | **~25 GB** |

The implementation loads models sequentially to fit within unified memory:
1. Load Gemma 3, encode text, unload
2. Load transformer, denoise, unload
3. Load VAE, decode video

## Precision

- **Weights**: BFloat16 (loaded from safetensors)
- **Computation**: BFloat16 for transformer, Float32 for VAE
- **Activations**: BFloat16 to reduce memory

## References

- [LTX-2 Technical Report](LTX_2_Technical_Report_compressed.pdf) - Official Lightricks paper
- [LTX-2 PyTorch](https://github.com/Lightricks/LTX-2) - Reference implementation
- [MLX Documentation](https://ml-explore.github.io/mlx/) - Apple MLX framework
