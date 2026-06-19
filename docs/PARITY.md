# PyTorch ↔ MLX Parity Testing

This document describes the parity testing methodology and results for verifying that the MLX implementation matches the PyTorch reference.

## Summary

**V2.3 (22B) Text Encoder Parity: 0.9996 video / 0.9931 audio** (May 21, 2026 audit)
**V2.3 (22B) Transformer Step-0 Parity: 0.9991** (May 21, 2026, 512x512x33)

The MLX implementation has been verified to produce outputs statistically equivalent to PyTorch across all pipeline stages.  The V2.3 hot path (distilled two-stage AV) has additionally been line-by-line audited against the canonical Lightricks `TI2VidTwoStagesPipeline` — see [Known Divergences (V2.3 Audit)](#known-divergences-v23-audit-may-21-2026) for the resolution of each finding.

## Methodology

### Two-Phase Comparison

Since running both PyTorch and MLX simultaneously would exceed memory limits, we use a sequential approach:

```
Phase 1: PyTorch Reference
├── Run inference with specific config
├── Save checkpoints at each stage
└── Clear memory

Phase 2: MLX Comparison
├── Load PyTorch checkpoints
├── Run MLX with identical config
└── Compare outputs at each stage
```

### Checkpoints Saved

| Checkpoint | Description |
|------------|-------------|
| `text_encoder_video_encoding.npy` | Text encoder output |
| `initial_latent.npy` | Starting noise (shared) |
| `positions.npy` | Pixel coordinates |
| `transformer_step_XXX.npy` | Transformer output at each step |
| `vae_decoder_input_latent.npy` | Final latent before VAE |
| `vae_decoder_output_pixels.npy` | Decoded video pixels |

### Comparison Metrics

| Metric | Description | Pass Threshold |
|--------|-------------|----------------|
| Correlation | Pearson correlation coefficient | ≥ 0.95 |
| Max Diff | Maximum absolute difference | Informational |
| Mean Diff | Average absolute difference | Informational |
| Shape Match | Tensor dimensions identical | Exact |

## Results

### Test Configuration

```
Prompt: "A golden retriever running through a meadow"
Resolution: 128x128 (for fast testing)
Frames: 17 (3 latent frames)
Steps: 8 (distilled)
Seed: 42
```

### Stage-by-Stage Results

| Stage | Correlation | Status |
|-------|-------------|--------|
| Text Encoder | 0.997 | ✅ PASS |
| Positions | 1.000 | ✅ PASS |
| Patchified Latent | 1.000 | ✅ PASS |
| Transformer Step 0 | 0.982 | ✅ PASS |
| Transformer Step 1 | 0.978 | ✅ PASS |
| Transformer Step 2 | 0.975 | ✅ PASS |
| Transformer Step 3 | 0.971 | ✅ PASS |
| Transformer Step 4 | 0.968 | ✅ PASS |
| Transformer Step 5 | 0.965 | ✅ PASS |
| Transformer Step 6 | 0.962 | ✅ PASS |
| Transformer Step 7 | 0.959 | ✅ PASS |
| VAE Input Latent | 0.957 | ✅ PASS |
| VAE Output Pixels | 0.954 | ✅ PASS |

### Visual Comparison

Frame-by-frame comparison at higher resolution (768x512, 65 frames):

| Frame | Correlation |
|-------|-------------|
| 0 | 0.962 |
| 16 | 0.966 |
| 32 | 0.963 |
| 48 | 0.962 |
| 64 | 0.964 |

**Mean Correlation: 0.963**

## V2.3 (22B) Parity Results — May 21, 2026

The V2.3 distilled-1.1 hot path was re-audited end-to-end against the canonical Lightricks reference (`Lightricks/LTX-2` at commit ~`1799988`, May 11 2026).  Two probes were run: stagewise text-encoder parity (Gemma3 → V2 connector → final AV encoding) and single-step transformer parity (LTXAVModel one forward through the full 22B model).

### Test Configuration

```
Model: ltx-2.3-22b-distilled-1.1.safetensors
Prompt: "a cat sitting on a windowsill watching rain"
Resolution: 512x512 (transformer probe)
Frames: 33 (5 latent frames)
Steps: 1 (isolated step-0 forward)
Seed: 42
Device: MPS (PyTorch reference), Metal (MLX)
Dtype: BFloat16 both sides
```

### Text Encoder Parity (Stagewise)

Stage-by-stage cosine similarity of Gemma3 hidden states on the real-token slice (left-padded `attention_mask.sum()` = 9 tokens):

| Stage | Output shape | cos sim (real tokens) | abs mean | Notes |
|---|---|---|---|---|
| `hidden_states[0]` (embeddings) | (1, 1024, 3840) | **1.0000000** | 0 | Bit-identical |
| `hidden_states[12]` | (1, 1024, 3840) | 0.998057 | 8.6e-2 | After 12 layers |
| `hidden_states[24]` | (1, 1024, 3840) | 0.997683 | 5.65e-1 | After 24 layers |
| `hidden_states[36]` | (1, 1024, 3840) | 0.992143 | 2.12 | After 36 layers |
| `hidden_states[48]` (post-norm) | (1, 1024, 3840) | 0.919992 | 5.27e-2 | bf16 noise across 48 layers |
| V2 feature extractor (real slice) | (1, 9, 4096) | 0.945 | 1.67 | Per-token RMSnorm + linear |
| **Final video encoding** | (1, 1024, 4096) | **0.9995924** | 3.07e-3 | Connector smooths the drift |
| **Final audio encoding** | (1, 1024, 2048) | **0.9931221** | 1.50e-2 | Smaller head_dim (64 vs 128) |

**Note on the 0.92 dip at hs[48]**: this is bf16 noise compounding across 48 transformer layers — not an implementation bug.  The V2 feature extractor's per-token RMSnorm + linear projection wipes magnitude differences and the connector attention averages across the 49 stacked hidden states, recovering to 0.9996 at the final video encoding.  Audio is slightly looser because the audio path has half the per-head capacity (32 heads x 64 dim vs video's 32 x 128) — same algorithm, less bf16 headroom.

### Transformer Step-0 Parity

Single forward pass through the full 22B V2.3 AudioVideo transformer, feeding identical (PyTorch-computed) text embedding + identical (numpy fp32 → bf16) noisy latent into both impls.  Output is the X0Model's denoised prediction.

| Stage | Shape | cos sim | abs mean | abs max |
|---|---|---|---|---|
| Positions | (1, 3, 1280, 2) | **1.0000000** | 0 | 0 |
| Input latent (patchified) | (1, 1280, 128) | **1.0000000** | 0 | 0 |
| **x0_video (transformer step-0 output)** | (1, 1280, 128) | **0.9991481** | 1.30e-2 | 1.17e-1 |

Reading: input alignment is bit-identical (positions and patchified latent cos sim = 1.0), so the 0.9991 cos sim on the output is **purely the transformer's contribution** — 48 transformer layers + AdaLN + cross-attention + per-head gated attention + the X0Model `latent - sigma*velocity` step add up to about 0.0009 of cross-impl drift.

### bf16 Noise Floor

To rule out "bf16 alone is causing the drift," ran MLX Gemma3 forward in bf16 vs fp32 (both on Metal):

| Layer | bf16 vs fp32 (real-token slice) | Notes |
|---|---|---|
| hs[0] | 0.9999986 | Within MLX, bf16 self-consistency is excellent |
| hs[12] | 1.0000000 | — |
| hs[24] | 0.9999996 | — |
| hs[36] | 0.9999986 | — |
| **hs[48]** | **0.9997602** | Final post-norm |

MLX bf16 vs MLX fp32 is **0.9998** at hs[48].  Compared to MLX bf16 vs PyTorch bf16 at hs[48] (0.9200), the cross-backend gap is **~250x larger** than the within-MLX bf16-vs-fp32 noise.  This means **bf16 is not the dominant source of cross-impl drift** — it's the kernel-implementation differences between MPS and Metal (different tile shapes, different reduction orders).  Going to fp32 in MLX alone would not close the gap with bf16 PyTorch; both backends would need to be fp32 (3.5x slower per forward, 2x memory) for cross-impl drift to drop to ~0.9999.

The 0.9996/0.9931 V2.3 cos sims **are roughly the floor** of "two different bf16 backends agreeing on the same algorithm on the same hardware family."  Probe scripts and saved sidecar tensors live under `$SHARED_TEMP_DIR/trace_analysis/` (or `${TMPDIR}/trace_analysis/` if `SHARED_TEMP_DIR` isn't set in your environment).

## Key Fixes for Parity

### 1. X0Model Wrapper

**Issue**: PyTorch LTXModel returns velocity, but comparison expected denoised (x0) output.

**Fix**: Added X0Model wrapper that converts velocity to denoised prediction:
```python
x0 = latent - sigma * velocity
```

### 2. context_mask Parameter

**Issue**: MLX was passing `context_mask=text_mask` while PyTorch uses `context_mask=None`.

**Fix**: Changed all Modality creations to use `context_mask=None`.

### 3. BFloat16 Precision

**Issue**: MLX was using float16 while PyTorch uses bfloat16.

**Fix**: Changed all dtype specifications to `mx.bfloat16`.

### 4. RoPE Dimension

**Issue**: Debug scripts used `dim=128` (head_dim) instead of `dim=4096` (inner_dim).

**Fix**: Corrected to `dim = num_heads x head_dim = 32 x 128 = 4096`.

### 5. Gemma3 RoPE Multiply Precision (V2.3 — May 21, 2026)

**Issue**: Our `apply_rotary_pos_emb` for the Gemma3 text encoder was downcasting cos/sin to bf16 before the multiply, while HF Gemma3 keeps cos/sin in fp32 and lets the multiply promote Q/K to fp32 (then downcasts before SDPA).  Mathematically lossier per layer of accumulated drift.

**Fix**: [`LTX_2_MLX/model/text_encoder/gemma3.py:117-141`](../LTX_2_MLX/model/text_encoder/gemma3.py) — cos/sin stay fp32, multiply runs in fp32, result downcasts back to bf16 before SDPA (so SDPA still picks its bf16 kernel — propagating fp32 to attention compiles a pure-fp32 steel kernel that's ~2x slower).  Mirrors the LTX connector RoPE pattern.  No measurable perf cost; the downcast happens once per attention block on a tiny tensor.

### 6. Tokenizer Whitespace Strip (V2.3 — May 21, 2026)

**Issue**: Our `encode_with_gemma` / `encode_av_gemma_batch` tokenized the raw prompt, while Lightricks' `LTXVGemmaTokenizer.tokenize_with_weights` does `text.strip()` first.

**Fix**: [`LTX_2_MLX/generate.py:1361,1539`](../LTX_2_MLX/generate.py) - apply `.strip()` to the prompt before passing it to the HF tokenizer.  Removes trailing-whitespace token mismatches.

### 7. Pad-to-Max Tokenization Default (V2.3 — May 21, 2026)

**Issue**: `encode_av_gemma_batch` defaulted to unpadded tokenization (variable-length, faster), but Lightricks always pads to `max_length=1024` before Gemma forward.  Real-token hidden states are bit-equivalent either way (RoPE-relative attention), but the canonical numerics path is padded.

**Fix**: [`LTX_2_MLX/generate.py:1532`](../LTX_2_MLX/generate.py) - `LTX_PAD_PROMPT_TO_MAX` now defaults to `"1"` (pad), opt out with `=0`.  Reproduces stock LTX-2 numerics exactly.

### 8. Audio Noise Normalization (V2.3 — May 21, 2026)

**Issue**: `AVPipeline._channelwise_normalize_audio_noise` unconditionally whitened the per-channel statistics of the audio noise tensor at stage-1 init.  Lightricks does not normalize the noise — the audio LoRA was trained on un-normalized `N(0,1) * sigma_max`.  Originally added as a workaround for a "duration-dependent amplitude bug" (see [AUDIO_ISSUES.md](AUDIO_ISSUES.md)) that has since been resolved by other fixes.

**Fix**: first env-gated behind `LTX_NORMALIZE_AUDIO_NOISE` (default OFF, May 21 2026), then **removed entirely** (June 18 2026) once the underlying bug was confirmed resolved -- the normalization diverged from upstream and perturbed the video branch via `a2v` cross-attention.  See [AUDIO_ISSUES.md](AUDIO_ISSUES.md) for the empirical re-test (20-second clip RMS 588 vs 608 between the two modes, both with healthy ~1000-1600 RMS dialog bursts).  Lock-in guard: `tests/test_audio_noise_parity.py`.

### 9. Keyframe Conditioning Routing (V2.3 — May 21, 2026)

**Issue**: `create_image_conditionings` unconditionally emitted `VideoConditionByLatentIndex`.  Lightricks routes the first frame through latent-replacement and any non-zero `frame_index` through `VideoConditionByKeyframeIndex` — different mask-blend semantics.  Silent semantic mismatch for multi-image conditioning with non-zero frame indices.

**Fix**: [`LTX_2_MLX/pipelines/common.py:139-152`](../LTX_2_MLX/pipelines/common.py) — branch on `frame_index`: zero → latent replacement, non-zero → `VideoConditionByKeyframeIndex` (the keyframe class already existed in `LTX_2_MLX/conditioning/keyframe.py`).  Test coverage added in `tests/test_pipelines.py::TestCreateImageConditionings`.

## Known Divergences (V2.3 Audit, May 21, 2026)

Subagent line-by-line audit of `AVPipeline.generate_distilled_two_stage` vs canonical `TI2VidTwoStagesPipeline.__call__` found four actionable items (all fixed in commit `c431b7b`) and a handful of structural observations that don't currently affect output but are worth knowing about:

### Resolved (commit `c431b7b`)

| # | Divergence | Status |
|---|---|---|
| 1 | Audio noise normalization (unconditional -> env-gated default OFF -> removed) | ✅ Fixed |
| 2 | Multi-image conditioning routing (always latent → keyframe for non-zero frame_index) | ✅ Fixed |
| 3 | Missing `stage_1_sigmas` / `stage_2_sigmas` override kwargs | ✅ Fixed |
| 4 | `distilled.py` dormant code retired (removed from working tree) | ✅ Fixed |

### Structural observations (no action — informational)

| Observation | Details |
|---|---|
| **`LatentState.attention_mask` field absent** | Canonical `LatentState` carries an optional `attention_mask: torch.Tensor \| None`.  Our MLX `LatentState` has no such field.  For the distilled two-stage hot path this is harmless — neither pipeline ever sets it.  Becomes relevant if we ever port a packing pipeline or anything using 2D self-attention masks. |
| **`LTX_VELOCITY_MODE` fastpath is MLX-specific** | Bypasses the X0Model wrapper and runs velocity model directly when latent mask is uniform-ones.  Algebraically identical to the canonical's wrap→unwrap→step round-trip, just skips the redundant `to_velocity` / `to_denoised` calls.  Subagent verified equivalence. |
| **`uniform_mask` flag footgun** | [`LTX_2_MLX/types.py:198-209`](../LTX_2_MLX/types.py) documents that `uniform_mask=True` must stay honest.  If conditioning ever produces a non-uniform mask but the flag isn't cleared, the scalar-timestep optimization picks up incorrect data.  Existing call sites are correct. |

### 2026-05-31 distilled two-stage precision scan

Scope: LTX-2.3 distilled two-stage AV hot path (`8+3` Euler steps, BF16
default compute, internal audio branch enabled), checked against the official
Lightricks checkout at `/Users/Shared/huggingface/reference/Lightricks-LTX-2`
and the local `ltx-2.3-22b-distilled-1.1.safetensors` metadata.

Official precision baseline:

- Transformer, latent payloads, attention, FFN, and AdaLN modulation use BF16
  payload compute by default.
- Sigma schedules, timestep basis, Euler / X0 denoising arithmetic, and
  noising scalars stay FP32, then cast back to the latent payload dtype.
- V2.3 RoPE frequency setup is configured as `frequencies_precision=float64`;
  upstream builds the frequency grid with the double-precision path, then casts
  final cos/sin to hidden dtype.
- BWE vocoder is a scoped FP32 island and then returns the input dtype.  This is
  official parity behavior, not a local bug.
- HQ / res2s is not the distilled two-stage path, but if ported it has deliberate
  FP64 sampler math.

Findings:

| Item | Triage | Details |
|---|---|---|
| **V2.3 RoPE double-precision setup** | **Fixed 2026-05-31** | The checkpoint metadata says `frequencies_precision=float64`, and local [`precompute_freqs_cis`](../LTX_2_MLX/model/transformer/rope.py) supports `use_double_precision`, but the video/audio/cross positional-embedding calls in [`model.py`](../LTX_2_MLX/model/transformer/model.py) did not pass the flag through.  A sparse stage-2-shape probe saw fp32-grid vs fp64-grid differences up to ~`0.0064` in cos and ~`0.0074` in sin before the hidden-dtype cast.  `load_av_transformer` now reads the transformer metadata (`frequencies_precision`, `rope_type`, `positional_embedding_max_pos`, `av_ca_timestep_scale_multiplier`) and wires double-precision RoPE into the simple and cross-modal preprocessors; covered by `tests/test_precision_plumbing.py`. |
| **A/V cross-attention timestep scale** | **Not a bug** | Local `generate.py` passes `av_ca_timestep_scale_multiplier=1000`.  The official safetensor metadata also has `av_ca_timestep_scale_multiplier: 1000.0`; upstream's `av_ca_factor = av_ca_timestep_scale_multiplier / timestep_scale_multiplier` makes the gate input equivalent to `cross_sigma * 1000`. |
| **Q/K RMSNorm dtype promotion** | **Guard fixed 2026-05-31** | `mx.fast.rms_norm(BF16 activation, FP32 weight)` promotes the result to FP32, which would bypass the BF16/FP16 STEEL attention selector.  The actual LTX-2.3 checkpoint has BF16 q/k norm weights (sampled count: 608 BF16), so the default path was already safe.  `RMSNorm.__call__` now casts back to `x.dtype`, preventing external/random-init weights from changing attention precision; covered by `tests/test_precision_plumbing.py`. |
| **Image/keyframe conditioning dtype promotion** | **Fixed 2026-05-31** | The native video encoder returns conditioning latents as FP32 to match the simple encoder, and `VideoConditionByLatentIndex` / `VideoConditionByKeyframeIndex` concatenated tokens without casting to the existing latent dtype.  Plain text-only distilled generation was unaffected, but image/keyframe-conditioned BF16 runs could silently move the denoise state to FP32.  Conditioning tokens now cast to `latent_state.latent.dtype` before concatenation, and keyframe masks keep the existing mask dtype; covered by `tests/test_conditioning.py`. |
| **BWE vocoder output dtype** | **Not a bug** | Both local and official code run the BWE chain in FP32 and cast the clipped waveform back to the input dtype.  Returning FP32 could be tested as a quality experiment, but it would be a deliberate divergence from official parity. |
| **Fused AdaLN env path** | **Guard fixed 2026-05-31** | `LTX_FUSED_ADALN=1` uses a custom kernel with hardcoded `eps=1e-6` and BF16 output.  That matches the default distilled BF16 path; non-default `norm_eps` and non-BF16/non-broadcast shapes now fall back to the MLX expression. |
| **Legacy interleaved RoPE** | **Removed 2026-06-16** | LTX-2.3 checkpoints use split RoPE, and official reference code treats interleaved RoPE as legacy. Local metadata parsing now warns when RoPE metadata is missing and rejects explicit `rope_type=interleaved`; only split RoPE remains implemented. |
| **2D float attention masks** | **Fixed 2026-05-31** | Float additive masks now preserve their values but normalize common `(B, S)` and `(B, T, S)` inputs to SDPA-broadcastable shapes, matching the bool/int mask path.  The distilled two-stage hot path still uses `context_mask=None`, so this remains defensive coverage. |

Full stage-2 RoPE A/B (same saved stage-1 latents, `--save-all-sidecars`,
audio enabled, 576x320x721, seed 42):

| Run | RoPE precision | Denoise | Total | Sidecar |
|---|---:|---:|---:|---|
| `stage2_rope_metadata_full_20260531_023011` | metadata / float64 grid | 300.7s | 391.0s | `/Users/Shared/huggingface/output/stage2_rope_metadata_full_20260531_023011.npz` |
| `stage2_rope_fp32_full_20260531_023700` | forced fp32 grid | 294.9s | 384.5s | `/Users/Shared/huggingface/output/stage2_rope_fp32_full_20260531_023700.npz` |

Stage-1 video/audio latents compared exactly equal between the two sidecars.
Final stage-2 drift was non-zero but modest: video latent
`max_abs=0.9765625`, `mean_abs=0.0192177`, `rms=0.0276312`,
`cos=0.999641218686`; audio latent `max_abs=2.17578125`,
`mean_abs=0.0133171`, `rms=0.0490743`, `cos=0.999144662049`.  Treat timing
as a busy-machine smoke read, not a performance result; this A/B is mainly
evidence that metadata-driven float64 RoPE is a real numerical parity change.

RoPE recompute follow-up: `LTX_ROPE_PRECOMPUTE=1` now precomputes both
self-RoPE and A/V cross temporal RoPE once per stage.  Keep it opt-in: a
stage-2 bench-mode pair on the same saved latents measured `96.945s` with
precompute vs `96.480s` with `LTX_DISABLE_ROPE_PRECOMPUTE=1`, so resident
FP32 RoPE tables did not produce an actionable denoise win on M1 Max.

### Upstream features not yet ported

Lightricks has shipped two pipeline-level features since our V2.3 baseline that we don't have.  Neither affects the distilled two-stage hot path (so they're not parity bugs), but they're net-new functionality:

| Feature | Where | Why we don't have it |
|---|---|---|
| **HDR IC-LoRA pipeline** | `packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py` (~886 lines, April 23) | Whole new pipeline class for HDR-aware image-conditioned LoRA generation.  Substantial port effort. |
| **Color conversion module** | `packages/ltx-pipelines/src/ltx_pipelines/utils/color_conversion.py` (~224 lines, May 11) | BT.709 / BT.2020 colorimetry for HDR output.  If a user reports "MLX output looks slightly washed out vs ComfyUI" this is likely the cause. |

## Status of the Parity Harness

The executable PyTorch-checkpoint harness this document originally described - a
`generate_pytorch_checkpoints.py` generator plus a `test_parity.py` comparison
suite - has been removed. Producing the golden checkpoints requires running the
PyTorch LTX reference, which does not run on macOS, so the suite only ever
skipped here, and it was still pinned to the retired 19B architecture.

The parity findings above remain the record of how MLX was validated against
PyTorch. Ongoing, runnable parity checks now compare against independent in-repo
references instead of the PyTorch reference - see `tests/test_fused_ops.py`
(fused kernels vs plain-MLX) and `tests/test_spatial_upscaler.py` (vs a
NumPy/einops reference).

## Interpreting Results

### Correlation Values

| Range | Interpretation |
|-------|----------------|
| 0.99+ | Excellent - numerical precision differences only |
| 0.95-0.99 | Good - functionally equivalent |
| 0.90-0.95 | Acceptable - minor implementation differences |
| < 0.90 | Investigate - potential bug |

### Expected Differences

Small differences are expected due to:
- Floating-point precision (bfloat16 vs float16 edge cases)
- Different BLAS implementations (MLX Metal vs PyTorch MPS/CPU)
- Operator fusion differences
- Random number generation

### When Correlation Drops

If correlation drops significantly at a specific step:
1. Check that step's inputs match the previous step's expected outputs
2. Verify operator implementations (attention, FFN, normalization)
3. Check for dtype mismatches
4. Verify RoPE and position encoding

## Conclusion

**V2.3 (22B)**: MLX achieves **0.9996 video / 0.9931 audio** text-encoder cross-impl cos sim and **0.9991** transformer step-0 cos sim against canonical Lightricks PyTorch.  These numbers are at the bf16 cross-kernel noise floor — MPS-PyTorch and Metal-MLX cannot agree more tightly without both being upgraded to fp32 (3.5x slower per forward, 2x memory).  All four substantive divergences from the May 2026 audit are resolved in `c431b7b`; remaining gaps are net-new upstream features (HDR IC-LoRA, color conversion), not parity bugs.
