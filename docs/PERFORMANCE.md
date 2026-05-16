# Performance Optimization Notes

This document tracks ways to make the existing MLX denoise path faster without
changing user-visible generation settings such as pipeline type, stage layout,
duration, frame count, resolution, sampler, or step count.

It is intentionally a working notebook. Some entries are proven knobs, some are
implementation candidates, and some are ideas that should not be treated as
free wins without evidence.

## Scope

In scope:

- MLX evaluation/materialization cadence.
- Reusing fixed per-run tensors instead of rebuilding them every denoise step.
- Reducing no-op casts, host/device transfers, and repeated small allocations.
- Narrow `mx.compile` experiments around pure repeated helper functions.
- Benchmark and logging improvements that make denoise timing easier to compare.

Out of scope for this document:

- Reducing steps, frames, duration, or resolution.
- Switching pipeline type or stage layout.
- Disabling audio when comparing audio-video runs.
- Quantized weights as a default path. Quantization can be useful, but it is a
  separate quality and checkpoint-format tradeoff.
- VAE-only visual changes as denoise-speed optimizations. For example,
  `--vae-spatial-padding zero` can improve boundary artifacts, but it does not
  make the transformer denoise loop faster.

## Measurement Rules

Use the same prompt, seed, weights, Gemma path, resolution, duration or frame
count, FPS, steps, dtype, audio setting, and VAE padding policy for each A/B.

Prefer:

```bash
--save-run-log
```

or:

```bash
--save-all-sidecars
```

The final timing summary currently groups denoising and VAE decode under
`generation + decode`. For denoise-only speed, use the live progress line:

```text
STEP1 ... | RUN ... | ETA ... | avg .../it
```

`RUN` and `avg` are the best quick comparison numbers for the transformer
denoise section. For long comparisons, capture the terminal output or keep the
`_run.json` sidecar alongside the output video.

Run each candidate at least twice when testing compilation changes. The first
iteration may include one-time graph construction or cache setup cost, and the
steady-state result is what matters for long runs.

## Current Baseline

The LTX-2.3 audio-video path uses the one-stage audio-video transformer. With
distilled checkpoints, CFG is forced to 1.0, so a normal distilled AV denoise
step is already a single joint transformer forward, not a positive/negative CFG
pair.

Already implemented baseline optimizations:

- Native BF16 compute is the default.
- Runtime loaders use `mx.load()` and preserve checkpoint BF16 tensors.
- Attention uses MLX fast scaled dot product attention.
- RMSNorm and several transformer helper paths use MLX fast or compiled helper
  functions.
- Reproducibility sidecars can save latents, text conditioning, and run metadata
  with `--save-all-sidecars`; distilled two-stage runs include both stage
  latents.

The main remaining hot path is therefore the repeated 48-layer AV transformer
forward inside the denoise loop.

## Transformer Profiling

Use this when the live progress line says denoise is slow and you need to know
where one or more transformer calls are spending time:

```bash
--profile-transformer-steps 1,2,8
```

`--profile-transformer-once` is kept as a convenience alias for step 1. Prefer
`--profile-transformer-steps` when you want to compare a cold first step with
warmer later steps after MLX/Metal graph and kernel setup has had a chance to
settle.

Each selected step prints two diagnostic sections:

- `Transformer profile`: internal model timing split into preprocessing,
  transformer block groups, and output projection.
- `AV denoise step N profile`: outer pipeline timing split into modality
  setup, transformer call, guidance/postprocess, scheduler step, and final state
  eval.

For a deeper split inside selected blocks, add:

```bash
--profile-transformer-blocks 0,40,47
```

Block numbers are 0-based to match the existing `blocks 40-47` timing labels.
The detailed block profile splits a block into video/audio self-attention, text
attention, audio-video cross attention, and feed-forward sections. Attention
sections are split further into setup/AdaLN, Q/K/V projections, Q/K norm, RoPE,
SDPA, gate, output projection, and residual checkpoints. Feed-forward sections
are split into AdaLN, `project_in`, GELU, `project_out`, and residual-gate
checkpoints so activation experiments can be separated from the two large
matmuls.

The profile deliberately inserts `mx.eval(...)` checkpoints so each section has
a real wall-clock boundary. That means it perturbs the exact timing it measures.
Use it to find which region is worth optimizing, then rerun without the flag for
clean benchmark numbers.

## MLX Runtime Notes

These notes come from the local MLX checkout under
`/Users/Shared/huggingface/lib/mlx-main`.

### Lazy evaluation

MLX records a computation graph and runs it when `mx.eval(...)`, printing,
NumPy conversion, memory access, or saving forces evaluation. The docs describe
the evaluation tradeoff directly:

- Too many small `mx.eval(...)` calls pay fixed scheduling overhead.
- Letting a graph grow extremely large can also become costly.
- A natural efficient boundary is the outer iteration of an iterative
  computation.

For LTX denoising, that maps to: one eval at the end of each denoise step is a
reasonable target. Extra evals inside the 48-layer transformer are a memory
tradeoff, not a free speed feature.

### Compilation

`mx.compile` compiles and caches computation graphs. It can reduce graph size by
merging common work and fusing operations. The docs call out a 5x GELU speedup
example on an M1 Max for a fusible elementwise function, but also document
several sharp edges:

- The first call pays compile overhead.
- Shape, dtype, or input-count changes can trigger recompilation.
- Avoid creating compiled lambdas/functions inside loops.
- Compiled functions should be pure: no printing, no side effects, no hidden
  mutable state unless captured through `inputs=` / `outputs=`.
- Compile the outer useful function when possible, but only when the function is
  stable enough to benefit from cache reuse.
- `shapeless=True` avoids shape-based recompilation, but can be wrong if the
  function uses static shape values in reshape/control logic.

For LTX denoising, this supports narrow repeated helper compilation first. A
whole-transformer compile may be possible, but only behind an opt-in benchmark
flag because the AV model forward has complex Python structure.

### Fast kernels

The local MLX docs expose:

- `mx.fast.scaled_dot_product_attention`
- `mx.fast.rms_norm`
- `mx.fast.layer_norm`
- `mx.fast.rope`
- `mx.fast.metal_kernel`
- `mx.fast.cuda_kernel`

The current code already uses fast SDPA and RMSNorm. `mx.fast.rope` exists and
supports `traditional`, `base`, `scale`, `offset`, and 1D `freqs` in the local
tests, but it is not obviously a drop-in replacement for LTX-2.3's 3D SPLIT
RoPE path with per-token composed cos/sin tensors. Treat it as a research item,
not an immediate swap.

If a custom Metal kernel is added, the MLX docs warn to build the kernel once
and reuse it. They also note that `fast.metal_kernel` defaults to
`ensure_row_contiguous=True`, which can insert copies for non-contiguous inputs.
For transformer hot paths, any custom kernel must account for layout and avoid
accidental copies.

### SDPA kernel tile floor on M1 Max

`mx.fast.scaled_dot_product_attention` dispatches to one of two Metal kernel
paths based on the sequence length `T`:

- `sdpa_full` — used when `T > 8`; covers all LTX latent grids.
- `sdpa_vector_2pass` — used when `T ≤ 8`; not relevant to LTX.

For `sdpa_full` on a non-NAX chip (M1 and M2 families,
`applegpu_g13s/g13g/g14s/g14g`), the tile sizes at `head_dim = 128` are
hardcoded in `mlx/backend/metal/scaled_dot_product_attention.cpp`:

```
bq = 32, bk = 16   # non-NAX, head_dim=128
```

For NAX-capable chips (M3 and later, `applegpu_g15s/g15g/g15p`):

```
bq = 64, bk = 32   # NAX
```

There is no env var or runtime flag to change tile sizes for `sdpa_full` in
MLX 0.31.2. MLX PR #3455 adds `MLX_SDPA_BLOCKS`, but that env var only
controls the `sdpa_vector_2pass` kernel (`T ≤ 8`) and has no effect on
`sdpa_full`. It is therefore irrelevant to any LTX denoise step.

At 1024×576×481, the video token count is `18 × 32 × 61 = 35,136` before audio
tokens. At 512×288 the same model runs at about `55s/it`. At 1024×576 the token
count is 4× larger; the observed `~425s/it` step time confirms that attention
scaling dominates and there is no software knob to recover the 4× token overhead
on this hardware.

**Implications:**

- On M1 Max, `sdpa_full` is at the hardware tile floor for `head_dim = 128`.
- Upgrading to M3+ (NAX) would double both tile dimensions and is the most
  direct path to faster SDPA at this resolution.
- A custom Metal kernel via `mx.fast.metal_kernel` could in principle implement
  a larger tile, but writing and validating a correct SDPA replacement is a
  significant engineering effort.
- Python-level chunked or tiled SDPA does not change the Metal tile width;
  prior experiments showed no runtime win at 1024×576 token counts.

Treat stage-2 denoise at 1024×576 on M1 Max as an optimization floor within
the current MLX 0.31.2 release.

### MLX 0.31.2 primitive scan

The local `mlx-main/docs/src/python` API source and the installed MLX 0.31.2
bindings expose a few primitives worth remembering for denoise optimization
research:

- `nn.gelu_fast_approx` / `nn.GELU(approx="fast")`: low-friction, non-parity
  activation experiment for the transformer feed-forward path. The current code
  uses the tanh/precise GELU approximation. Fast GELU uses
  `x * sigmoid(1.702 * x)`, with a larger documented approximation error.
- `mx.qqmm`: quantizes activations on the fly and multiplies by a quantized or
  non-quantized weight matrix. It supports `nvfp4` and `mxfp8`, but on the
  current local Metal runtime both quantized-weight and non-quantized-weight
  tests on LTX FFN-shaped matrices failed with
  `RuntimeError: [QQMatmul] NYI for the general case`. Keep this on the list for
  future MLX releases, but do not wire it into the current denoise path.
- `mx.quantize`, `mx.quantized_matmul`, `nn.QuantizedLinear`, and
  `nn.quantize(...)`: useful for a separate quantized-transformer experiment.
  Weight-only quantization is easier to wire than `qqmm`, runs on the current
  Metal runtime for LTX FFN-shaped 3D token tensors, and may trade speed,
  memory, and quality differently from activation quantization.
- `mx.block_masked_mm`: dense matmul with optional block masks. It is only useful
  if a structured sparsity or pruning experiment introduces block masks; it is
  not a faster replacement for the current dense transformer.
- `mx.gather_mm` / `mx.gather_qmm`: efficient gather-plus-matmul primitives for
  batched selected matrices, MoE-style routing, or similar designs. They are not
  relevant to the current dense LTX denoise pass.
- `mx.segmented_mm`: exposed in the MLX 0.31.2 Python binding but not listed in
  `docs/src/python/ops.rst`. It computes per-segment matrix products along the
  inner dimension. It is interesting for diagnostics or decomposed matmul
  experiments, not an immediate speed knob.
- `mx.hadamard_transform`: fast Walsh-Hadamard transform along the final axis.
  Potentially useful for quantization/rotation experiments, not a direct
  transformer-forward replacement.
- `mx.depends`: exposed in the binding and useful for graph dependency control.
  Keep it in mind for precise async/stream experiments, but avoid adding graph
  dependencies to the hot path without a measured reason.
- `mx.to_fp8` / `mx.from_fp8`: exposed in the binding for e4m3 conversion. These
  belong to a separate FP8 storage/compute experiment rather than the current
  BF16 parity path.

Confirmed present in the diffusers venv:

```bash
/Users/diffuser/.venvs/diffusers/bin/python - <<'PY'
import mlx.core as mx
for name in [
    "block_masked_mm", "gather_mm", "gather_qmm", "hadamard_transform",
    "quantized_matmul", "quantize", "dequantize", "qqmm", "segmented_mm",
    "depends", "to_fp8", "from_fp8",
]:
    print(name, hasattr(mx, name))
PY
```

### Memory and profiling

MLX exposes active, peak, and cache memory counters, plus cache/memory limit
controls. The local Metal debugger docs also show `mx.metal.start_capture(...)`
and `mx.metal.stop_capture(...)` for GPU traces when running with
`MTL_CAPTURE_ENABLED=1`.

For this repo, use lightweight timing first, then add Metal capture only for a
small repro. Capturing a full long generation is usually too noisy.

The distributed docs also mention `MLX_METAL_FAST_SYNCH=1`, which enables a
different CPU/GPU synchronization path and is not specific to distributed
backends. In a local single-process LTX-2.3 AV smoke test at 352x192, 15s,
8 steps, `--low-memory`, and `--save-all-sidecars`, it did not improve denoise
speed: the default sync run reported `RUN 3m14s` / `avg 24.3s/it`, while the
FAST_SYNCH run reported `RUN 3m16s` / `avg 24.5s/it`. Treat it as tested with
no local win for this workload, and only retest if the workload or MLX runtime
changes materially.

### Conversion, indexing, and loops

The remaining usage docs add a few performance guardrails:

- NumPy conversion evaluates the graph. For BF16 arrays it also requires an
  explicit cast to FP32 or FP16 first, because NumPy does not support BF16.
  Keep NumPy stats and frame/audio analysis out of hot denoise timing unless
  they are explicitly part of the measurement.
- MLX slicing creates a copy, not a view. Do not assume slice-heavy hot-path
  code is free just because the equivalent NumPy code would be view-based.
- `vmap` can remove Python loops and can be dramatically faster for repeated
  elementwise/batched work. It is not an obvious transformer-forward fix, but it
  is worth considering for probe scripts, batched analysis utilities, or any
  repeated per-token/per-frame helper that is currently expressed as Python
  iteration.
- MLX export/import can serialize graph traces and inspect graph primitives via
  an export callback. This is more useful for graph inspection or C++ reuse than
  immediate Python denoise speed, but it could help inspect a narrowed helper.

## Candidate Optimizations

### 1. Compare `--low-memory` against default eval cadence

Status: tested on the small AV smoke, no observed win; experimental patch not
retained.

`--low-memory` reduces peak memory by forcing more frequent intermediate
`mx.eval()` calls inside the transformer. That can make long runs fit, but it can
also slow denoising because it materializes the lazy graph more often.

Current behavior in `LTXModel`:

| Mode | Transformer eval cadence | Expected effect |
| --- | ---: | --- |
| `--low-memory` | every 4 transformer blocks | lower peak memory, slower |
| default | every 8 transformer blocks | higher peak memory, faster |
| `fast_mode` | no intermediate evals | highest peak memory, potentially fastest |

Local result:

- `--low-memory`: `RUN 3m14s`, `avg 24.3s/it`, `generation + decode 3m35.4s`.
- Default eval cadence: `RUN 3m17s`, `avg 24.7s/it`,
  `generation + decode 3m38.5s`.

The tested command was the 352x192, 15s, 8-step AV smoke with BF16, zero VAE
padding, and `--save-all-sidecars`. At this size, removing `--low-memory` was
slightly slower rather than faster, so default eval cadence is not an immediate
local win. Retest on larger shapes only if the run fits and memory pressure or
graph size changes the tradeoff.

Benchmark recipe:

```bash
# Baseline: current memory-safe path
python scripts/generate.py "Your prompt" \
  --generate-audio \
  --low-memory \
  --save-run-log \
  --output outputs/baseline_low_memory.mp4

# Same settings, but let MLX keep larger lazy graphs
python scripts/generate.py "Your prompt" \
  --generate-audio \
  --save-run-log \
  --output outputs/default_eval_cadence.mp4
```

Keep every other flag identical.

### 2. `MLX_METAL_FAST_SYNCH=1`

Status: tested locally, no observed win.

The MLX distributed docs describe `MLX_METAL_FAST_SYNCH=1` as a faster
synchronization path for cases where CPU and GPU collaborate. It is not claimed
as a general inference speedup, but LTX denoise progress, explicit evals, and
host-side loop control do involve CPU/GPU synchronization.

Local result:

- Default sync: `RUN 3m14s`, `avg 24.3s/it`, `generation + decode 3m35.4s`.
- `MLX_METAL_FAST_SYNCH=1`: `RUN 3m16s`, `avg 24.5s/it`,
  `generation + decode 3m37.4s`.

The tested command was the 352x192, 15s, 8-step AV kitten smoke with
`--low-memory`, BF16, zero VAE padding, and `--save-all-sidecars`. The result is
close enough to be noise, but it was slightly slower rather than faster, so this
should not be a priority candidate on the current local setup.

### 3. Benchmark AV `fast_mode`

Status: wired through the AV transformer path, tested on the small AV smoke
with no material speedup.

The video-only loader already accepted `fast_mode` and passed it into
`LTXModel`. The audio-video loader now does the same, so `--fast-mode` reaches
the LTX-2.3 AV path and sets the transformer's intermediate eval cadence to 0.

Expected benefit:

- Fewer forced materializations inside each denoise step.
- Best chance of improving denoise speed without changing output settings.

Local result:

- `--low-memory`: `RUN 3m14s`, `avg 24.3s/it`, `generation + decode 3m35.4s`.
- Default eval cadence: `RUN 3m17s`, `avg 24.7s/it`,
  `generation + decode 3m38.5s`.
- `--fast-mode`: `RUN 3m15s`, `avg 24.3s/it`,
  `generation + decode 3m35.8s`.

The tested command was the same 352x192, 15s, 8-step AV smoke with BF16, zero
VAE padding, and `--save-all-sidecars`. The load banner confirmed the flag was
active: `Loading AudioVideo transformer (BF16 (fast mode) (V2))...`. At this
size, removing intermediate evals is effectively tied with the low-memory
baseline, so eval cadence is probably not the bottleneck for this workload.
Retest on larger shapes if memory allows.

Risks:

- Higher peak unified-memory pressure.
- Possible tiny numeric drift from changed lazy evaluation boundaries. Treat
  visual/audio A/B as required, even if the math is intended to be equivalent.

Benchmark recipe:

```bash
python scripts/generate.py "Your prompt" \
  --generate-audio \
  --fast-mode \
  --save-run-log \
  --output outputs/fast_mode_test.mp4
```

Keep every other flag identical to the baseline. Watch active, peak, and cache
memory while testing. Higher peak memory is the expected cost of fewer
intermediate evals.

### 4. Precompute per-run RoPE and positional embeddings

Status: tested on the small AV smoke, no observed win.

Video and audio token positions do not change across denoise steps for a fixed
run. The current transformer preprocessor can rebuild RoPE and cross-modal
position embeddings each forward. Those tensors should be reusable across all
steps as long as shape, FPS, positions, rope type, and precision settings stay
the same.

Tested implementation:

- `TransformerArgsPreprocessor` caches self-attention RoPE for the current
  `positions` MLX array object.
- `MultiModalTransformerArgsPreprocessor` caches cross-modal temporal RoPE for
  the current `positions` MLX array object.
- Cache keys include object identity, shape, and dtype.
- Each cache is one-entry and local to the model preprocessor; there is no
  global or cross-run cache.
- Cached cos/sin arrays are materialized once with `mx.eval(...)`.

Expected benefit:

- Removes repeated fixed-shape setup work from every denoise step.
- Probably smaller than eval-cadence changes, but cleaner and lower memory risk.

Local result:

- `--low-memory` baseline: `RUN 3m14s`, `avg 24.3s/it`,
  `generation + decode 3m35.4s`.
- RoPE cache: `RUN 3m16s`, `avg 24.6s/it`,
  `generation + decode 3m37.0s`.

The tested command used the same 352x192, 15s, 8-step AV smoke with BF16, zero
VAE padding, and a precomputed text-conditioning sidecar. At this size, the
cache did not improve denoise speed; the repeated RoPE setup is either too
small to matter or hidden by larger transformer work. The temporary cache patch
was removed after this result.

Implementation cautions:

- Preserve the exact existing dtype and precision path.
- If retesting on a larger shape, watch peak memory. The caches should be small
  relative to transformer activations, but they deliberately keep materialized
  RoPE arrays alive for the run.
- Benchmark with the same text-conditioning sidecar if possible so prompt
  encoding variance stays out of the comparison.

### 5. Remove repeated no-op casts and small allocations

Status: low-risk cleanup candidate.

The denoise path repeatedly constructs small arrays such as sigma/timestep
inputs and may cast contexts or latents even when they are already in the
configured compute dtype.

Expected benefit:

- Small by itself, but useful combined with other cleanup.
- Reduces allocation noise and makes profiling easier.

Implementation cautions:

- Keep scheduler/time/position math in the precision required for stability.
- Do not reintroduce the audio position-cast issue fixed during audio debugging.
- Guard casts by dtype when the result would be identical.

### 6. Quantize video FF projections

Status: wired as an opt-in research flag; all-layer `mxfp8` is faster but not
parity.

The FFN sub-profile on the 352x192, 15s, 8-step AV kitten smoke showed GELU is
not a meaningful target:

- Clean blocks 40 and 44 were about `0.51-0.52s`.
- `video ff gelu` was about `0.00s` / `0.7%`.
- `video ff project_in` was about `0.06s` / `12%`.
- `video ff project_out` was about `0.21-0.22s` / `42%`.

So the first useful FFN target was the second linear projection, not the
activation. The first projection is also selectable for isolated A/Bs, but it
is expected to be more quality-sensitive because its quantization error lands
before GELU.
`mx.qqmm` would have been the most direct activation-plus-weight quantization
test, but it is currently unavailable for this Metal workload. The practical
local experiment is weight-only `nn.QuantizedLinear` on video FF projections:

```bash
--video-ff-quantize project_out:mxfp8
```

The flag keeps the stock checkpoint path. For normal non-streaming loads it
loads BF16 weights first, then replaces selected video transformer block FF
projections with in-memory MLX `QuantizedLinear` layers. For block streaming it
builds a separate cache artifact with the selected FF tensors already quantized
so resident blocks can stream quantized weights directly. It does not quantize
audio FF layers. If the mode is omitted for a target, it defaults to `mxfp8`;
the public API is `--video-ff-quantize target:mode[,target:mode]`.

Local `project_out` layer-range results on the fixed 352x192, 15s, 8-step AV
kitten smoke with cached text conditioning, BF16 compute, low-memory mode, and
zero VAE padding:

| Quantized layers | Denoise `RUN` | Average | `generation + decode` | Visual read |
| --- | ---: | ---: | ---: | --- |
| none | `3m15s` | `24.4s/it` | `3m35.3s` | reference |
| 40-47 | `3m13s` | `24.1s/it` | `3m33.4s` | minute differences; no useful speedup |
| 32-39 | `3m19s` | `24.9s/it` | `3m40.8s` | slightly different; no speed win |
| 32-47 | `2m57s` | `22.1s/it` | `3m17.6s` | slightly different; useful speedup |
| 24-47 | `2m55s` | `21.9s/it` | `3m15.6s` | visibly different, roughly all-layer level |
| 0-23 | `2m57s` | `22.1s/it` | `3m17.4s` | visibly different, roughly all-layer level |
| 0-47 | `2m21s` | `17.6s/it` | `2m41.5s` | visibly different, not parity |

Current read: `32-47` is the best observed quality/speed tradeoff. Expanding
down to `24-47` only saved about two additional denoise seconds over `32-47`
while moving the output into the visibly-different bucket. Quantizing `0-23`
also produced all-layer-like visual differences at about the same speed as
`32-47`, so early-layer quantization is not a good way to recover more speed.
Quantizing `32-39` alone was slightly different but slower than the baseline,
so the `32-47` result should be treated as the current conservative candidate
rather than proof of an additive per-layer speed model. All-layer `0-47` should
be evaluated as a separate performance mode: non-parity, but potentially usable
if full-video quality is acceptable.

A larger 512x288, 20s, 8-step bakery AV smoke with seed 124 tested all-layer
`0-47` quantization against the earlier BF16 bakery run:

| Quantized layers | Denoise `RUN` | Average | `generation + decode` | Visual read |
| --- | ---: | ---: | ---: | --- |
| none | `10m22s` | `77.8s/it` | `11m03.2s` | reference |
| 0-47 `mxfp8` | `7m31s` | `56.3s/it` | `8m11.1s` | different faces; possibly less clear faces; may be seed-like rather than artifact-like |
| 0-47 `mxfp4` | `7m50s` | `58.8s/it` | `8m30.6s` | bread-making mechanics more confused and less natural; laptop felt laggier despite lower memory pressure |
| 0-47 `nvfp4` | `7m55s` | `59.4s/it` | `8m36.1s` | probably about the same mechanics as `mxfp8`, but no speed win in this in-memory partial-quant path |
| 0-47 `project_in:mxfp8` | `11m11s` | `83.9s/it` | `11m52.4s` | slower than BF16; younger baker identity became less stable |

The bakery all-layer `mxfp8` result is therefore a plausible fast/draft mode but
not a parity mode. The `mxfp4` result is not currently attractive: it was slower
than `mxfp8`, felt worse interactively, and damaged scene mechanics more
visibly. `nvfp4` is worth remembering because official LTX checkpoints exist in
that format, but this partial in-memory `project_out` test did not show a speed
benefit over `mxfp8`. Quantizing `project_in` alone is a dead end for this
runtime path: it was slower than BF16 and degraded identity stability.
Judge any fast mode by end-to-end acceptability: face clarity, lip/audio
coherence, hands/props, edge stability, and whether the scene still satisfies
the prompt.

Because all-layer quantization is visibly non-parity, use the layer selection
flag to find a quality/speed tradeoff:

```bash
--video-ff-quantize project_out:mxfp8 \
--video-ff-quantize-layers 32-47
```

Stronger compression modes can be tested explicitly:

```bash
--video-ff-quantize project_out:affine
```

The broader Comfy-style MXFP8 policy is exposed as a separate cache mode:

```bash
--stream-transformer \
--transformer-cache-quantize mxfp8-blocks
```

This does not consume Comfy checkpoint tensors directly. It rebuilds the local
converted transformer cache from the stock BF16 checkpoint and stores heavy
block linears as MLX-native `QuantizedLinear` tensors. The policy mirrors the
downloaded `mxfp8_block32` transformer split: attention Q/K/V/out/gate and
video/audio FF linears are quantized, while biases, Q/K norm weights, AdaLN
tables, connector weights, patch/output projections, VAE, audio VAE, and
vocoder remain full precision. Same-math FF/attention layout caches are disabled
for this mode because those weights are replaced by quantized linears.

To test whether the quant path mostly lost the same-math pretranspose win, use:

```bash
--stream-transformer \
--transformer-cache-quantize mxfp8-blocks-pretranspose
```

This packs `weight.T` and uses `mx.quantized_matmul(..., transpose=False)`.
It is a deliberately separate A/B mode because the quantization groups are now
formed along the transposed matrix axis, so quality and error distribution can
move relative to the plain `mxfp8-blocks` cache.

Risks:

- This is non-canonical and changes weights used by the denoise transformer.
- Speed may improve, tie, or regress depending on Metal quantized-matmul
  kernels and shape.
- Quality may change, so compare generated video, audio perception, and saved
  latents before keeping it.

### 7. Compile narrow repeated helper functions

Status: research candidate.

MLX `mx.compile` is most appropriate for pure functions that are called
repeatedly with compatible shapes. The safest targets are small helper functions
around denoise math, not the full AV transformer on the first pass.

Good candidates:

- Velocity to denoised conversion.
- Euler latent update helpers.
- Denoise-mask and timestep preparation, if measurable.
- Latent post-processing helpers that are pure and shape-stable.

Benchmark cautions:

- Compilation has overhead.
- A short smoke test can look worse if compile cost dominates.
- Measure steady-state behavior on an 8-step run.
- Do not create compiled lambdas inside the denoise loop.
- Avoid `shapeless=True` for helpers that use static shape values in reshapes.
- Do not print, save, convert to NumPy, or call `.item()` inside compiled
  helpers.

### 8. Historical compile experiments

Status: tried and removed from the public CLI after no useful speedup.

Some MLX projects compile larger model prediction functions, but this AV
transformer forward crosses modality dataclasses, lists of modules, optional
branches, conditioning objects, and per-block control flow. The first direct
attempt to compile the `X0Model` wrapper was rejected by MLX because `Modality`
dataclasses are not valid compiled function arguments. An array-only wrapper
could run after suppressing transformer-internal `mx.eval()` checkpoints, but
the bakery smoke tied baseline speed while carrying higher memory risk.

Measured results:

- Full transformer compile: 512x288, 20s, 8-step bakery AV smoke with cached
  text conditioning completed at about 77.1s/it (`RUN 10m17s`, total 11m05s).
  This tied the earlier uncompiled bakery run at about 77.8s/it.
- Video FF subpath compile: the same bakery smoke was still about 77.7s/it
  after three denoise steps, close enough to baseline to stop early.

Conclusion: keep these results as negative evidence. Do not reintroduce compile
flags unless a future MLX release changes the compilation behavior enough to
justify a fresh A/B.

### 9. Pretranspose video FF `project_out`

Status: default same-math layout optimization. Use `--video-ff-layout off` for
baseline A/Bs.

`nn.Linear` computes `x @ weight.T`. The video FF `project_out` is the largest
same-math linear hotspot in the clean block profile, so this experiment caches a
contiguous `weight.T` after loading stock BF16 weights and calls `mx.addmm`
against that cached layout. It does not quantize weights or intentionally change
precision.

The default generator path enables:

```bash
--video-ff-layout project_in:pretranspose,project_out:pretranspose
```

Notes:

- `project_in:pretranspose` is also enabled by default for the video FF input
  projection. Unlike the earlier `project_in:mxfp8` quantization run, this is
  same-math.
- Use `--video-ff-layout-layers 0-47` or a narrower range to control where the
  extra cached transposes are created.
- The first implementation duplicated selected `project_out` weights; the
  current path materializes each transposed weight layer by layer and drops the
  original weight immediately afterward. Peak memory can still rise during the
  transform, but steady-state should be much closer to memory-neutral.
- Keep this separate from `--video-ff-quantize`; the CLI rejects combining them
  so each run answers one benchmark question.

Measured result:

- 512x288, 20s, 8-step bakery AV smoke with cached text conditioning and
  `--video-ff-layout project_out:pretranspose --video-ff-layout-layers 0-47`
  completed at about 60.3s/it (`RUN 8m02s`, total 8m51s). That is materially
  faster than the roughly 77s/it BF16 bakery baseline, but the original
  duplicate-cache implementation raised process memory by about 5GB and memory
  pressure spiked early before settling. Retest this result after the
  replacement-layout cleanup.
- After changing the layout transform to replace the original `project_out`
  weight instead of keeping both layouts, the same all-layer bakery run was
  stable around 55s/it with about 44GB process memory. This is the current best
  same-math denoise optimization.
- Adding `project_in:pretranspose` on top of `project_out:pretranspose` showed
  about the same memory and about the same 55s/it bakery speed. It appears safe
  for inference, but not an additional speed win over `project_out` alone.
- A 24-47 layer-only bakery run was stopped after three denoise steps: first
  step was about 67s, later steps slowed toward 74s/it, process memory was about
  47GB, and memory pressure remained bad. This range is not a promising
  compromise.

### 10. Pretranspose video attention `to_out`

Status: default same-math layout optimization. Use `--video-attn-layout off` for
baseline A/Bs.

Attention output projections also compute `x @ weight.T`. This experiment
materializes contiguous transposed `to_out` weights after loading stock BF16
weights, then drops the original projection weights. For AV blocks it applies to
video self-attention, video text-attention, and audio-to-video attention; it
intentionally skips audio-only output projections.

The default generator path enables:

```bash
--video-attn-layout to_out:pretranspose
```

Recommended baseline A/B:

- Disable the default same-math layouts with
  `--video-ff-layout off --video-attn-layout off`.
- Watch both denoise `avg .../it` and steady process memory. The attention
  `to_out` projections are smaller than FF `project_out`, so the upside is
  modest, but the replacement layout should avoid the duplicate-cache problem.

Measured result:

- Combined with `--video-ff-layout project_out:pretranspose` on the same
  512x288, 20s, 8-step bakery AV smoke, all-layer `to_out:pretranspose` ran at
  about 54s/it with the same steady memory as FF `project_out` layout alone.
  This is a marginal positive result: keep it available, but treat FF
  `project_out:pretranspose` as the main same-math win.
- A step-8 block profile after both layouts showed video self-attention as the
  largest remaining per-block bucket at about 39-41% for normal sampled blocks,
  followed by video text-attention and then the split FF matmuls.
- The finer attention profile on blocks 16 and 40 showed video self-attention
  SDPA as the largest single sub-piece: about 0.36-0.39s, or 22-25% of the
  forced-eval block profile. Video self-attention Q/K/V were about 0.06s each,
  `to_out` was about 0.05-0.06s, and RoPE/norm/gating were tiny. The combined
  video FF `project_in` + `project_out` path was still about 0.47-0.48s per
  sampled block. This points away from more output-layout work and toward either
  SDPA/token-count limits or a very selective Q/K/V layout experiment.

### 11. Cap the MLX allocator cache

Status: enabled by default as a same-math memory-pressure knob.

MLX keeps an allocator cache to avoid returning buffers to the system between
operations. That can improve reuse, but on unified-memory Macs it can also keep
process memory and swap pressure higher than the active graph requires. This is
separate from the on-disk `--weights-cache`: it controls MLX's in-memory
allocator cache only.

This is enabled by default:

```bash
--mlx-cache-limit-gb 1
```

Measured result:

- On the 512x288, 20s, 8-step bakery AV smoke using the current same-math
  `project_out:pretranspose` layout, `--mlx-cache-limit-gb 1` reduced average
  process RAM from about 44GB to about 40GB.
- The same run did not show an added time cost at a 1GB cache limit.
- This does not change model math, checkpoint precision, or output quality. Keep
  it independent from `--low-memory`, quantization, and layout experiments.

The current default same-math constrained-memory stack is equivalent to:

```bash
--weights-cache auto \
--mlx-cache-limit-gb 1 \
--video-ff-layout project_in:pretranspose,project_out:pretranspose \
--video-attn-layout to_out:pretranspose
```

`--weights-cache auto` now caches non-transformer weights as named families:
connector, video VAE, audio VAE, and vocoder. `--weights` remains the normal
full-bundle source, while `--transformer-weights`, `--connector-weights`,
`--vae-weights`, `--audio-vae-weights`, `--vocoder-weights`, and
`--config-weights` allow mixed-source A/B runs without pretending all
non-transformer weights are one opaque component blob.

The attention layout can still be A/B tested with `--video-attn-layout off`;
it remains enabled by default because it is same-math and measured
neutral-to-small-positive with the FF layout stack.

### 12. Stream transformer blocks from the weights cache

Status: useful constrained-memory path, not the fast path.

Cache-backed block streaming keeps only a small resident pool of transformer
blocks and rotates cached block weights through it. This is a memory tradeoff,
not a free speed win. It preserves model math, but it repeatedly rebinds 48
block weight sets per denoise step, so latency is worse than keeping the full
transformer resident.

The practical advantage is SSD write avoidance under constrained memory. After
the one-time weights-cache build, streaming reuses read-only safetensors pages
and can lean on the macOS file cache instead of forcing the full transformer and
MLX allocator cache to stay resident. If that prevents swap pressure, it avoids
pummeling the internal SSD with swap writes. Cold or evicted file-cache pages can
still cost read bandwidth and latency, so this is an SSD-friendly/constrained-RAM
mode rather than a throughput mode.

Enable the recommended preset with:

```bash
--stream-transformer
```

That expands to the current known-good default shape:

```bash
--weights-cache auto \
--transformer-block-resident-blocks 16 \
--transformer-block-compile \
--transformer-block-compile-group-size 4
```

Measured bakery AV results with `project_out:pretranspose`,
`to_out:pretranspose`, `--mlx-cache-limit-gb 1`, 512x288, 20s, 8 steps,
seed 124:

- `--transformer-block-resident-blocks 4`: process RAM was around 8GB average,
  denoise RUN `9m24s`, average `70.5s/it`, total `10m27.9s`.
- Adding the original per-resident-block `--transformer-block-compile`: no
  compile fallback warning, denoise RUN `9m01s`, average `67.6s/it`, total
  `10m10.3s`. It was faster on earlier steps, then slowed in later steps.
- `--transformer-block-resident-blocks 8`: about the same speed class as r4;
  without the MLX cache cap it drifted from roughly 60s early steps to roughly
  67s later steps.
- r8 resident-group compile without `--low-memory`: first attempt aborted
  before denoise step 1 with Metal `Impacting Interactivity` despite no visible
  memory pressure; immediate retry completed with STEP1 `0m58s`, denoise RUN
  `8m10s`, average `61.2s/it`, total `9m07.7s`. This points at file-cache /
  command-buffer state, not a deterministic allocator-capacity result.
- `--transformer-block-resident-blocks 16`: around 16GB process RAM and about
  the same time class in normal desktop conditions. With essentially no other
  foreground workload, r16 reached stable `54s/it`, showing that block streaming
  can approach the full-resident same-math speed when the transformer cache
  stays hot and the system avoids memory/cache contention.
- r16 resident-group compile with live Gemma, `--mlx-cache-limit-gb 1`,
  `project_in:pretranspose,project_out:pretranspose`, and `to_out:pretranspose`
  completed at STEP1 `0m54s`, denoise RUN `7m20s`, average `55.0s/it`,
  `generation + decode 8m09.6s`, total `8m57.4s`. This run included a one-time
  fresh transformer cache build (`25.61s`, 240 layout tensors); a warm cache
  should remove most of that load cost.
- 1024x576x481 with the same r16 resident-group compile shape completed step 1
  at `6m23s`, then aborted with Metal `Impacting Interactivity`. The latent
  grid was `61x18x32`, or 4x the 512x288 spatial token count, and attention work
  scales roughly quadratically with video tokens. For this size, full r16
  compiled groups are too large for the watchdog. Prefer keeping the resident
  window large enough for weight-cache locality, but splitting compiled/eval
  command-buffer groups with `--transformer-block-compile-group-size 4` or `8`.
  If that still stalls the desktop, fall back to smaller resident windows (`r4`
  first, then `r1` if needed) and `--mlx-cache-limit-gb 0` for stability, or add
  denoise/modality tiling before treating this as a viable full-size mode.
- 1024x576x481 with r16 resident blocks and `--transformer-block-compile-group-size 4`
  completed without a watchdog abort. It remained visibly laggy but usable, and
  the resulting video looked good enough to keep this as a known-good direct
  1024 render recipe rather than a failed path. Timing was STEP1 `6m49s`,
  denoise RUN `56m38.5s`, average `424.8s/it`, tiled native Conv3d VAE decode
  `5m01.2s` (`temporal=12`, `spatial=1x3`, `36` tiles), audio decode `13.0s`,
  output save `25.5s`, total `62m21.7s`. Treat direct 1024 as a premium
  walk-away mode; try compile group `8` only as a speed/stability A/B, and
  compile group `2` only if desktop interactivity matters more than runtime.
- [MLX issue #3267](https://github.com/ml-explore/mlx/issues/3267) points at
  active-display / WindowServer contention as a trigger for the same Metal
  `Impacting Interactivity` failure, even when memory is not the limiting
  factor. Local MLX source exposes two process-start env knobs,
  `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`, with Metal defaults
  around `40` ops / `40 MB` on base/pro architectures. These can split command
  buffers between ops, but cannot split a single oversized op; treat them as a
  watchdog-pressure experiment to combine with smaller compile groups or smaller
  resident windows, not as a replacement for denoise/modality tiling at larger
  token counts.
- The SDPA-specific watchdog path is plausibly the same failure class discussed
  in [MLX issue #3302](https://github.com/ml-explore/mlx/issues/3302) and the
  closed chunked-SDPA PR #3307. Current MLX 0.31.2 does not expose
  `MLX_SDPA_CHUNK_THRESHOLD` / `MLX_SDPA_CHUNK_SIZE`, and the PR's default
  `65536` key threshold would not trigger for the 1024x576 LTX latent grid
  (`35136` tokens). Local SDPA-query and FF-token chunking experiments matched
  baseline math on tiny tests, but did not show a useful runtime win and added
  more hot-path branches than they earned, so they were removed.
- Added `--stream-transformer` as the user-facing preset for r16 resident blocks,
  resident-group compile, and 4-block compile groups. Keep the lower-level flags
  for A/B tests and watchdog tuning.
- Added `--transformer-block-compile-group-size` as the cleaner watchdog lever
  for cache-backed block streaming. It keeps the resident block window unchanged
  for weight locality, but splits compiled/eval command-buffer groups into
  smaller subgroups. Example: `--transformer-block-resident-blocks 16
  --transformer-block-compile --transformer-block-compile-group-size 4` keeps
  16 resident blocks hot while materializing four blocks at a time.
- Tried packed attention layouts on the quiet r16 path. `self_qkv:pack` plus
  `to_out:pretranspose` reached stable `53s/it`; adding `kv:pack` also landed
  around `53s/it`. Both packing paths were removed because the measured speedup
  versus `to_out:pretranspose` was neutral-to-tiny while adding extra runtime
  branches, cache variants, resident-block state, and CLI surface area.
- Forcing a cache clear after each full block sweep made the first step worse
  at about 71s, so do not clear MLX cache inside the denoise loop.

Follow-up implementation notes:

- Streaming now uses the resident window as its sync boundary instead of
  inheriting `--low-memory`'s every-4-block eval cadence. This should avoid
  redundant evals for r8/r16 while still materializing before resident slots are
  reused.
- `--transformer-block-compile` now compiles the whole resident window as one
  group (`inputs=blocks`) instead of compiling each block separately. Initial
  r8 testing without `--low-memory` improved the completed run to about
  `61.2s/it`, but a preceding attempt hit the Metal interactivity watchdog
  before step 1. Treat it as promising but cache/watchdog-sensitive.

Conclusion: keep block streaming as an opt-in memory mode. r16 looks like the
best practical compromise so far: it can avoid swap-write pressure while still
approaching full-resident speed on a quiet system. Packed attention QKV/KV
experiments were removed after neutral-to-tiny measured payoff. Resident-group
compile is also neutral-to-small-positive, but none of these has removed the
broader dependence on macOS file-cache and memory pressure. If memory
comfortably fits, the full-resident same-math layout path remains the known fast
path.

### 13. Use `vmap` for repeated helper/probe loops

Status: opportunistic cleanup candidate.

MLX docs show `vmap` can remove Python loop overhead for batched array work. The
main transformer forward is already vectorized, so this is not a first-line
denoise optimization. It may still help:

- Analysis/probe scripts that loop over frames, crops, or channels.
- Batched prompt or text-conditioning comparisons.
- Any future helper that applies the same pure function across many positions
  using a Python loop.

Keep this separate from the main denoise hot path unless profiling points at a
real Python loop.

### 14. Transformer cache quantize (`mxfp8-blocks`) at 1024×576

Status: tested at 1024×576×481, no net speed win over the pretranspose-layout
path; auto-disables same-math layouts.

`--transformer-cache-quantize mxfp8-blocks` quantizes KV cache residual tensors
to mxfp8 after each block using block-granularity scaling. When this mode is
active the runtime sets `transformer_cache_quantize_layouts_disabled: true`
automatically, skipping the pretranspose FF and attention layouts because the
quantized cache already changes the in-memory layout contract.

Stage-2-only result at 1024×576×481 frames, full-resident (no streaming),
seed 124, distilled 3-step stage-2 denoise:

| Mode | Stage-2 denoise | Avg per step |
| --- | ---: | ---: |
| `mxfp8-blocks` (no layouts) | `1380.8s` | `460s/it` |

The existing `--stream-transformer` preset at the same resolution (r16,
compile, group-4, with pretranspose layouts) completed at about `424.8s/it`.
The mxfp8-blocks full-resident path is slower than the streaming pretranspose
path, not faster.

`mxfp8-blocks-pretranspose` (a separate cache mode that packs `weight.T` before
quantizing, so `mx.quantized_matmul` receives a pre-transposed matrix) was also
tested. That result matched `mxfp8-blocks`-only speed: the layout benefit does
not stack on top of a quantized cache at this resolution.

Conclusion: `mxfp8-blocks` does not improve denoise throughput at 1024×576 on
M1 Max under MLX 0.31.2. Keep the flag as a memory-pressure and hardware
experiment, but do not promote it as a speed preset.

### 15. Add a fused split-RoPE kernel

Status: only consider after RoPE precompute/profiling.

The codebase has a fused interleaved RoPE path, but LTX-2.3 uses split RoPE.
Precomputing RoPE is a simpler first move. A custom split-RoPE kernel is only
worth it if profiling still shows split-RoPE application as a significant cost.

Before writing a custom kernel, test whether `mx.fast.rope` can represent the
needed LTX-2.3 SPLIT 3D RoPE exactly. If it cannot, a custom Metal kernel should
be built once and should avoid hidden row-contiguity copies.

### 16. Distributed tensor parallelism

Status: separate project, not a local optimization.

MLX has distributed communication, tensor-parallel linear layers, and
Thunderbolt-oriented backends. The tensor-parallel example shows the usual
transformer pattern: shard Q/K/V and FFN up/gate projections, then aggregate
output/down projections.

This could be relevant for very large Apple Silicon setups, but it is outside
the current goal of making the existing single-process path faster. It would
require a distributed launch workflow, sharded loader/weights, communication
benchmarking, and quality/performance validation. Track it separately if the
project ever targets multi-Mac inference.

### 17. Defer AV text encoder load until after Gemma

Status: implemented for prompt-encode peak memory.

The AV text-conditioning path now runs Gemma for the positive and negative
prompts first, trims/materializes only the real-token hidden states, frees Gemma,
and only then loads the AV text encoder. This avoids overlapping Gemma 3 12B
weights with AV connector weights during prompt encoding.

Measured bakery AV run:

- Previous live-Gemma path sometimes peaked above 30GB process RAM during prompt
  encoding.
- Deferred connector path peaked around 24GB process RAM in Activity Monitor.
- Text sidecar parity against the previous live-Gemma run was exact: all 15
  `.npz` keys matched, including positive/negative video encodings, audio
  encodings, masks, dtype metadata, prompt strings, and schema version.

Do not remove Gemma's per-layer `mx.eval(hidden_states)` as a first move for
memory. It may help speed, but it may also allow a much larger lazy graph to
accumulate across the 48-layer text model.

### 18. Throttle terminal redraws

Status: implemented.

Terminal.app and WindowServer are GPU-accelerated on macOS. Any time the
denoise progress line repaints, that paint runs through Core Text and Metal on
the same GPU MLX is using for the diffusion forward pass. Activity Monitor on
the bakery 1024x576 distilled run showed Terminal at ~15% GPU and WindowServer
at ~15% during stage 2; both dropped to zero the moment the denoise loop ended
and VAE decode started, confirming the contention.

The previous `DenoiseProgress` class ran a daemon thread that called `_render`
every `0.12s` (~8 Hz). Each render printed a ~120-char line with eight ANSI
color escapes and a `\033[2K` full-line clear, then `flush=True`. For a 6-min
stage-2 step that is roughly `3000` GPU-accelerated repaints per step, all
contending with MLX. Other progress bars in the project used `tqdm` defaults
(10 Hz refresh, unicode block-char bar) for VAE encode/decode, tiled decode,
Gemma forward, weight loading, upscaler loading, and frame saving.

Changes:

- `DenoiseProgress` no longer spawns a heartbeat thread. The line is repainted
  only on actual step boundaries via `update()`, plus once at `start()` and
  `finish()`. The spinner glyph, all ANSI color escapes, and the `\033[2K`
  clear are gone. Repaints `\r`-overwrite and pad to the previous line length
  with spaces, so Terminal only re-rasterizes the changed characters
  (typically the right-hand `RUN`/`ETA`/pace fields). A byte-equality cache on
  the rendered line skips the `write(2)+flush` entirely when nothing changed.
- All `tqdm(...)` call sites in `scripts/` and `LTX_2_MLX/` pass
  `ascii=True` (plain `#` bar instead of unicode block chars, which Terminal
  routes through font fallback and glyph shaping) plus `mininterval=2.0` for
  hot paths during MLX GPU work (VAE encode/decode, tiled decode, denoise
  helper) and `mininterval=1.0` for cold paths (weight/shard/upscaler load,
  Gemma forward, frame saving).

Measured bakery 1024x576x481 distilled AV (same prompt, seed 124, weights,
flags, machine; only diff is the redraw policy):

| Phase                     | Before    | After     | delta     |
|---------------------------|-----------|-----------|-----------|
| Stage 1 denoise (8 steps) | 7m 04.2s  | 7m 00s    | -4.2s     |
| Stage 2 denoise (3 steps) | 21m 10.7s | 19m 29s   | -101.7s   |
| VAE decode (4 tiles)      | 2m 19.1s  | 2m 15s    | -4.1s     |
| Total                     | 31m 28.9s | 29m 38s   | -1m 50.9s |

Stage 2 went from ~423.6 s/it to ~389.5 s/it. The win scales with step
duration: longer steps had more per-step redraws crammed into them, so killing
the heartbeat reclaims more wall time per step. ~6% off total wall time and
~8% off the dominant phase from a terminal-redraw fix is unusual but
reproducible on a GPU-accelerated terminal stack.

If profiling on a non-macOS or non-GPU-accelerated terminal, expect the gain
to be much smaller — most of the cost was Terminal.app's Metal repaint, not
syscall overhead.

## Ideas To Avoid As First Moves

### KV caching text cross-attention

Not obviously valid for the current AV transformer. Text cross-attention is
conditioned through per-step/per-block modulation, so K/V reuse is not a simple
drop-in cache. The MLX Llama example uses KV cache for autoregressive token
generation, but that does not transfer directly to diffusion denoising where the
latent state and timestep conditioning change every step.

### Scheduler rewrites

Scheduler math is tiny compared with the 48-layer AV transformer forward. Keep
it correct and readable unless profiling proves otherwise.

### Quantization as a default speed fix

Quantization may reduce memory and can help some workloads, but it changes
precision behavior and may require checkpoint conversion or quality validation.
Track it separately from "same checkpoint, same pipeline, faster MLX runtime."

### Decode-only changes

VAE tiled decode, VAE spatial padding, and output encoding affect decode quality,
memory, or save time. They are useful, but they are not denoise-speed fixes.

Native Conv3d VAE decode is the default generator decode path. It keeps the
existing final-latent and tiling machinery while swapping the per-tile decoder
from the PyTorch-layout slice-conv implementation to MLX's native channel-last
`Conv3d`. Use `--vae-decoder simple` only when you want the older baseline for
A/B testing. `scripts/compare_vae_decoders.py` can compare timing, MLX
active/cache/peak memory, sampled luma stats, a contact-sheet diff, and a
full-motion side-by-side MP4 on a saved latent. The comparison script auto-muxes
a sibling `.wav` sidecar when present; pass `--no-audio` to keep the video silent
or `--audio /path/to/file` to choose a specific track.

Initial bakery latent A/B at 512x288x481:

- Auto temporal tiling: simple decode `38.1s`, peak `5.9GB`; native Conv3d
  `61.3s`, peak `3.0GB`.
- No tiling: simple decode `23.5-26.2s`, peak `32.1GB`; native Conv3d
  `29.4-36.4s`, peak `10.4GB`.
- Full-video uint8 diff between simple and native was tiny (`p95=2`,
  `p99=3`, max `37` in the no-tiling MP4 comparison). Subjectively this looks
  close enough to make native Conv3d the default lower-memory decoder while
  keeping `simple` available as the baseline.

End-to-end bakery AV smoke with warm caches, a precomputed text sidecar, `r16`
resident-group compile, FF `project_in`/`project_out` pretranspose, attention
`to_out` pretranspose, `--vae-decoder native-conv3d`, and `--vae-tiling off`:
denoise RUN `7m08s`, average `53.4s/it`; `generation + decode` `7m59.5s`;
total `8m07.3s`. This validates the native Conv3d path in the real generator at
512x288. At this size, native no-tiling is the fastest native decode mode
measured so far; use custom temporal tiling when decode peak headroom matters
more than decode time.

Isolated 1024x576x481 decode-only tests on the saved direct-1024 latent found a
native Conv3d full-volume tail failure, so do not use `--vae-tiling off` for
direct 1024 renders yet. The no-tiling run completed in `119.4s` with
`39.2GB` MLX-reported peak, but the final tail clipped white starting around
frame `455-456` and the luma mean jumped to `0.3306`. A one-tile custom run
(`--vae-temporal-tile-frames 512 --vae-temporal-overlap-frames 8`) reproduced
the same white tail (`118.8s`, `43.4GB`, luma `0.3306`).

Temporal-only multi-tile native Conv3d decode avoids that tail failure at
1024x576:

- `32/8`: `175.1s`, `7.1GB`, clean tail.
- `40/8`: `159.1s`, `7.5GB`, clean tail.
- `64/8`: `149.9s`, `9.7GB`, clean tail.
- `128/8`: `126.5s`, `14.9GB`, clean tail.
- `256/8`: `125.1s`, `25.8GB`, clean tail.

For now, `128/8` is the best 1024 native Conv3d safety/speed tradeoff on the
64GB test machine. `256/8` is only about a second faster in the decode-only
probe while using much more memory. The default native Conv3d `--vae-tiling
auto` uses a small RAM-derived planner: it keeps the simple decoder's legacy
conservative policy when `--vae-decoder simple` is requested, but the default
native path prefers the fastest temporal-only tile that stays under both the MLX
`0.31.2` `2^31` Conv3d output boundary and an estimated VAE decode budget. On
the 64GB machine, direct 1024x576x481 auto-selects `128/8` with no spatial
tiling. A live native-only auto run on the saved 1024 bakery latent selected
`128/8`, decoded in `126.54s`, peaked at `14.93GB`, and had a clean tail
(`first_bright_frame=null`).

The tail probe currently points at the final spatial upsample's native
Conv3d (`07_upsample_conv`): input after `06_res` is still healthy, depth-to-
space is not the cause, and both cache-clearing and causal temporal padding
diagnostics failed to fix the full-volume white tail.

A full-vs-window probe on `07_upsample_conv` made the failure deterministic:
running the same conv on frames `417:481` stayed healthy, while running it on
the full `481` frames collapsed after frame `455`. That lines up with a signed
32-bit element-index boundary. The `07_upsample_conv` output is
`481x72x128x512`; one frame is `4,718,592` elements, and
`2^31 / 4,718,592 = 455.11`. The last fully good frame was `454`, frame `455`
was a transition, and frame `456+` collapsed. Treat native Conv3d output tensors
above `2^31` elements as unsafe on MLX `0.31.2`. MLX PR
`ml-explore/mlx#3524` appears to fix this implicit Conv3d pointer-offset
overflow; once a local MLX release includes that PR, first rerun the standalone
Conv3d reproducer and then retest 1024x576 native Conv3d with
`--vae-tiling off`.

Custom VAE tiling controls are available for middle-ground tests:

```bash
--vae-decoder native-conv3d \
--vae-tiling custom \
--vae-temporal-tile-frames 128 \
--vae-temporal-overlap-frames 8
```

For decode-only A/B, the same knobs exist on
`scripts/compare_vae_decoders.py`. Pass `--decoder native-conv3d` to run only
the native decoder when testing tile sizes where the simple decoder might exceed
memory. `--vae-tiling off` is the fastest path when the peak memory fits and
the native Conv3d output stays under the MLX `0.31.2` int32 boundary. Use
`custom` only when you want to override the RAM-derived auto choice or sweep new
tile sizes. Avoid direct full-volume `--vae-tiling off` at 1024x576 until a
local MLX build includes the `ml-explore/mlx#3524` fix and the standalone Conv3d
reproducer passes.

A temporary native Conv3d eval-cadence switch was tested and removed. On the
512x288x481 no-tiling native decode, evaluating only at the end measured
`29.07s` and `10.44GB` peak versus the normal per-block materialization at
`29.77s` and `10.44GB`. The tiny speed delta was not worth an extra CLI branch:
even when reported peak memory is unchanged, fewer materialization boundaries can
increase graph size, reduce watchdog safety margin, and behave worse at larger
resolutions or under system pressure.

## 2026-05-15 Session: Audio pretranspose, skip-negative, and bakery parity

Investigation kicked off from a measured per-step gap to mlx-video at small T
(distilled `--generate-audio` at 256x256x25). Starting baseline was 46.85s
end-to-end (8+3 step distilled two-stage). Bakery (1024x576x481 frames) was
about 28-30 minutes total, with stage 2 alone at 18m 41s (~373s/it).

### What shipped (defaults changed)

#### `--internal-audio auto` and reverse-engineered audio-branch dispatch

`internal_audio_active = is_av_model and (use_internal_audio_branch or
audio_enabled)` in `pipelines/one_stage.py` made the legacy
`LTX_DISABLE_INTERNAL_AUDIO=1` knob silently ignored whenever
`--generate-audio` was set, because `audio_enabled` short-circuits the `or`.
Without `--generate-audio` the AV transformer model was still running the full
audio branch (audio self-attn + audio text-attn + A2V/V2A cross-modal) every
denoise step and discarding the result. The transformer call site dispatches
`self.transformer(video, audio)` whenever `audio_state is not None`; the
audio_state was always being created.

Fix: new `--internal-audio auto|on|off` flag in `scripts/generate.py`.
`auto` (default) resolves to `on` only when `--generate-audio` is set. A loud
startup banner now prints `Internal audio branch: ON|OFF [source]`. The
`--internal-audio off --generate-audio` combination errors at argv parse with
a clear message (you cannot output audio without running the branch). The
`LTX_DISABLE_INTERNAL_AUDIO=1` env var still works as a legacy override.

Impact on the 256x256x25 small bench (no `--generate-audio`): `47.50s ->
30.91s` (-34.9%) by skipping audio attention/FFN/cross-modal across every
step. This is a pure correctness fix: prior behavior was paying for audio
compute that was then thrown away.

mlx-video does the same thing already (see
`mlx_video/models/ltx_2/generate.py:1985-1991`): the distilled branch
encodes a single positive prompt, the dev branch encodes positive and negative
for CFG. Our prior behavior was strictly more work than the reference.

#### Audio module pretranspose (cache-integrated)

The existing same-math layout pipeline only walked the *video-Q* attention
modules (`attn1`, `attn2`, `audio_to_video_attn`) and `block.ff`. The
audio-side modules (`audio_attn1`, `audio_attn2`, `video_to_audio_attn`,
`audio_ff`) were running `mx.addmm(bias, x, weight.T)` with an implicit
transpose op on every per-step call.

Per AV block: 6 attention modules, each previously contributing 1
un-pretransposed `to_out` plus the 3 Q/K/V projections, plus 2
un-pretransposed audio FF projections. Across 48 blocks that is hundreds of
implicit transpose ops per step that the GPU pays for and then throws away.

Wiring touched `LTX_2_MLX/loader/transformer_cache.py` (added
`audio_ff_layout_specs` and `audio_attn_layout_specs` payload entries +
recognition in `_layout_cache_key`), `LTX_2_MLX/model/transformer/model.py`
(new `apply_audio_ff_layout` and `apply_audio_attn_layout`), and the
`load_av_transformer` call path. The cache hash bumps automatically because
the payload now includes the audio layout specs; old caches stay valid for
callers that haven't updated.

Opt out with `LTX_DISABLE_AUDIO_PRETRANSPOSE=1`.

Impact on the 256x256x25 small bench: `46.85s -> 41.61s` (-11.2%) with audio
pretranspose alone after audio FF + audio attn `to_out` were cached.

#### QKV pretranspose

Extended the attention layout machinery to `to_q`, `to_k`, `to_v` in addition
to `to_out`. Per AV block: 6 attention modules x 3 QKV projections = 18 more
matmuls per block per step where the implicit `weight.T` is now eliminated.
Defaults updated so all four attention projections are pretransposed by
default for both video and audio. Cache hash bumps; previously-built caches
under the old hash stay valid.

Impact: `41.61s -> 40.09s` (-3.7%) on top of the audio pretranspose, for a
combined `-14.4%` from the original 46.85s baseline.

#### Skip negative prompt encoding for distilled two-stage

`generate_distilled_two_stage` does not accept a negative encoding at all; its
docstring even says "There is intentionally no CFG in either stage." The
prompt encoding path was still tokenizing + running Gemma 3 + running the AV
text encoder twice (positive AND negative) and then passing `null_encoding`
to a function signature that never read it.

Fix in `scripts/generate.py`: when `distilled_two_stage_requested` and
`save_text_embeddings` is off, encode only `[prompt]` not `[prompt,
neg_prompt]`. Sets `null_encoding = null_audio_encoding = null_mask = None`
since they are never consumed. Verified mlx-video does the same.

Impact: `prompt encoding 12.20s -> 8.25s`, total `41.61s -> 36.21s` (-10%).
Re-enable the full encoding with `LTX_ENCODE_UNUSED_NEGATIVE=1` if you need
the negative encoding in a sidecar.

#### Tokenize without `padding="max_length"`

For a 2-token real prompt, `encode_av_gemma_batch` was padding to 1024
tokens and running the 12B Gemma forward on the full padded sequence before
trimming. The trim happened after the O(N^2) attention had already done the
work.

Fix: `padding=False` (default) in the tokenizer call inside
`encode_av_gemma_batch`. The hidden-state trim that already existed becomes a
no-op when the input is already minimal. Re-enable padding with
`LTX_PAD_PROMPT_TO_MAX=1` for debugging.

Impact: `prompt encoding 8.25s -> 6.24s`, total `36.21s -> 34.30s` (-5%).
Cumulative on the small-T AV bench from baseline: `46.85s -> 34.30s`
(-26.8%).

### Bakery (1024x576x481)

Re-run with the cumulative defaults: `28m 24s` total. Reference mlx-video
bakery time was `28m 20s`. We are at parity at the bakery scale; the small-T
gap was the symptomatic edge that the larger workload mostly masked. Stage 2
denoise specifically: `18m 41s` (~373s/it for 3 steps), with current
defaults, no compile.

### Experiments tried and found neutral (kept env-toggleable)

- `LTX_VELOCITY_MODE=1` (in `pipelines/one_stage.py`) bypasses `X0Model` and
  does the velocity-form Euler update inline. Mathematically identical;
  measured zero change at both 256x256 (46.85s -> 46.85s) and bakery scale.
  Confirms `X0Model` wrapper does not add measurable overhead. Kept gated
  for future MLX versions that may handle it differently.
- `LTX_COMPILE_BLOCK_GROUPS=N` (eager-path block-group `mx.compile`).
  Tested at 256x256 with `N=4` (40.21s warm vs 40.09s no-compile = neutral)
  and at the bakery with `N=48` (stage 2 18m 29s vs 18m 41s baseline =
  -1%). Compile-trace cost (5+ minutes at bakery scale) is paid up front and
  per-step does not recover it. mlx-video uses zero `mx.compile` and gets
  the same numbers. Earlier MEMORY note that `compile-group > 4` watchdog-
  hangs did not reproduce in this session at 1024x576; the warning is kept
  but the clamp is lifted.
- `LTX_ADALN_PRETRANSPOSE=1` extends the pretranspose to the eight
  `AdaLayerNormSingle.linear` projections (cache-integrated). At small T
  measured `+0.7s` vs baseline (slight regression). The aggregate adaln
  matmuls are too few per step (8/step) on a tiny batch to recover the
  per-tensor dispatch overhead. Code kept; not in defaults.
- `LTX_ROPE_PRECOMPUTE=1` matches mlx-video's per-stage RoPE precompute
  pattern. `Modality.positional_embeddings` field accepts an optional
  precomputed `(cos, sin)` tuple; the preprocessor honors it. Measured
  neutral (within noise) at small T - MLX's lazy graph already deduplicates
  the per-step `precompute_freqs_cis` calls.
- `to_gate_logits` pretranspose: machinery added under
  `ATTN_LAYOUT_SPECS["to_gate_logits"]`, removed from the default attention
  layout spec set. Weight is too small (4096x32 / 2048x32) for the implicit
  transpose to matter; measured `+0.81s` when included. Opt-in via explicit
  `--video-attn-layout to_out:pretranspose,...,to_gate_logits:pretranspose`.

### Experiments tried and removed

- `LTX_DISABLE_BLOCK_OVERHEAD=1` stripped-down `_fast_call` path in
  `BasicAVTransformerBlock` (removed `mark_profile` closure allocation,
  perturbation checks, asserts, the `_cross_attn_scale` getattr lookup, and
  the `TransformerArgs.replace()` kwargs.get loop). Measured zero change at
  small T and 159 lines of duplicate hot-loop code was not worth keeping.
  Theory: Python overhead between block calls is not the bottleneck. MLX's
  lazy graph already pipelines around Python-side dispatch.

### Profile tooling notes

`scripts/profile-watch.sh` (py-spy auto-attach watcher) was removed.
`scripts/profile.sh` keeps only the macOS `sample` backend; the py-spy
backend was removed because SIP gates `task_for_pid` on Darwin and the
no-root-for-child workaround is Linux-only. `sample` is sufficient for
"where is the time going" questions; use Instruments for deeper Metal
traces.

For Python flame graphs of MLX work specifically, note that `py-spy`
(without `--native`, which is Linux/Windows only) collapses almost all
time onto the `mx.eval` line because MLX is lazy and the actual execution
is on a different thread. Block-level `--profile-transformer-steps` +
`--profile-transformer-blocks` instrumentation is a better tool for
attributing time to specific phases inside one step.

### Monolithic-inlined transformer experiment (negative result)

After ruling out every cheap structural change to the per-block forward
(`LTX_DISABLE_BLOCK_OVERHEAD`, `LTX_VELOCITY_MODE`, `LTX_COMPILE_BLOCK_GROUPS`
all neutral), the remaining steelman was: collapse the entire 48-block
forward + preprocess + output projection into one inlined function with
flat pretransposed weights, no `nn.Module` dispatch.  The hypothesis: if
our `BasicAVTransformerBlock` / `LTXAVModel` / `X0Model` /
`TransformerArgsPreprocessor` chain costs anything at the MLX-graph level,
inlining should expose it.

Implemented in `scripts/mono_pipeline.py` (`InlinedAVModel` +
`transformer_step`).  `stage2_harness.py` swaps the pipeline's transformer
when `LTX_MONO_INLINED=1` is set so the rest of the run (spatial upscale,
sigma loop, modality construction, Euler step, VAE decode) stays
byte-identical.

Result on bakery 704x384 (3-step stage-2):

- **Math is equivalent.** Source modular `stage_2_video_latent` vs inlined
  `final_video_latent`: cosine similarity 0.99922 for video, 0.99977 for
  audio.  Mean/std match to 4 decimals.  p99 abs diff < 0.15.  Visually
  indistinguishable; classic BF16 rounding-order drift.
- **Wall clock is neutral.** Inlined 6m04s vs modular ~6m at this shape,
  within FaceTime/music GPU-contention noise.

That confirms the abstraction-cost hypothesis is empirically wrong on this
MLX version: the lazy graph optimizes through the `nn.Module` dispatch
chain just as well as flat-function code.  The remaining per-step gap to
mlx-video (when it exists) is structural at the MLX kernel / Metal
command-buffer level — not anything to chase by restructuring Python.

The code is left in place as a same-math reference implementation.  It
also doubles as a documented worked example of "what the V2.3 distilled AV
block actually computes," useful for future investigators who want to know
the exact op sequence without reading the modular Module tree.

Bug-hunt notes that may help if anyone resurrects the experiment:

- `Attention.q_norm` and `k_norm` are `RMSNorm` modules with **learned
  weights**, not just `mx.fast.rms_norm` with `weight=None`.  Missing this
  produced uniform brown-noise output and was the first bug to fix.
- `_prepare_timestep` multiplies the timestep by
  `timestep_scale_multiplier` (= 1000 for V2.3) for **every** AdaLN call,
  including `prompt_adaln`.  Forgetting the scale on prompt_adaln produces
  subtly broken output that looks worse than just-noise (was the second
  bug).
- The latent that reaches the transformer is already patchified ((B, N,
  128) by `create_initial_state`), not the 5D (B, C, F, H, W) the VAE
  decoder consumes.  Don't re-patchify.
- Cross-modal AdaLN is conditioned on the *other* modality's sigma:
  video's cross-attn uses audio's sigma and vice versa.  Easy to swap.

## Benchmark Matrix

Use a fixed command and change only one thing at a time.

| Experiment | Code change? | Memory risk | Expected denoise speed impact | Notes |
| --- | --- | --- | --- | --- |
| `--internal-audio auto` (default) | yes | none | -35% wall on video-only (256x256x25) | New CLI flag. Default `auto` resolves to `on` iff `--generate-audio`. Off otherwise. Replaces the silently-overridden `LTX_DISABLE_INTERNAL_AUDIO=1` legacy env. Matches mlx-video's distilled path. |
| Audio module pretranspose (default) | yes | medium | -11% AV (256x256x25) | `audio_attn1/2.to_*`, `video_to_audio_attn.to_*`, `audio_ff.project_*` now go through the same `mx.contiguous(weight.T)` cache path the video modules use. Cache hash bumps; old caches stay valid. Opt out with `LTX_DISABLE_AUDIO_PRETRANSPOSE=1`. |
| QKV pretranspose (default) | yes | medium | -4% additional AV | `to_q/to_k/to_v` added to the default attention layout spec. 18 more matmuls per AV block per step with the implicit transpose eliminated. Cache hash bumps. |
| Skip negative prompt encoding (distilled two-stage) | yes | none | -10% AV (256x256x25) | `generate_distilled_two_stage` doesn't accept a negative encoding, so we no longer encode one. Re-enable with `LTX_ENCODE_UNUSED_NEGATIVE=1` for sidecar debugging. Matches mlx-video. |
| Tokenize without max-length padding | yes | none | -5% AV (256x256x25) | Tokenizer was padding 2 real tokens to 1024 before the Gemma forward; trim happened after the wasted O(N^2) attention. Re-enable with `LTX_PAD_PROMPT_TO_MAX=1`. |
| `LTX_VELOCITY_MODE=1` | yes | low | neutral | Bypasses `X0Model` and does the velocity-form Euler update inline in `_denoise_loop_simple_av`. Same math. Tested at small T and bakery; both neutral. Kept env-gated for future MLX versions. |
| `LTX_COMPILE_BLOCK_GROUPS=N` | yes | medium | neutral at all tested scales | Eager-path `mx.compile` over N-block groups. `N=4` tested at small T (neutral), `N=48` at bakery (neutral: 18m 29s vs 18m 41s baseline). Compile-trace cost paid up front; per-step does not recover it. mlx-video uses zero compile and gets the same numbers. |
| `LTX_ADALN_PRETRANSPOSE=1` | yes | low | slight regression at small T | Cache-integrated pretranspose for the 8 `AdaLayerNormSingle.linear` projections. The 8 calls per step on a tiny batch don't amortize the per-tensor dispatch overhead. Kept opt-in. |
| `LTX_ROPE_PRECOMPUTE=1` | yes | none | neutral | mlx-video pattern: compute RoPE once per stage and pass through `Modality.positional_embeddings`. MLX's lazy graph already deduplicates per-step `precompute_freqs_cis` calls, so the savings don't show. |
| `to_gate_logits` pretranspose (opt-in) | yes | low | slight regression | Pretranspose support for the V2 per-head gate-logits Linear. Weight (4096x32 / 2048x32) is too small for the implicit transpose to matter. Opt-in via `--video-attn-layout to_out:pretranspose,...,to_gate_logits:pretranspose`. |
| `LTX_MONO_INLINED=1` (stage2_harness only) | yes | none | neutral | Replaces `av_pipeline.transformer` with `mono_pipeline.InlinedAVModel`: same math, 48-block forward + AdaLN preprocess + output projection all inlined into one function with flat pretransposed weights. Latent diff vs modular: cosine sim 0.999+, BF16 rounding-order noise only. Wall clock within measurement noise. Confirms `nn.Module` dispatch is free at the MLX-graph level. See section above. |
| `--profile-transformer-steps` | yes | low | diagnostic only | Forces eval checkpoints during selected denoise steps to locate hotspots. Do not use for final timing. |
| `--profile-transformer-blocks` | yes | low | diagnostic only | Adds forced eval checkpoints inside selected blocks for already-profiled steps. Block profiles now split attention into setup/AdaLN, Q/K/V, Q/K norm, RoPE, SDPA, gate, output, and residual sections. |
| FFN sub-profile | yes | low | diagnostic only | Selected block profiles also split FFN into AdaLN, `project_in`, GELU, `project_out`, and residual gate. |
| Remove `--low-memory` | no | medium | none observed in small run | 352x192 15s AV smoke was slightly slower. Retest larger shapes if they fit. |
| `MLX_METAL_FAST_SYNCH=1` | no | low | none observed | 352x192 15s AV smoke was slightly slower. |
| AV `fast_mode` | yes | high | none observed in small run | 352x192 15s AV smoke tied low-memory baseline. Retest larger shapes if they fit. |
| Defer AV text encoder load until after Gemma | yes | low | none, prompt-encode memory only | Reduces prompt-encode peak by avoiding overlap between Gemma 3 12B weights and AV connector weights. Gemma hidden states are trimmed to real tokens, materialized, then Gemma is freed before the connector loads. |
| Per-run RoPE precompute | yes | low | none observed in small run | Temporary one-entry cache patch was removed after a slightly slower 352x192 15s AV smoke. |
| No-op cast/allocation cleanup | yes | low | low | Good cleanup after larger wins. |
| Fast GELU approximation | no | low | unlikely | FFN sub-profile showed GELU at about 0.7% of a clean block; skip unless future profiles differ. |
| Historical compile experiments | yes | medium to high | none observed | Full transformer and video FF subpath compile were removed from the CLI after bakery AV smokes tied baseline speed while adding complexity and memory risk. |
| `mx.qqmm` FF linears | no | medium to high | blocked locally | Current Metal runtime throws `[QQMatmul] NYI for the general case` on LTX FFN-shaped tests. |
| `--video-ff-quantize` | yes | medium | useful in selected ranges | `project_out:mxfp8` layers 32-47 improved 352x192 15s AV denoise from about 24.4s/it to 22.1s/it with slight visual differences. Layers 0-23 and 24-47 were both visibly different; all-layer `mxfp8` is much faster but non-parity. With block streaming, quantized FF tensors are now stored in the converted cache and can keep resident-group compile for all-layer quantization; partial-layer streaming quant disables compile to avoid resident-slot type changes. All-layer `mxfp4` and `nvfp4` were slower than `mxfp8` on the bakery smoke, so keep `mxfp8` as the runtime-quant candidate for now. `project_in:mxfp8` was slower than BF16 and hurt identity stability. Official NVFP4 checkpoint support remains a separate loading/compatibility experiment. |
| `--transformer-cache-quantize mxfp8-blocks` | yes | medium to high | new non-parity cache experiment | Mirrors the downloaded Comfy MXFP8 block32 transformer policy in MLX-native cache form: quantize heavy attention and FF block linears, keep biases/norms/AdaLN/glue full precision, and require block streaming. This disables same-math layouts for the quantized weights and should be judged as a fast/draft-quality mode, not parity. |
| `--transformer-cache-quantize mxfp8-blocks-pretranspose` | yes | medium to high | new A/B mode | Same target set as `mxfp8-blocks`, but packs `weight.T` and calls quantized matmul with `transpose=False` to probe whether the plain quant path is losing the layout win. It changes quantization grouping orientation, so compare output quality separately. |
| `--video-ff-layout project_out:pretranspose` | yes | medium | default same-math win | Replacement-layout all-layer bakery AV smoke improved from roughly 77s/it BF16 baseline to about 55s/it, stable around 44GB process memory. Original duplicate-cache implementation was slower and more memory hungry. |
| `--video-ff-layout project_in:pretranspose,project_out:pretranspose` | yes | medium | default | Same-math and safe for inference. Bakery smoke showed about the same memory and about the same 55s/it as `project_out` alone, so it is included in the default layout stack for simplicity. |
| `--video-attn-layout to_out:pretranspose` | yes | medium | default marginal positive | Combined with `project_out:pretranspose`, all-layer bakery AV smoke improved slightly to about 54s/it with the same steady memory. |
| `--mlx-cache-limit-gb 0` or `1` | no | low | default 1GB | Same-math allocator-cache cap. `1` GB dropped bakery AV average process RAM from about 44GB to about 40GB with no observed time penalty. `0` returns freed buffers immediately and is worth trying for watchdog/cache pressure, but it will not make oversized compiled groups safe by itself. Keep separate from `--weights-cache`, which is an on-disk converted-weight cache. |
| `MLX_MAX_OPS_PER_BUFFER=1 MLX_MAX_MB_PER_BUFFER=10` | no | low | unknown | Must be set before Python starts. Real MLX command-buffer split knobs; useful for watchdog-pressure A/B after a Metal `Impacting Interactivity` abort. They can split between ops, not inside one huge op, so pair with smaller compile groups, smaller resident windows, or denoise/modality tiling for larger token grids. |
| `--stream-transformer` | yes | low to medium | constrained-memory preset | Expands to r16 resident blocks, resident-group compile, and 4-block compile groups. This is the preferred user-facing switch before reaching for the lower-level block streaming flags. |
| Stage-2 SVD/residual cache experiment | no | medium to high | removed | Removed wholesale after local A/Bs showed the quality/speed tradeoff was upside down. Broad block-window probes were fast but visibly distorted: `8-39` plus final-step reuse produced video relative MAE around 0.46, and `16-31` with the final step exact still produced foreground noise with video relative MAE around 0.26. The narrow `24-27` probe saved only about 15s on stage 2 while still showing moving-foreground noise and video relative MAE around 0.18. Stage-2 full-resolution refinement is too sensitive for this hidden-residual reuse path; keep it exact. |
| `--transformer-block-resident-blocks` | yes | low | slower | Cache-backed block streaming cuts process RAM dramatically, e.g. r4 around 8GB average on the bakery smoke, but denoise slowed to about 70.5s/it. Use as a constrained-memory mode, not a fast path. |
| `--transformer-block-compile` | yes | low to medium | mixed | Resident-group compile r8 without `--low-memory` completed at about 61.2s/it after one prior Metal watchdog abort, so it is promising but cache/watchdog-sensitive. Pair with compile group sizing for larger token grids. |
| `--transformer-block-compile-group-size` | yes | low to medium | stabilizes larger shapes, adds overhead | Splits compiled/eval command-buffer groups while keeping the resident block window unchanged. At 1024x576x481, resident 16 with group 4 completed without watchdog abort but still had usable desktop lag and took about 424.8s/it. |
| `--vae-decoder native-conv3d` | yes | medium | none for denoise | Default decode path. Validated in full generate with `--vae-tiling off`: bakery AV smoke total `8m07.3s`, denoise avg `53.4s/it`. Compare against `--vae-decoder simple` on the same saved final latent; do not count this as a transformer speed win. Pair with `--mlx-cache-limit-gb` if allocator cache growth matters. |
| `--transformer-cache-quantize mxfp8-blocks` | yes | medium | none observed at 1024×576 | Full-resident test at 1024×576×481 stage 2: `460s/it`. Auto-disables same-math layouts. Slower than `--stream-transformer` pretranspose path (`~425s/it`). Keep as memory/hardware experiment only. |
| `--transformer-cache-quantize mxfp8-blocks-pretranspose` | yes | medium | none observed at 1024×576 | Packs `weight.T` before quantizing. Tested alongside `mxfp8-blocks`; matched its speed — the layout gain does not stack on a quantized cache at this resolution. |
| `MLX_SDPA_BLOCKS` (MLX PR #3455) | no | none | inapplicable | Controls `sdpa_vector_2pass` (T≤8) only. LTX uses `sdpa_full` (T=35,136 tokens at 1024×576). No effect on any LTX denoise step. |
| Weight-only quantized transformer | yes | medium | unknown | Broader quantization than project_out only; separate quality/checkpoint tradeoff. |
| `mx.block_masked_mm` | no | high | unknown | Only relevant if we introduce structured block sparsity or pruning. |
| `mx.gather_mm` / `mx.gather_qmm` | no | high | unknown | Relevant to MoE/routing or selected batched matrices, not current dense LTX. |
| `vmap` helper/probe loops | yes | low | low | Useful only where Python loops show up. |
| `mx.fast.rope` / split-RoPE kernel | yes | low to medium | unknown | Only after proving exact RoPE parity. |
| Distributed tensor parallelism | yes | high | unknown | Separate multi-machine project. |
| FP8 conversion primitives | no | high | unknown | `mx.to_fp8` / `mx.from_fp8` are storage/compute research items, not safe BF16 path changes. |
| Terminal redraw throttling | yes | none | default win on macOS | Removed `DenoiseProgress` heartbeat thread; all `tqdm` use `ascii=True` plus `mininterval=1-2s`. Bakery 1024x576x481 distilled total `31m 28.9s` -> `29m 38s` (-5.9%); stage 2 alone `21m 10.7s` -> `19m 29s` (-8.0%). Win comes from killing GPU contention with Terminal.app/WindowServer; expect smaller gain on non-macOS terminals. See section 18. |

## Experiment Log Template

Copy this section when testing a candidate.

```text
### Experiment: <name>

Branch:
Commit:
MLX version:
Machine:
Weights:
Gemma:
Prompt:
Resolution:
Frames or duration:
FPS:
Steps:
Seed:
Audio:
Dtype:
VAE padding:

Baseline command:
Candidate command:

Baseline denoise RUN / avg:
Candidate denoise RUN / avg:
Baseline total:
Candidate total:
Peak memory notes:
Visual/audio notes:
Result:
Next action:
```
