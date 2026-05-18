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

These notes come from a local MLX checkout (e.g. `mlx-main`).

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
python - <<'PY'
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

`scripts/bench-process-watch.sh` is a tiny loop that pkill's
`mediaanalysisd`, `mediaanalysisd-access`, and `photoanalysisd` every
N seconds (default 2 s).  Those macOS background agents periodically
hammer the GPU with image-similarity / OCR work — visible in Metal
System Trace as random WindowServer-adjacent compute dispatches that
contend with our denoise steps.  Killing them in a loop (they'll be
relaunched by launchd, but the loop kills the relaunches too) gives
both cleaner traces and meaningfully more stable per-step timing on
the bench.  Use during any perf measurement or trace capture; stop
when done so the agents can do their normal background work.
Replaces an earlier `bench-quiet.sh` that tried to disable the
launchd jobs themselves (more invasive and didn't actually stick).

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

## 2026-05-16 Session: apples-to-apples Metal traces

Continuation of the perf investigation.  Goal: stop guessing about
"is mlx-video really faster, and if so why," and get matched Metal
System Trace captures that cover identical work windows.

### Profile hooks: pause-then-stop

Two env-gated helpers added to `OneStagePipeline.generate_distilled_two_stage`
in `LTX_2_MLX/pipelines/one_stage.py`:

- `LTX_PROFILE_PAUSE_BEFORE_DENOISE=1` — blocks on stdin once, immediately
  before stage 1's first denoise step starts.  Prints `pid=N` so a
  second-terminal `xcrun xctrace record --attach <PID>` can connect at
  a precise point, after model load / prompt encoding, before any
  denoise work.
- `LTX_PROFILE_STOP_AFTER_STEPS=N` — `sys.exit(0)`s cleanly after step N
  of the current denoise loop.  `N=2` gives one warmup step + one
  steady-state step, the minimum useful capture window for kernel-mix
  and dispatch-distribution analysis.

Both zero-cost when unset.  Mirroring hooks with identical env-var names
were added to `mlx-video` so the same shell incantation produces directly
comparable captures from both projects.

Capture template (works on either project):

```
LTX_PROFILE_PAUSE_BEFORE_DENOISE=1 LTX_PROFILE_STOP_AFTER_STEPS=2 \
caffeinate -di python -m <generate-entrypoint> --pipeline distilled \
  --height 576 --width 1024 --num-frames 481 --fps 24 --audio ...
```

Wait for the `[LTX_PROFILE_PAUSE_BEFORE_DENOISE] pid=N` print, then
in a second terminal:

```
xcrun xctrace record --template "Metal System Trace" \
  --attach <PID> --output my.trace --no-prompt
```

Then hit Enter in the original terminal.  Recording stops when the
process exits via `LTX_PROFILE_STOP_AFTER_STEPS`.

### 3-way apples-to-apples results (bakery 1024×576×481, distilled AV)

All three traces captured with `LTX_PROFILE_PAUSE_BEFORE_DENOISE=1
LTX_PROFILE_STOP_AFTER_STEPS=2`.  Thermal state "Nominal" throughout
all three runs — no throttling.  Same machine (M1 Max), same prompt,
same seed, same model weights.

| Metric                          | mlx-video | LTX w/ compile | LTX no-compile |
| ------------------------------- | --------- | -------------- | -------------- |
| **2-step wall time**            | **98.18s**| **103.47s**    | **106.80s**    |
| Per-step avg                    | 49.1 s    | 51.7 s         | 53.4 s         |
| GPU busy total                  | 99.88 s   | 101.67 s       | 101.87 s       |
| GPU utilization                 | 102 %     | 98 %           | 95 %           |
| Total dispatches                | 12143     | 12625          | 9085           |
| Dispatches / sec                | 124       | 122            | 85             |
| Median dispatch dur             | 962 µs    | 994 µs         | 918 µs         |
| **p99 dispatch dur**            | **44 ms** | **62 ms**      | **187 ms**     |
| Max dispatch dur                | 168 ms    | ~150 ms        | 325 ms         |
| Dispatch gaps > 20 ms           | **1**     | 38             | 50             |
| Largest dispatch gap            | 24 ms     | 100 ms         | **2283 ms**    |

The "no-compile" column was captured with `LTX_DISABLE_COMPILED_ATTN=1
LTX_DISABLE_COMPILED_HELPERS=1 LTX_DISABLE_FUSED_ROPE=1`.

### What this overturns from prior sessions

**`mx.compile` is not a no-op.**  Prior bench tables marked compile-vs-no
neutral.  In the apples-to-apples 2-step window it shaves **3.3 s
(–3.1 %)** off total wall time, cuts the p99 tail **3×** (187 ms → 62 ms),
and drops the largest dispatch gap from **2.28 s to 0.10 s**.  The earlier
"neutral" reading came from longer runs where the per-step delta got
amortized against larger absolute timings and warmup variance.  **Default
to compile on; do not disable it for perf reasons.**

**"mlx-video has more fused kernels" is backwards.**  In steady state
mlx-video runs **124 dispatches / sec** while we run **122 dps with
compile, 85 dps without**.  They are *less* fused than us-on-compile,
not more.  The "8 steel_attention specializations vs our 2–3" thing
isn't fusion — it's the opposite, finer-grained dispatch.  Compile is
working as designed: it bundles work into bigger kernels.

**GPU busy time is essentially equal across all three** (within 2 %).
The 5.4 % wall gap to mlx-video is *not* "they run faster math."  It's
entirely in dispatch dynamics — they have **zero meaningful warm-up**
(step 1 ≈ step 2 in their trace because they JIT-compile eagerly during
model load, before our pause point fires), and they have **one** 24 ms
dispatch gap across both steps where we have 37–50 gaps > 20 ms.

**Long-tail dispatches are the real bottleneck.**  p99 = mlx-video 44 ms
→ LTX w/ compile 62 ms → LTX no-compile 187 ms.  When one of our bundled
dispatches hits its slow case it stalls the queue for tens to hundreds
of ms.  mlx-video's smaller dispatches give them a much tighter
distribution.  This is the right place to spend the next batch of
investigation cycles.

### Signposts: attribute the long-tail dispatches

Three env vars added for per-phase trace attribution.  All zero-cost
when unset, all eager-init at module import (so the init prints don't
clobber the denoise progress bar):

- `LTX_PROFILE_SIGNPOSTS=1` — wraps the 8 sub-ops in
  `BasicAVTransformerBlock.__call__` (`video_self_attn`,
  `video_text_ca`, `audio_self_attn`, `audio_text_ca`, `a2v_cross`,
  `v2a_cross`, `video_ff`, `audio_ff`) with `os_signpost` intervals
  in subsystem `ltx`, category `OS_LOG_CATEGORY_POINTS_OF_INTEREST`.
- `LTX_PROFILE_SIGNPOSTS_SYNC=1` — also forces `mx.eval()` on each
  phase's output at the signpost end via a `signpost_barrier()` call.
  Required for **time-based attribution** to work — without this,
  MLX's lazy graph queues phase N+1's ops microseconds after phase N
  on the Python side while the GPU dispatches lag by seconds, so
  signpost intervals don't bracket the actual dispatches they
  produced.  `mx.synchronize()` alone is NOT enough: it only waits
  for already-dispatched GPU work, not for lazy ops still pending.
  Adds ~10 % wall-clock overhead at sync points.
- `LTX_PROFILE_SIGNPOST_LOG=/path` — writes a sidecar log with one
  line per begin/end event (`<monotonic_ns> <begin|end> <phase>`).
  Reliable belt-and-suspenders source if the trace's `os-signpost`
  table loses events under buffer pressure.

Implementation: `LTX_2_MLX/utils/signpost.py` ctypes-loads a tiny
C shim (`LTX_2_MLX/utils/_signpost.c`, auto-built on first import via
clang) that exposes the per-phase `begin`/`end` symbols.  The shim is
needed because `os_signpost`'s macro-based API embeds the calling
image's `__dso_handle`, which can't be passed from ctypes.

Disabled overhead: ~0.6 µs per context-manager enter (~5 ms total for a
2-step capture).  Enabled overhead: ~3 µs per enter (~17 ms total).
Both negligible vs ~50 s/step.

Capture recipe (apples-to-apples with mlx-video):

```
LTX_PROFILE_PAUSE_BEFORE_DENOISE=1 \
LTX_PROFILE_STOP_AFTER_STEPS=2 \
LTX_PROFILE_SIGNPOSTS=1 \
LTX_PROFILE_SIGNPOSTS_SYNC=1 \
LTX_PROFILE_SIGNPOST_LOG=/tmp/ltx_signposts.log \
caffeinate -di python scripts/generate.py "..." --pipeline distilled \
  --height 576 --width 1024 --frames 481 --fps 24 --generate-audio \
  --output /path/to/out.mp4
```

**Capture via the `xctrace record` CLI.**  After the process prints
`[LTX_PROFILE_PAUSE_BEFORE_DENOISE] pid=<N>` and before hitting
Enter, run in a second terminal:

```
xcrun xctrace record \
  --template "Metal System Trace" \
  --instrument "Points of Interest" \
  --attach <PID> \
  --output "${TMPDIR:-/tmp}/my.trace" \
  --no-prompt
```

This writes straight to the trace file as events come in and
captures every signpost.  Hit Enter in the original terminal to
start denoise; the recording stops when the process exits (which
happens automatically via `LTX_PROFILE_STOP_AFTER_STEPS=N`).

Don't use the Instruments GUI for this.  Its Deferred recording
mode drops signposts under buffer pressure (our app-level signposts
compete with ~40k Metal Shader Compiler signposts/run from
WindowServer + other Metal-using apps) and even Immediate mode is
lossy and slower to set up.  The CLI is strictly better.

After capture, export signposts + GPU intervals:

```
xcrun xctrace export --input my.trace \
  --xpath '//trace-toc/run/data/table[@schema="os-signpost"]' \
  --output signposts.xml
xcrun xctrace export --input my.trace \
  --xpath '//trace-toc/run/data/table[@schema="metal-gpu-intervals"]' \
  --output gpu_intervals.xml
```

Time-based attribution: each GPU dispatch is assigned to whichever
phase's `Begin` interval most recently preceded it.  With
`SYNC=1` the begin/end timestamps tightly bracket the actual GPU
work for that phase, so attribution is order-preserving.

If the trace's `os-signpost` table is short on events (drops under
buffer pressure), fall back to the sidecar log — it captures every
event regardless of OS buffer state.  Anchor by setting trace_t for
sidecar's last event = trace total duration.

### What we learned about signpost capture

Tried os_signpost-based phase attribution via the Instruments GUI
first.  Captures missed ~600 of our 1536 events per 2-step run, all
concentrated at the start of denoise.  Root cause: Metal Shader
Compiler emits ~40k `FunctionCompiled` signposts during a typical
capture from system-wide Metal use (WindowServer, Messages,
Instruments itself, AppKit XPC services), which crowds our
per-process signposts out of whichever shared buffer the GUI
processes them through.

Two false-start fixes that didn't help (and were reverted):

- **Pre-warming the transformer before the pause hook** to move our
  cold-start Metal compile outside the traced window.  Saved 1-2 s
  on step 1 but didn't help the trace gap because the source of
  buffer pressure isn't our compile, it's everyone else's.
- **Suppressing signposts during prewarm** to avoid pre-pause buffer
  pollution.  Same outcome.

Real fix: capture via the `xctrace record` CLI (documented above).
Captures every signpost reliably; the GUI is not worth bothering
with for this workflow.

### Per-phase attribution results (stage 1, 288×512, 2-step sync capture)

With sidecar-anchored attribution (every emitted signpost accounted for):

| Phase            | LTX %GPU | LTX p99 | LTX max | mlxv %GPU | mlxv p99 | mlxv max |
| ---------------- | -------- | ------- | ------- | --------- | -------- | -------- |
| video_self_attn  | 37.8 %   | 74 ms   | **180** | 42.3 %    | 43 ms    | 133 ms   |
| video_ff         | 37.7 %   | 42 ms   | 74 ms   | 32.9 %    | 50 ms    | **147**  |
| video_text_ca    | 11.8 %   | 65 ms   | 112 ms  | 10.1 %    | 38 ms    | 41 ms    |
| a2v_cross        | 4.5 %    | 39 ms   | 58 ms   | 8.6 %     | **147**  | **149**  |
| v2a_cross        | 2.5 %    | 39 ms   | 39 ms   | 3.4 %     | 108 ms   | 149 ms   |
| audio_ff         | 3.2 %    | 37 ms   | 38 ms   | 1.4 %     | 67 ms    | 68 ms    |
| audio_self_attn  | 1.6 %    | 38 ms   | 38 ms   | 0.8 %     | 21 ms    | 25 ms    |
| audio_text_ca    | 1.0 %    | 37 ms   | 62 ms   | 0.6 %     | 24 ms    | 24 ms    |
| **total**        | 101.6 s  |         |         | 99.8 s    |          |          |

Top 30 longest dispatches per project:
- **LTX-2-MLX:** 30/30 are `video_self_attn`, max 180 ms.
- **mlx-video:** 18/30 `a2v_cross`, 7/30 `video_ff`, 3/30 `v2a_cross`, 1/30
  `video_self_attn`, max 149 ms.

The two projects have **different tail shapes**, not the same shape at
different magnitudes:

- LTX-2-MLX's bottleneck is `video_self_attn` — a single phase that
  owns the entire long tail.  Likely a specific SDPA gemm shape
  combination at our token count that hits a worst-case kernel
  selection.
- mlx-video's bottleneck is spread across cross-modal attention
  (`a2v_cross` / `v2a_cross`) and `video_ff`.  Their cross-modal
  attention dispatches have noticeably worse tail behavior than ours
  (147 ms p99 vs our 39 ms).
- Audio is < 6 % of GPU time in both projects — not a useful target.
- `video_ff` is comparable in total time and **our tail is actually
  cleaner** (74 ms max vs theirs at 147 ms), so quantizing `video_ff`
  won't close the gap.

### What this points to next (not done yet)

1. **Hunt the `video_self_attn` 180 ms outliers** in LTX-2-MLX.  30
   of them per 2-step capture — roughly 15/step, suggestive of one
   per block.  These are likely the SDPA call at the full token grid.
   Worth trying alternative `mx.fast.scaled_dot_product_attention`
   tile/sub-tile parameters or `MLX_SDPA_BLOCKS` if relevant at our
   T (currently controls 2-pass vector path only — see matrix).

2. **Investigate mlx-video's a2v_cross tail** (147 ms p99 vs our
   39 ms).  If we can understand why they're slower at this specific
   op while we're slower at video_self_attn, we may converge by
   borrowing tactics in both directions.

3. **Hunt the > 20 ms inter-phase gaps** if any remain in non-sync
   runs.  Sync-mode captures show no gaps because we drain the queue
   per phase, so this analysis needs a different capture mode.

4. **Update the `LTX_COMPILE_BLOCK_GROUPS` matrix row** when the next
   session retests at the steady-state window — the "neutral" reading
   from earlier sessions doesn't match the apples-to-apples 3 % win
   from this session.

### Why the "monolithic-inlined" negative result is consistent with this

The 05-15 inlined-transformer experiment (`scripts/mono_pipeline.py`)
found that collapsing the 48-block module dispatch into one inlined
function moved per-step time by ~0 %.  This trace data explains why:
the per-step gap to mlx-video isn't in Python or in module dispatch —
it's in Metal-level dispatch dynamics that inlining can't touch.  The
inlined reference is still useful as same-math documentation of the
AV block, but it isn't a speed lever.

## 2026-05-17 Session: AdaLN/RoPE dtype-promotion fix (-16.8 % bakery total)

The actual cause of the per-step gap to mlx-video turned out to be a
silent BF16 → FP32 dtype promotion through the AdaLN modulation path,
which forced SDPA to compile and dispatch pure
`steel_attention_float32_*_maskfloat32_*` kernels — roughly 2× the
data movement and compute of the BF16 equivalents.

### The dispute that led here

The 05-16 traces had two competing analyses:

- Mine: "video_text_ca is 5.2× more expensive in LTX, the
  `_apply_text_cross_attention` helper is fragmenting dispatch."
- Codex's correction: my "GPU time per phase" wasn't elapsed wall
  time (it summed dispatch durations across parallel GPU channels and
  exceeded the phase wall span).  Sidecar interval math is the
  honest primary metric.

After accepting the correction, the real story was: LTX is slower on
every phase, not just one — distributed overhead, not a single
hotspot.  Three env-toggle experiments confirmed:

- `LTX_DISABLE_COMPILED_ATTN=1`: neutral (+1.5 s)
- `LTX_DISABLE_COMPILED_HELPERS=1`: neutral (–0.6 s)
- `--video-attn-layout off`: regression (+3.1 s — pretranspose IS a win)

That ruled out three obvious causes.  Then Codex spotted the actual
signal in the trace's shader **inventory** (the
`metal-shader-profiler-shader-list` table, not the empty
`metal-shader-profiler-intervals` table):

- LTX inventory: 30+ pure `float32` steel kernels, including 6
  `steel_attention_float32_*_maskfloat32_*` variants.
- mlx-video inventory: zero pure `float32` steel kernels — only BF16
  or `bfloat16_float32`-accumulator variants.

Codex's first hypothesis was that split-RoPE was promoting BF16
activations to FP32 by multiplying against FP32 cos/sin tables and
not casting back.  We fixed that.  Result: same FP32 kernels still in
the inventory, wall time unchanged.

### Probe: prove what dtype actually reaches SDPA

Instead of guessing, monkey-patch
`mx.fast.scaled_dot_product_attention` BEFORE importing any
LTX-2-MLX modules, log every call's `(q.dtype, k.dtype, v.dtype,
mask.dtype, shapes)`, then run one denoise step:

```python
# scripts/sdpa_dtype_probe.py — runs scripts/generate.py under a
# monkey-patched SDPA that records every call's dtype.
```

First run (RoPE fix only) revealed: **every per-block SDPA call had
q/k/v dtype `float32`**, including the V tensor — which never goes
through RoPE.  So RoPE wasn't the source; promotion happened
*upstream* of attention.

### Root cause: AdaLN scale/shift/gate are FP32 tables

The 6 `scale_shift_table` tensors in `BasicTransformerBlock` /
`BasicAVTransformerBlock` are explicitly typed
`mx.zeros((..., dim), dtype=mx.float32)` (kept FP32 for sincos /
timestep-embedding numerical stability).  Then the inline math:

```python
normed * (1 + scale) + shift   # AdaLN
x + residual * gate            # residual gate
```

promotes BF16 activations to FP32 by broadcasting against the FP32
scale/shift/gate.  The FP32 result propagates through `to_q/to_k/to_v`
into SDPA, which compiles pure `steel_attention_float32_*` kernels.

mlx-video's `rope.py` and equivalent paths have always cast back to
the input dtype (`output.astype(input_dtype)`).  LTX-2-MLX was missing
this cast at five call sites:

1. `_adaln_inline` (transformer.py)
2. `_residual_gate_inline` (transformer.py)
3. `_apply_text_cross_attention` V2 modulation (transformer.py:466-479)
4. Cross-modal A2V/V2A inline scale-shift (transformer.py:685-690, 716-717)
5. Cross-modal A2V/V2A residual gates (transformer.py:704, 737)

Plus `apply_split_rotary_emb` and `apply_interleaved_rotary_emb`
fallback in rope.py (kept the RoPE fix even though it wasn't the
bottleneck — same pattern, correct hygiene).

### Verification (post-fix probe)

Re-running the SDPA dtype probe with the AdaLN fix: **all 9 unique
SDPA signatures are now bfloat16 q/k/v** — connector, every
per-block phase, video and audio.  No more pure FP32 paths.

### Numbers

**2-step sync-mode capture (288×512 stage 1, signposts on, mx.eval
barriers — same protocol as the 05-16 captures):**

| Phase            | baseline | +rope-only | +adaln (fix) | Δ baseline | mlxv r2 | vs mlxv |
| ---------------- | -------- | ---------- | ------------ | ---------- | ------- | ------- |
| video_self_attn  | 46.01 s  | 46.12 s    | **40.80 s**  | **−5.21**  | 39.04 s | +1.76   |
| video_ff         | 38.19 s  | 38.71 s    | **33.98 s**  | **−4.21**  | 35.58 s | **−1.60** |
| video_text_ca    | 14.46 s  | 14.79 s    | **12.98 s**  | **−1.48**  | 11.40 s | +1.58   |
| audio_self_attn  |  1.40 s  |  1.52 s    |   1.24 s     |  −0.16     |  0.65 s | +0.59   |
| audio_text_ca    |  1.04 s  |  1.18 s    |   1.19 s     |  +0.14     |  0.67 s | +0.52   |
| a2v_cross        |  5.82 s  |  5.88 s    |   5.76 s     |  −0.06     |  5.07 s | +0.69   |
| v2a_cross        |  5.24 s  |  5.33 s    |   5.39 s     |  +0.15     |  4.87 s | +0.52   |
| audio_ff         |  2.52 s  |  2.86 s    |   2.23 s     |  −0.28     |  0.97 s | +1.27   |
| **TOTAL**        | **114.69** | 116.37   | **103.58 s** | **−11.11** | 98.26 s | **+5.32** |

Shader inventory: pure-FP32 steel kernels **20 → 0**, pure FP32
attention kernels **6 → 0**.

**Full bakery 1024×576×481 distilled (fast mode, end-to-end,
non-sync — the user-visible workload):**

|                          | baseline (per prior PERFORMANCE.md) | post-fix | Δ      |
| ------------------------ | ----------------------------------- | -------- | ------ |
| Stage 1 (8 steps, 288×512) | n/a (was ~10 s/it pre-fast-mode notes) | 45.3 s/it = 6m 02s | |
| Stage 2 (3 steps, 576×1024) | ~50 s/it baseline → ~26 m total stage 2 | 313.5 s/it = **15m 40s** | |
| VAE decode (tiled)       |                                       | 2m 10s   |        |
| **Total**                | **29m 38s**                           | **24m 40.6s** | **−4m 58s, −16.8 %** |

The win scales *better* at full bakery than the sync-mode-stage-1
measurement predicted (9.7 % → 16.8 %) because stage-2 token counts
(~280 k tokens) make the per-op FP32-vs-BF16 difference matter more,
and because fast mode + smaller BF16 intermediates lets MLX's lazy
graph fuse more without OOM risk.

### Quality

Visual inspection on the bakery output: indistinguishable from prior
runs.  Pixel-level diffs are real (the prior path held FP32
intermediates inside AdaLN before settling back to BF16 at layer
boundaries; the new path is BF16 end-to-end), but content and
structure are unchanged.  Matches mlx-video's reference precedent —
they've always done it this way.

If a stricter quality check is needed: `--save-latents` writes the
stage-1 and stage-2 latents; compare cosine sim against a known-good
pre-fix `.npz`.  Expect ~0.999 with BF16-rounding-noise scale of
1e-3 in p99 abs diff.

### What's left in the gap to mlx-video

Down from +16.4 s to +5.3 s over 2 sync-mode stage-1 steps.
Residual per-phase:

- video_self_attn: +1.76 s — modest, likely structural per-call cost.
- video_text_ca: +1.58 s — the `_apply_text_cross_attention` path
  emits ~7.6× more GPU dispatches than mlx-video's inline equivalent
  (303 vs 40 in the per-phase dispatch bucket).  Still a real
  fragmentation signal worth investigating, but lower priority now
  that the FP32 promotion is gone.
- audio phases combined: +2.4 s.  Audio is small in absolute terms
  (~5 s total), so further chasing has limited payoff ceiling.
- video_ff: **−1.60 s** — we now win this phase outright.

**Extrapolated to full bakery.** The +5.3 s/2-steps measured at
288×512 implies a per-step delta of ~2.65 s.  Stage 1 is 8 steps and
stage 2 is 3 steps at ~4× the tokens.  Naïve linear extrapolation
puts us within **~1–3 minutes** of mlx-video's wall time on the full
bakery 1024×576×481 workload — a small enough gap that this is no
longer a "lose to mlx-video" situation, it's a polish target.

**Ranked next steps (carries over the 05-16 hunt list, now prioritized
post-AdaLN-fix):**

1. **video_self_attn 180 ms outliers.** This was the top item at 05-16
   (lines 1747-1755) and remains the largest single residual.  The
   05-16 trace showed 30 of 30 longest dispatches in our project were
   `video_self_attn`, max 180 ms vs mlx-video's max 133 ms.  Likely a
   specific SDPA gemm-shape hitting a worse kernel selection at our
   full token grid.  Re-trace post-AdaLN-fix to confirm the 180 ms
   tail is still there or whether the BF16 path picked a better
   kernel.  Try `MLX_SDPA_BLOCKS` and tile/sub-tile env overrides.
2. **video_text_ca dispatch fragmentation.** 303 vs 40 dispatches in
   the per-phase bucket is real and could be flattened by inlining
   `_apply_text_cross_attention`.  Worth ~1.5 s/2-steps so secondary
   to (1).
3. **Leave audio alone.** ~5 s total budget across all audio phases;
   even a 2× win is ~2.4 s, smaller than the easier wins above and
   risks audio-quality regressions.
4. **`video_ff` quant is now a quality trade-off, not a speed-gap
   closer.** Past `--video-ff-quantize project_out:mxfp8` layers 32-47
   at 352×192 ran ~10 % faster than BF16 with slight visual
   differences (line 2007).  That speed lever still exists.  But
   pre-AdaLN-fix the framing was "trade quality for speed to catch
   mlx-video"; post-fix we already beat mlx-video at this phase by
   −1.60 s in BF16.  So mxfp8 here now means "draft-quality mode that
   gets even faster," not "way to close a gap."  Two real caveats to
   re-test before deploying: (a) the plain quant path loses the
   pretranspose layout win unless you use `mxfp8-blocks-pretranspose`
   (line 2009), and (b) partial-layer streaming quant disables
   resident-group compile (line 2007) — both can flip the speed sign.

The probe tool (`scripts/sdpa_dtype_probe.py`) is reusable for any
future "what dtype is actually reaching kernel X" question.

### 2026-05-17 follow-up: SDPA probe + non-sync A/B (gap was signpost overhead)

The "+5.3 s residual to mlx-video" framing above turned out to be
**signpost/sync-mode overhead**, not real GPU work.  Two new measurements
disproved the gap and reframed the brutal-efficiency hunt.

**(a) Per-call SDPA head-to-head, identical workloads, eval-barrier mode.**

Extended `sdpa_dtype_probe.py` with `LTX_PROBE_TIME_SDPA=1` to wrap
every `mx.fast.scaled_dot_product_attention` call in `mx.eval` barriers
+ timer.  Ran both projects at distilled stage 1, 1024×576 → 288×512
latent, 2 steps, identical seed/prompt.  Required
`LTX_DISABLE_COMPILED_ATTN=1 LTX_DISABLE_COMPILED_HELPERS=1` on the
LTX side so `mx.eval` could break MLX's compiled regions.

| Phase (shape)                                        | LTX total | mlxv total | Δ          |
| ---------------------------------------------------- | --------- | ---------- | ---------- |
| video_self_attn  q/k/v=(1,32,8784,128), mask=None    | 22,287 ms | 22,743 ms  | **−456**   |
| video_text_ca    q=(1,32,8784,128), kv=(1,32,1024,128) | 2,696 ms | 2,791 ms   | **−96**    |
| v2a_cross        q=(1,32,501,64), kv=(1,32,8784,64)  | 660 ms    | 703 ms     | **−43**    |
| a2v_cross        q=(1,32,8784,64), kv=(1,32,501,64)  | 640 ms    | 650 ms     | **−10**    |
| audio_text_ca    q=(1,32,501,64), kv=(1,32,1024,64)  | 108 ms    | 119 ms     | **−11**    |
| audio_self_attn  (1,32,501,64)                       | 80 ms     | 82 ms      | tied       |
| **SDPA TOTAL**                                       | **26,470** | **27,088** | **−618 (LTX wins by 2.3 %)** |

Both projects produce **identical SDPA shapes** and **identical kernel
selection** (every call BF16, no mask, same dims).  The "180 ms vs
133 ms" gap from the 05-16 trace was a sampling-bucket artifact of
the `metal-shader-profiler-dispatches` table, not a real per-call
difference.  Per-block SDPA distribution is flat 232-245 ms (LTX) vs
235-272 ms (mlxv) — LTX has the tighter tail.

**(b) Non-sync end-to-end wall-time A/B.**

Ran both projects in production lazy-graph mode (no signposts, no
eval barriers) for 8 stage-1 steps each at the same workload, via
the new script `scripts/bench_ab_wall_time.sh`:

| Metric                      | LTX-2-MLX | mlx-video | Δ          |
| --------------------------- | --------- | --------- | ---------- |
| Total process wall (8 steps + load) | 6m 10s   | 6m 58s    | **+48 s**  |
| Per-step denoise (8 steps sum)      | 364.40 s | 396.11 s  | **+31.7 s (+8.7 %)** |
| Per-step average                    | 45.5 s/it | 49.5 s/it | **+4.0 s/it** |

Per-step trajectory (LTX is consistent; mlxv drifts upward after step 4):

| step | LTX (s) | mlxv (s) | Δ      |
| ---- | ------- | -------- | ------ |
| 1    | 46.30   | 48.62    | +2.32  |
| 2    | 45.90   | 48.48    | +2.58  |
| 3    | 45.60   | 48.43    | +2.83  |
| 4    | 45.50   | 48.42    | +2.92  |
| 5    | 45.40   | 49.25    | +3.85  |
| 6    | 45.30   | 50.51    | +5.21  |
| 7    | 45.20   | 51.17    | +5.97  |
| 8    | 45.20   | 51.23    | +6.03  |

**LTX-2-MLX is 8.7 % faster than mlx-video end-to-end in production.**
The +5.3 s "residual" the prior section chased was the cost of:

- 384 signpost emit pairs per step (8 phases × 48 blocks)
- Per-phase `mx.eval` barriers in sync mode (lazy graph fully broken)
- Profile/event overhead specific to the LTX trace protocol

mlx-video's traces showed less overhead because they emit fewer
signposts per block (they instrument at the block level, not the 8
sub-ops).

### Reframed brutal-efficiency hunt

With the gap-to-mlxv question closed, the new question is: **how far
below 45 s/it can we drive per-step wall on the same workload**?
We're at 26 % of M1 Max BF16 peak on SDPA (2.7 TFlops/s of 10.4) and
~64 % of peak on FF (6.7 TFlops/s).  Plenty of headroom.

Per-step decomposition (from sync-mode 2-step capture, **divide by 2
for per-step**):

| Phase           | sync-mode wall / step | SDPA-only / step | non-SDPA / step |
| --------------- | --------------------- | ---------------- | --------------- |
| video_self_attn | 20.3 s                | 11.1 s           | **9.2 s**       |
| video_ff        | 17.0 s                | (no SDPA)        | **17.0 s**      |
| video_text_ca   | 6.5 s                 | 1.3 s            | **5.2 s**       |
| a2v_cross       | 2.9 s                 | 0.3 s            | **2.6 s**       |
| v2a_cross       | 2.7 s                 | 0.3 s            | **2.4 s**       |
| audio combined  | 2.3 s                 | 0.1 s            | **2.2 s**       |
| **TOTAL (sync)** | **51.7 s**           | **13.1 s**       | **38.6 s**      |

(Non-sync per-step is 45 s; the ~6.6 s sync-mode delta is signpost
overhead.  In non-sync mode the budget is ~38.6 s × (45/51.7) ≈
33.6 s of non-SDPA work the lazy graph fuses, plus ~11.4 s of SDPA
the graph cannot fuse.)

**The 9.2 s/step "non-SDPA in video_self_attn phase" and 17 s/step
video_ff are the two biggest unattributed buckets**.  Sub-phase
signposts (added in this same commit) will break them down into:

- **Attention internals**: `attn_qkv` (V/Q/K + gate_logits projections),
  `attn_sdpa` (just the SDPA call, redundant with the per-call probe
  but useful in-trace), `attn_out` (gate apply + output projection).
  Fires from every attention call site.
- **FF internals**: `v_ff_adaln` (the AdaLN modulation), `v_ff_inner`
  (the `self.ff(...)` call — project_in + GELU + project_out fused
  inside the FeedForward module).

After capturing a fresh sync-mode 2-step trace with the new
sub-phases, the rollup will show which of:

- Q/K/V projections (4096 → 12288 matmul × 48 blocks × 8 steps)
- Output projection (4096 → 4096 × 48 × 8)
- FF project_in (4096 → 16384)
- FF GELU
- FF project_out (16384 → 4096)

is the largest single addressable target.  Likely candidates:

1. **Q/K/V fusion** — three separate matmuls into one combined matmul
   if not already done by MLX.  ~30-40 % of that phase if applicable.
2. **FlashAttention-2 custom Metal kernel** for video_self_attn at
   T=8784.  At 26 % of peak we have ~3× headroom on this kernel
   alone.  Bounded engineering effort: the shape is known and stable.
3. **FF quant at acceptable quality** — mxfp8 `project_out` 32-47
   already tested at 10 % faster than BF16.  Now a draft-quality
   lever (no longer "catch up to mlxv"), but ~2 s/step on the table.

### Tooling artifacts from this session

- `scripts/sdpa_dtype_probe.py` — monkey-patches
  `mx.fast.scaled_dot_product_attention` BEFORE any LTX import,
  records every call's (dtype, shape) signature with caller stack.
  Set `LTX_PROBE_TIME_SDPA=1` (requires `LTX_DISABLE_COMPILED_ATTN=1`
  and `LTX_DISABLE_COMPILED_HELPERS=1`) to also wrap each call in
  `mx.eval` barriers + timer for per-call wall time.  Set
  `LTX_PROBE_TIME_LOG=/path/to/sdpa.jsonl` for one record per call.
  `LTX_PROBE_MODULE=mlx_video.models.ltx_2.generate` runs against
  mlx-video instead (uses `runpy.run_module` for package-relative
  imports).  Reports: per-signature dtype histogram, per-signature
  timing summary sorted by total ms, top-30 slowest individual calls.
- `scripts/bench_ab_wall_time.sh` — production-mode A/B wall-time
  comparison.  Sequential (no GPU contention).  Required env vars:
  `LTX_REPO`, `MLXV_REPO`, `MLXV_MODEL_REPO`.  Optional:
  `LTX_VENV_BIN`, `MLXV_VENV_BIN`, `STEPS`, `SEED`, `AB_OUTDIR`.
  Default `STEPS=4` runs in ~10 min; `STEPS=8` for full stage-1
  in ~16 min.  Parses per-step times from both projects' progress
  output and prints a side-by-side comparison plus per-step trajectory.
- Sub-phase signposts in `LTX_2_MLX/utils/_signpost.c` and
  `LTX_2_MLX/utils/signpost.py` (`attn_qkv`, `attn_sdpa`,
  `attn_out`, `v_ff_adaln`, `v_ff_inner`) for the brutal-efficiency
  next step.  Wired into `attention.py` (`Attention.__call__`) and
  `transformer.py` (around the video_ff block).  Nest inside the
  existing 8 parent phase signposts; aggregate across all attention
  call sites.
- `scripts/analyze_signpost_subphases.py` — walks a sidecar log
  (`LTX_PROFILE_SIGNPOST_LOG`), attributes each sub-phase to its
  currently-open parent phase, prints a per-(parent, sub-phase)
  rollup with `n`, total, mean, p50/p99/max, and an
  `[unaccounted]` budget per parent.  Calls fired outside any
  transformer parent (text encoder, AV connector during prompt
  encode) get bucketed under `[no_parent]`.  Also prints the
  top-N slowest individual intervals.

### Sub-phase validation: 2-step sync-mode capture, 1024×576 stage 1

Sanity-check run with `LTX_PROFILE_SIGNPOSTS=1 LTX_PROFILE_SIGNPOSTS_SYNC=1`
and the new sub-phase signposts enabled.  All 13 phases emit cleanly,
all begin/end pairs matched (no orphans), and sums reconcile with their
parents within 1 % (the unaccounted residual is the unwrapped
`residual_gate` at the end of each phase).

**Per-parent sub-phase attribution (per step — divide totals by 2):**

| Parent          | Sub-phase     | per-step | mean/block | p99/block | max/block |
| --------------- | ------------- | --------:| ----------:| ---------:| ---------:|
| video_self_attn | (total)       | 21.3 s   | 443.5 ms   | 510.2 ms  | 602.9 ms  |
|                 | attn_sdpa     | 11.0 s   | 230.0 ms   | 241.5 ms  | 242.6 ms  |
|                 | attn_qkv      | 7.7 s    | 161.0 ms   | 229.9 ms  | 322.5 ms  |
|                 | attn_out      | 2.4 s    | 49.8 ms    | 59.6 ms   | 59.8 ms   |
|                 | [unaccounted] | 0.1 s    |            |           | 0.6 %     |
| video_ff        | (total)       | 18.9 s   | 392.8 ms   | 532.8 ms  | 777.1 ms  |
|                 | v_ff_inner    | 18.7 s   | 389.3 ms   | 529.5 ms  | 774.4 ms  |
|                 | v_ff_adaln    | 0.06 s   | 1.3 ms     | 2.0 ms    | 2.3 ms    |
|                 | [unaccounted] | 0.1 s    |            |           | 0.6 %     |
| video_text_ca   | (total)       | 7.5 s    | 156.8 ms   | 178.6 ms  | 202.6 ms  |
|                 | attn_qkv      | 3.5 s    | 73.8 ms    | 94.3 ms   | 118.3 ms  |
|                 | attn_out      | 2.4 s    | 49.6 ms    | 57.4 ms   | 58.7 ms   |
|                 | attn_sdpa     | 1.3 s    | 27.9 ms    | 29.9 ms   | 30.1 ms   |
|                 | [unaccounted] | 0.3 s    |            |           | 3.5 %     |
| a2v_cross       | (total)       | 3.8 s    | 79.8 ms    | 128.1 ms  | 205.0 ms  |
|                 | attn_qkv      | 1.7 s    | 35.1 ms    | 77.3 ms   | 136.3 ms  |
|                 | attn_out      | 1.2 s    | 25.1 ms    | 27.6 ms   | 28.1 ms   |
|                 | attn_sdpa     | 0.3 s    | 6.7 ms     | 7.6 ms    | 7.7 ms    |
|                 | [unaccounted] | 0.6 s    |            |           | 16.1 %    |
| v2a_cross       | (total)       | 3.2 s    | 66.1 ms    | 74.9 ms   | 77.3 ms   |
|                 | attn_qkv      | 2.6 s    | 53.4 ms    | 62.2 ms   | 64.7 ms   |
|                 | attn_sdpa     | 0.3 s    | 7.2 ms     | 8.2 ms    | 9.2 ms    |
|                 | attn_out      | 0.2 s    | 4.5 ms     | 6.7 ms    | 7.1 ms    |
|                 | [unaccounted] | 0.1 s    |            |           | 1.6 %     |
| audio_ff        | (total)       | 1.6 s    | 33.7 ms    | 86.3 ms   | 114.6 ms  |
| audio_self_attn | (total)       | 0.8 s    | 16.6 ms    | 32.4 ms   | 56.8 ms   |
| audio_text_ca   | (total)       | 0.8 s    | 16.7 ms    | 22.9 ms   | 30.2 ms   |

**Two rankings of the brutal-efficiency targets (they don't agree, by design):**

| Sub-phase                       | per-step | rank by size | rank by actionability | why |
| ------------------------------- |---------:|:------------:|:---------------------:| --- |
| video_ff / v_ff_inner           | 18.7 s   | **#1**       | **#1**                | Two non-conflicting levers: (a) mxfp8 quant on project_in/project_out — tested at ~10 % faster (line 2007), quality cost; (b) custom Metal kernel fusing project_in + GELU + project_out into one pass so the 16384-dim intermediate never materializes — bandwidth win at T=8784, similar idea to FlashAttention.  Most leverage with least new engineering on the quant side; biggest theoretical ceiling on the fused-kernel side.  Combinable. |
| video_self_attn / attn_sdpa     | 11.0 s   | **#2**       | **#3**                | Big but at-parity with mlx-video (per-call probe).  Only path forward is a custom FlashAttention-2 Metal kernel — bounded but high-effort.  Worth doing eventually but is the hardest target. |
| video_self_attn / attn_qkv      | 7.7 s    | #3           | **#2**                | 3 separate matmuls (V, Q, K) + 2 RMSNorms + 2 RoPE calls per attention.  Fusing Q+K+V into one combined matmul would collapse most of this with bounded engineering (need to check whether MLX is already doing this internally before refactoring). |
| video_text_ca / attn_qkv        | 3.5 s    | #4           | #5                    | Same lever as above but smaller — Q on T=8784, K/V on T=1024. |
| v2a_cross / attn_qkv            | 2.6 s    | #5           | #6                    | Q on T=501, K/V on T=8784 — interesting bandwidth pattern.  Small absolute. |
| video_self_attn / attn_out      | 2.4 s    | #6           | #4                    | 4096→4096 single matmul + per-head gate apply.  If the gate is close to identity (post-zero-init weights), the gate-apply could be elided — easy lever for a marginal win. |
| video_text_ca / attn_out        | 2.4 s    | tied         | tied                  | Same as above, different parent. |

**Three sanity-check observations:**

1. **video_ff is almost entirely the inner FF matmul** — 99.1 %
   of video_ff time is in `v_ff_inner` (project_in + GELU +
   project_out); AdaLN modulation is 1.3 ms/block (0.3 %).
   "Fuse AdaLN INTO the FF matmul" is dead-on-arrival because
   AdaLN has no time to save (note: this is different from
   internal FF fusion — project_in + GELU + project_out into one
   kernel — which IS a live lever per the table above).
   v_ff_inner owns 18 of the top 20 slowest individual sub-phase
   intervals at this workload.
2. **attn_sdpa cross-validates the per-call probe** — 22.08 s in
   this sync-mode capture vs 22.29 s in the eval-barrier probe.
   Same kernel, same shape, same speed.  Per-block SDPA wall is
   bounded at 230 ms by MLX's current implementation.
3. **a2v_cross has 16 % unaccounted time** — the AdaLN modulation
   at `transformer.py:691-692` (`vx_norm3 * (1 + scale) + shift`)
   is inside the `a2v_cross` parent signpost but NOT wrapped in a
   sub-phase.  Worth wrapping if a2v_cross ever becomes a real
   target (currently 3.8 s/step total — probably not).

**Video ops dominate by a wide margin** — video_self_attn +
video_ff + video_text_ca alone is 47.7 s/step out of ~50 s.
Audio + cross-modal combined is < 10 s/step.  Optimization ROI
continues to be video > everything else.

The sub-phase signposts are now part of the default profile-mode
toolkit; future captures will produce the same 13-phase rollup
automatically without code changes.

### Scaling validation: 30-second 1024×576 (721 frames)

Same fix applied, different prompt + seed, longer duration to confirm
the win holds at higher token counts:

| Stage                          | Bakery (481 frames) | Kitten (721 frames) | Ratio |
| ------------------------------ | ------------------- | ------------------- | ----- |
| Frames                         | 481                 | 721                 | 1.50× |
| Latent shape                   | 61×18×32            | 91×18×32            | 1.49× tokens |
| Stage 1 s/it (288×512)         | 45.3 s              | 75.9 s              | 1.68× |
| Stage 2 s/it (576×1024)        | 313.5 s             | 602.1 s             | 1.92× |
| Total wall                     | 24m 40s             | 44m 41s             | 1.81× |
| Per-token-step cost (stage 2)  | ~8.9 ms             | ~11.5 ms            | 1.29× |

Stage 2 scales 1.92× for 1.49× tokens — expected attention quadratic
dilution.  Going from 35,136 → 52,416 tokens means each token does
~49 % more SDPA work; in stage 2 attention is a big fraction of total
time, so per-token-step cost rises 29 %.  Pre-fix this run would have
been ~53–55 minutes by the same −16.8 % ratio; post-fix is 44m 41s
(saved ~9 minutes on a 30-second video).

Memory: **43 GB peak process RAM** during stage 2 (Activity Monitor).
No memory pressure on a 64 GB M1 Max with nothing else running.  Fast
mode + AdaLN fix lets the lazy graph fit comfortably even at 91-frame
latents (1.49× the bakery token count).

Visual / audio sync: confirmed good on the first seed.  Cross-seed
quality check still TODO.

## Benchmark Matrix

Use a fixed command and change only one thing at a time.

| Experiment | Code change? | Memory risk | Expected denoise speed impact | Notes |
| --- | --- | --- | --- | --- |
| `--internal-audio auto` (default) | yes | none | -35% wall on video-only (256x256x25) | New CLI flag. Default `auto` resolves to `on` iff `--generate-audio`. Off otherwise. Replaces the silently-overridden `LTX_DISABLE_INTERNAL_AUDIO=1` legacy env. Matches mlx-video's distilled path. |
| Audio module pretranspose (default) | yes | medium | -11% AV (256x256x25) | `audio_attn1/2.to_*`, `video_to_audio_attn.to_*`, `audio_ff.project_*` now go through the same `mx.contiguous(weight.T)` cache path the video modules use. Cache hash bumps; old caches stay valid. Opt out with `LTX_DISABLE_AUDIO_PRETRANSPOSE=1`. |
| QKV pretranspose (default) | yes | medium | -4% additional AV | `to_q/to_k/to_v` added to the default attention layout spec. 18 more matmuls per AV block per step with the implicit transpose eliminated. Cache hash bumps. |
| AdaLN/RoPE dtype cast-back (default) | yes | none | **-16.8% bakery total** | `scale_shift_table` tensors are FP32 (kept for sincos precision).  The inline math `normed * (1 + scale) + shift` and `x + residual * gate` silently promoted BF16 activations to FP32, forcing SDPA to compile pure `steel_attention_float32_*_maskfloat32_*` kernels (~2× the data movement of BF16).  Fixed at five call sites in `transformer.py` plus `rope.py` — cast back to input dtype after each modulation, matching mlx-video's pattern.  Verified via monkey-patched `mx.fast.scaled_dot_product_attention` probe: every per-block SDPA call dropped from float32 to bfloat16 q/k/v.  Full bakery 1024x576x481 distilled with fast-mode: 29m 38s → 24m 40.6s.  See "2026-05-17 Session". |
| Skip negative prompt encoding (distilled two-stage) | yes | none | -10% AV (256x256x25) | `generate_distilled_two_stage` doesn't accept a negative encoding, so we no longer encode one. Re-enable with `LTX_ENCODE_UNUSED_NEGATIVE=1` for sidecar debugging. Matches mlx-video. |
| Tokenize without max-length padding | yes | none | -5% AV (256x256x25) | Tokenizer was padding 2 real tokens to 1024 before the Gemma forward; trim happened after the wasted O(N^2) attention. Re-enable with `LTX_PAD_PROMPT_TO_MAX=1`. |
| `LTX_VELOCITY_MODE=1` | yes | low | neutral | Bypasses `X0Model` and does the velocity-form Euler update inline in `_denoise_loop_simple_av`. Same math. Tested at small T and bakery; both neutral. Kept env-gated for future MLX versions. |
| `LTX_COMPILE_BLOCK_GROUPS=N` | yes | medium | neutral at all tested scales (but see 2026-05-16 update) | Eager-path `mx.compile` over N-block groups. `N=4` tested at small T (neutral), `N=48` at bakery (neutral: 18m 29s vs 18m 41s baseline). Compile-trace cost paid up front; per-step does not recover it. **2026-05-16 update:** apples-to-apples Metal trace shows the default compile path (not this experimental block-group compile, but the underlying `mx.compile` wrappers on attention/RoPE/helpers) is worth ~3% wall and 3× tail-latency reduction in the 2-step capture window. `LTX_DISABLE_COMPILED_*=1` is a measurable regression, not a no-op. See "2026-05-16 Session" section. |
| `LTX_ADALN_PRETRANSPOSE=1` | yes | low | slight regression at small T | Cache-integrated pretranspose for the 8 `AdaLayerNormSingle.linear` projections. The 8 calls per step on a tiny batch don't amortize the per-tensor dispatch overhead. Kept opt-in. |
| `LTX_ROPE_PRECOMPUTE=1` | yes | none | neutral | mlx-video pattern: compute RoPE once per stage and pass through `Modality.positional_embeddings`. MLX's lazy graph already deduplicates per-step `precompute_freqs_cis` calls, so the savings don't show. |
| `to_gate_logits` pretranspose (opt-in) | yes | low | slight regression | Pretranspose support for the V2 per-head gate-logits Linear. Weight (4096x32 / 2048x32) is too small for the implicit transpose to matter. Opt-in via `--video-attn-layout to_out:pretranspose,...,to_gate_logits:pretranspose`. |
| `LTX_MONO_INLINED=1` (stage2_harness only) | yes | none | neutral | Replaces `av_pipeline.transformer` with `mono_pipeline.InlinedAVModel`: same math, 48-block forward + AdaLN preprocess + output projection all inlined into one function with flat pretransposed weights. Latent diff vs modular: cosine sim 0.999+, BF16 rounding-order noise only. Wall clock within measurement noise. Confirms `nn.Module` dispatch is free at the MLX-graph level. See section above. |
| `LTX_PROFILE_PAUSE_BEFORE_DENOISE=1` | yes | none | diagnostic only | Blocks on stdin once, immediately before stage 1's first denoise step in `generate_distilled_two_stage`. Prints `pid=N` so a second-terminal `xcrun xctrace record --attach <PID>` can connect at a precise point after model load and prompt encoding. Zero cost when unset. Pair with `LTX_PROFILE_STOP_AFTER_STEPS=N` for fixed-window captures. See "2026-05-16 Session". |
| `LTX_PROFILE_STOP_AFTER_STEPS=N` | yes | none | diagnostic only | `sys.exit(0)` after step N of the current denoise loop. `N=2` gives one warmup + one steady-state step — the minimum useful capture window for kernel-mix and dispatch-distribution analysis. Same hooks exist in mlx-video under identical env-var names so traces from the two projects are directly comparable. |
| `LTX_PROFILE_SIGNPOSTS=1` | yes | none | diagnostic only | Wraps the 8 sub-ops in `BasicAVTransformerBlock.__call__` (`video_self_attn`, `video_text_ca`, `audio_self_attn`, `audio_text_ca`, `a2v_cross`, `v2a_cross`, `video_ff`, `audio_ff`) with `os_signpost` intervals in subsystem `ltx`, category Points of Interest. Auto-builds a tiny ctypes-loaded C shim on first import (`LTX_2_MLX/utils/_signpost.{c,py}`). Disabled overhead ~5 ms/run, enabled ~17 ms/run. Capture via `xcrun xctrace record --template "Metal System Trace" --instrument "Points of Interest" --attach <PID>` — see "2026-05-16 Session". |
| `LTX_PROFILE_SIGNPOSTS_SYNC=1` | yes | medium | diagnostic only | Forces `mx.eval()` on each phase's output at signpost end so begin/end intervals tightly bracket the actual GPU dispatches.  Required for time-based attribution — without it, lazy MLX queues phases microseconds apart while GPU lags by seconds and signpost intervals don't correspond to dispatches.  Adds ~10 % wall-clock overhead.  Pair with `LTX_PROFILE_SIGNPOSTS=1`. |
| `LTX_PROFILE_SIGNPOST_LOG=/path` | yes | none | diagnostic only | Writes a sidecar log (`<monotonic_ns> <begin\|end> <phase>` per line) alongside the os_signpost emission.  Ground-truth source for attribution when the trace's `os-signpost` table is short on events; anchor by aligning sidecar's last event with trace's total duration. |
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
