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
- Final reproducibility sidecars can save latents, text conditioning, and run
  metadata with `--save-all-sidecars`.

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

The flag keeps the stock checkpoint path. It loads normal BF16 weights first,
then replaces selected video transformer block FF projections with in-memory
MLX `QuantizedLinear` layers. It does not quantize audio FF layers. If the mode
is omitted for a target, it defaults to `mxfp8`; the public API is
`--video-ff-quantize target:mode[,target:mode]`.

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

Status: implemented as an opt-in same-math layout experiment.

`nn.Linear` computes `x @ weight.T`. The video FF `project_out` is the largest
same-math linear hotspot in the clean block profile, so this experiment caches a
contiguous `weight.T` after loading stock BF16 weights and calls `mx.addmm`
against that cached layout. It does not quantize weights or intentionally change
precision.

Enable it with:

```bash
--video-ff-layout project_out:pretranspose
```

Notes:

- `project_in:pretranspose` is also supported for the video FF input
  projection. Test it separately or in combination with `project_out`; unlike
  the earlier `project_in:mxfp8` quantization run, this is same-math.
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

Status: implemented as an opt-in same-math layout experiment.

Attention output projections also compute `x @ weight.T`. This experiment
materializes contiguous transposed `to_out` weights after loading stock BF16
weights, then drops the original projection weights. For AV blocks it applies to
video self-attention, video text-attention, and audio-to-video attention; it
intentionally skips audio-only output projections.

Enable it with:

```bash
--video-attn-layout to_out:pretranspose
```

Recommended A/B:

- Start by combining it with the current best same-math FF layout:
  `--video-ff-layout project_out:pretranspose --video-attn-layout to_out:pretranspose`.
- Use `--video-attn-layout-layers 0-47` for the all-layer test, then narrow only
  if memory pressure appears.
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

Status: implemented as an opt-in same-math memory-pressure knob.

MLX keeps an allocator cache to avoid returning buffers to the system between
operations. That can improve reuse, but on unified-memory Macs it can also keep
process memory and swap pressure higher than the active graph requires. This is
separate from the on-disk `--weights-cache`: it controls MLX's in-memory
allocator cache only.

Enable it with:

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

Recommended current same-math constrained-memory stack:

```bash
--weights-cache auto \
--mlx-cache-limit-gb 1 \
--video-ff-layout project_out:pretranspose \
--video-ff-layout-layers 0-47
```

The attention layout can still be A/B tested, but it is not part of the minimal
recommended stack because its benefit was marginal.

### 12. Use `vmap` for repeated helper/probe loops

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

### 13. Add a fused split-RoPE kernel

Status: only consider after RoPE precompute/profiling.

The codebase has a fused interleaved RoPE path, but LTX-2.3 uses split RoPE.
Precomputing RoPE is a simpler first move. A custom split-RoPE kernel is only
worth it if profiling still shows split-RoPE application as a significant cost.

Before writing a custom kernel, test whether `mx.fast.rope` can represent the
needed LTX-2.3 SPLIT 3D RoPE exactly. If it cannot, a custom Metal kernel should
be built once and should avoid hidden row-contiguity copies.

### 13. Distributed tensor parallelism

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

## Benchmark Matrix

Use a fixed command and change only one thing at a time.

| Experiment | Code change? | Memory risk | Expected denoise speed impact | Notes |
| --- | --- | --- | --- | --- |
| `--profile-transformer-steps` | yes | low | diagnostic only | Forces eval checkpoints during selected denoise steps to locate hotspots. Do not use for final timing. |
| `--profile-transformer-blocks` | yes | low | diagnostic only | Adds forced eval checkpoints inside selected blocks for already-profiled steps. Block profiles now split attention into setup/AdaLN, Q/K/V, Q/K norm, RoPE, SDPA, gate, output, and residual sections. |
| FFN sub-profile | yes | low | diagnostic only | Selected block profiles also split FFN into AdaLN, `project_in`, GELU, `project_out`, and residual gate. |
| Remove `--low-memory` | no | medium | none observed in small run | 352x192 15s AV smoke was slightly slower. Retest larger shapes if they fit. |
| `MLX_METAL_FAST_SYNCH=1` | no | low | none observed | 352x192 15s AV smoke was slightly slower. |
| AV `fast_mode` | yes | high | none observed in small run | 352x192 15s AV smoke tied low-memory baseline. Retest larger shapes if they fit. |
| Per-run RoPE precompute | yes | low | none observed in small run | Temporary one-entry cache patch was removed after a slightly slower 352x192 15s AV smoke. |
| No-op cast/allocation cleanup | yes | low | low | Good cleanup after larger wins. |
| Fast GELU approximation | no | low | unlikely | FFN sub-profile showed GELU at about 0.7% of a clean block; skip unless future profiles differ. |
| Historical compile experiments | yes | medium to high | none observed | Full transformer and video FF subpath compile were removed from the CLI after bakery AV smokes tied baseline speed while adding complexity and memory risk. |
| `mx.qqmm` FF linears | no | medium to high | blocked locally | Current Metal runtime throws `[QQMatmul] NYI for the general case` on LTX FFN-shaped tests. |
| `--video-ff-quantize` | yes | medium | useful in selected ranges | `project_out:mxfp8` layers 32-47 improved 352x192 15s AV denoise from about 24.4s/it to 22.1s/it with slight visual differences. Layers 0-23 and 24-47 were both visibly different; all-layer `mxfp8` is much faster but non-parity. All-layer `mxfp4` and `nvfp4` were slower than `mxfp8` on the bakery smoke, so keep `mxfp8` as the runtime-quant candidate for now. `project_in:mxfp8` was slower than BF16 and hurt identity stability. Official NVFP4 checkpoint support remains a separate loading/compatibility experiment. |
| `--video-ff-layout project_out:pretranspose` | yes | medium | best same-math win so far | Replacement-layout all-layer bakery AV smoke improved from roughly 77s/it BF16 baseline to about 55s/it, stable around 44GB process memory. Original duplicate-cache implementation was slower and more memory hungry. |
| `--video-ff-layout project_in:pretranspose,project_out:pretranspose` | yes | medium | neutral vs project_out only | Same-math and safe for inference, but bakery smoke showed about the same memory and about the same 55s/it as `project_out` alone. Keep as supported, not the recommended minimal setting. |
| `--video-attn-layout to_out:pretranspose` | yes | medium | marginal positive | Combined with `project_out:pretranspose`, all-layer bakery AV smoke improved slightly to about 54s/it with the same steady memory. Keep available, but FF `project_out` remains the main same-math win. |
| `--mlx-cache-limit-gb 1` | no | low | no cost observed | Same-math allocator-cache cap. On the bakery AV smoke with `project_out:pretranspose`, average process RAM dropped from about 44GB to about 40GB with no observed time penalty. Keep separate from `--weights-cache`, which is an on-disk converted-weight cache. |
| Weight-only quantized transformer | yes | medium | unknown | Broader quantization than project_out only; separate quality/checkpoint tradeoff. |
| `mx.block_masked_mm` | no | high | unknown | Only relevant if we introduce structured block sparsity or pruning. |
| `mx.gather_mm` / `mx.gather_qmm` | no | high | unknown | Relevant to MoE/routing or selected batched matrices, not current dense LTX. |
| `vmap` helper/probe loops | yes | low | low | Useful only where Python loops show up. |
| `mx.fast.rope` / split-RoPE kernel | yes | low to medium | unknown | Only after proving exact RoPE parity. |
| Distributed tensor parallelism | yes | high | unknown | Separate multi-machine project. |
| FP8 conversion primitives | no | high | unknown | `mx.to_fp8` / `mx.from_fp8` are storage/compute research items, not safe BF16 path changes. |

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
