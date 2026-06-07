# Performance Optimization Notes

This document tracks ways to make the existing MLX denoise path faster without
changing user-visible generation settings (pipeline type, stage layout,
duration, frame count, resolution, sampler, step count).

It is intentionally a working notebook.  Some entries are shipped defaults,
some are env-toggle experiments, some are diagnostic-only, and some are ideas
that should not be treated as free wins without evidence.  Sessions at the
top are most recent; older history is preserved at the bottom for context.

---

## TL;DR — current state (2026-05-18)

**Production baseline at bakery (1024×576×481, distilled two-stage,
`--fast-mode`, default flags):**

| Metric                       | Current   |
| ---------------------------- | --------- |
| Stage 1 (8 steps, 288×512)   | 45.5 s/it = 6m 04s |
| Stage 2 (3 steps, 576×1024)  | 313.5 s/it = 15m 40s |
| VAE decode (tiled)           | 2m 10s    |
| **Bakery total**             | **24m 40s** |

**Head-to-head vs mlx-video (same workload, production lazy-graph mode,
8 stage-1 steps via `scripts/bench_ab_wall_time.sh`):**

| Metric                       | LTX-2-MLX | mlx-video | Δ     |
| ---------------------------- | --------- | --------- | ----- |
| Per-step denoise (8 steps)   | 364.4 s   | 396.1 s   | **+31.7 s (+8.7 %)** |
| Per-step average             | 45.5 s/it | 49.5 s/it | **+4.0 s/it** |
| Total process wall           | 6m 10s    | 6m 58s    | **+48 s** |

**LTX-2-MLX is 8.7 % faster than mlx-video end-to-end in production.**  The
gap-to-mlxv question, which drove most of the 05-15/05-16 investigation, is
closed.

### Default flag stack (auto-enabled)

The following ship enabled by default:

- BF16 compute throughout, native checkpoint dtype preserved.
- Video FF layout: `project_out:pretranspose` only.  (The historical
  default also included `project_in:pretranspose`, but microbench
  evidence — `scripts/bench_ff_microbench.py bf16_layout` — showed
  project_in pretranspose is a 2.5 % regression in isolation and
  neutral end-to-end.  Only project_out earns its keep, with a 35 %
  isolated-matmul win that rescues a BlockLoader coalescing cliff in
  MLX's `_nt_` GEMM template — see `PERFORMANCE_NOTES.md`
  "BlockLoader cliff characterization" entry for the mechanism.)
- Video attention layout: OFF by default.  (The historical default
  was `to_q/to_k/to_v/to_out:pretranspose`; same microbench showed
  these are tied with naive BF16 within ±1 % at the 4096×4096 shape.)
- Audio pretranspose (audio attn `to_*`, video→audio attn `to_*`, audio FF).
- AdaLN/RoPE dtype cast-back (no FP32 leakage into SDPA).
- Skip negative prompt encoding for distilled mode (cfg=1.0 → no-op).
- Non-padded tokenizer (no `padding="max_length"`).
- `--internal-audio auto` (on iff `--generate-audio`).
- `--mlx-cache-limit-gb 1`.
- `--vae-decoder native` (the only supported backend; the historical `legacy` path was archived 2026-05-23).
- `--output-backend auto` — `--encode-tier default` (HEVC Main10 + ALAC)
  goes through AVAssetWriter; other tiers stay on ffmpeg.  Removes a
  raw-frames-to-ffmpeg subprocess pipe from the hot path on macOS.
- Terminal redraw throttling (`DenoiseProgress` no heartbeat thread,
  `tqdm` with `ascii=True` + `mininterval=1-2s`).

### Active brutal-efficiency targets

**Update 2026-06-06:** one actionable target survived deeper testing:
a default-on `mx.fast.metal_kernel` wrapper around MLX STEEL attention
helpers, specialized for the LTX-2.3 no-mask hot path.  D128 uses the
source-packaged `BQ=80, BK=40, q8k2v8` reducer+scalefold path.  D64 uses an
adaptive path: self-attention keeps `BQ=64, BK=32`, audio-to-video uses
`BQ=64, BK=24, q8k2` reducer+scalefold, and video-to-audio uses
`BQ=64, BK=32, q8k4`.  Disable all local STEEL attention with
`LTX_DISABLE_STEEL_ATTN=1` or `LTX_STEEL_ATTN=0`; `LTX_STEEL_ATTN_PROBE=1`
prints hit/fallback counts.  D64 escape hatches are
`LTX_STEEL_ATTN_DISABLE_D64=1`, `LTX_STEEL_ATTN_D64_BK32=1`, and
`LTX_STEEL_ATTN_D64_Q8K4=1`.  The implementation lives in
`LTX_2_MLX/kernels/metal/`; the older compact and vendored snapshots are
archived under `archive/steel_attention/`.

Prior negative candidates remain preserved below (full reasoning in
`PERFORMANCE_NOTES.md` Archive):

| Previous candidate | Status | Evidence |
|---|---|---|
| **mxfp8 quant on `project_*`** | **DEAD** — +10-45 % SLOWER post-AdaLN-fix | `bench_ff_microbench.py quant_matmul`: `mx.quantized_matmul` is structurally a BF16 matmul with on-the-fly dequant (`fp_quantized.h:139`, `:663`).  Hits 4.7 TFlops/s vs steel_gemm's 7.95.  Will always lose on M1; flips on M3+/M5 hardware.  Empirically verified 2026-05-23 via `scripts/bench_int8_alu.py`: **no INT8 hardware path exists on Apple7**.  Metal has no `dot(char4,char4)` intrinsic, `simdgroup_matrix` accepts only `half/bfloat/float`, and `mpp::tensor_ops::matmul2d` (Metal 4 cooperative-tensor with INT8 inputs) is M5+ only.  Microbench: INT8 4-way unrolled "dot" achieves 0.80 TOps/s vs FP16 scalar fma 0.76 TOps/s — per-MAC tied, no advantage.  The "21 TFlops/s theoretical INT8 peak" claim in prior PERFORMANCE_NOTES was wishful. |
| **Q+K+V fusion into one matmul** | **DEAD** — −1.2 % regression | `bench_ff_microbench.py qkv`: 3 separate (111.75 ms) is faster than 1 packed (113.10 ms).  Bandwidth saving on input dominated by larger output's tile-alignment + post-split overhead. |
| **Custom STEEL SDPA tile wrapper** | **DEFAULT WIN** — opt out with `LTX_DISABLE_STEEL_ATTN=1` | Supersedes the earlier D-sweep conclusion.  A literal wrapper around MLX's own STEEL body hits D=128 video self-attn and D=64 no-mask audio/cross-modal attention.  D128 uses `BQ=80, BK=40, q8k2v8` reducer+scalefold.  D64 selects by direction: self `BK=32`, audio-to-video `BK=24, q8k2` reducer+scalefold, video-to-audio `BK=32, q8k4`.  Fresh 576×320×721 distilled AV smoke (`8+3`, seed 42, BF16, audio on): lean D64/D128 stage-2 denoise 294.9s; q8k2 D128 + lean D64 stage-2 denoise 288.1s.  Saved-reference MP4 comparison: PSNR 47.64 dB, SSIM 0.99449, decoded RGB8 mean abs 1.31/255.  Latents are not bit-exact (`stage_2_video_latent` cos 0.99925), and the older stock-vs-lean path already had small BF16/tile-order drift; visual parity is the rollout gate. |
| **Custom fused / streamed BF16 FFN kernel** | **ABANDONED** — feasibility bench shows 3.8-7.5 % per-call ceiling | `bench_ff_microbench.py fused`: stock FF runs at 91-95 % of `steel_gemm` ceiling.  Conservative recoverable wall (vs both-matmuls-chained-no-GELU floor) = +12 ms = +3.8 % per call.  Optimistic recoverable wall (vs a perfect tiled fused kernel that elides hidden HBM round-trip AND retains GEMM efficiency) = +24 ms = +7.5 %.  Per-step ceiling: +1.3-2.6 % of 45.5 s.  Streamed-FFN sketch (per-chunk inner-dim streaming on top of MLX) additionally pays an output-accumulation tax and many-small-GEMMs efficiency loss; would not clear the bar.  See `PERFORMANCE_NOTES.md` Archive "Custom fused / streamed BF16 FFN kernel". |
| **Custom AdaLN+residual Metal kernel** | **NOT WORTH IT** — ~0.10 % step headroom | `bench_ff_microbench.py adaln`: production compiled chain runs at 285 GB/s = 84 % of `pointwise_bw` 340 GB/s peak.  Inline-vs-compiled gap is 5.5× (`mx.compile` is already doing the heavy lifting).  A custom Metal kernel can at most recover the 16 % gap from 285→340 GB/s × 0.64 % of step the chain costs = ~0.10 % of step.  Keep `@mx.compile` on `_adaln_inline` / `_residual_gate_inline` — defends the current production default with a number. |
| **RoPE BF16 cos/sin (Lever A)** | **DEAD** — tested 2026-05-18, visible output drift | `bench_ff_microbench.py rope` predicted -0.55 % step (-252 ms) if BF16 cos/sin (cast inside `apply_split_rotary_emb`) preserved quality.  Empirical one-stage A/B (288x512x721, seed 42): speed delta within single-sample noise once layout-default change is controlled for; visible same-seed output drift.  RoPE precision change cascades into the render via AdaLN-conditioned attention.  Patch reverted, env var removed.  See `PERFORMANCE_NOTES.md` Archive "Consolidated quiet-machine microbench sweep" → Lever A. |
| **RoPE merged into QKV matmul epilogue (Lever B)** | **NOT WORTH IT** — up to 0.90 % step, multi-day kernel work | Same `rope` bench: even a perfect kernel that eliminates RoPE entirely (folded into Q/K projection) caps at the 0.90 % step that production RoPE costs.  Multi-day Metal effort for sub-1 % return.  See `PERFORMANCE_NOTES.md` Archive "Consolidated quiet-machine microbench sweep". |
| **VAE Conv3d (upstream `mx.conv_general` int-overflow fix)** | **BLOCKED upstream** — 2.4-3.5 % end-to-end ceiling | `bench_ff_microbench.py vae`: `nn.Conv3d` achieves 56-72 % of `steel_gemm` ceiling across three decoder shapes; dominates resnet-block time at ~80-95 %.  If upstream fix lifts Conv3d to GEMM ceiling: ~1.5-2x on Conv3d × ~7 % VAE share of total wall = ~2.4-3.5 % end-to-end ceiling.  Re-run `vae_ops` after fix lands to confirm.  Pointwise norm/silu inside the block are negligible. |

**Achievement summary:** stock MLX BF16 remains hard to beat on M1 Max,
but the STEEL wrapper found one narrow default software win.
Further substantial wins still require:
- (a) **Silicon upgrade to M3+ (INT8 GEMM beats BF16) or M5+ (NAX
  cooperative-tensor matmul)** — the same `--video-ff-quantize` and
  `--transformer-cache-quantize` flags become positive levers on
  M5+ once NAX routes `fp_quantized_nax.metal`.
- (b) **Upstream MLX kernel improvements** — VAE int-overflow fix
  (already pending), possible SDPA tile retuning.  (Earlier text
  mentioned a future "native INT8 GEMM kernel using `dot4I8Packed`";
  empirically refuted on 2026-05-23 — Apple7 has no INT8 hardware
  path.  See `scripts/bench_int8_alu.py` + `PERFORMANCE_NOTES.md`
  mxfp8 entry.)

See "2026-05-17 follow-up" session below for the per-parent rollup and
the original detailed analysis that drove this hunt.

### Where to look for what

- **What's the current production wall time?** This section (TL;DR).
- **What's currently enabled by default?** This section (default flag stack).
- **What env knobs / CLI flags exist and what do they do?**
  [Benchmark Matrix](#benchmark-matrix) below.
- **Why is X enabled / disabled by default?** Find it in the matrix, then
  follow the "see session" pointer.
- **How do I profile or trace a specific step?**
  [Transformer Profiling](#transformer-profiling) and
  [Trace capture recipes](#trace-capture-recipes).
- **What did we try and abandon?**
  [Optimization History (archived)](#optimization-history-archived) at the
  bottom.

---

## Scope

In scope:

- MLX evaluation/materialization cadence.
- Reusing fixed per-run tensors instead of rebuilding them every denoise step.
- Reducing no-op casts, host/device transfers, and repeated small allocations.
- Narrow `mx.compile` experiments around pure repeated helper functions.
- Benchmark and logging improvements that make denoise timing easier to compare.

Out of scope:

- Reducing steps, frames, duration, or resolution.
- Switching pipeline type or stage layout.
- Disabling audio when comparing audio-video runs.
- Quantized weights as a *default* path (it remains a useful opt-in fast/draft
  mode — see matrix).
- VAE-only visual changes as denoise-speed optimizations.

## Measurement Rules

Use the same prompt, seed, weights, Gemma path, resolution, duration or frame
count, FPS, steps, dtype, audio setting, and VAE padding policy for each A/B.

Prefer `--save-run-log` or `--save-all-sidecars` (the latter is the canonical
choice for reproducibility — see `docs/TEST_PROMPTS.md`).

The final timing summary currently groups denoising and VAE decode under
`generation + decode`.  For denoise-only speed, use the live progress line:

```text
STEP1 ... | RUN ... | ETA ... | avg .../it
```

`RUN` and `avg` are the best quick comparison numbers.  For long
comparisons, keep the `_run.json` sidecar alongside the output video.

Run each candidate at least twice when testing compilation changes.  The first
iteration may include one-time graph construction or cache setup cost; steady
state is what matters.

---

## Benchmark Matrix

The authoritative current status of every knob.  Change ONE thing at a time
from the default stack when running A/Bs.

| Experiment | Status | Memory risk | Speed impact | Notes |
| --- | --- | --- | --- | --- |
| **--- shipped defaults ---** | | | | |
| `--internal-audio auto` | default | none | -35 % wall on video-only (256×256×25) | Resolves to `on` iff `--generate-audio`.  Legacy `LTX_DISABLE_INTERNAL_AUDIO=1` is honored but the new flag is preferred.  Matches mlx-video. |
| Audio module pretranspose | **partial default** (trimmed 2026-05-17) | none | -11 % AV (256×256×25) historical, neutral at bakery T=502 | The audio cache build inherits from `DEFAULT_VIDEO_FF_LAYOUT_SPECS` and `DEFAULT_VIDEO_ATTN_LAYOUT_SPECS`.  When those were trimmed (project_out:pretranspose only for FF, attn fully off), audio inherited the trimming.  `bf16_layout_audio` microbench at T=502 (bakery scale) confirmed all audio projections are tied with naive within ±1 % so the trimming is safe at bakery scale.  Historical −11 % at small T (256×256×25) is NOT re-verified — small-T workloads may regress.  Opt out the broader mechanism: `LTX_DISABLE_AUDIO_PRETRANSPOSE=1`.  See `PERFORMANCE_NOTES.md`. |
| QKV pretranspose | **opt-in (was default pre-2026-05-17)** | medium | -4 % additional AV at small T (256x256x25) | `to_q/to_k/to_v` were in the default attention layout, but `scripts/bench_ff_microbench.py bf16_layout` clean run showed they're tied with naive BF16 at the LTX-2.3 4096x4096 attention shape (within ±1 %).  The -4 % win was likely small-T-specific.  Re-enable via `--video-attn-layout to_q:pretranspose,to_k:pretranspose,to_v:pretranspose,to_out:pretranspose`. |
| `--video-ff-layout project_out:pretranspose` | default | none | -28 % bakery vs no-layout (77 → 55 s/it) | Same-math.  ONLY `project_out` is enabled by default: per `scripts/bench_ff_microbench.py bf16_layout`, project_out alone is a 35 % isolated-matmul win (rescues a kernel-selection cliff at K=16384,N=4096 where naive falls to 5.17 TFlops/s vs 7.95 with pretranspose).  `project_in` was the historical default but is +2.5 % in isolation and neutral end-to-end; opt back in via `--video-ff-layout project_in:pretranspose,project_out:pretranspose`.  Implementation drops the original weight after transposing (memory-neutral steady state).  See `PERFORMANCE_NOTES.md` "Pretranspose default cleanup" entry. |
| `--video-attn-layout` (default OFF, `()` empty) | default | none | none | Pre-2026-05-17 default was `to_out,to_q,to_k,to_v:pretranspose`.  Microbench (`bench_ff_microbench.py bf16_layout`) shows all four attention projections at the 4096x4096 shape are tied with naive BF16 (within ±1 %, all at ~7.9 TFlops/s).  Default flipped to OFF.  Opt back in via `--video-attn-layout to_out:pretranspose,to_q:pretranspose,to_k:pretranspose,to_v:pretranspose`.  End-to-end A/B with the old default still to be measured. |
| AdaLN/RoPE dtype cast-back | default | none | **-16.8 % bakery total** | The `scale_shift_table` tensors are FP32 (sincos precision).  Inline math `normed * (1 + scale) + shift` and `x + residual * gate` was promoting BF16 → FP32, forcing SDPA to compile `steel_attention_float32_*_maskfloat32_*` kernels (~2× the data movement of BF16).  Fixed at 5 sites in `transformer.py` + `rope.py`.  Bakery 29m 38s → 24m 40s.  See [2026-05-17 AdaLN/RoPE fix](#2026-05-17-adalnrope-dtype-promotion-fix). |
| Skip negative prompt encoding (distilled, cfg=1.0) | default | none | -10 % AV (256×256×25) + ~7 s/run on prompt encode | Distilled pipelines never use the negative encoding (cfg=1.0 makes it a no-op).  Both two-stage and one-stage now skip it.  Re-enable: `LTX_ENCODE_UNUSED_NEGATIVE=1`. |
| AVPipeline CFG short-circuit | default | none | ~2× denoise vs CFG-on | When `cfg_scale==1.0 and audio_cfg_scale==1.0 and rescale_scale==0.0` for euler+no-STG runs, `AVPipeline.__call__` routes to `_denoise_loop_simple_av` — one transformer pass per step instead of two.  Log line "CFG disabled (scale 1.0) - Running optimized single-pass inference" confirms the short-circuit fired. |
| Tokenize without max-length padding | default | none | -5 % AV (256×256×25) | Tokenizer was padding 2 real tokens to 1024 before Gemma forward.  Re-enable: `LTX_PAD_PROMPT_TO_MAX=1`. |
| `--mlx-cache-limit-gb 1` | default | low | neutral | Same-math allocator-cache cap.  Bakery AV process RAM 44 GB → 40 GB with no time penalty.  `0` returns freed buffers immediately. |
| Defer AV text encoder load until after Gemma | default | low | neutral; -6 GB prompt-encode peak | Trim Gemma hidden states, materialize, free Gemma, then load AV connector. |
| `--vae-decoder native` | only option | medium | none for denoise | The historical `legacy` (`SimpleVideoDecoder`, PyTorch-layout slice-conv) backend was archived 2026-05-23 after the native Conv3d decoder had been the production default for an extended bake-in period — source lives in `archive/simple_decoder.py.bak` for reference.  The matching `SimpleVideoEncoder` was archived to `archive/simple_encoder.py.bak` at the same time and replaced by `NativeConv3dVideoEncoder` (parity cos sim 0.99965 FP32; ~2-3× per-call speedup). |
| Terminal redraw throttling | default | none | -5.9 % bakery total on macOS | `DenoiseProgress` no longer spawns a heartbeat thread; `tqdm` uses `ascii=True + mininterval=1-2s`.  Bakery 31m 28s → 29m 38s.  Stage 2 alone -8 %.  See [Terminal redraw throttling](#terminal-redraw-throttling) for the full story. |
| **--- compute-precision opt-ins ---** | | | | |
| `--video-ff-dtype float16` | opt-in | none | **-4.6 % bakery 384x640x20s (8m 00s → 7m 38s)** | Cache-baked: cast project_in and project_out weights to FP16 at cache-build time, run the FF interior in FP16 (silu+geglu+matmuls), cast the residual back to BF16 at FF exit.  Attention/RMSNorm/RoPE/SDPA stay BF16.  ~15 % per-matmul FP16 win at production FF shapes (BF16/FP16 ratio 1.17–1.18×) translates to 4.6 % wall savings because FF is ~22 % of total compute and the FF-block win includes boundary casts.  Cosine sim ~0.71 vs BF16 — perceptually close, not bit-equivalent.  **Auto-pairs** with `project_in:pretranspose,project_out:pretranspose` (enforced in `scripts/generate.py:_ensure_ff_layout_for_dtype`).  FP16 *requires* pretranspose: at project_out (K=16384) the FP16 naive kernel falls off a deeper kernel-selection cliff than BF16 (4.95 vs 5.86 TFlops/s, then both recover to >8 TFlops/s with pretranspose).  Disabling pretranspose with FP16 on would be -18 % regression vs BF16 baseline.  Tried `--video-attn-dtype float16` too: net regression (+0.6 % wall) because the SDPA boundary's projection casts outweigh the matmul win; flag dropped.  See "FF FP16 + FP16 cliff" entry in `PERFORMANCE_NOTES.md` and `scripts/bench_pretranspose_dtype.py`. |
| `--audio-ff-dtype float16` | opt-in (curiosity) | none | **none measurable** (real-world A/B: +4.6 s within noise on bakery) | Mirror of `--video-ff-dtype` for the audio branch.  Per-matmul FP16 win at audio shapes is real (~10–13 % on audio.FF.project_in K=2048 N=8192 and audio.FF.project_out K=8192 N=2048) but per-call wall is ~2 ms vs video's 200+ ms.  Microbench predicted ~0.32 s savings (0.07 %); bakery A/B (video+audio-FF-FP16 7m 42.2s vs video-only-FF-FP16 7m 37.6s = +4.6s) is fully within single-run noise — output save phase alone varied by +2.6 s between the two runs.  Audio latent cos sim ≥0.997 vs the video-only-FP16 reference; perceptually unchanged by listening test on bakery dialogue.  **No kernel cliff at audio K=8192** — unlike video K=16384, FP16 naive is actually FASTER than BF16 naive there.  Auto-pairs audio FF pretranspose (avoids FP16 × BF16 → FP32 mixed-dtype promotion, not because of a cliff).  Kept as a documented opt-in for future hardware / workload shifts that might change the math.  See "Audio FF FP16" entry in `PERFORMANCE_NOTES.md` and `scripts/bench_pretranspose_dtype.py`. |
| **--- env-toggle opt-ins ---** | | | | |
| `LTX_VELOCITY_MODE=1` | opt-in | low | neutral | Inline velocity-form Euler update in `_denoise_loop_simple_av`.  Same math.  Kept env-gated for future MLX versions. |
| `LTX_ROPE_PRECOMPUTE=1` | opt-in | none | neutral | mlx-video pattern: per-stage RoPE precompute via `Modality.positional_embeddings`.  MLX's lazy graph already dedupes per-step RoPE calls. |
| `LTX_ADALN_PRETRANSPOSE=1` | opt-in | low | slight regression at small T | Cache-integrated pretranspose for the 8 `AdaLayerNormSingle.linear` projections.  Too few per step to amortize per-tensor dispatch overhead. |
| `to_gate_logits` pretranspose | opt-in | low | slight regression | Weight (4096×32 / 2048×32) too small for the implicit transpose to matter.  Enable: `--video-attn-layout to_out:pretranspose,...,to_gate_logits:pretranspose`. |
| `LTX_COMPILE_BLOCK_GROUPS=N` | opt-in | medium | neutral at tested scales | Eager-path `mx.compile` over N-block groups.  `N=4` at small T neutral; `N=48` at bakery neutral (18m 29s vs 18m 41s).  Compile-trace cost paid up front. |
| `LTX_MONO_INLINED=1` (stage2_harness only) | opt-in | none | neutral | Inlined 48-block forward + AdaLN preprocess + output projection.  Latent diff vs modular: cosine sim 0.999+.  Same math.  Confirms `nn.Module` dispatch is free at the MLX-graph level.  See [Monolithic-inlined transformer](#monolithic-inlined-transformer-experiment-negative-result). |
| **--- quantization opt-ins ---** | | | | |
| `--video-ff-quantize project_out:mxfp8` (+ `--video-ff-quantize-layers RANGE`) | **regression post-AdaLN-fix** | medium | **+10-45% SLOWER than baseline** | Pre-AdaLN-fix the 352×192 smoke saw -10% (vs broken 77.8 s/it BF16 baseline).  Post-fix at the bakery shape: variant B (project_out only, no layout) is +10.2% slower.  Per-matmul microbench (`scripts/bench_ff_microbench.py quant_matmul`) shows `mx.quantized_matmul` is consistently ~66% slower than `steel_gemm` at our shapes (4.7 vs 7.95 TFlops/s).  Quant pays bandwidth-savings cost but we're compute-bound on weights at our matmul shapes.  Flag still works for future MLX/hardware where quant kernel improves; see `PERFORMANCE_NOTES.md` Archive "mxfp8 draft mode is DEAD". |
| `--transformer-cache-quantize mxfp8-blocks` | **regression post-AdaLN-fix** | medium-high | **+43% SLOWER than baseline** | Pre-AdaLN-fix at stage 2 was 460 s/it vs 425 s/it for streaming-pretranspose (better than alternatives at the time).  Post-fix at stage 1: 65.0 s/it vs 45.5 s/it baseline (+42.9%).  Auto-disables same-math layouts.  Broader scope than `--video-ff-quantize` (attention + both FF) so the +66% per-matmul quant penalty compounds across more of the step.  Non-parity.  See `PERFORMANCE_NOTES.md` Archive. |
| `--transformer-cache-quantize mxfp8-blocks-pretranspose` | **regression post-AdaLN-fix** | medium-high | **+45% SLOWER than baseline** | Pre-AdaLN-fix matched `mxfp8-blocks` speed.  Post-fix at stage 1: 66.0 s/it vs 45.5 baseline (+45.0%).  Packing `weight.T` before quantizing does NOT recover the layout win on top of quantized matmul.  See `PERFORMANCE_NOTES.md` Archive. |
| **--- memory-constrained ---** | | | | |
| `--stream-transformer` | opt-in | low-medium | constrained-memory preset | Expands to `--transformer-block-resident-blocks 16 --transformer-block-compile --transformer-block-compile-group-size 4`.  Preferred user-facing switch.  See [Block streaming](#block-streaming-constrained-memory-mode). |
| `--transformer-block-resident-blocks` | opt-in | low | slower | Cache-backed streaming.  `r4` ~8 GB process RAM but ~70.5 s/it on bakery.  Constrained-memory mode, not fast path. |
| `--transformer-block-compile` | opt-in | low-medium | mixed | Resident-group compile r8 completed at ~61.2 s/it after one prior Metal watchdog abort.  Cache/watchdog-sensitive. |
| `--transformer-block-compile-group-size` | opt-in | low-medium | stabilizes larger shapes | Splits compiled/eval command-buffer groups.  1024×576 with r16/group-4 completes without watchdog abort (~425 s/it). |
| `MLX_MAX_OPS_PER_BUFFER=1 MLX_MAX_MB_PER_BUFFER=10` | opt-in | low | unknown | Must be set before Python starts.  Real MLX command-buffer split knobs for watchdog-pressure A/B.  Cannot split inside a single huge op. |
| **--- diagnostic / profiling ---** | | | | |
| `--profile-transformer-steps N[,M,...]` | diagnostic | low | perturbs measurement | Forces eval checkpoints on selected denoise steps. |
| `--profile-transformer-blocks N[,M,...]` | diagnostic | low | perturbs measurement | Adds forced evals inside selected blocks for already-profiled steps.  See [Transformer Profiling](#transformer-profiling). |
| `LTX_PROFILE_PAUSE_BEFORE_DENOISE=1` | diagnostic | none | blocks once | Prints `pid=N` and waits for Enter, before denoise step 1.  Pair with `xcrun xctrace record --attach <PID>`.  See [Trace capture recipes](#trace-capture-recipes). |
| `LTX_PROFILE_STOP_AFTER_STEPS=N` | diagnostic | none | exits early | `sys.exit(0)` after step N.  `N=2` is the minimum useful capture window.  Same hooks exist in mlx-video for direct A/B captures. |
| `LTX_PROFILE_SIGNPOSTS=1` | diagnostic | none | ~17 ms/run overhead | Wraps 8 parent phases + 5 sub-phases (`attn_qkv`, `attn_sdpa`, `attn_out`, `v_ff_adaln`, `v_ff_inner`) with `os_signpost` intervals in subsystem `ltx`, category Points of Interest.  Auto-builds a ctypes-loaded C shim on first import. |
| `LTX_PROFILE_SIGNPOSTS_SYNC=1` | diagnostic | medium | ~10 % wall overhead | Forces `mx.eval()` on phase output at signpost end.  Required for time-based attribution. |
| `LTX_PROFILE_SIGNPOST_LOG=/path` | diagnostic | none | none | Writes `<monotonic_ns> <begin\|end> <phase>` per line.  Ground-truth source for attribution when the trace's `os-signpost` table drops events under buffer pressure. |
| `LTX_DISABLE_COMPILED_ATTN=1`, `LTX_DISABLE_COMPILED_HELPERS=1` | diagnostic | none | regression | Strips the default `mx.compile` wrappers — needed for `scripts/sdpa_dtype_probe.py` to break the compiled regions with `mx.eval` barriers.  Production: -1.5 s neutral, -0.6 s neutral.  Apples-to-apples trace: leaving them ON saves ~3 s in 2-step window. |
| **--- tested neutral or removed ---** | | | | |
| `MLX_METAL_FAST_SYNCH=1` | tested | low | none | Slightly slower on 352×192 15s AV smoke. |
| Remove `--low-memory` | tested | medium | none in small run | 352×192 15s AV smoke slightly slower. |
| `--fast-mode` (AV `fast_mode`) | opt-in user flag | high | none in small run; material at bakery | Wired through the AV transformer path; sets the transformer's intermediate eval cadence to 0.  User passes the flag explicitly — not auto-enabled.  Tied low-memory baseline at 352×192; real win shows up at bakery scale (post-AdaLN-fix bakery numbers in the TL;DR were captured with `--fast-mode`).  Higher peak unified-memory pressure is the cost. |
| Per-run RoPE precompute | tested | low | none | Removed after slightly slower 352×192 smoke. |
| No-op cast/allocation cleanup | tested | low | low | Good hygiene after larger wins. |
| Fast GELU approximation | not pursued | low | unlikely | FFN sub-profile showed GELU at 0.7 % of clean block. |
| Historical full-transformer compile experiments | removed | medium-high | none | Bakery AV smokes tied baseline while adding complexity. |
| `LTX_DISABLE_BLOCK_OVERHEAD=1` | removed | low | none | Stripped-down `_fast_call` path in `BasicAVTransformerBlock`.  Python-side overhead is not the bottleneck. |
| Packed attention layouts (`self_qkv:pack`, `kv:pack`) | removed | low | neutral | Reached ~53 s/it on quiet r16 path — matching `to_out:pretranspose` alone.  Removed for not justifying the added CLI/runtime surface. |
| Stage-2 SVD/residual cache | removed | medium-high | bad quality | Broad block windows visibly distorted (MAE 0.18-0.46).  Narrow probes saved ~15 s with foreground noise.  Keep stage-2 refinement exact. |
| `mx.qqmm` FF linears | blocked | medium-high | n/a | Current Metal runtime: `[QQMatmul] NYI for the general case` on LTX FFN shapes. |
| `mx.block_masked_mm` | not applicable | high | n/a | Only for structured block sparsity. |
| `mx.gather_mm` / `mx.gather_qmm` | not applicable | high | n/a | MoE/routing, not dense LTX. |
| `vmap` helper/probe loops | candidate | low | low | Useful only where Python loops show up. |
| `mx.fast.rope` / custom split-RoPE kernel | candidate | low-medium | unknown | Only after proving exact RoPE parity. |
| Distributed tensor parallelism | separate project | high | unknown | Multi-machine, out of scope. |
| FP8 conversion primitives | research | high | unknown | `mx.to_fp8` / `mx.from_fp8` are storage/compute research items. |
| `MLX_SDPA_BLOCKS` (MLX PR #3455) | inapplicable | none | none | Controls `sdpa_vector_2pass` (T ≤ 8) only.  LTX uses `sdpa_full` (T = 35,136 at 1024×576). |

---

## Recent Sessions

Reverse chronological.  Most recent at top.

### 2026-05-23: FF compute in FP16 (4.6 % bakery win, attn FP16 dropped)

Added `--video-ff-dtype float16` as an opt-in.  Cache-baked: the transformer
cache safetensors hold project_in/project_out pretransposed weights cast to
FP16; the residual stream stays BF16, FF entry/exit casts handle the
boundary.  Cosine sim ~0.71 vs BF16 baseline — perceptually close but not
bit-equivalent.

**Bakery 384x640x20s timings** (distilled two-stage, idle machine, fast-mode):

| variant                         | wall      | Δ vs BF16   |
| ------------------------------- | --------- | ----------- |
| BF16 baseline                   | 7m 59.6s  | (reference) |
| `--video-ff-dtype float16`      | **7m 37.6s** | **-22s (-4.6 %)** |
| `--video-ff-dtype float16 --video-attn-dtype float16` | 7m 56.7s  | -3s |
| `--video-attn-dtype float16` (attn-only) | 8m 17.0s  | **+18s** |

Attention FP16 is a NET REGRESSION even though the matmul is faster: the
4 BF16↔FP16 boundary casts per attention block (~528 attention calls per
generation) cost more than the ~10 ms/projection FP16 win, plus
`mx.fast.scaled_dot_product_attention` doesn't accept FP16 inputs cleanly.
The `--video-attn-dtype` flag was tried and removed.

**Why only 4.6 % when microbench shows FP16 matmul is 15 % faster:**

Microbench at production shapes (M=14640, K=4096, N=16384 / 16384→4096):

| projection                       | BF16 pretrans | FP16 pretrans | FP16 saves |
| -------------------------------- | ------------- | ------------- | ---------- |
| FF.project_in   (K=4096 N=16384) | 7.99 TFlops/s | 9.38 TFlops/s | +14.8 %    |
| FF.project_out  (K=16384 N=4096) | 8.10 TFlops/s | 9.51 TFlops/s | +14.9 %    |
| attn.QKV/O      (K=N=4096)       | 8.09 TFlops/s | 9.53 TFlops/s | +15.0 %    |

Wall savings ≈ (per-matmul FP16 win) × (FF compute share).  FF is ~22 % of
stage-2 compute (attention is ~3.6× more compute per layer at T=14640), so
~15 % × 22 % ≈ 3.3 % expected; observed denoise savings ~12 s out of
~6m 27s = 3.1 %.  Microbench prediction lines up to within 1 s.  The
remaining ~10 s of the headline 22 s "total wall" delta is noise in
pre/post phases (audio decode, save, encode) that aren't dtype-sensitive.

**FP16 has a worse kernel-selection cliff than BF16.**  At project_out's
K=16384 in the naive layout, BF16 falls to 5.86 TFlops/s (the historical
2026-05-17 cliff this codebase already mitigates via pretranspose).  FP16
naive falls *further* to 4.95 TFlops/s — i.e. FP16 naive is 18 % slower
than BF16 naive on this shape.  Both layouts recover to >8 TFlops/s with
pretranspose.  The auto-pair helper `_ensure_ff_layout_for_dtype` in
`scripts/generate.py` enforces `project_in:pretranspose,project_out:pretranspose`
whenever `--video-ff-dtype float16` is set, so a user can't accidentally
trip the cliff.  Tested via
`scripts/bench_pretranspose_dtype.py`.

Full bench logs and additional detail in `PERFORMANCE_NOTES.md` "FF FP16
+ FP16 kernel cliff" entry.

**Audio-side followup (`--audio-ff-dtype float16`):** same plumbing
mirrored for the audio branch and shipped as a curiosity opt-in.
Microbench predicted ~0.32 s savings per generation (0.07 % of 7m 38s)
because audio FF per-call wall is small (~2 ms vs video's 200+ ms).
Real-world bakery A/B confirmed: video+audio-FF-FP16 7m 42.2s vs
video-only-FF-FP16 7m 37.6s = +4.6 s, fully within the ~3 s single-run
noise floor (output save phase alone varied by +2.6 s).  Audio latent
cos sim ≥0.997 vs the video-only-FP16 reference; audio dialogue
perceptually unchanged by listening test.  No kernel cliff at audio
K=8192 — FP16 naive is faster than BF16 naive there.  Kept as
documented opt-in for future hardware / workload shifts.  See "Audio
FF FP16" entry in `PERFORMANCE_NOTES.md`.

### 2026-05-17 Follow-up: probe + non-sync A/B + sub-phase signposts (gap closed)

The +5.3 s "residual to mlx-video" from the AdaLN session below turned out to
be **signpost/sync-mode overhead**, not real GPU work.  Two new measurements
disproved the gap and reframed the brutal-efficiency hunt.

#### Per-call SDPA head-to-head (identical workloads, eval-barrier mode)

Extended `scripts/sdpa_dtype_probe.py` with `LTX_PROBE_TIME_SDPA=1` to wrap
every `mx.fast.scaled_dot_product_attention` call in `mx.eval` barriers
+ timer.  Ran both projects at distilled stage 1, 1024×576 → 288×512
latent, 2 steps, identical seed/prompt.  Required
`LTX_DISABLE_COMPILED_ATTN=1 LTX_DISABLE_COMPILED_HELPERS=1` on the
LTX side so `mx.eval` could break MLX's compiled regions.

| Phase (shape)                                          | LTX total | mlxv total | Δ          |
| ------------------------------------------------------ | --------- | ---------- | ---------- |
| video_self_attn  q/k/v=(1,32,8784,128), mask=None      | 22,287 ms | 22,743 ms  | **−456**   |
| video_text_ca    q=(1,32,8784,128), kv=(1,32,1024,128) | 2,696 ms  | 2,791 ms   | **−96**    |
| v2a_cross        q=(1,32,501,64), kv=(1,32,8784,64)    | 660 ms    | 703 ms     | **−43**    |
| a2v_cross        q=(1,32,8784,64), kv=(1,32,501,64)    | 640 ms    | 650 ms     | **−10**    |
| audio_text_ca    q=(1,32,501,64), kv=(1,32,1024,64)    | 108 ms    | 119 ms     | **−11**    |
| audio_self_attn  (1,32,501,64)                         | 80 ms     | 82 ms      | tied       |
| **SDPA TOTAL**                                         | **26,470** | **27,088** | **−618 (LTX +2.3 %)** |

Both projects produce **identical SDPA shapes** and **identical kernel
selection** (every call BF16, no mask, same dims).  The "180 ms vs 133 ms" gap
in the [2026-05-16 Apples-to-apples](#2026-05-16-apples-to-apples-metal-traces)
trace was a sampling-bucket artifact of the
`metal-shader-profiler-dispatches` table, not a real per-call difference.
Per-block SDPA distribution is flat 232-245 ms (LTX) vs 235-272 ms (mlxv) —
LTX has the tighter tail.

#### Non-sync end-to-end wall-time A/B

`scripts/bench_ab_wall_time.sh` runs both projects in production lazy-graph
mode (no signposts, no eval barriers) for 8 stage-1 steps each at the same
workload.  Results in TL;DR above; per-step trajectory:

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

LTX is consistent; mlxv drifts upward after step 4.  The prior section's
"+5.3 s residual" was the cost of:

- 384 signpost emit pairs per step (8 phases × 48 blocks).
- Per-phase `mx.eval` barriers in sync mode (lazy graph fully broken).
- Profile/event overhead specific to the LTX trace protocol.

mlx-video's traces showed less overhead because they emit fewer signposts per
block (instrument at block level, not the 8 sub-ops).

#### Sub-phase signposts (5 new phases) + validation

Added `attn_qkv`, `attn_sdpa`, `attn_out`, `v_ff_adaln`, `v_ff_inner` inside
the existing 8 parent phases.  Wired into `attention.py`
(`Attention.__call__`) and `transformer.py` (around the `video_ff` block).
Nest cleanly inside parent signposts; aggregate across all attention call
sites.

Validated via 2-step sync-mode capture, 1024×576 stage 1:

- All 13 phases emit with matched begin/end pairs.
- Sums reconcile with parent phase walls within 1 %.
- Unaccounted residual is the unwrapped `residual_gate` at phase end.

`scripts/analyze_signpost_subphases.py` walks the sidecar log and attributes
each sub-phase to its currently-open parent phase.  Per-(parent, sub-phase)
rollup with `n`, total, mean, p50/p99/max + `[unaccounted]` budget per
parent.  Calls outside any parent (text encoder, AV connector during prompt
encode) bucket under `[no_parent]`.  Top-N slowest individual intervals also
reported.

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

#### Reframed brutal-efficiency hunt

With the gap-to-mlxv question closed, the new question is: **how far below
45 s/it can we drive per-step wall on the same workload**?  We're at 26 % of
M1 Max BF16 peak on SDPA (2.7 TFlops/s of 10.4) and ~64 % of peak on FF
(6.7 TFlops/s).  Plenty of headroom.

**Two rankings of the targets (they don't agree, by design):**

| Sub-phase                       | per-step | rank by size | rank by actionability | why |
| ------------------------------- |---------:|:------------:|:---------------------:| --- |
| video_ff / v_ff_inner           | 18.7 s   | **#1**       | **#1**                | Two compatible levers: (a) `mxfp8` quant on `project_in/project_out` — tested at ~10 % faster, quality cost; (b) custom Metal kernel fusing `project_in + GELU + project_out` into one pass so the 16384-dim intermediate never materializes — bandwidth win at T=8784, similar to FlashAttention.  Combinable. |
| video_self_attn / attn_sdpa     | 11.0 s   | **#2**       | **#3**                | At-parity with mlx-video per the per-call probe.  Only path forward is a custom FlashAttention-2 Metal kernel — bounded but high-effort. |
| video_self_attn / attn_qkv      | 7.7 s    | #3           | **#2**                | 3 separate matmuls (V, Q, K) + 2 RMSNorms + 2 RoPE calls.  Fuse Q+K+V into one combined matmul if MLX isn't doing it internally. |
| video_text_ca / attn_qkv        | 3.5 s    | #4           | #5                    | Same fusion lever, smaller scale (Q on T=8784, K/V on T=1024). |
| v2a_cross / attn_qkv            | 2.6 s    | #5           | #6                    | Q on T=501, K/V on T=8784.  Small absolute. |
| video_self_attn / attn_out      | 2.4 s    | #6           | #4                    | 4096→4096 matmul + per-head gate apply.  If gate is close to identity post-zero-init, gate-apply could be elided. |
| video_text_ca / attn_out        | 2.4 s    | tied         | tied                  | Same as above, different parent. |

**Three sanity-check observations:**

1. **video_ff is almost entirely the inner FF matmul** — 99.1 % is
   `v_ff_inner` (project_in + GELU + project_out); AdaLN modulation is
   1.3 ms/block (0.3 %).  "Fuse AdaLN INTO the FF matmul" is
   dead-on-arrival (AdaLN has no time to save).  This is different from
   the FF-internal fusion lever above, which IS live.
2. **attn_sdpa cross-validates the per-call probe** — 22.08 s sync vs
   22.29 s eval-barrier.  Same kernel, same shape, same speed.
3. **a2v_cross has 16 % unaccounted time** — the AdaLN modulation at
   `transformer.py:691-692` is inside the `a2v_cross` parent signpost but
   NOT wrapped in a sub-phase.  Worth wrapping if a2v_cross becomes a
   target (currently only 3.8 s/step — probably not).

**Video ops dominate by a wide margin** — video_self_attn + video_ff +
video_text_ca alone is 47.7 s/step out of ~50 s.  Audio + cross-modal
combined is < 10 s/step.

#### Tooling added in this session

- `scripts/sdpa_dtype_probe.py` — monkey-patches
  `mx.fast.scaled_dot_product_attention` before any LTX import; records
  every call's (dtype, shape) signature with caller stack.  Optional
  per-call timing via `LTX_PROBE_TIME_SDPA=1` (requires compile-disable
  env vars).  Works against mlx-video via `LTX_PROBE_MODULE=mlx_video.models.ltx_2.generate`.
- `scripts/bench_ab_wall_time.sh` — production-mode wall-time A/B.
  Sequential.  Required env: `LTX_REPO`, `MLXV_REPO`, `MLXV_MODEL_REPO`.
  Optional: `LTX_VENV_BIN`, `MLXV_VENV_BIN`, `STEPS`, `SEED`, `AB_OUTDIR`.
- Sub-phase signposts in `_signpost.c` / `signpost.py` + wiring in
  `attention.py` and `transformer.py`.
- `scripts/analyze_signpost_subphases.py` — walks sidecar log and
  attributes each sub-phase to its currently-open parent.

### 2026-05-17: AdaLN/RoPE dtype-promotion fix

Silent BF16 → FP32 dtype promotion through the AdaLN modulation path was
forcing SDPA to compile and dispatch pure `steel_attention_float32_*` kernels
— roughly 2× the data movement and compute of the BF16 equivalents.

**Note:** the 05-17 follow-up above showed we were never actually behind
mlx-video in non-sync production mode.  The AdaLN fix is real and valuable
(it changes the SDPA kernel selection and reduces bakery wall by 16.8 %), but
the original framing "this was THE gap-causer" was wrong — see follow-up
section above.

#### The dispute that led here

The 2026-05-16 traces had two competing analyses:

- Mine: "video_text_ca is 5.2× more expensive in LTX, the
  `_apply_text_cross_attention` helper is fragmenting dispatch."
- Codex's correction: my "GPU time per phase" wasn't elapsed wall time
  (it summed dispatch durations across parallel GPU channels and exceeded
  the phase wall span).  Sidecar interval math is the honest primary metric.

After accepting the correction, the story was: LTX was slower on every
phase, not just one — distributed overhead, not a single hotspot.  Three
env-toggle experiments ruled out obvious causes:

- `LTX_DISABLE_COMPILED_ATTN=1`: neutral (+1.5 s)
- `LTX_DISABLE_COMPILED_HELPERS=1`: neutral (−0.6 s)
- `--video-attn-layout off`: regression (+3.1 s — pretranspose IS a win)

Then Codex spotted the actual signal in the trace's shader **inventory**
(the `metal-shader-profiler-shader-list` table, not the empty
`metal-shader-profiler-intervals` table):

- LTX inventory: 30+ pure `float32` steel kernels, including 6
  `steel_attention_float32_*_maskfloat32_*` variants.
- mlx-video inventory: zero pure `float32` steel kernels — only BF16 or
  `bfloat16_float32`-accumulator variants.

#### Probe: prove what dtype actually reaches SDPA

Instead of guessing, monkey-patch `mx.fast.scaled_dot_product_attention`
BEFORE importing any LTX-2-MLX modules, log every call's
`(q.dtype, k.dtype, v.dtype, mask.dtype, shapes)`, then run one denoise step:

```python
# scripts/sdpa_dtype_probe.py — runs scripts/generate.py under a
# monkey-patched SDPA that records every call's dtype.
```

First hypothesis was that split-RoPE was promoting BF16 to FP32 by
multiplying against FP32 cos/sin tables and not casting back.  We fixed
that.  Result: same FP32 kernels still in the inventory, wall time
unchanged.

Re-ran the probe.  **Every per-block SDPA call had q/k/v dtype `float32`**,
including the V tensor — which never goes through RoPE.  So RoPE wasn't the
source; promotion happened *upstream* of attention.

#### Root cause: AdaLN scale/shift/gate are FP32 tables

The 6 `scale_shift_table` tensors in `BasicTransformerBlock` /
`BasicAVTransformerBlock` are explicitly typed
`mx.zeros((..., dim), dtype=mx.float32)` (kept FP32 for sincos /
timestep-embedding numerical stability).  Then the inline math:

```python
normed * (1 + scale) + shift   # AdaLN
x + residual * gate            # residual gate
```

promotes BF16 activations to FP32 by broadcasting against the FP32
scale/shift/gate.  The FP32 result propagates through `to_q/to_k/to_v` into
SDPA, which compiles pure `steel_attention_float32_*` kernels.

mlx-video's `rope.py` and equivalent paths have always cast back to the
input dtype (`output.astype(input_dtype)`).  LTX-2-MLX was missing this cast
at five call sites:

1. `_adaln_inline` (transformer.py)
2. `_residual_gate_inline` (transformer.py)
3. `_apply_text_cross_attention` V2 modulation
4. Cross-modal A2V/V2A inline scale-shift
5. Cross-modal A2V/V2A residual gates

Plus `apply_split_rotary_emb` and `apply_interleaved_rotary_emb` fallback in
rope.py (kept the RoPE fix even though it wasn't the bottleneck — same
pattern, correct hygiene).

#### Verification (post-fix probe)

Re-running the SDPA dtype probe with the AdaLN fix: **all 9 unique SDPA
signatures are now bfloat16 q/k/v** — connector, every per-block phase,
video and audio.  No more pure FP32 paths.

#### Numbers

**2-step sync-mode capture (288×512 stage 1, signposts on, mx.eval
barriers):**

| Phase            | baseline | +rope-only | +adaln (fix) | Δ baseline | mlxv r2 |
| ---------------- | -------- | ---------- | ------------ | ---------- | ------- |
| video_self_attn  | 46.01 s  | 46.12 s    | **40.80 s**  | **−5.21**  | 39.04 s |
| video_ff         | 38.19 s  | 38.71 s    | **33.98 s**  | **−4.21**  | 35.58 s |
| video_text_ca    | 14.46 s  | 14.79 s    | **12.98 s**  | **−1.48**  | 11.40 s |
| audio_self_attn  |  1.40 s  |  1.52 s    |   1.24 s     |  −0.16     |  0.65 s |
| audio_text_ca    |  1.04 s  |  1.18 s    |   1.19 s     |  +0.14     |  0.67 s |
| a2v_cross        |  5.82 s  |  5.88 s    |   5.76 s     |  −0.06     |  5.07 s |
| v2a_cross        |  5.24 s  |  5.33 s    |   5.39 s     |  +0.15     |  4.87 s |
| audio_ff         |  2.52 s  |  2.86 s    |   2.23 s     |  −0.28     |  0.97 s |
| **TOTAL**        | **114.69** | 116.37   | **103.58 s** | **−11.11** | 98.26 s |

(The remaining +5.3 s vs mlxv in this sync-mode capture is signpost-emit
overhead — see [follow-up section above](#2026-05-17-follow-up-probe--non-sync-ab--sub-phase-signposts-gap-closed).)

Shader inventory: pure-FP32 steel kernels **20 → 0**, pure FP32 attention
kernels **6 → 0**.

**Full bakery 1024×576×481 distilled (fast mode, end-to-end, non-sync):**

|                          | baseline (per prior PERFORMANCE.md) | post-fix       | Δ      |
| ------------------------ | ----------------------------------- | -------------- | ------ |
| Stage 1 (8 steps, 288×512) | n/a (was ~10 s/it pre-fast-mode)  | 45.3 s/it = 6m 02s | |
| Stage 2 (3 steps, 576×1024) | ~50 s/it baseline → ~26 m stage 2 | 313.5 s/it = 15m 40s | |
| VAE decode (tiled)       |                                     | 2m 10s         |        |
| **Total**                | **29m 38s**                         | **24m 40.6s**  | **−4m 58s, −16.8 %** |

The win scales *better* at full bakery than the sync-mode-stage-1
measurement predicted (9.7 % → 16.8 %) because stage-2 token counts
(~280 k tokens) make the per-op FP32-vs-BF16 difference matter more, and
because fast mode + smaller BF16 intermediates lets MLX's lazy graph fuse
more without OOM risk.

#### Quality

Visual inspection on the bakery output: indistinguishable from prior runs.
Pixel-level diffs are real (the prior path held FP32 intermediates inside
AdaLN before settling back to BF16 at layer boundaries; the new path is BF16
end-to-end), but content and structure are unchanged.  Matches mlx-video's
reference precedent — they've always done it this way.

If a stricter quality check is needed: `--save-latents` writes the stage-1
and stage-2 latents; compare cosine sim against a known-good pre-fix `.npz`.
Expect ~0.999 with BF16-rounding-noise scale of 1e-3 in p99 abs diff.

#### Scaling validation: 30-second 1024×576 (721 frames)

Same fix applied, different prompt + seed, longer duration to confirm the
win holds at higher token counts:

| Stage                          | Bakery (481 frames) | Kitten (721 frames) | Ratio        |
| ------------------------------ | ------------------- | ------------------- | ------------ |
| Frames                         | 481                 | 721                 | 1.50×        |
| Latent shape                   | 61×18×32            | 91×18×32            | 1.49× tokens |
| Stage 1 s/it (288×512)         | 45.3 s              | 75.9 s              | 1.68×        |
| Stage 2 s/it (576×1024)        | 313.5 s             | 602.1 s             | 1.92×        |
| Total wall                     | 24m 40s             | 44m 41s             | 1.81×        |
| Per-token-step cost (stage 2)  | ~8.9 ms             | ~11.5 ms            | 1.29×        |

Stage 2 scales 1.92× for 1.49× tokens — expected attention quadratic
dilution.  Memory: **43 GB peak process RAM** during stage 2 (Activity
Monitor).  No memory pressure on a 64 GB M1 Max with nothing else running.

### 2026-05-16: Apples-to-apples Metal traces

Continuation of the perf investigation.  Goal: stop guessing about "is
mlx-video really faster, and if so why," and get matched Metal System Trace
captures that cover identical work windows.

> **Status note (added 2026-05-17 follow-up):** The "+5.4 % wall gap to
> mlx-video" finding from this session was overturned.  In non-sync
> production mode (what users actually run), LTX-2-MLX is **8.7 % faster**
> than mlx-video, not slower.  The sync-mode capture protocol itself added
> ~6 s/run of overhead from per-phase eval barriers + signpost emission.
> The trace findings about dispatch dynamics and kernel inventories are
> still valid — they just don't translate to a production gap.

#### Profile hooks: pause-then-stop

Two env-gated helpers added to `AVPipeline.generate_distilled_two_stage`:

- `LTX_PROFILE_PAUSE_BEFORE_DENOISE=1` — blocks on stdin once, immediately
  before stage 1's first denoise step.  Prints `pid=N` so a second-terminal
  `xcrun xctrace record --attach <PID>` can connect at a precise point.
- `LTX_PROFILE_STOP_AFTER_STEPS=N` — `sys.exit(0)`s after step N.  `N=2`
  is the minimum useful capture window.

Both zero-cost when unset.  Identical-name hooks were added to `mlx-video`
so the same shell incantation produces directly comparable captures.

#### 3-way apples-to-apples results (bakery 1024×576×481, distilled AV)

| Metric                          | mlx-video | LTX w/ compile | LTX no-compile |
| ------------------------------- | --------- | -------------- | -------------- |
| **2-step wall time**            | **98.18s**| **103.47s**    | **106.80s**    |
| Per-step avg                    | 49.1 s    | 51.7 s         | 53.4 s         |
| GPU busy total                  | 99.88 s   | 101.67 s       | 101.87 s       |
| GPU utilization                 | 102 %     | 98 %           | 95 %           |
| Total dispatches                | 12,143    | 12,625         | 9,085          |
| Dispatches / sec                | 124       | 122            | 85             |
| Median dispatch dur             | 962 µs    | 994 µs         | 918 µs         |
| **p99 dispatch dur**            | **44 ms** | **62 ms**      | **187 ms**     |
| Max dispatch dur                | 168 ms    | ~150 ms        | 325 ms         |
| Dispatch gaps > 20 ms           | **1**     | 38             | 50             |
| Largest dispatch gap            | 24 ms     | 100 ms         | **2283 ms**    |

The "no-compile" column was captured with `LTX_DISABLE_COMPILED_ATTN=1
LTX_DISABLE_COMPILED_HELPERS=1 LTX_DISABLE_FUSED_ROPE=1`.

#### What this overturned from earlier sessions

**`mx.compile` is not a no-op.**  Earlier bench tables marked compile-vs-no
neutral.  Apples-to-apples 2-step window: compile shaves **3.3 s (–3.1 %)**
off wall, cuts p99 tail **3×** (187 ms → 62 ms), drops largest dispatch gap
from 2.28 s to 0.10 s.  Default to compile on.

**"mlx-video has more fused kernels" is backwards.**  In steady state
mlx-video runs **124 dps** while we run **122 dps with compile, 85 dps
without**.  They are *less* fused than us-on-compile, not more.  The
"8 steel_attention specializations vs our 2-3" thing isn't fusion — it's
the opposite, finer-grained dispatch.

**GPU busy time is essentially equal across all three** (within 2 %).  The
~5 % sync-mode wall gap to mlx-video was *not* "they run faster math" —
it was dispatch dynamics + signpost overhead.  (See 05-17 follow-up: in
production non-sync mode the gap is reversed; we're faster.)

#### Signposts: attribute the long-tail dispatches

Three env vars added for per-phase trace attribution.  All zero-cost when
unset, eager-init at module import:

- `LTX_PROFILE_SIGNPOSTS=1` — wraps the 8 sub-ops in
  `BasicAVTransformerBlock.__call__` with `os_signpost` intervals in
  subsystem `ltx`, category Points of Interest.
- `LTX_PROFILE_SIGNPOSTS_SYNC=1` — also forces `mx.eval()` on each phase's
  output via `signpost_barrier()`.  Required for time-based attribution.
  Without this, MLX's lazy graph queues phase N+1's ops microseconds after
  phase N on the Python side while the GPU lags by seconds, so signpost
  intervals don't bracket the actual dispatches.  `mx.synchronize()` alone
  is NOT enough.
- `LTX_PROFILE_SIGNPOST_LOG=/path` — sidecar log with `<monotonic_ns>
  <begin|end> <phase>` per line.  Belt-and-suspenders source if the
  trace's `os-signpost` table loses events under buffer pressure.

Implementation: `LTX_2_MLX/utils/signpost.py` ctypes-loads a tiny C shim
(`LTX_2_MLX/utils/_signpost.c`, auto-built on first import via clang) that
exposes the per-phase `begin`/`end` symbols.  The shim is needed because
`os_signpost`'s macro-based API embeds the calling image's `__dso_handle`,
which can't be passed from ctypes.

Disabled overhead: ~0.6 µs per context-manager enter (~5 ms total for a
2-step capture).  Enabled overhead: ~3 µs per enter (~17 ms total).  Both
negligible vs ~50 s/step.

(Sub-phase signposts added in the 05-17 follow-up — see that section for
`attn_qkv`, `attn_sdpa`, `attn_out`, `v_ff_adaln`, `v_ff_inner`.)

#### What we learned about signpost capture

The Instruments GUI is lossy under buffer pressure (system-wide Metal use
crowds out per-process signposts).  Two false-start fixes (pre-warming the
transformer; suppressing signposts during prewarm) didn't help and were
reverted.

**Real fix: capture via the `xctrace record` CLI** (see [Trace capture
recipes](#trace-capture-recipes)).  Captures every signpost reliably.

#### Per-phase attribution results (stage 1, 288×512, 2-step sync capture)

Pre-AdaLN-fix data.  See 05-17 follow-up for post-fix numbers.

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

> Note: the "180 ms vs 133 ms video_self_attn max" gap looked actionable
> at the time and drove the "hunt for SDPA outliers" plan.  The 05-17
> per-call probe disproved it as a sampling artifact (every call is flat
> 232-245 ms in both projects).

#### Why the "monolithic-inlined" negative result is consistent

The 2026-05-15 inlined-transformer experiment found that collapsing the
48-block module dispatch into one inlined function moved per-step time by
~0 %.  This trace data explains why: per-step time isn't bottlenecked by
Python or module dispatch — it's at the Metal kernel / command-buffer
level.  The inlined reference is useful as same-math documentation, not
as a speed lever.

### 2026-05-15: Audio pretranspose, skip-negative, bakery parity

Investigation kicked off from a measured per-step gap to mlx-video at small T
(distilled `--generate-audio` at 256×256×25).  Starting baseline 46.85 s
end-to-end (8+3 step distilled two-stage); bakery 28-30 min total, stage 2
alone 18m 41s (~373 s/it).

#### What shipped (defaults changed)

**`--internal-audio auto`** — `internal_audio_active = is_av_model and
(use_internal_audio_branch or audio_enabled)` made the legacy
`LTX_DISABLE_INTERNAL_AUDIO=1` knob silently ignored whenever
`--generate-audio` was set.  Without `--generate-audio` the AV transformer
was still running the full audio branch every denoise step and discarding
the result.  Fix: new `--internal-audio auto|on|off` flag; `auto` (default)
resolves to `on` only when `--generate-audio` is set.  Impact on 256×256×25:
**47.50s → 30.91s (-34.9 %)**.

**Audio module pretranspose (cache-integrated)** — the existing same-math
layout pipeline only walked video-Q attention modules and `block.ff`.
Audio-side modules (`audio_attn1`, `audio_attn2`, `video_to_audio_attn`,
`audio_ff`) were running `mx.addmm(bias, x, weight.T)` with implicit
transpose on every per-step call.  Wired into the cache; hash bumps.  Opt
out: `LTX_DISABLE_AUDIO_PRETRANSPOSE=1`.  Impact 256×256×25:
**46.85s → 41.61s (-11.2 %)**.

**QKV pretranspose** — extended the attention layout machinery to
`to_q`, `to_k`, `to_v` in addition to `to_out`.  18 more matmuls per AV
block per step.  Impact: **41.61s → 40.09s (-3.7 %)** on top of audio
pretranspose.

**Skip negative prompt encoding (distilled two-stage)** —
`generate_distilled_two_stage` doesn't accept a negative encoding at all;
the prompt-encoding path was still tokenizing + running Gemma 3 + running
the AV text encoder twice and then passing `null_encoding` to a function
signature that never read it.  Fix: encode only `[prompt]` not
`[prompt, neg_prompt]`.  Impact: prompt encoding 12.20s → 8.25s, total
41.61s → 36.21s (-10 %).  Re-enable: `LTX_ENCODE_UNUSED_NEGATIVE=1`.

**Tokenize without `padding="max_length"`** — for a 2-token real prompt,
the tokenizer was padding to 1024 before the Gemma forward.  Trim happened
after the O(N²) attention.  Fix: `padding=False`.  Impact: prompt
encoding 8.25s → 6.24s, total 36.21s → 34.30s (-5 %).  Re-enable:
`LTX_PAD_PROMPT_TO_MAX=1`.

Cumulative on the small-T AV bench: **46.85s → 34.30s (-26.8 %)**.

#### Bakery (1024×576×481)

Re-run with cumulative defaults: **28m 24s total**.  Reference mlx-video
bakery: 28m 20s.  At parity at bakery scale.

#### Monolithic-inlined transformer experiment (negative result)

After ruling out cheap structural changes
(`LTX_DISABLE_BLOCK_OVERHEAD`, `LTX_VELOCITY_MODE`,
`LTX_COMPILE_BLOCK_GROUPS` all neutral), the remaining steelman was:
collapse the entire 48-block forward + preprocess + output projection into
one inlined function with flat pretransposed weights, no `nn.Module`
dispatch.

Implemented in `scripts/mono_pipeline.py` (`InlinedAVModel` +
`transformer_step`).  `stage2_harness.py` swaps the pipeline's transformer
when `LTX_MONO_INLINED=1` is set.

Result on bakery 704×384 (3-step stage-2):

- **Math is equivalent.**  Source modular `stage_2_video_latent` vs
  inlined `final_video_latent`: cosine similarity 0.99922 video, 0.99977
  audio.  Visually indistinguishable; classic BF16 rounding-order drift.
- **Wall clock is neutral.**  Inlined 6m04s vs modular ~6m at this shape.

Confirms the abstraction-cost hypothesis is empirically wrong on this MLX
version: the lazy graph optimizes through the `nn.Module` dispatch chain
just as well as flat-function code.  Per-step gap (when it exists) is
structural at the Metal level — not anything to chase by restructuring
Python.

Code is left in place as a same-math reference implementation.  Doubles as
a worked example of "what the V2.3 distilled AV block actually computes."

Bug-hunt notes if anyone resurrects the experiment:

- `Attention.q_norm` and `k_norm` are `RMSNorm` modules with **learned
  weights**, not just `mx.fast.rms_norm` with `weight=None`.
- `_prepare_timestep` multiplies the timestep by
  `timestep_scale_multiplier` (= 1000 for V2.3) for **every** AdaLN call,
  including `prompt_adaln`.
- The latent that reaches the transformer is already patchified
  ((B, N, 128) by `create_initial_state`), not the 5D shape.
- Cross-modal AdaLN is conditioned on the *other* modality's sigma.

---

## Transformer Profiling

Use when the live progress line says denoise is slow and you need to know
where one or more transformer calls are spending time:

```bash
--profile-transformer-steps 1,2,8
```

`--profile-transformer-once` is a convenience alias for step 1.  Prefer
`--profile-transformer-steps` when comparing cold first step vs warmer
later steps.

Each selected step prints two diagnostic sections:

- `Transformer profile`: internal model timing split into preprocessing,
  transformer block groups, and output projection.
- `AV denoise step N profile`: outer pipeline timing split into modality
  setup, transformer call, guidance/postprocess, scheduler step, and final
  state eval.

For a deeper split inside selected blocks:

```bash
--profile-transformer-blocks 0,40,47
```

Block numbers 0-based.  The detailed block profile splits a block into
video/audio self-attention, text attention, audio-video cross attention,
and feed-forward sections.  Attention sections split further into
setup/AdaLN, Q/K/V projections, Q/K norm, RoPE, SDPA, gate, output
projection, and residual checkpoints.  FF sections split into AdaLN,
`project_in`, GELU, `project_out`, and residual-gate checkpoints.

The profile inserts `mx.eval(...)` checkpoints so each section has a real
wall-clock boundary.  That means it perturbs the exact timing it measures.
Use it to find which region is worth optimizing, then rerun without the
flag for clean benchmark numbers.

### Trace capture recipes

For Metal System Trace captures with per-phase signpost attribution, use
the `xctrace record` CLI (do NOT use the Instruments GUI — it drops
signposts under buffer pressure).

```bash
LTX_PROFILE_PAUSE_BEFORE_DENOISE=1 \
LTX_PROFILE_STOP_AFTER_STEPS=2 \
LTX_PROFILE_SIGNPOSTS=1 \
LTX_PROFILE_SIGNPOSTS_SYNC=1 \
LTX_PROFILE_SIGNPOST_LOG="${TMPDIR:-/tmp}/ltx_signposts.log" \
caffeinate -di python scripts/generate.py "$BAKERY" \
  --pipeline distilled --height 576 --width 1024 --duration 20 --seed 124 \
  --generate-audio --fast-mode --save-all-sidecars \
  --output-prefix bakery_trace
```

When you see `[LTX_PROFILE_PAUSE_BEFORE_DENOISE] pid=<N>`, in a second
terminal:

```bash
xcrun xctrace record \
  --template "Metal System Trace" \
  --instrument "Points of Interest" \
  --attach <PID> \
  --output "${TMPDIR:-/tmp}/my.trace" \
  --no-prompt
```

Then hit Enter in the original terminal.  Recording stops when the process
exits via `LTX_PROFILE_STOP_AFTER_STEPS`.

After capture, export signposts + GPU intervals:

```bash
xcrun xctrace export --input my.trace \
  --xpath '//trace-toc/run/data/table[@schema="os-signpost"]' \
  --output signposts.xml
xcrun xctrace export --input my.trace \
  --xpath '//trace-toc/run/data/table[@schema="metal-gpu-intervals"]' \
  --output gpu_intervals.xml
```

For per-(parent, sub-phase) attribution of the sidecar log:

```bash
python scripts/analyze_signpost_subphases.py "${TMPDIR:-/tmp}/ltx_signposts.log"
```

If the trace's `os-signpost` table is short on events (drops under buffer
pressure), fall back to the sidecar log — it captures every event
regardless of OS buffer state.

### A/B against mlx-video

```bash
# Required env (set in your shell rc):
export LTX_REPO=/path/to/LTX-2-MLX
export MLXV_REPO=/path/to/mlx-video
export MLXV_MODEL_REPO=/path/to/local-mlx-video-format-model-dir
# Optional:
export LTX_VENV_BIN=/path/to/ltx-venv/bin/activate
export MLXV_VENV_BIN=/path/to/mlxv-venv/bin/activate

bash scripts/bench_ab_wall_time.sh           # 4 steps each (~10 min total)
STEPS=8 bash scripts/bench_ab_wall_time.sh   # full stage-1 (~16 min)
```

### Per-call SDPA probe

```bash
LTX_PROBE_TIME_SDPA=1 \
LTX_PROBE_TIME_LOG="${TMPDIR:-/tmp}/sdpa_per_call.jsonl" \
LTX_PROFILE_STOP_AFTER_STEPS=2 \
LTX_DISABLE_COMPILED_ATTN=1 LTX_DISABLE_COMPILED_HELPERS=1 \
python scripts/sdpa_dtype_probe.py "$BAKERY" \
  --pipeline one-stage --height 288 --width 512 --duration 20 --seed 124 \
  --generate-audio --fast-mode --output-prefix probe
```

For the mlx-video side, set `LTX_PROBE_MODULE=mlx_video.models.ltx_2.generate`
and run from the mlx-video repo root.

---

## MLX Runtime Notes

Reference material from a local MLX checkout (e.g. `mlx-main`).

### Lazy evaluation

MLX records a computation graph and runs it when `mx.eval(...)`, printing,
NumPy conversion, memory access, or saving forces evaluation.  Tradeoffs:

- Too many small `mx.eval(...)` calls pay fixed scheduling overhead.
- Letting a graph grow extremely large can also become costly.
- A natural efficient boundary is the outer iteration of an iterative
  computation.

For LTX denoising: one eval at the end of each denoise step is a reasonable
target.  Extra evals inside the 48-layer transformer are a memory tradeoff,
not a free speed feature.

### Compilation

`mx.compile` compiles and caches computation graphs.  Sharp edges:

- First call pays compile overhead.
- Shape, dtype, or input-count changes trigger recompilation.
- Avoid creating compiled lambdas/functions inside loops.
- Compiled functions should be pure: no printing, no side effects, no hidden
  mutable state unless captured through `inputs=` / `outputs=`.
- Compile the outer useful function when possible, but only when stable
  enough to benefit from cache reuse.
- `shapeless=True` avoids shape-based recompilation but can be wrong if the
  function uses static shape values in reshape/control logic.

For LTX: narrow repeated helper compilation first.  Whole-transformer
compile is possible but only behind an opt-in benchmark flag.

### Fast kernels

MLX exposes:

- `mx.fast.scaled_dot_product_attention`
- `mx.fast.rms_norm`
- `mx.fast.layer_norm`
- `mx.fast.rope`
- `mx.fast.metal_kernel`
- `mx.fast.cuda_kernel`

Current code uses fast SDPA and RMSNorm.  `mx.fast.rope` exists and
supports `traditional`, `base`, `scale`, `offset`, and 1D `freqs` in the
local tests, but it is not obviously a drop-in replacement for LTX-2.3's
3D SPLIT RoPE path with per-token composed cos/sin tensors.  Research item.

For `mx.fast.metal_kernel`: build the kernel once and reuse it.  Defaults
to `ensure_row_contiguous=True`, which can insert copies for non-contiguous
inputs.  Any custom kernel must account for layout to avoid accidental
copies.

### SDPA kernel tile floor on M1 Max

**2026-05-28 correction:** the earlier D-sweep was a useful warning
but not a final answer.  Directly retile-testing the stock STEEL body
showed that `BQ=64, BK=32` is faster at the actual LTX shapes on M1
Max.  The first production branch used this as the fixed local path for
supported D64 shapes; unsupported shapes fall back to stock MLX SDPA.  Later
updates below keep `BK=32` for D64 self-attention but adapt cross-modal D64 by
direction:

**2026-06-06 update:** D128 moved to the faster source-packaged
`BQ=80, BK=40, q8k2v8` reducer+scalefold path.  D64 now uses the quiet
KinoMLX adaptive winner set: self-attention keeps `BQ=64, BK=32`,
audio-to-video uses `BQ=64, BK=24, q8k2` reducer+scalefold, and
video-to-audio uses `BQ=64, BK=32, q8k4`.  The newer D128 path is not
latent-bit-exact against the older lean path, but visual MP4 comparison on
the saved 576x320x721 kitten stage-2 run was clean (PSNR 47.64 dB, SSIM
0.99449).  Stock-vs-lean already showed small BF16/tile-order drift, so
treat local STEEL as a visually equivalent performance path, not a
latent-exact replacement for stock MLX SDPA.

```bash
LTX_STEEL_ATTN_PROBE=1 \
PYTHONDONTWRITEBYTECODE=1 \
python scripts/generate.py "$PROMPT" \
  --pipeline distilled \
  --height 320 --width 576 --duration 30 --seed 42 \
  --generate-audio --fast-mode --save-all-sidecars
```

Measured on the standard smoke prompt:

| Mode | Total | Stage 1 denoise | Stage 2 denoise | Denoise total | Probe |
|---|---:|---:|---:|---:|---|
| stock MLX same patch | 562.6s | 155.1s | 310.2s | 465.4s | n/a |
| STEEL retile D128 only | 546.5s | 152.7s | 292.0s | 444.6s | `hit_d128=2`, fallback=10 before probe split |
| STEEL retile D64 default | 535.7s | 151.5s | 286.9s | 438.4s | `hit_d128=4`, `hit_d64=6`, fallback=2 masked text |

D64 self/cross-modal shapes were neutral in the first isolation pass, but
won in the full integrated run and then improved in the D64 variant sweep.
D64 is therefore default-on with adaptive selection; use
`LTX_STEEL_ATTN_DISABLE_D64=1`, `LTX_STEEL_ATTN_D64_BK32=1`, or
`LTX_STEEL_ATTN_D64_Q8K4=1` only for quick bisects.  Masked text
cross-attention remains on stock MLX.

`mx.fast.scaled_dot_product_attention` dispatches to one of two Metal
kernel paths based on sequence length `T`:

- `sdpa_full` — used when `T > 8`; covers all LTX latent grids.
- `sdpa_vector_2pass` — used when `T ≤ 8`; not relevant to LTX.

SDPA dispatch logic at `mlx-main/mlx/backend/metal/scaled_dot_product_attention.cpp:177`:

```cpp
if (metal::is_nax_available() && q.shape(3) != 80 &&
    (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(...);  // NAX path: bq=64, bk=32
}
// else fall through to the regular path:
int bq = 32;
int bk = bd < 128 ? 32 : 16;
```

So there are two SDPA tile-size regimes:

| Path | Tile sizes | Kernel | Required hardware |
|---|---|---|---|
| **NAX** | `bq=64, bk=32` | `sdpa_full_self_attention_nax` — uses `mpp::tensor_ops::matmul2d` (Apple Metal Performance Primitives, dedicated tensor-multiply hardware) | M5+, macOS 26.2+, head_dim ≠ 80, not pure float32 (unless TF32 enabled) |
| **Non-NAX** | `bq=32, bk=16` for `head_dim=128` (and `bk=32` for smaller head_dim) | regular `steel_attention_*` — uses `metal::simdgroup_matrix<T>` (general SIMD ops, no dedicated matrix hardware) | M1+ |

`is_nax_available()` at `mlx-main/mlx/backend/metal/device.cpp:828`
requires:

- **macOS 26.2+** (where the `mpp::tensor_ops` headers exist), AND
- **GPU generation 17+** (gen >= 17 for base/Pro variants, >= 18
  for `p` variants).  GPU-gen mapping per
  `device.cpp:489-497`: M1=13, M2=14, M3=15, M4=16, **M5=17**.

**M1 Max is non-NAX** (gen 13) and uses the `bq=32, bk=16` path for
our `head_dim=128` shape.  On M5+ the same workload would use the
`bq=64, bk=32` tensor-core path — both bigger tiles AND dedicated
matrix hardware, a step-function speedup.

**Important earlier-doc correction:** previous versions of this
section said "M3+ (NAX)" / "M3+ would double tile dimensions".  That
was wrong inference — NAX requires **M5+**, not M3+.  M2/M3/M4 get
modest improvements from broader architectural gains (more cores,
more bandwidth, better cache) but do NOT unlock the NAX kernel path.

No env var or runtime flag changes tile sizes for `sdpa_full` in MLX
0.31.2.  MLX PR #3455 adds `MLX_SDPA_BLOCKS` but it only controls
`sdpa_vector_2pass` (T ≤ 8) — irrelevant to LTX.

At 1024×576×481, video token count is `18 × 32 × 61 = 35,136` before
audio tokens.  At 512×288 the same model runs at ~55s/it.  At 1024×576 the
4× token count produces ~425s/it stage-2 — attention scaling dominates and
there is no software knob to recover the 4× token overhead.

**Implications:**

- On M1 Max, stock `sdpa_full` is a strong baseline but not the final
  tile floor for these LTX shapes; local STEEL wrappers beat the
  packaged `BQ=32, BK=16` kernel in full 8+3 smoke tests.  Current D64
  defaults adapt between `BK=32`, `BK=24/q8k2` reducer+scalefold, and
  `BK=32/q8k4`; D128 defaults to `BQ=80, BK=40, q8k2v8`
  reducer+scalefold.
- Upgrading to **M5+** (the actual NAX-capable family, NOT M3) would
  unlock the `mpp::tensor_ops::matmul2d` hardware path entirely —
  dedicated tensor-multiply units replacing the SIMD-group matrix
  shim that M1-M4 use.  M2/M3/M4 buy modest improvements from
  broader architectural gains (more cores, more bandwidth) but do
  NOT unlock NAX.
- A custom Metal kernel via `mx.fast.metal_kernel` can implement larger
  tiles without rebuilding MLX.  The current local wrapper does exactly
  that with lean MLX-derived Metal resources loaded by the Python launcher.
- Python-level chunked or tiled SDPA does not change the Metal tile width;
  prior experiments showed no runtime win at 1024×576.

**Negative result: SDPA chunking from MLX issue #3302 / PR #3307.**  The
closed PR adds `MLX_SDPA_CHUNK_THRESHOLD` / `MLX_SDPA_CHUNK_SIZE` with a
default `65536` key threshold — which would not trigger for the
1024×576 LTX latent grid (35,136 tokens) anyway.  Local SDPA-query and
FF-token chunking experiments matched baseline math on tiny tests but
showed no useful runtime win at our scale; they added hot-path branches
without earning their keep and were removed.  Don't reintroduce chunking
as a perf experiment unless a future MLX release ships it as a default
kernel-level change.

### MLX 0.31.2 primitive scan

The local MLX API and installed 0.31.2 bindings expose:

- `nn.gelu_fast_approx` / `nn.GELU(approx="fast")` — low-friction
  non-parity activation experiment for FF.  Current code uses tanh/precise
  GELU.  Fast GELU uses `x * sigmoid(1.702 * x)`.
- `mx.qqmm` — activation + weight quantization.  Supports `nvfp4` and
  `mxfp8`, but current Metal runtime fails LTX FFN-shaped tests with
  `RuntimeError: [QQMatmul] NYI for the general case`.  Keep on the list
  for future MLX releases.
- `mx.quantize`, `mx.quantized_matmul`, `nn.QuantizedLinear`,
  `nn.quantize(...)` — weight-only quantization, runs on current Metal
  runtime.
- `mx.block_masked_mm` — only useful for structured sparsity.
- `mx.gather_mm` / `mx.gather_qmm` — MoE/routing, not dense LTX.
- `mx.segmented_mm` — per-segment matrix products.  Diagnostic / decomposed
  matmul experiments.
- `mx.hadamard_transform` — Walsh-Hadamard transform.  Quantization/rotation
  experiments.
- `mx.depends` — graph dependency control for async/stream experiments.
- `mx.to_fp8` / `mx.from_fp8` — e4m3 conversion for FP8 research.

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

MLX exposes active, peak, and cache memory counters, plus cache/memory
limit controls.  Local Metal debugger docs show `mx.metal.start_capture(...)`
and `mx.metal.stop_capture(...)` for GPU traces when running with
`MTL_CAPTURE_ENABLED=1`.

Use lightweight timing first, then add Metal capture only for small repros.
Capturing a full long generation is usually too noisy.

`MLX_METAL_FAST_SYNCH=1` enables a different CPU/GPU synchronization path.
Tested locally: no win on small AV smoke; slightly slower in one A/B.

### Conversion, indexing, and loops

- NumPy conversion evaluates the graph.  BF16 arrays require explicit cast
  to FP32 or FP16 first.  Keep NumPy stats out of hot denoise timing.
- MLX slicing creates a copy, not a view.
- `vmap` removes Python loop overhead for batched array work.  Useful in
  probe/analysis scripts, not on the main transformer forward (already
  vectorized).
- MLX export/import can serialize graph traces — useful for graph
  inspection or C++ reuse.

---

## Decode-time notes (not denoise speed)

VAE tiled decode, VAE spatial padding, and output encoding affect decode
quality, memory, or save time.  They are useful but **not denoise-speed
fixes**.  See the matrix entry for `--vae-decoder native`.

Native Conv3d VAE decode is the only supported generator decode path.
The historical `--vae-decoder legacy` (`SimpleVideoDecoder`) backend
and its companion `scripts/compare_vae_decoders.py` A/B script were
archived 2026-05-23 — see `archive/simple_decoder.py.bak`
for the legacy source.

### Native Conv3d output-size limit on MLX 0.31.2

Direct 1024×576 full-volume native Conv3d decode hits a tail failure: the
final spatial upsample's `07_upsample_conv` output is `481×72×128×512`;
one frame is `4,718,592` elements, and `2^31 / 4,718,592 = 455.11`.  The
last fully good frame is `454`; frame `455` is a transition; frame `456+`
collapses (white tail).  MLX PR `ml-explore/mlx#3524` appears to fix this
implicit Conv3d pointer-offset overflow; once a local MLX release includes
that PR, retest 1024×576 native Conv3d with `--vae-tiling off`.

Temporal-only multi-tile native Conv3d decode avoids the tail at 1024×576:

| Tile / overlap | Time   | Peak    | Tail  |
| -------------- | ------ | ------- | ----- |
| 32/8           | 175.1s |  7.1 GB | clean |
| 40/8           | 159.1s |  7.5 GB | clean |
| 64/8           | 149.9s |  9.7 GB | clean |
| 128/8          | 126.5s | 14.9 GB | clean |
| 256/8          | 125.1s | 25.8 GB | clean |

The default `--vae-tiling auto` for native Conv3d uses a small RAM-derived
planner that prefers the fastest temporal-only tile under both the MLX
`2^31` Conv3d output boundary and an estimated decode budget.  On a 64 GB
machine, direct 1024×576×481 auto-selects `128/8`.

Custom controls for middle-ground tests:

```bash
--vae-decoder native \
--vae-tiling custom \
--vae-temporal-tile-frames 128 \
--vae-temporal-overlap-frames 8
```

---

## Optimization History (archived)

The detailed prose for each numbered candidate is preserved here for
reference.  Current status of every knob is in the [Benchmark
Matrix](#benchmark-matrix) above; only read this section if you need the
original experiment context (specific resolution + memory measurements,
why a particular path was abandoned, recipe details).

### Pretranspose layouts

- **`--video-ff-layout`** — `nn.Linear` computes `x @ weight.T`.  Caches
  contiguous `weight.T` after loading stock BF16 weights and calls
  `mx.addmm` against the cached layout.  Same-math.  Bakery 1024×576
  smoke: 77 → 55 s/it.  Original duplicate-cache implementation was
  slower and more memory hungry; current path materializes each
  transposed weight layer by layer and drops the original.  **Default
  as of 2026-05-17 is `project_out:pretranspose` only** — the
  `bf16_layout` microbench showed that's the single matmul shape
  (K=16384, N=4096) where pretranspose rescues a 5.17 → 7.95 TFlops/s
  kernel-selection cliff.  `project_in` was in the pre-2026-05-17
  default but measures as +2.5 % in isolation and neutral
  end-to-end.  Opt back in via
  `--video-ff-layout project_in:pretranspose,project_out:pretranspose`.
- **`--video-attn-layout`** — same idea applied to attention output AND
  Q/K/V projections.  For AV blocks: video self-attention, video
  text-attention, audio-to-video attention; skips audio-only output
  projections (those are covered by audio pretranspose).  **Default as
  of 2026-05-17 is OFF (empty)** — the `bf16_layout` microbench
  showed all four projections at the 4096×4096 attention shape are
  tied with naive BF16 within ±1 % noise.  End-to-end "marginal
  positive" observation in earlier sessions was likely measurement
  noise.  Opt back in per-target via
  `--video-attn-layout to_q:pretranspose,to_k:pretranspose,to_v:pretranspose,to_out:pretranspose`.

### Block streaming (constrained-memory mode)

`--stream-transformer` is the user-facing preset; expands to
`--transformer-block-resident-blocks 16 --transformer-block-compile
--transformer-block-compile-group-size 4`.

Keeps a small resident pool of transformer blocks and rotates cached block
weights through it.  Preserves model math.  Repeatedly rebinds 48 block
weight sets per denoise step, so latency is worse than keeping the full
transformer resident.

Practical advantage: SSD write avoidance under constrained memory.  After
the one-time weights-cache build, streaming reuses read-only safetensors
pages and can lean on the macOS file cache.

Measured bakery AV results with `project_out:pretranspose`,
`to_out:pretranspose`, `--mlx-cache-limit-gb 1`, 512×288, 20s, 8 steps,
seed 124:

| Variant                                  | Process RAM | Per-step  | Total      |
| ---------------------------------------- | ----------- | --------- | ---------- |
| `r4`                                     | ~8 GB       | 70.5 s/it | 10m 28s    |
| `r4 + --transformer-block-compile`       | ~8 GB       | 67.6 s/it | 10m 10s    |
| `r8` (no `--low-memory`)                 | ~14 GB      | 61.2 s/it | 9m 08s     |
| `r16` (quiet system)                     | ~16 GB      | 55 s/it   | 8m 57s     |
| `r16 + compile + group-4` @ 1024×576     | ~?          | 424.8 s/it| 62m 22s    |

1024×576 with r16 resident blocks and `--transformer-block-compile-group-size 4`
completed without watchdog abort but remained visibly laggy.  Direct 1024
is a premium walk-away mode.  See
[MLX #3267](https://github.com/ml-explore/mlx/issues/3267) for the
active-display contention pattern.

### Quantization opt-ins

**`--video-ff-quantize`** layer-range results on 352×192×15s AV smoke:

| Quantized layers | Per-step  | Visual read                          |
| ---------------- | --------- | ------------------------------------ |
| none             | 24.4 s/it | reference                            |
| 32-47            | 22.1 s/it | slightly different; useful speedup   |
| 24-47            | 21.9 s/it | visibly different; all-layer level   |
| 0-23             | 22.1 s/it | visibly different; all-layer level   |
| 0-47             | 17.6 s/it | visibly different; not parity        |

Bakery 512×288 all-layer comparison:

| Mode             | Per-step  | Visual                                       |
| ---------------- | --------- | -------------------------------------------- |
| none             | 77.8 s/it | reference                                    |
| 0-47 `mxfp8`     | 56.3 s/it | different faces; possibly less clear; usable |
| 0-47 `mxfp4`     | 58.8 s/it | mechanics more confused; laggier             |
| 0-47 `nvfp4`     | 59.4 s/it | similar to mxfp8 but no speed win            |
| `project_in:mxfp8` | 83.9 s/it | slower than BF16, identity degraded        |

Conservative candidate: `project_out:mxfp8 layers 32-47`.  Fast/draft mode:
`project_out:mxfp8 layers 0-47`.  `mxfp4` not currently attractive.
`project_in:mxfp8` is a dead end.

### Decode-time options

See [Decode-time notes](#decode-time-notes-not-denoise-speed) above.  Brief
results on bakery 512×288×481:

| Tiling mode             | Simple decoder | Native Conv3d |
| ----------------------- | -------------- | ------------- |
| Auto temporal tiling    | 38.1s / 5.9 GB | 61.3s / 3.0 GB |
| No tiling               | 23.5-26.2s / 32.1 GB | 29.4-36.4s / 10.4 GB |

End-to-end bakery AV smoke with warm caches, `r16` resident-group compile,
FF pretranspose, attn pretranspose, `--vae-decoder native`,
`--vae-tiling off`: denoise RUN 7m08s avg 53.4 s/it; total 8m07s.  Native
Conv3d at 512×288 with no tiling is the fastest decode mode measured.

### Terminal redraw throttling

The old `DenoiseProgress` class ran a daemon thread that called `_render`
every `0.12s` (~8 Hz).  Each render printed a ~120-char line with eight
ANSI color escapes and a `\033[2K` full-line clear, then `flush=True`.
Terminal.app and WindowServer are GPU-accelerated on macOS — they were
contending with MLX for GPU.  Activity Monitor on bakery 1024×576
distilled showed Terminal ~15 % GPU and WindowServer ~15 % during stage 2;
both dropped to zero when the denoise loop ended and VAE decode started.

Changes:

- `DenoiseProgress` no longer spawns a heartbeat thread.  Repaints only on
  actual step boundaries via `update()`, plus once at `start()` and
  `finish()`.  Spinner glyph, ANSI colors, `\033[2K` clear gone.  Repaints
  `\r`-overwrite and pad to previous line length, so Terminal only
  re-rasterizes changed characters.  Byte-equality cache on the rendered
  line skips the `write(2)+flush` entirely when nothing changed.
- All `tqdm(...)` call sites pass `ascii=True` (plain `#` bar instead of
  unicode block chars) plus `mininterval=2.0` for hot paths during MLX GPU
  work and `mininterval=1.0` for cold paths.

Measured bakery 1024×576×481 distilled AV (same prompt/seed/weights/flags;
only diff is redraw policy):

| Phase                     | Before    | After     | Δ         |
| ------------------------- | --------- | --------- | --------- |
| Stage 1 denoise (8 steps) | 7m 04.2s  | 7m 00s    | -4.2s     |
| Stage 2 denoise (3 steps) | 21m 10.7s | 19m 29s   | -101.7s   |
| VAE decode (4 tiles)      | 2m 19.1s  | 2m 15s    | -4.1s     |
| **Total**                 | 31m 28.9s | 29m 38s   | -1m 50.9s |

Stage 2 went from ~423.6 s/it to ~389.5 s/it.  Win scales with step
duration: longer steps have more per-step redraws.

On non-macOS or non-GPU-accelerated terminals, expect a much smaller gain —
most of the cost was Terminal.app's Metal repaint, not syscall overhead.

### Experiments tried and found neutral (kept env-toggleable)

- `LTX_VELOCITY_MODE=1` — inline velocity-form Euler update.  Same math.
  Neutral at small T and bakery.
- `LTX_COMPILE_BLOCK_GROUPS=N` — eager-path block-group `mx.compile`.
  Neutral at all tested scales (compile cost paid up front; per-step
  doesn't recover it).  See 2026-05-16 update: the underlying `mx.compile`
  wrappers ARE worth ~3 %, just not this experimental block-group form.
- `LTX_ADALN_PRETRANSPOSE=1` — slight regression at small T (8 calls/step
  too few to amortize per-tensor dispatch overhead).
- `LTX_ROPE_PRECOMPUTE=1` — neutral (MLX lazy graph already dedupes
  per-step RoPE calls).
- `to_gate_logits` pretranspose — slight regression (weight too small for
  implicit transpose to matter).
- Per-run RoPE precompute (one-entry cache patch) — removed after slightly
  slower 352×192 smoke.

### Experiments tried and removed

- `LTX_DISABLE_BLOCK_OVERHEAD=1` — stripped-down `_fast_call` in
  `BasicAVTransformerBlock` (removed `mark_profile` closure, perturbation
  checks, asserts, `_cross_attn_scale` getattr, `TransformerArgs.replace()`
  kwargs.get loop).  Measured zero change.  Python overhead between blocks
  isn't the bottleneck.
- Packed attention layouts (`self_qkv:pack`, `kv:pack`) — reached
  ~53 s/it on quiet r16 path, matching `to_out:pretranspose` alone.
  Removed for not justifying added CLI/runtime surface.
- Forcing a cache clear after each full block sweep — made first step
  worse (~71 s).  Don't clear MLX cache inside denoise loop.
- Stage-2 SVD/residual cache — all probes hit foreground noise + visible
  MAE (0.18-0.46).  Removed wholesale.

### Profile tooling notes

`scripts/profile-watch.sh` (py-spy auto-attach watcher) was removed.
`scripts/profile.sh` keeps only the macOS `sample` backend; py-spy
removed because SIP gates `task_for_pid` on Darwin.  `sample` is
sufficient for "where is the time going" questions; use Instruments for
deeper Metal traces.

`scripts/bench-process-watch.sh` pkill's `mediaanalysisd`,
`mediaanalysisd-access`, and `photoanalysisd` every N seconds.  Those
macOS background agents periodically hammer the GPU with image-similarity /
OCR work — visible in Metal System Trace as random WindowServer-adjacent
compute dispatches that contend with denoise steps.  Use during any perf
measurement or trace capture.  Replaces earlier `bench-quiet.sh`.

---

## Ideas To Avoid As First Moves

### KV caching text cross-attention

Not obviously valid for the current AV transformer.  Text cross-attention is
conditioned through per-step/per-block modulation, so K/V reuse is not a
simple drop-in cache.  The MLX Llama example uses KV cache for
autoregressive token generation; that doesn't transfer directly to diffusion
denoising where the latent state and timestep conditioning change every step.

### Scheduler rewrites

Scheduler math is tiny compared with the 48-layer AV transformer forward.
Keep it correct and readable unless profiling proves otherwise.

### Quantization as a default speed fix

Quantization may reduce memory and can help some workloads, but it changes
precision behavior and may require checkpoint conversion or quality
validation.  Track it separately from "same checkpoint, same pipeline,
faster MLX runtime."  See `--video-ff-quantize` and
`--transformer-cache-quantize` matrix entries for opt-in modes.

### Compile lambdas inside hot loops

`mx.compile` of a lambda created inside the denoise loop produces a fresh
graph every iteration.  Always compile the outer stable function once.

### Restructure Python to remove `nn.Module` dispatch

The 2026-05-15 monolithic-inlined experiment showed this moves per-step
time by ~0 %.  MLX's lazy graph optimizes through the module chain just
as well as flat-function code.
