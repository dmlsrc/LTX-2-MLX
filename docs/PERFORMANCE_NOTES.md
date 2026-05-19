# Performance Notes (scratchpad)

In-progress thinking, open investigations, half-baked ideas, research summaries,
and failed experiments worth remembering.  This is the working notebook;
`PERFORMANCE.md` is the canonical "what's true right now" document.

## Workflow rules

1. **Entries are dated.**  Top of each entry: `## YYYY-MM-DD: <topic>`.  Older
   entries naturally sink down.
2. **Status markers** make it clear what's live vs settled vs dead:
   - `[OPEN]` — actively investigating
   - `[BLOCKED]` — waiting on something (decision, hardware, MLX version)
   - `[PROMOTED]` — findings moved to `PERFORMANCE.md`; entry kept as a stub
     pointing there
   - `[ABANDONED]` — explored and not pursuing; kept as negative-result note
3. **Promotion direction is one-way and explicit.**  When something becomes a
   default or a confirmed result, it moves to `PERFORMANCE.md` and the
   scratchpad entry is reduced to a one-line stub with the link.
4. **No duplication.**  If it's already in `PERFORMANCE.md`, the scratchpad
   doesn't repeat it — just refers.
5. **Periodic prune.**  `[ABANDONED]` entries get archived to the Archive
   section at the bottom or deleted entirely.  `[PROMOTED]` entries get
   stubbed.

---

## Open

_(empty — investigation converged 2026-05-18; no actionable hypotheses
remain.  See Promoted section below for what landed in
`PERFORMANCE.md`, and Archive section for what was tested and dropped.)_

---

## Promoted

(Entries moved to `PERFORMANCE.md` — kept here as one-line stubs for
cross-reference.)

### 2026-05-17: Microbench results for FF / attention candidate optimizations `[PROMOTED 2026-05-18]`

Findings from `scripts/bench_ff_microbench.py` (Q+K+V refutation,
`mx.compile`+FF refutation, SDPA T-padding refutation, GELU cost
bound, FF matmul-bound confirmation, SDPA tile-floor confirmation)
landed in `PERFORMANCE.md` "Active brutal-efficiency targets" -- now
shows the converged "investigation closed" framing with each
previous candidate's status + microbench evidence.  Full original
detail in this scratchpad's git history (commits 2026-05-17/18).

### 2026-05-17: Pretranspose default cleanup -- project_out is the only real win `[PROMOTED]`

Defaults trimmed in `scripts/generate.py`:
- `DEFAULT_VIDEO_FF_LAYOUT_SPECS = (("project_out", "pretranspose"),)`
  (dropped `project_in:pretranspose`)
- `DEFAULT_VIDEO_ATTN_LAYOUT_SPECS = ()` (dropped all attention layouts)

Microbench evidence: `bench_ff_microbench.py bf16_layout` showed
project_out is a 35 % win (rescues kernel-selection cliff:
5.17 → 7.95 TFlops/s) but project_in is +2.5 % regression and all
four attention projections are tied within ±1 %.  Kitten smoke
verified safe at bakery scale: 75.4 s/it vs 75.9 s/it (-0.7 %, no
regression).  Documented in `PERFORMANCE.md` matrix rows for
`--video-ff-layout` and `--video-attn-layout`.

### 2026-05-17: FlashAttention-2 custom Metal kernel for attn_sdpa `[ABANDONED 2026-05-18]`

D-sweep microbench refuted the tile-size hypothesis.  Full detail in
Archive entry "2026-05-18: FA-2 custom kernel -- D-sweep refutes
tile-size hypothesis".  Status reflected in `PERFORMANCE.md` "Active
brutal-efficiency targets" table.

---

## Archive

(Abandoned investigations kept for negative-result evidence.)

### 2026-05-18: Consolidated quiet-machine microbench sweep -- RoPE has the only marginal remaining lever `[ABANDONED]`

**Bottom line:** ran all five new microbenches
(`fused_ffn_feasibility`, `pointwise_bw`, `adaln_residual`, `rope`, `vae_ops`)
on a quiet machine at ITERS=50 WARMUP=5.  Five independent
calibration + lever-search results.  Only one surfaced any remaining
lever (RoPE precision), and it tops out below 1 % of step time.  No
further microbench candidates remain that haven't been measured.

**Log:** `$SHARED_TEMP_DIR/trace_analysis/ff_microbench_new_20260518_221142.log`
**Code:** `scripts/bench_ff_microbench.py {fused,pointwise_bw,adaln,rope,vae}`

**Pointwise BF16 bandwidth ceiling on this hardware** (calibration):
- Peak achieved: **340.6 GB/s** at 512 MB `x+x` (~85 % of M1 Max ~400 GB/s nominal HBM).
- LTX residual (72 MB):    219.7 `x+x` / 247.0 `x+y` GB/s (65-73 % of peak).
- LTX FF hidden (288 MB):  305.6 `x+x` / 290.7 `x+y` GB/s (85-92 % of peak).
- Small tensors (1-16 MB): dispatch-bound at 8-150 GB/s — informational.
- **Use 340 GB/s as the divisor when grading any bandwidth-bound op below.**

**Fused FFN feasibility re-confirmed** (tighter numbers vs the earlier
exploratory ITERS=10 run):
- Stock FF: 311.6 ms at **7.57 TFlops/s = 95 % of `steel_gemm` ceiling**.
- project_out alone: 7.96 TFlops/s = **100 % of GEMM ceiling**.
- Recoverable: +10.9 ms (3.5 % conservative) to +21.1 ms (6.8 % optimistic) per call.
- Per-step ceiling: **+521 ms to +1011 ms = +1.15 % to +2.22 % of 45.5 s/step**.
- Same verdict: NOT WORTH IT.  See "Custom fused / streamed BF16 FFN
  kernel" entry below for the full reasoning.

**AdaLN + residual chain** (defends the production default):
- Compiled chain (production): 1.52 ms at **285 GB/s = 84 % of pointwise peak** -- already near ceiling.
- Inline chain (no compile):    8.29 ms at  52 GB/s.  `mx.compile` gives a 5.5x lift.
- Per-run cost of compiled chain: 0.58 s = 0.64 % of step.
- If inline ran instead, would lose **1.3 s/step = 2.86 % of step**.
- **KEEP `mx.compile` wrappers on `_adaln_inline` / `_residual_gate_inline`**;
  defends `transformer.py:57-58` defaults with a number.
- Custom AdaLN+residual Metal kernel headroom = gap from 285 to 340 GB/s
  = ~16 % of chain time = ~0.10 % of step.  **NOT WORTH IT.**

**RoPE -- the only remaining lever** (production:
`LTX_2_MLX.model.transformer.rope.apply_split_rotary_emb`):
- Production (FP32 cos/sin + cast back to BF16): 4.28 ms at **67.2 GB/s
  = 20 % of pointwise peak**.  By far the worst-efficiency op benched.
- All-BF16 freqs (lower precision):              1.66 ms at 130 GB/s = 38 % of peak.
  Saves **2.62 ms (61 % of RoPE time)** per call.
- Bare BF16 multiply at half-shape:              0.57 ms at 188 GB/s
  (close to per-shape pointwise ceiling -- 92 % of the 205 GB/s achieved
  at the comparable LTX residual size).
- Bare concatenate at full shape:                0.56 ms at 259 GB/s.
- Per-step: production RoPE = **0.82 s/run = 0.90 % of step**.
- **Lever A (BF16 cos/sin):** TESTED 2026-05-18, **DEAD**.  Patched
  `apply_split_rotary_emb` with env-gated cast (`LTX_ROPE_BF16_FREQS=1`),
  ran kitten one-stage (288x512x721, 8 steps, seed 42) baseline vs
  experimental.  Speed: 75.9 → 74.5 s/it = -1.85 %, but baseline ran
  with the older `project_in+project_out+full-attention` layout stack
  vs the trimmed default — comparison confounded, real RoPE delta is
  within noise of single-sample runs.  Quality: same-seed output
  shows visible drift (e.g. light switch on the wall renders
  differently).  Precision change cascades into the render through
  AdaLN-conditioned attention.  Patch reverted, env var removed, A/B
  script deleted.
- **Lever B (merge RoPE into QKV matmul epilogue):** caps at **0.90 %
  of step** even with a perfect kernel.  Multi-day Metal effort for
  sub-1 % return on a hot path whose precision is already proven
  sensitive (per Lever A's visible drift).  NOT WORTH IT.

**VAE Conv3d -- blocked on upstream fix**:
- `nn.Conv3d` achieves **4.47-5.72 TFlops/s = 56-72 % of `steel_gemm`
  ceiling** across three decoder shapes (bottleneck → mid → late).
- Conv3d dominates the resnet block (~80-95 % of its time).
- Pixel-norm + silu are negligible (<0.42 ms even at the late stage).
- Headroom IF the pending `mx.conv_general` int-overflow fix lifts
  Conv3d to GEMM ceiling: ~1.5-2x on Conv3d × ~7 % VAE share of total
  wall = **~2.4-3.5 % end-to-end ceiling**.
- **Blocked on upstream MLX.**  Re-run `vae_ops` after the fix lands
  to confirm the lift; numbers above set the upper bound.

**Consolidated remaining-lever ceiling on M1 Max:**

| Lever                                       | Status                                  | Per-step ceiling |
|---|---|---|
| Custom fused / streamed BF16 FFN kernel     | DEAD (re-confirmed quiet-machine)       | 1.15-2.22 %      |
| Custom AdaLN+residual Metal kernel          | NOT WORTH IT                            | ~0.10 %          |
| RoPE BF16 cos/sin (Lever A)                 | DEAD — tested, visible output drift     | up to 0.55 %     |
| RoPE merged into QKV epilogue (Lever B)     | NOT WORTH IT (kernel effort)            | up to 0.90 %     |
| VAE Conv3d int-overflow fix                 | BLOCKED upstream                        | 2.4-3.5 %        |

**Total combined remaining lever headroom on M1 Max:** ~2.4-3.5 % of
step, entirely gated on the upstream `mx.conv_general` int-overflow
fix.  All five candidates surfaced by the consolidated microbench
sweep have now been tested or are blocked.  **The investigation is
fully closed.**

**Side finding -- compile-block-groups watchdog memory now stale**
(2026-05-18, end-to-end `scripts/bench_compile_groups.sh`):

| group   | step1 wall | steady (n=1) | vs off  |
|---|---|---|---|
| off     | 46.10 s    | 45.90 s      | (ref)   |
| N=4     | 45.90 s    | 45.50 s      | -0.87 % |
| N=20\*  | 45.80 s    | 45.30 s      | -1.31 % |
| N=48    | 45.90 s    | 45.40 s      | -1.09 % |

\*N=20 was a bash-builtin-collision accident (script's `GROUPS`
shadowed by bash's `GROUPS` array → "20" = staff GID).  The accident
turned into useful data: **the historical "compile-group > 4 causes
watchdog hangs" memory is now stale** -- since the BF16/AdaLN dtype
cast fix landed (2026-05-17), N=20 and N=48 STEPS=2 both ran to
completion with no hang and no desktop-interactivity regression.

All four values land in a 0.6 s band (45.30-45.90 s) = ~1.3 % range
= run-to-run noise.  **`mx.compile` over the LTX block stack does
literally nothing detectable at this resolution**, regardless of
group size.  Bonus finding: step1 at N=48 (whole-transformer compile
trace) is 45.90 s vs off's 46.10 s — `mx.compile`'s trace cost is
essentially free at the LTX scale.  No "amortize compile over many
steps" argument left.

Feedback memory `feedback_perf_flags.md` updated; bench script's
watchdog warning softened.  The stale warning string is still printed
by the model code somewhere (saw it fire at N=20 and N=48) — needs a
follow-up grep + edit pass when convenient.  If a higher-precision
answer than ±1.3 % is wanted, re-run at
`STEPS=4 COMPILE_GROUPS=off,4,8,16,48` (~75 min, 3 steady samples per
run) — but absent a hypothesis the curve hides anything, this looks
like a fully closed door.

### 2026-05-18: Custom fused / streamed BF16 FFN kernel -- feasibility microbench shows ~3-7 % ceiling `[ABANDONED]`

**Bottom line:** the stock matmul → GELU → matmul FF chain already runs at
**91-95 % of MLX's own `steel_gemm` ceiling**; a custom fused Metal kernel
(or any pure-MLX streamed/tiled variant on top) can save at most the GELU
pass and the hidden-tensor HBM round-trip.  Per-call ceiling: **3.8 %
(conservative)** to **7.5 % (optimistic)** of stock wall.  Per-step ceiling:
**+1.3 % to +2.6 % of a 45.5 s step**.  Combined with the requirement that
a tiled kernel match `steel_gemm` GEMM efficiency (which tiling typically
loses), this is a clear pass.

**Trigger:** offline patch sketch for a "streamed BF16 FFN" (per-chunk
streaming over the inner dim to avoid the full `T × inner_dim`
intermediate) circulated for review.  Rather than build the prototype, we
wrote `bench_fused_ffn_feasibility` to bound the maximum possible win
first.

**Bench:** `scripts/bench_ff_microbench.py fused` -- new mode added
2026-05-18.  At LTX-2.3 stage-1 shape (B=1 T=8784 D=4096 INNER=16384 BF16),
20 iters, M1 Max:

| mode | mean ms | TFlops/s | % of GEMM ceiling | peak MB |
|---|---|---|---|---|
| stock FF (matmul → GELU → matmul) | 324.5 | 7.27 | 91 % | 287.8 |
| project_in matmul only            | 156.4 | 7.38 | 93 % | -- |
| project_out matmul only           | 158.2 | 7.61 | 96 % | -- |
| both matmuls chained, no GELU     | 312.3 | 7.55 | 95 % | 287.8 |
| gelu_approx at hidden shape       |  12.3 | n/a (50.5 GB/s) | -- | 0 |

**Floors (per FF call):**
- floor A (project_in + project_out, sum): 314.6 ms
- floor B (chained, no GELU; still writes/reads hidden via HBM): 312.3 ms
  -> stock-B recoverable = **+12.2 ms = +3.8 %**
- floor C (B minus hidden HBM round-trip, perfect tiled fused kernel):
  ~300.1 ms  -> stock-C recoverable = **+24.4 ms = +7.5 %**
- floor C is an upper bound and assumes tiling does NOT drop GEMM
  efficiency below the 95 % `steel_gemm` ceiling currently observed.

**Extrapolation** (48 blocks × 2 steps = 96 FF calls per stage-1 2-step run):
- conservative (vs floor B): +585 ms / step = +1.29 % of 45.5 s
- optimistic   (vs floor C): +1173 ms / step = +2.58 % of 45.5 s
- 30 s clip (~91 / 61 = 1.49× tokens, both stages): ~15-30 s total wall ceiling

**Why the streamed-FFN sketch specifically does NOT clear that bar:**

The circulated prototype computed FFN by streaming over the inner
dimension as N smaller MLX GEMMs with output-tensor accumulation
(`out = out + (hidden_chunk @ w_out_chunk)` per chunk).  Concretely:

1. **Output accumulation kills the memory story.**  Each chunk reads and
   writes the full `tokens × dim_out` output (~72 MB).  At 16 chunks
   that's ~1.1 GB of extra output traffic the stock path doesn't have.
2. **N small GEMMs vs 1 well-tuned `steel_gemm`** -- empirically the
   matmuls are already at 91-96 % of ceiling; chunking can only lose.
3. **Weight memory roughly doubles** -- chunks are `mx.contiguous(...)`
   copies of `_project_*_weight_t` slices; originals are not freed.
4. **Silent correctness hole for non-GELU FFNs** -- the sketch
   hardcoded `nn.gelu_approx` but didn't check whether the target
   `FeedForward` was the GELU variant or `SwiGLU` (which exists in the
   codebase and is the activation for SwiGLU FFN variants).

A real fused Metal kernel could in principle save the +1.3 % to +2.6 %
ceiling above, but the engineering cost (multi-day Metal work, riskier
than the FA-2 attempt because two GEMMs must serialize on one kernel
issue) is not justified for that envelope.

**What this kills:**
- Custom fused BF16 FFN Metal kernel
- Streamed / tiled FFN on top of MLX (any chunk size)
- `mx.compile`-based FFN fusion (already refuted by `ff_chain` bench
  on 2026-05-17 -- `is_fusable()` excludes matmul)

**What it leaves untested:**
- A fused kernel that ALSO drops to FP16 / FP8 (out of scope: dtype
  change).
- Architectural reductions (fewer FFNs, MoE routing, layer skipping).
- Hardware upgrades (M3+ INT8 GEMM, M5+ NAX cooperative-tensor matmul)
  -- the recoverable envelope is so small that any of these dwarfs it.

**Bench code preserved:** `scripts/bench_ff_microbench.py fused` is in
the repo and re-runnable if anyone wants to verify on different M1 Max
units or after MLX upstream kernel improvements.  If the
`steel_gemm`-to-stock-FF gap widens past ~10 % in a future MLX release,
this hypothesis becomes worth revisiting.

### 2026-05-18: FA-2 custom kernel -- D-sweep refutes tile-size hypothesis `[ABANDONED]`

**Bottom line:** the cheapest possible test of the FA-2 tile-size
lever (a D-dimension sweep through MLX's bk=32/bk=16 kernel-selection
boundary at D=128) refuted the hypothesis.  MLX SDPA at our shape is
already at ~75% of MLX's own steel_gemm ceiling -- the theoretical
maximum lift from a custom kernel is ~1.34x per-call = ~5.5%
end-to-end.  Combined with a refuted tile-size lever and 2-5 days of
implementation work with high risk, this is a clear pass.

**D-sweep result** (`bench_sdpa_d_sweep` at B=1 H=32 T=8784, BF16
non-causal, 50 iters, log at `$SHARED_TEMP_DIR/trace_analysis/sdpa_d_sweep.log`):

| D | MLX bk | mean ms | TFlops/s | vs D=128 |
|---|---|---|---|---|
| 64 | 32 | 96.54 | 6.55 | +10.0% |
| 80 | 32 | 121.08 | 6.53 | +9.6% |
| 96 | 32 | 185.11 | 5.12 | -13.9% |
| 112 | 32 | 195.37 | 5.66 | -4.9% |
| 120 | 32 | 209.26 | 5.66 | -4.8% |
| **128** | **16** | **212.40** | **5.95** | **(baseline)** |
| 136 | 16 | 257.79 | 5.21 | -12.5% |
| 160 | 16 | 263.35 | 6.00 | +0.8% |
| 192 | 16 | 275.76 | 6.88 | +15.5% |
| 256 | 16 | 352.36 | 7.18 | +20.6% |

**Hypothesis was:** MLX's bk=16 path at D=128 leaves compute on the
table; D=120 (bk=32) should achieve meaningfully higher TFlops/s.

**Reality:** D=120 (bk=32) achieves 5.66 TFlops/s vs D=128 (bk=16) at
5.95 TFlops/s -- the bk=16 kernel is actually 4.8% MORE efficient per
FLOP at the boundary, NOT less.  MLX's `bk = bd < 128 ? 32 : 16`
dispatch heuristic is empirically correct at the D=128 transition.

**Pattern in the broader table** (interesting but not actionable):
- D=64/80 well-tuned (canonical LLM head dims, most kernel-tuning love)
- D=96 / D=136 fall off cliffs to ~5.2 TFlops/s -- likely fallback to
  less-optimized template specializations at non-standard dims
- D=192 / D=256 climb back up (higher arithmetic intensity amortizes
  overhead, hits 7.18 TFlops/s = 90% of GEMM ceiling)
- D=128 (us) sits in the middle plateau at 5.95 TFlops/s = 75% of
  GEMM ceiling

**Achieved-vs-peak at D=128:**
- M1 Max BF16 hardware peak: ~10 TFlops/s -> 60% achieved
- MLX `steel_gemm` ceiling (FF shape): 7.95 TFlops/s -> 75% achieved
- Theoretical max FA-2 lift (match GEMM): 1.34x per-call
- End-to-end if achieved: ~5.5% wall (SDPA is ~22% of step time)

**What this leaves untested:**

- **Q-tile size** (MFA's 96-row layout via TQ>1 or WM=8): D-sweep
  doesn't test this dimension.  MLX always uses BQ=32 regardless of
  D.  No free signal that BQ=64+ would help; would require custom
  kernel writing to find out.
- **Architectural changes** (KV aliasing, register softmax): possibly
  meaningful at very long T but no direct evidence at our shape.
- **Other unexplored micro-optimizations**: open-ended.

None of these are pursued.  The combination of (a) most likely lever
empirically refuted, (b) ~5.5% wall ceiling even if a custom kernel
matched GEMM throughput, (c) 2-5 days of high-risk kernel work, makes
this a clear pass.

---

**Research that's worth preserving** (kept here as negative-result
context for future investigators; the lever may become viable on
future hardware or with future MLX kernel improvements):

**Tile choices across three projects** (head_dim=128, M1 Max / Apple7
/ Max tier):

| Project | bq × bk | Source |
|---|---|---|
| **MLX** `sdpa_full` non-NAX | **32 × 16** | `mlx/backend/metal/scaled_dot_product_attention.cpp:198` |
| **pmetal** FlashAttention | **32 × 32** (NOT 64×32 as their docs claim) | `crates/pmetal-metal/src/kernels/metal/flash_attention.metal:271-272` |
| **MFA** AttentionKernel | **96 × 32** | `kernels/AttentionDescriptor.cpp:522-630` Apple7 fork |

**TGSM strategy comparison** (head_dim=128, half precision):

| Kernel | Q tile | K tile | V tile | S/P | O accum | Aliasing | Total TGSM |
|---|---|---|---|---|---|---|---|
| **pmetal** | 8 KB | 8 KB | 8 KB | regs | regs | **none** | **~24 KB** |
| **MLX Steel** BQ=32 BK=16 | 8.5 KB | KV_smem 6.1 KB | alias of K | regs | regs | **K/V share** | **~14.6 KB** |
| **MFA** | **regs** | 8 KB | reuses arena | regs | **regs** | **all via `std::max`** | **~8 KB** |

M1 Max has 32 KB TGSM available; MLX uses 14.6 KB.  **MLX is at the
kernel-template-rigidity frontier (`static_assert(TQ == 1)` at
`mma.h:184`), NOT the hardware frontier.**  Bigger tiles are
physically possible -- the question (which the D-sweep refuted as
profitable) was whether they'd deliver more compute.

**`mx.fast.metal_kernel` capability survey** (relevant if revisiting):
- Accepts arbitrary Metal source -- `simdgroup_multiply_accumulate`,
  `simd_shuffle_xor`, threadgroup arrays, manual aliasing, async
  `simdgroup_event` copies, `<metal_simdgroup_matrix>` all available
- CANNOT import `mlx::steel` internal templates (`BlockLoaderT`,
  `MMATile`, `row_reduce<Op>`) -- those would need re-implementation
  in standalone Metal (~500 LOC total per the TGSM agent)
- Requires compile-time-sized threadgroup arrays; template_args
  mechanism (`custom_kernel.cpp:266-275`) supports int template
  parameters for size-at-instantiation

**pmetal FA kernel reusability:** NOT directly callable from MLX-land
(Rust API takes IEEE f16 not BF16; no standalone FA bench binary;
pyo3 module exposes only high-level APIs).  BUT their Metal shader
source is plain text and portable:
`crates/pmetal-metal/src/kernels/metal/flash_attention.metal`
(1639 LOC), with a non-causal `flash_attention_forward_d128` kernel
at line ~259.  Credits philipturner/metal-flash-attention as
lineage.  Could be ported into `mx.fast.metal_kernel` if a future
investigation finds reason to retry.

**Three-phase investigation arc** (preserved for honesty about how
the understanding evolved):

1. **Phase 1 -- pmetal docs read:** Looked like MLX picked 32×16 vs
   pmetal's 64×32 vs MFA's 96×32.  Hypothesis: "MLX is leaving
   headroom."  (Later: pmetal docs were misleading -- their actual
   code is 32×32, not 64×32.)
2. **Phase 2 -- kernel-build experiment:** Background agent built a
   naive BQ=64 BK=32 kernel via `mx.fast.metal_kernel`, hit TGSM
   budget concerns (~47 KB vs 32 KB limit without aliasing).
   Hypothesis reframed: "MLX may be at TGSM frontier."  (Later:
   it's not -- MLX uses 14.6 KB of 32 KB available.)
3. **Phase 3 -- TGSM-strategy comparison:** Background agent
   confirmed MLX is at kernel-template-rigidity frontier, not TGSM
   frontier; `mx.fast.metal_kernel` CAN express the tricks pmetal
   and MFA use.  Hypothesis upgraded to "GREEN actionable" with
   2-5 day effort estimate.
4. **Phase 4 -- D-sweep microbench (THIS abandonment):** the
   cheapest possible test of the K-tile hypothesis refuted it.
   Hypothesis killed before kernel work began.

**Artifacts** (kept on local /tmp -- delete at session end if not
needed):
- `/tmp/pmetal_dig/pmetal/` -- pmetal repo checkout
- `/tmp/mlx_sdpa_tile_experiment/` -- phase-2 naive kernel artifacts

**Bench code preserved:** `scripts/bench_ff_microbench.py
sdpa_d_sweep` -- the bench that closed this hypothesis is in the
repo and re-runnable if anyone wants to verify on different M1 Max
units or after MLX upstream changes.

### 2026-05-18: Draw Things / MFA research summary -- nothing portable for M1 Max `[ABANDONED]`

Triggered by the "LTX 2.3 became 1.7x faster in just one month" Reddit
headline and the Draw Things blog posts on Metal Quantized Attention +
Metal FlashAttention v2.5 w/ Neural Accelerators.  Spawned two parallel
agents to read the public posts and dig into the source of
[liuliu/ccv](https://github.com/liuliu/ccv) (MFA lives at
`lib/nnc/mfa/`) and
[drawthingsai/draw-things-community](https://github.com/drawthingsai/draw-things-community).

**The 1.7x is plausibly explained by ONE M3+ lever alone -- Draw
Things' own blog cites Int8 attention as 1.19-1.76x e2e on M3+ hardware.
M1 Max stops at the Apple7 gate for that lever and several others.**
From `ccv_nnc_mfa.cpp` capability gates (verified via direct source
read):

| Lever | Source claim | MFA hardware gate | M1 Max? |
|---|---|---|---|
| Metal Quantized Attention (Int8) -- `NAInt8AttentionKernel` | 1.19-1.76x e2e, 1.43-1.95x single-step | `supportsFamily(1009)` = Apple9 (**M3+**) | ❌ |
| FA v2.5 w/ Neural Accelerators (`mpp::tensor_ops::matmul2d`) | "4.6x on M5 over M4" | `ccv_nnc_mfa_has_neural_accelerators` = `supportsFamily(1010)` (Apple10, **M5 only**) | ❌ |
| ANE rowwise GEMM (CoreML mlprogram, FF projections) | implicit in stack | `ccv_nnc_mfa_supports_int8_ane` = Apple9 (**M3+**) + macOS 26.1 for BF16 | ❌ |
| Codex-authored VAE shader | "2.4x on M1-M4, 4.7x on M5" | none -- generic Metal | ✅ |
| Classic MFA FA via `simdgroup_matrix_storage` (`AttentionKernel.cpp`) | shipped since 2023 | Apple7 (**M1+**) with M1-tuned tiles `parallelization=96 traversal=32` | ✅ |

The Apple7 tile fork is explicit in `AttentionDescriptor::forwardMixed`
(`kernels/AttentionDescriptor.cpp:522-630`): M1/M2 gets larger tiles
(96x32 vs 32x16 on M3+) -- well-tuned, not a degraded fallback.

**Nothing in MFA is directly portable for our M1 Max LTX-2.3 work, in
ranked likelihood:**

1. **MFA classic FA via `simdgroup_matrix_storage`**:  MLX's
   `mx.fast.scaled_dot_product_attention` already routes through
   `simdgroup_matrix` on M1 (verified in MLX source at
   `mlx/backend/metal/scaled_dot_product_attention.cpp:177`).
   Same intrinsics, same approximate ceiling.  Porting MFA's version
   would be a multi-week rewrite into MLX's Steel template
   infrastructure for at best modest gain.  Stays in scope only via
   the FA-2 entry above (which now references this finding).
2. **Conv3DKernel for VAE decode** (`kernels/Conv3DKernel.cpp`, 503
   lines, no hardware gating):  Draw Things' "Codex-authored VAE
   shader" claim of 2.4x on M1-M4 is presumably this.  But VAE
   decode in our pipeline is only 49 s of 11m 30s total (~7 %) and
   we're already on the "native Conv3d VAE decoder" path -- MLX is
   presumably already using competitive `mx.conv_general` / Steel
   kernels.  Recommendation per user: wait for the int-overflow fix
   in upstream MLX VAE rather than port.  Listed for completeness.
3. **Small ops** (`RMSNormKernel`, `RopeKernel`, `GELUKernel`,
   `CastKernel`, `ExpKernel`, `CMulKernel`, `AddKernel`):  duplicates
   of `mx.fast.rms_norm`, `mx.fast.rope`, etc.  Microsecond kernels.
   Not worth replacing.
4. **`GEMMKernel`**:  same regime as MLX's `steel_gemm` (~7.95 TFlops/s
   at our FF shape, ~77 % of M1 Max BF16 peak per Pretranspose entry).
5. **Hardware-gated kernels** (`NAAttentionKernel`, `NAConv3DKernel`,
   `NAInt8AttentionKernel`, `NAMatMulSmallMKernel`, `ANERowwiseTransform*`):
   physically unreachable on Apple7.

**ANE on M1 Max specifically -- DEAD for LTX-2.3, three independent
reasons:**

1. **Working-set size:** 22B params, dynamic T=8784 sequence, large
   activations -- the data-shuttling cost to/from system memory eats
   the compute win.  Even on M3+ where Draw Things does use ANE, only
   FF rowwise GEMM tail is offloaded -- never attention.
2. **Dtype:** M1 Max ANE doesn't have native BF16.  BF16 on ANE
   requires macOS 26.1 + M3+.  Extra FP16↔BF16 conversion would
   kill the throughput case.
3. **Op coverage:** ANE has limited 3D conv, no native attention, no
   RoPE.  For a video transformer with audio sidecar you'd CoreML-
   convert maybe 10 % of the graph at best; cross-boundary overhead
   dominates.

Where M1 Max ANE *is* useful (just not for us): `whisper.cpp`
(~3x over CPU for speech), small CLIP / vision encoders, Apple
Intelligence on-device LMs.  Pattern: static-shape, INT8/FP16,
≤1-2 GB working set.

**Why FA-2 isn't in MLX core** (informational, useful context):

1. FA-2 *has* been ported to Metal -- by ccv/MFA, not MLX.
2. FA-1 → FA-2's headline 2x on H100 came from NVIDIA-specific
   warp scheduling + tensor cores; doesn't translate 1:1 to Apple
   simdgroups (no warp shuffle in the same form, no tensor cores
   until M5 NAX).  Realistic Apple FA-2 lift is 1.2-1.5x on
   favorable shapes.
3. MLX has been absorbing the *parts* of FA-2 that matter: tile-
   floor logic for NAX (bq=64/bk=32 vs bq=32/bk=16), M5 cooperative-
   tensor SDPA in late 0.30.x.  Priority has been M3+/M5 hardware
   paths, not squeezing the last 20 % out of M1 simdgroup_matrix.
4. MLX team is small; roadmap is dominated by training/grad,
   distributed, mixed precision, new hardware.  Video diffusion
   on M1 Max is a niche compute-bound regime that doesn't drive
   their roadmap (most users run bandwidth-bound 7-13B LLMs where
   existing SDPA is "good enough").
5. License/code-style mismatch: MFA is C++/ObjC++ with template
   metaprogramming generating Metal source.  MLX kernels live in
   `mlx/backend/metal/kernels/*.metal` + Steel templates.  A copy
   isn't a copy-paste -- it's a rewrite.

**"S models" terminology:** Draw Things' Quantized Attention post:
*"Int8 matrix multiplication applies only to the 8-bit S models we
recently added."*  It's their Int8 quantization model variant -- not
a chip suffix, not an `applegpu_g13s`-style GPU codename, not an
LTX-2 sub-variant.

**Key reference files for future investigation** (paths relative to
each upstream repo root; clone from links above):

ccv (`lib/nnc/mfa/`):

- `ccv_nnc_mfa.cpp` -- capability gates lines 23-56, 141
- `ccv_nnc_mfa_attention.cpp` -- NA/NAInt8 dispatch at lines 179, 381
- `kernels/AttentionDescriptor.cpp` -- tile schedule fork lines 522-630
- `kernels/AttentionKernel.cpp` -- classic FA via simdgroup_matrix, ~2400 lines
- `kernels/Conv3DKernel.cpp` -- VAE-relevant, no hw gate, 503 lines
- `ccv_nnc_mfa_ane_rowwise_coreml.mm:818` -- CoreML mlprogram + `MLComputeUnitsCPUAndNeuralEngine`

draw-things-community:

- `Libraries/SwiftDiffusion/Sources/Models/LTX2.swift` -- LTX2 graph, `FlashAttentionLevel` consumer, 2351 lines
- `Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift:14` -- `FlashAttentionLevel` enum

**Bottom line:** the 1.7x is mostly an M3+ Int8-attention win, not a
software lever we're leaving on the table for M1 Max.  On M1 Max the
realistic same-direction levers are (a) the Conv3DKernel VAE shader
(~7 % of wall, possibly redundant with MLX-native VAE which the user
notes is awaiting an upstream int-overflow fix) and (b) the FA-2 entry
above (1.2-1.5x SDPA at best, multi-week effort).  No new action items.

### 2026-05-17: Audio pretranspose microbench -- neutral at bakery T, defaults unchanged `[ABANDONED]`

Ran `scripts/bench_ff_microbench.py bf16_layout_audio` three times
(clean 50-iter runs, bench-process-watch active, logs in
`$SHARED_TEMP_DIR/trace_analysis/bf16_layout_audio_clean*.log`)
at the bakery audio shapes (T=502, D_audio=2048, FF_inner=8192).
All four audio projections show pretranspose as **neutral within
noise** — averaging across 3 runs:

| Shape | Δ vs naive (3-run avg) | Verdict |
|---|---|---|
| audio.FF.project_in (2048→8192) | +0.6% | tied |
| audio.FF.project_out (8192→2048) | +0.2% | tied |
| audio.attn.to_q/k/v (2048→2048) | +0.6% | tied |
| audio.attn.to_out (2048→2048) | +0.4% | tied |

TFlops/s is consistent across modes too: audio FF achieves 6.4-6.9
TFlops/s and audio attn achieves 5.0-5.2 TFlops/s regardless of
pretranspose.  Audio is naturally less efficient than video on a
per-TFlops basis because the small T=502 gives less parallelism, but
the per-matmul time is tiny (0.8-2.7 ms) so the total cost is small.

**No kernel-selection cliff** at audio shapes -- contrast with
video.project_out where naive falls to 5.17 TFlops/s vs pretranspose
7.95 TFlops/s (35 % win).

**Defaults changed via inheritance from video defaults.**  When
`DEFAULT_VIDEO_FF_LAYOUT_SPECS` and `DEFAULT_VIDEO_ATTN_LAYOUT_SPECS`
were trimmed (see "Pretranspose default cleanup" above), audio
inherited the same trimming via the shared cache-build path: audio FF
is now `project_out:pretranspose` only, audio attn is OFF.  Kitten
smoke (one-stage 288×512, 721 frames, 8 steps) confirms this is safe
at bakery scale: 75.4 s/it vs 75.9 s/it baseline = **−0.7 %** (within
noise but at minimum no regression).

The historical 2026-05-15 −11.2 % small-T audio-pretranspose win is
NOT re-verified.  At 256×256×25, audio T is much smaller (~25 tokens)
and per-call dispatch overhead approaches per-call matmul time —
pretranspose may genuinely save dispatch cost there.  Users running
small-T workloads who notice a regression can restore the audio
layouts manually (no LTX-side env-var-only opt-out exists; would
need to pass `--video-attn-layout to_q:pretranspose,...` and similar
audio flags, or reintroduce the historical defaults in
`scripts/generate.py`).

Opt out the broader audio-pretranspose mechanism via
`LTX_DISABLE_AUDIO_PRETRANSPOSE=1` (legacy env, still honored).

### 2026-05-17: mxfp8 draft mode is DEAD post-AdaLN-fix `[ABANDONED]`

All three mxfp8 variants tested are slower than the BF16 production
baseline (45.5 s/it stage 1):

| Variant | Per-step | vs baseline | Per-matmul evidence |
|---|---|---|---|
| **A. baseline** BF16 + FF pretranspose | 45.5 s/it | reference | 7.95 TFlops/s on project_out |
| **B. `--video-ff-quantize project_out:mxfp8 --video-ff-layout off`** | 50.2 s/it | **+10.2%** | mxfp8 = 4.70 TFlops/s on project_out (vs 7.95 BF16) |
| **C. `--transformer-cache-quantize mxfp8-blocks`** | 65.0 s/it | **+42.9%** | broader scope (attn + both FF), all at ~4.7 TFlops/s |
| **D. `--transformer-cache-quantize mxfp8-blocks-pretranspose`** | 66.0 s/it | **+45.0%** | same as C, pretranspose-baked-in does NOT recover |

**Per-matmul microbench (`scripts/bench_ff_microbench.py quant_matmul`)
nails the mechanism:** mxfp8 is ~66% slower than BF16+pretranspose
at every LTX matmul shape we tested:

| Shape | BF16+pre | mxfp8 | regression | TFlops/s ratio |
|---|---|---|---|---|
| FF.project_in (4096→16384) | 159.6 ms | 265.4 ms | +66.3% | 7.39 → 4.44 |
| FF.project_out (16384→4096) | 151.1 ms | 250.8 ms | +65.9% | 7.80 → 4.70 |
| attn.to_q/k/v (4096→4096) | 37.6 ms | 63.0 ms | +67.6% | 7.84 → 4.68 |
| attn.to_out (4096→4096) | 37.6 ms | 63.0 ms | +67.4% | 7.84 → 4.68 |

End-to-end math closes perfectly:
- B quantizes only project_out (~16% of step) → 66% × 16% = +10.6% (measured +10.2%)
- C/D quantize attn + both FF (~60% of step) → 66% × 60% = +40% (measured +43-45%)

**The "22% of INT8 peak" framing earlier was a category error -- there
is no INT8 matmul on M1 Max in MLX.**  Direct source read of
`mlx/backend/metal/kernels/fp_quantized.h:139` and `:663` shows the
non-NAX `mx.quantized_matmul` is a **BF16 matmul with on-the-fly
dequant**, not an INT8 matmul:

```cpp
// fp_quantized.h:139 -- dequant happens inline, output is U-typed
inline void dequantize(uint8_t w, U scale, threadgroup U* w_local) {
  ...
  w_local[0] = scale * Dequantize<8, U>{}(w);   // mxfp8 path
}
// fp_quantized.h:663 -- same BlockMMA / simdgroup_matrix as steel_gemm
using mma_t = mlx::steel::BlockMMA<...>;
mma_op.mma(Xs, Ws);                              // Ws is dequantized BF16
```

`U` is the activation dtype (BF16 for us).  `BlockMMA` is the same
`simdgroup_matrix`-based MMA primitive that `steel_gemm` uses.  So the
pipeline is: read packed FP8 weight → read per-block scale → multiply
scale × dequantized byte → write BF16 into threadgroup memory → run
BF16 MMA over the dequantized tile.  **The matmul itself runs at the
BF16 ceiling minus dequant overhead -- M1's INT8 instruction throughput
(`dot4I8Packed`, ~21 TFlops/s theoretical) is never reached because
the kernel doesn't use it.**

**Why mxfp8 will ALWAYS be slower than BF16 on M1 Max for our shapes**
(three additive sources of the 7.95 → 4.7 TFlops/s gap):

1. **Inline dequant in the inner loop.**  Every BF16 weight loaded into
   the MMA tile costs an extra global scale-load (amortized 1/32),
   an FP8→BF16 conversion, a multiply with the scale, and a
   threadgroup-memory store.  The pure-BF16 path skips all of this
   and loads BF16 weights directly into the MMA tile.
2. **Disrupted MMA scheduling.**  `steel_gemm` keeps the MMA pipe full
   via register tiling, double-buffered loads, and careful
   threadgroup-memory layout.  Inserting dequant work between loads
   and MMA stalls the pipe more often -- lower achievable arithmetic
   intensity.
3. **Less mature kernel tuning.**  The quantized kernel exists in two
   variants: `quantized.h` (M1 path) and `quantized_nax.h` (M5-only).
   Apple/MLX's optimization gradient is firmly on the NAX path
   (cooperative-tensor `matmul2d`); the M1 fallback is the older,
   less-tuned dequant-into-BF16-MMA approach.

**Why bandwidth savings don't rescue this.**  In principle mxfp8 cuts
weight-read bandwidth by ~50% vs BF16:

| Shape | BF16 weight | mxfp8 weight + scales | Saving |
|---|---|---|---|
| project_out (16384×4096) | 128 MB | ~68 MB | ~150 µs at 400 GB/s |

But `project_out` runs in **151 ms BF16 vs 251 ms mxfp8** -- the entire
weight read is only ~0.3 ms in either case.  At our matmul shapes we
are **compute-bound by a factor of ~500×**, not bandwidth-bound.
Saving 150 µs of bandwidth is invisible when the compute regression is
100 ms.

**Structural conclusion**: mxfp8 cannot beat BF16 on M1 Max via the
current MLX kernel, because the kernel reduces to "BF16 matmul +
extra work."  There is no instruction path it can switch to that
would deliver more compute than the BF16 ceiling itself.

**Why the historical "mxfp8 ~13% draft mode" was real**: it was
measured against a pre-AdaLN-fix BF16 baseline of 77.8 s/it
(FP32-polluted SDPA path).  The race that mxfp8 was winning was
broken-BF16 vs quant.  Post-AdaLN-fix BF16 is 45.5 s/it (-42%) but
the mxfp8 path got essentially no benefit (it doesn't hit the same
SDPA path).  So the relative ordering flipped.

**Verdict:** mxfp8 is structurally a net regression on M1 Max -- not
fixable by tuning, only by either:

- (a) A from-scratch native-INT8 kernel using M1's `dot4I8Packed`
  intrinsics (~21 TFlops/s theoretical INT8 peak).  Same multi-week
  kernel-writing effort class as the FA-2 hypothesis.  Apple, MLX, and
  Draw Things have all independently chosen NOT to do this for Apple7
  -- the engineering cost doesn't justify the win, and on M1 even a
  perfect INT8 kernel would only modestly beat BF16 because Apple
  didn't invest in INT8 throughput improvements until M3.
- (b) M3+ hardware where Apple-tuned INT8 kernels start to beat BF16
  (Draw Things' Int8 attention claims 1.19-1.76× e2e on M3+, which
  arithmetically requires INT8 GEMM > BF16 GEMM).
- (c) M5+ NAX hardware where the `mpp::tensor_ops::matmul2d`
  cooperative-tensor path delivers tensor-core-class INT8 throughput
  (per source verification: NAX requires GPU gen 17+, M5/M5 Pro --
  NOT M3 as earlier text incorrectly claimed).

CLI flags (`--video-ff-quantize`, `--transformer-cache-quantize`) and
their underlying code paths are kept intact for future research and
because the cache infrastructure is reusable for other quant modes
that might land later (e.g. INT4, FP4, future MLX kernels that
actually use M1's INT8 instructions).

**Bench script `scripts/bench_mxfp8_draft.sh`** is reusable for
future quant experiments — runs four variants A/B/C/D with
PREWARM=1 (excludes one-time cache build cost) and SKIP_* env vars.

### 2026-05-17: Q+K+V fusion candidate `[ABANDONED]`

**Microbench killed it.**  Packed Q+K+V matmul is 1.34 ms slower than
3 separate matmuls per call (113.10 vs 111.75 ms mean, 50 iters in
`scripts/bench_ff_microbench.py` clean run with bench-process-watch
killing background macOS GPU agents).  Tails are tight (p99 113.49 vs
113.18 — the gap is real, not noise).

Reason: bandwidth savings on the input read (~0.5 ms / call saved at
300 GB/s) are dominated by the overhead of the larger output's worse
tile alignment + the post-matmul split.  MLX's existing 3-separate-matmul
dispatch is already at or near optimal for our shape.

This also explains the earlier `self_qkv:pack` / `kv:pack` "neutral-to-tiny"
results in `PERFORMANCE.md`'s archive.  Same underlying outcome,
measured directly this time.

### 2026-05-17: `mx.compile` around FF chain `[ABANDONED]`

**Microbench killed it.**  Lazy 309.16 ms vs compiled 309.51 ms — within
0.11 % (lazy actually marginally faster, well within measurement noise).
50 iters with bench-process-watch.  Confirms MLX source inspection
(`is_fusable()` in `mlx/compile.cpp` excludes matmul).  `mx.compile`
only fuses pointwise ops; it cannot fuse matmul into a GELU activation.

The existing `mx.compile` wrappers on attention/RoPE helpers (the ones the
2026-05-16 trace showed are worth ~3 % wall) work because those wrap
pointwise-heavy regions.  Wrapping the FF chain does not produce a similar
win because the FF is matmul-dominated.

### 2026-05-17: SDPA T-alignment padding `[ABANDONED]`

**Microbench killed it.**  T=8784 (only 16-aligned) was the hypothesis
for a hidden tile-alignment penalty in `mx.fast.scaled_dot_product_attention`.
Sweep across T = 8704, 8736, 8768, 8784, 8800, 8832, 8896, 8960, 9216
(50 iters each with bench-process-watch, clean run logged at
`$SHARED_TEMP_DIR/trace_analysis/sdpa_t_sweep_clean.log`) shows:

- T=8784 carries only a **0.27 % per-tile-alignment penalty** vs the best
  neighbor (T=8768 at 2.7432 ns/T² vs our 2.7505 ns/T²).
- Every padding-up target is SLOWER in absolute time.  Best pad
  candidate (T=8800, +16 tokens) is +0.4 ms/call.  T=8832 (128-aligned,
  +48 tokens) is +2.0 ms/call.
- Normalized cost spread across the full T=8704–9216 range is only
  1.5 % — SDPA scales nearly perfectly O(T²).

**Verdict:** MLX `sdpa_full` handles non-32-aligned T well at our (H=32,
D=128) shape; there's no cliff to dodge by padding.  The tile-floor
analysis in `PERFORMANCE.md` stands: SDPA is at the MLX hardware-tile
floor on M1 Max non-NAX; only a custom Metal kernel or M5+ hardware
(NAX-capable, gen >= 17, per `mpp::tensor_ops::matmul2d`) will change
it.
