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

_(empty — investigation converged; see Promoted section below.)_

---

## Promoted

(Entries moved to `PERFORMANCE.md` — kept here as one-line stubs for
cross-reference.)

### 2026-05-23: Audio FF FP16 -- shipped as curiosity opt-in, no measurable wall savings `[PROMOTED]`

Added `--audio-ff-dtype {bfloat16, float16}` mirroring `--video-ff-dtype`.
Microbench at bakery T=502 predicted per-generation savings ~0.32 s
(0.07 % of 7m 38s) — well below noise.  No kernel-selection cliff at
audio K=8192 (FP16 naive is FASTER than BF16 naive there, unlike video
K=16384).  Per-matmul FP16 win is real (~10–13 %) but per-call wall is
tiny (0.7–2.4 ms vs video's 60–250 ms).

Exposed anyway because (a) implementation cost was low once the video
FF FP16 plumbing was in place — `audio_ff` uses the same FeedForward
class so the runtime boundary-cast code is shared, and (b) the dtype
dimension is interesting to keep available for future workloads where
audio is a larger compute share.  `_AUDIO_FF_KEY_PATTERNS` and
`_ensure_audio_ff_layout_for_dtype` mirror their video equivalents in
`LTX_2_MLX/loader/transformer_cache.py` and `scripts/generate.py`.
`_video_cache_dtype_for_key` was renamed to `_ff_cache_dtype_for_key`
to dispatch on both video and audio FF prefixes.

Auto-pair: when `--audio-ff-dtype float16` is set,
`project_in:pretranspose,project_out:pretranspose` is added to the
audio FF layout regardless of what video FF layout is doing.  Primary
motivation is the FP16 × BF16 → FP32 mixed-dtype promotion fallback,
not the kernel cliff (which doesn't exist at audio K=8192).

Bench evidence: `scripts/bench_pretranspose_dtype.py` audio rows.

| audio shape                          | BF16 pretrans | FP16 pretrans | FP16 saves |
| ------------------------------------ | ------------- | ------------- | ---------- |
| audio.FF.project_in   (K=2048 N=8192) | 2.43 ms       | 2.10 ms       | +13.6 %    |
| audio.FF.project_out  (K=8192 N=2048) | 2.61 ms       | 2.33 ms       | +10.7 %    |
| audio.attn  (K=N=2048)               | 0.82 ms       | 0.74 ms       | +9.2 %     |

**Real-world bakery A/B (idle machine, distilled 384x640x20s):**

| variant                          | wall      | stage 1   | stage 2   | output save | Δ wall vs video-only-FP16 |
| -------------------------------- | --------- | --------- | --------- | ----------- | ------------------------- |
| BF16 baseline                    | 7m 59.6s  | 2m 09s    | 4m 18s    | 56.10s      | (baseline)                |
| `--video-ff-dtype float16` only  | 7m 37.6s  | 2m 07.7s  | 4m 07.3s  | 56.15s      | (reference)               |
| video + audio FF FP16            | 7m 42.2s  | 2m 07.5s  | 4m 09.8s  | 58.76s      | **+4.6s**                 |

Output save (pure VAE + encoder, untouched by the transformer) varies
by +2.6s between the two FP16 runs.  That's the noise floor.  Stage 2
denoise +2.5s sits within it.  **No measurable wall savings from audio
FP16 on bakery.**

**Latent comparison (cos sim):**

| key                  | video-FP16 vs BF16 | video+audio-FP16 vs BF16 | video+audio vs video-only-FP16 |
| -------------------- | ------------------ | ------------------------ | ------------------------------ |
| stage_1_video_latent | 0.78008            | 0.78030                  | 0.95847                        |
| stage_2_video_latent | 0.70955            | 0.71348                  | 0.88612                        |
| stage_1_audio_latent | 0.98973            | 0.98943                  | **0.99903**                    |
| stage_2_audio_latent | 0.98411            | 0.98028                  | **0.99662**                    |

Audio FP16 keeps the audio latent at cos sim ≥0.997 vs the video-only-
FP16 reference — effectively untouched.  Audio listening test confirms
no perceptual change in the bakery dialogue.

The apparent stage-2 video latent drift between the two FP16 configs
(cos 0.886) looks alarming but is consistent with geometric distance
between two FP16 runs both sitting at cos ~0.71 from BF16 baseline —
two points at angular distance ~45° from a third reference can be
~27° apart from each other.  Cross-attention feedback from audio to
video is real but small (audio latent only perturbed by cos 0.997
from adding audio FP16, propagated through cross-attn into video).

**Verdict: flag shipped as a curiosity opt-in.**  Documented marginal
savings, no quality concern, no measurable wall improvement.  Kept
because the dtype dimension is interesting to have available for
future hardware / workload changes that might shift the math.

### 2026-05-23: BlockLoader cliff characterization (project_out K-sweep + MLX source) `[CHARACTERIZED]`

The "kernel-selection cliff" we've been writing about since 2026-05-17 is
finally pinned down to a specific mechanism in the MLX `steel_gemm`
BlockLoader, plus an empirical K-sweep that reveals it's neither a clean
slope nor a single hard step.  Reference bench:
`scripts/bench_pretranspose_dtype.py --k-sweep`.

**The mechanism (from MLX source).**  At all M=14640, N=4096 matmuls
tested, MLX dispatches identical `steel_matmul_regular_axpby` with
tile params `bm=64, bn=64, bk=16, wm=2, wn=2` — the "Medium device"
defaults (`mlx/backend/metal/matmul.cpp:163-168`).  The ONLY runtime
difference between cliff and fast paths is the `transpose_b` template
flag selecting between two kernel-name suffixes:

- **`_nt_` (naive `x @ w.T`)** — `BlockLoader` (`kernels/steel/gemm/loader.h:14-80`)
  puts 128 threads in a **2-across × 64-down** layout in B's (N, K)
  view, touching 64 distinct B-rows per warp-cycle, each row separated
  by `ldb=K` elements.  As K grows, the **64-row × ldb-byte span grows
  linearly** and coalescing within a SIMD group breaks.
- **`_nn_` (pretranspose `x @ w_pt`)** — same template family, but
  with `transpose_b=false` the loader flips to **8-across × 16-down**
  along contiguous memory.  Eight adjacent threads read 8 adjacent
  BF16 elements = full 128-bit coalesced burst, and only 16 B-rows
  are touched per warp-cycle, K-independent.

No K-based branching in the dispatcher (`matmul.cpp:88-169` and
`device.cpp:485-520`); no FP16/BF16 split in dispatch either.  The
cliff is **purely in the kernel's memory access pattern**, not in
template selection.  Same kernel template for every shape we tested,
just different `ldb` and `transpose_b` flag.

**The K-sweep (empirical).**  At M=14640, N=4096, varying K from
10240 to 18432 in 1024 steps:

| K     | ldb (B bytes) | 64-row span | BF16 naive  | BF16 pre   | BF16 Δ%   | FP16 naive  | FP16 pre   | FP16 Δ%  |
|-------|---------------|-------------|-------------|------------|-----------|-------------|------------|----------|
| 10240 | 20 KB         | 1.25 MB     | 7.92        | 8.09       | +2.1      | 8.11        | 9.52       | +14.8    |
| 11264 | 22 KB         | 1.38 MB     | 7.87        | 8.08       | +2.6      | 7.84        | 9.52       | +17.7    |
| 12288 | 24 KB         | 1.50 MB     | 7.73        | 8.08       | +4.4      | 7.58        | 9.51       | +20.3    |
| **13312** | **26 KB** | **1.63 MB** | **7.13 ★** | 8.08       | +11.7     | **6.81 ★**  | 9.51       | +28.4    |
| 14336 | 28 KB         | 1.75 MB     | 7.13 (plat) | 7.97       | +10.5     | 6.81 (plat) | 9.38       | +27.4    |
| 15360 | 30 KB         | 1.88 MB     | 6.68 ★     | 7.52       | +11.1     | 6.26 ★      | 8.73       | +28.3    |
| 16384 | 32 KB         | 2.00 MB     | 5.84 ★     | 7.10       | +17.8     | 5.67 ★      | 8.54       | +33.6    |
| 17408 | 34 KB         | 2.13 MB     | 5.93 (plat) | 6.96       | +14.9     | 5.71 (plat) | 7.96       | +28.3    |
| 18432 | 36 KB         | 2.25 MB     | 6.03 (plat) | 6.79       | +11.2     | 5.74 (plat) | 7.68       | +25.2    |

(★ = step in naive throughput. "plat" = plateau, throughput unchanged
from previous K despite span growing.)

**Structure of the cliff (the K-sweep surprise).**  It's not a single
step at K=16384, and it's not a smooth slope either.  It's **slope
+ steps + plateaus + recovery plateau**:

```
TF/s
 8.0  ●●●                                ← gradual slope K=10240-12288
 7.5      ●
 7.0          ●●                         ← plateau 1 K=13312-14336
 6.5             ●                       ← step to 6.68 at K=15360
 6.0                  ●
 5.5                     ●●●             ← plateau 2 K=17408-18432
 5.0
      10  11  12  13  14  15  16  17  18  K/1024
       └── slope ──┘ └─── steps ────┘ └─ plateau ─┘
```

Reading: each step is one cache-residency tier being crossed by the
64-row × ldb-byte working set.  Plateaus are "this many rows still
fit, drop happens when one more falls out."  Largest single drop is
K=12288 → 13312 (-0.60 TF/s BF16, span 1.50 → 1.63 MB) — one specific
cache tier exhausted there.

**Pretranspose also degrades past K≈14336, but smoothly.**  BF16
pretrans drops from 8.08 (K=13312) to 6.79 (K=18432) = -16 %.  Same
kernel template, no loader cliff (it's K-invariant by construction).
This is **the second mechanism**: longer K-loop, FLOPs/byte ratio
shifts as the matmul becomes compute-bound at lower effective
throughput.  Independent of the strided-loader cliff.  At K=16384
both mechanisms stack but only the loader cliff is rescued by
pretranspose; the K-loop slope affects both paths equally.

**FP16 is consistently deeper, starts earlier.**

| dtype | "production meaningful" cliff (≥5 % pretrans Δ) starts at |
|-------|-------------------------------------------------------------|
| BF16  | K ≈ 13312 (3.3× hidden_size 4096)                          |
| FP16  | K ≈ 10240 already (2.5× hidden_size 4096) — visible at every K tested |

Pretranspose-vs-naive Δ scales by roughly 2× FP16-over-BF16 across the
sweep (BF16 +11.7 % vs FP16 +28.4 % at K=13312; BF16 +17.8 % vs FP16
+33.6 % at K=16384).  Hypothesis (un-confirmed in source): same
absolute byte penalty (coalescing breaks identically since ldb_bytes
is dtype-independent for 16-bit), but FP16 has slightly higher
compute throughput on Apple7 → kernel is more bandwidth-bound → same
penalty as larger fraction of total wall.  Numbers consistent: BF16
naive/pretrans ratio at K=16384 is 5.84/7.10 = 82 %; FP16 is 5.67/8.54
= 66 %.  Penalty applied to a faster compute baseline.

**Audio (K=8192, M=502) escape clause.**  Audio matmuls don't hit the
cliff because (a) K=8192 = 16 KB stride is below the cliff onset
even for video M, AND (b) audio runs at M=502, ~29× fewer warp
launches than video M=14640 → far less L2 contention pressure even at
the same per-warp footprint.  Audio FP16 naive is faster than BF16
naive at K=8192 — the matmul is firmly on the upside of the cliff.

**Threshold guidance.**

- LTX production project_out is K=16384, which sits squarely on the
  cliff for both dtypes.  Pretranspose is **mandatory**, not optional.
- For any new layer with K ≥ 13312 in BF16 or K ≥ 10240 in FP16,
  audit the layout — pretranspose is going to pay off.
- For square or wide-output shapes (K ≤ 4096), pretranspose is
  neutral or a slight regression; don't enable by default.

**Cross-reference with M1 Max GPU cache hierarchy.**  The plateau-step
structure pins to specific cache tiers when reconciled with M1 Max
specs (Apple G13X / AGX2 / apple7 family):

| Cache tier                | Capacity                  | Source              |
|---------------------------|---------------------------|---------------------|
| L1 data per GPU core      | 8 KB                      | RE (Chips & Cheese, Philip Turner) |
| L2 per shader cluster     | ~256 KB–1.5 MB (disputed) | RE (Chips & Cheese: M1 Pro/Max dropped from M1's 768 KB-chip-wide to 256 KB-per-cluster) |
| SLC (System Level Cache)  | 48 MB chip-wide           | Apple documented    |
| DRAM bandwidth            | 408 GB/s, LPDDR5 512-bit  | Apple documented    |
| Page size                 | 16 KB                     | Apple documented (Asahi)  |
| GPU TLB                   | unknown size, ~32 MB total reach observed on M1 Ultra | RE (HN 31020484) |
| Threadgroup memory        | 32 KB Metal limit         | Apple documented    |
| Simdgroup width           | 32 lanes                  | Apple documented    |

(RE = reverse-engineered, not Apple-published.)

Mapped to our K-sweep step structure at M=14640:

| K-step                        | 64-row span change       | Hypothesized cache event |
|-------------------------------|--------------------------|--------------------------|
| K=10240→12288 (slope, no step)| 1.25 → 1.50 MB          | Working set growing inside per-cluster L2; coalescing degrades smoothly but most rows still resident |
| **K=12288→13312** (-0.60 BF16)| **1.50 → 1.63 MB**       | **Per-cluster L2 partition spills.**  Best match to the ~1.5 MB reverse-engineered per-cluster L2 capacity.  Stridedly-touched B rows start missing into SLC. |
| K=13312→14336 (plateau)       | 1.63 → 1.75 MB           | Hold pattern: L2 already spilled; SLC absorbing misses |
| **K=14336→15360** (-0.45 BF16)| **1.75 → 1.88 MB**       | **Probable L1 TLB exhaustion.**  16 KB pages, 64 distinct B-rows separated by ldb > 16 KB at K ≥ 8192 → 64 distinct page touches per gather.  L1 TLB capacity exceeded somewhere in this window. |
| **K=15360→16384** (-0.84 BF16)| **1.88 → 2.00 MB**       | **Probable L2 TLB pressure.**  Compounding page-walk cost as the multi-level TLB structure runs out of headroom.  Apple GPU TLB sizes for M1 Max not publicly pinned down; this is the inference, not a confirmation. |
| K=17408+ (recovery plateau)   | ≥ 2.13 MB                | Saturated state: L2 spilled + TLB saturated + SLC absorbing the entire 64-row span; new K just adds proportional miss cost.  The curve flattens because nothing new is being spilled. |

Pretranspose's smooth slope past K≈14336 is **independent of all the
above**: contiguous loads don't hit cluster-L2 partition or TLB
boundaries the same way.  The pretranspose curve degrades because the
contiguous tile size grows vs L2 — pure DRAM-bandwidth scaling, no
multi-level cache event.

**Why MLX won't auto-mitigate.**  `matmul.cpp:88-169` GEMM tile
selection has **zero cache awareness**: M1 Max hits the Medium-device
unconditional branch (`bm=64, bn=64, bk=16, wm=2, wn=2`) regardless
of K, dtype, or transpose flag.  No L1/L2/SLC/page-size constants in
the dispatcher.  No `ldb`-based heuristic.  `BlockLoader` at
`kernels/steel/gemm/loader.h:25-134` walks K via `src += tile_stride`
with no L2-blocking logic.  The cliff is therefore **structurally
unavoidable** for any (large-M × strided-large-K) matmul on M1 Max in
current MLX; pretranspose at the LTX-2-MLX cache layer is the
load-bearing mitigation.  No upstream MLX change is in flight that
would address this — the structural cause is "the tile selector
doesn't know the GPU's caches exist."

**Honesty about sources.**

- Apple-documented: SLC 48 MB, page 16 KB, simdgroup 32, threadgroup
  mem 32 KB, DRAM bandwidth.
- Reverse-engineered (Chips & Cheese, Philip Turner, Dougall Johnson,
  Asahi Linux): L1 8 KB, per-cluster L2 256 KB–1.5 MB, 24 concurrent
  simdgroups/core.
- Not pinned down publicly: exact M1 Max GPU TLB sizes/structure,
  exact L2 partition policy per cluster, whether a separate texture
  cache is reused by `device T*` matmul loads.  The step at K=15360
  and K=16384 is *attributed* to TLB pressure based on the only
  public M1-family TLB data point (HN 31020484, ~32 MB total reach on
  M1 Ultra) and the page-count arithmetic, but is not confirmed.

**Files (MLX source).**

- `mlx-main/mlx/backend/metal/matmul.cpp:88-169` —
  `GEMM_TPARAM_MACRO`, tile-param selection per device tier.  M1 Max
  hits the Medium-device fallback at lines 163-168 with no cache
  awareness.
- `mlx-main/mlx/backend/metal/matmul.cpp:842,1225` —
  `steel_matmul_axpby` dispatcher, `Matmul::eval_gpu`.
- `mlx-main/mlx/backend/metal/device.cpp:485-520,828,841` —
  arch suffix detection (M1 Max returns `'s'`), NAX availability
  (requires `arch_gen >= 17`, M1 is 14).
- `mlx-main/mlx/backend/metal/kernels/steel/gemm/loader.h:25-134` —
  `BlockLoader` template, the actual seat of the cliff.  Walks K via
  `src += tile_stride` with no L2-blocking logic.
- `mlx-main/mlx/backend/metal/kernels/steel/gemm/gemm.h:38-46,48-61` —
  `tgp_mem_size` calculation (~5 KB for Medium tier, well under
  32 KB Metal limit so not the bottleneck), `loader_b_t`
  parameterization on `transpose_b`.
- `mlx-main/mlx/backend/metal/kernels/steel/gemm/mma.h:440-505` —
  `BlockMMA`, same on both paths (so the cliff isn't in the MMA).

**External sources** (M1 Max GPU specs):

- Chester Lam / Chips & Cheese — *iGPU Cache Setups Compared, Including M1* — per-cluster L2 reverse-engineering.
- Asahi Linux — *Tales of the M1 GPU*, *Apple GPU (AGX) docs* — 16 KB page size confirmed.
- Dougall Johnson — `applegpu/docs.html` — G13 architecture reference, simdgroup-per-core counts.
- Philip Turner — `metal-benchmarks` — 8 KB L1 measurement, simdgroup throughput.
- AnandTech — M1 Max performance review — 48 MB SLC, 408 GB/s bandwidth.
- HN 31020484 — M1 Ultra GPU TLB ~32 MB reach.

**Future levers (tempting but probably not worth it).**  Captured here
as thinking, NOT planned work.  Re-evaluate only if MLX upstream stays
cache-blind through several releases or if a similar cliff appears in a
new layer.

1. **Manual K-splitting on project_out.**  Process K=16384 as K=8192 +
   K=8192 with accumulation.  Each chunk sits on the upside of the
   cliff (full ~8 TF/s pretranspose), combined effective throughput
   should beat the K=16384 pretranspose 7.10 TF/s.  Estimated ceiling:
   ~10-15 % speedup on project_out alone, ~2-3 % wall savings on
   stage 2.
   - **Costs:** custom kernel or graph rewrite (MLX `splitk` won't
     auto-trigger here per the agent's read of `matmul.cpp:919` —
     `min_tmn_threshold = 2048` and our `_tm*_tn = 915 * 256 = 234240`
     vastly exceeds it), accumulation precision drift, ~few days work.
   - **Risks:** the K-loop slope analysis shows even K=8192 isn't at
     ceiling (~7.95 TF/s); the win might be smaller than predicted.
     Also adds memory traffic for the intermediate accumulator.
   - **Verdict:** not worth pursuing.  ~2-3 % is in the noise floor
     we just measured for the FF FP16 work.  Also MLX could land a
     fix upstream any release and obsolete this.

2. **Reduce video M via M-chunking.**  M=14640 → two M=7320 launches
   would halve L2 contention per launch and might escape the cliff
   entirely.
   - **Costs:** ~2× kernel launch overhead, transformer-block surgery
     to chunk attention/FF/residual cleanly.  Cross-attention can't
     be cleanly chunked (mixes M dimensions).
   - **Verdict:** not worth pursuing.  Architecturally invasive for
     unclear benefit.

3. **Upstream a cache-aware tile selector to MLX.**  Patch
   `matmul.cpp:88-169` to add an `ldb`-based heuristic that bumps
   `bk` (from 16 to 32 or 64) when the strided-B span would exceed
   estimated per-cluster L2.  Specifically, if `transpose_b &&
   bn * ldb_bytes > L2_per_cluster_estimate`, increase `bk` so fewer
   B-rows are touched per warp-cycle (at the cost of more arithmetic
   per loaded tile).
   - **Costs:** MLX PR review cycle, hardware-specific tuning that
     would need a "Medium" tier sub-classification, no Apple-published
     cache sizes to anchor the heuristic.
   - **Verdict:** not our PR to write as downstream consumers.  Worth
     mentioning if anyone from MLX team asks about M1 Max perf cliffs.

4. **Belt-and-suspenders layout sanity check.**  Emit a one-line
   warning at model construction time if any `Linear` in the
   transformer has `weight.shape[1] >= 13312` (BF16) / `>= 10240`
   (FP16) AND lacks a pretranspose layout spec.  ~30 lines in
   `loader/transformer_cache.py` or `scripts/generate.py`.
   - **Cost:** trivial.
   - **Benefit:** protects future LTX versions or alternate models
     from silently regressing if a new large-K matmul is introduced
     without a layout audit.
   - **Verdict:** could land cheaply; current LTX-2.3 doesn't need it
     (project_out is the only K≥13312 linear in the stack and it's
     already pretransposed).  Defer until a non-LTX model is loaded
     into this codebase or someone reports a cliff regression.

### 2026-05-23: FF FP16 (--video-ff-dtype, --audio-ff-dtype) `[PROMOTED]`

Implemented `--video-ff-dtype float16` and `--audio-ff-dtype float16`
as cache-baked opt-ins.  Lands in `PERFORMANCE.md` matrix rows and the
"2026-05-23: FF compute in FP16" Recent Sessions entry.

**Headline:** `--video-ff-dtype float16` saves **-4.6 % bakery
384x640x20s wall** (8m 00s → 7m 38s) at cosine sim ~0.71 vs BF16.
`--audio-ff-dtype float16` adds no measurable wall savings (within
3 s noise on the same workload) but is exposed as a curiosity opt-in;
audio quality unchanged (latent cos sim ≥0.997 vs the video-only-FP16
reference, perceptually unchanged by listening test).

The microbench prediction matches production within 1 s.  Per-matmul
FP16 win ~15 %, FF share of stage-2 compute ~22 % → 3.3 % expected
wall savings; observed denoise savings ~12 s on ~6m 27s denoise = 3.1
%.  Remaining ~10 s of headline 22 s "total wall" delta is noise in
pre/post phases (audio decode, save, encode).

**Auto-pair is mandatory for FP16:** `_ensure_ff_layout_for_dtype` /
`_ensure_audio_ff_layout_for_dtype` in `scripts/generate.py` force
`project_in:pretranspose,project_out:pretranspose` whenever the
corresponding FF dtype is FP16.  Two reasons stack: (a) without
dtype-baked pretransposed cache, project_in does FP16 × BF16 → FP32
mixed-dtype promotion fallback; (b) FP16 has a deeper BlockLoader
cliff than BF16 (see characterization entry above) — the same K=16384
naive path runs 4.95 TFlops/s in FP16 vs 5.86 TFlops/s in BF16, so
FP16 naive is actually 18 % *slower* than BF16 naive on this shape.
Both failure modes are closed by the auto-pair.

**Attention FP16 was tried and dropped.**  `--video-attn-dtype float16`
gave +18 s regression on bakery despite matmuls being faster.  Two
contributing factors: (a) 4 BF16↔FP16 boundary casts per attention block
(~500 attention calls / generation) dominate the per-matmul win;
(b) `mx.fast.scaled_dot_product_attention` doesn't accept FP16 inputs
cleanly, forcing additional cast at the SDPA boundary.  Per-projection
matmul savings exist (1.18× per the bench) but they don't compose into
a wall-time win once the SDPA boundary tax is paid.  Flag removed in
2026-05-23 cleanup.

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
four attention projections are tied within ±1 %.  A standard smoke run
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

### 2026-05-28: Stage-2 video self-attn K/V token reduction `[ABANDONED]`

**Bottom line:** real speed win, unacceptable quality loss.  Do not wire
this into `generate.py`.

Experiment: `scripts/stage2_harness.py` gained a diagnostic-only
`--stage2-video-attn-kv-pool HxW` path that spatially reduces K/V tokens
only for stage-2 D128 video self-attention.  The guard behaved correctly:
on the 576x320x721 stage-2 harness it hit exactly one video
self-attention call per block per pooled step (`48 layers * steps`) and
fell back for D64/audio and cross/text attention.

Timing evidence:

| Variant | Measured stage-2 step | Denoise wall | K/V hits |
| --- | ---: | ---: | --- |
| baseline | 96.425s | not run full quality in this pass | none |
| `1x2` mean, all 3 steps | 80.296s | 4m 04.7s | 144 / 864 no-mask calls |
| `1x2` mean, first 2 steps only | not bench-mode isolated | 4m 20.7s | 96 / 864 no-mask calls |

The all-step `1x2` path saved ~16.7% on the measured stage-2 step, but
the decoded output was a blurred mess.  Limiting pooling to the first
two stage-2 steps did not recover quality: the output still showed
smearing and vertical-line ghosting.  The diagnostic count confirmed the
final stage-2 refinement step was full-resolution (`budget` fallback for
the last 48 video self-attention calls), so the early pooled trajectory
had already moved into the wrong basin.

Failure mode is consistent with the operation.  `1x2` mean pooling
averages adjacent latent-width tokens, which low-passes horizontal detail
and creates vertical-edge ghost/echo artifacts.  `2x1` would predictably
move the smear axis, `2x2` should be worse, and stride selection would
likely trade blur for aliasing.  This is not a VAE artifact: the harness
was corrected to match the real distilled path (`--vae-decoder native`,
auto tiling `temporal=256/8`, streaming VAE decode into VideoToolbox)
before the quality verdict.

The code path is intentionally kept as a harness diagnostic / negative
evidence tool, not a production candidate.

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
  ran a one-stage smoke A/B (288x512x721, 8 steps, seed 42) baseline vs
  experimental.  Speed: 75.9 → 74.5 s/it = -1.85 %, but baseline ran
  with the older `project_in+project_out+full-attention` layout stack
  vs the trimmed default — comparison confounded, real RoPE delta is
  within noise of single-sample runs.  Quality: same-seed output
  shows visible same-seed drift.  Precision change cascades into the render through
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

### 2026-05-28: MLX STEEL attention wrapper retile -- BQ64/BK32 wins on M1 Max `[DEFAULT]`

Follow-up to the abandoned SDPA tile hypothesis.  The D-sweep was too
indirect: it tested MLX's built-in `bk` selection through different head
dimensions, but it did not directly test the actual LTX `D=128` kernel
with a larger Q tile.

Implementation:

- Added `LTX_2_MLX/kernels/steel_attention.py`.
- The default path now uses the lean LTX-specific STEEL subset in
  `LTX_2_MLX/kernels/metal/` (no mask, no causal, no sinks, B=1/H=32,
  D=64/128).  The older compact subset and full vendored MLX snapshot have
  been moved to `archive/steel_attention/` as historical parity references.
- Apple's MIT license notice is preserved in
  `LTX_2_MLX/kernels/STEEL_ATTENTION_LICENSE.md`; the Metal resources also
  carry SPDX license comments.
- Uses `mx.fast.metal_kernel(..., ensure_row_contiguous=False)` so the
  kernel accepts the real reshape/transpose strides.
- Emits a row-contiguous physical `(B, L, H, D)` output and returns
  `.transpose(0, 2, 1, 3)`, matching MLX full-attention's physical output
  layout trick.
- Specializes no-mask `BQ=64, BK=32` for `D=128` and `D=64`.
- Default-on for supported no-mask D128/D64 shapes.  Disable with
  `LTX_DISABLE_STEEL_ATTN=1` or `LTX_STEEL_ATTN=0`.
  `LTX_STEEL_ATTN_PROBE=1` prints `hit_d128`, `hit_d64`, fallback
  reasons and sample shapes.
- D64 is default-on; use `LTX_STEEL_ATTN_DISABLE_D64=1` only for
  bisects.

Compact-source follow-up (2026-06-01):

- Removed the runtime local-MLX-reference source splicer from the default
  module path.
- `steel_attention.py` became just the shape gate and kernel launcher;
  `_steel_attention_ltx.py` was a 16-line resource loader; the compact Metal
  subset is split into a 941-line header and 277-line body fragment.
- `_steel_attention_vendor.py` was also just a resource loader; the full
  retile fallback lived in `steel_attention_vendor_header.metal` and
  `steel_attention_vendor_body.metal`.
- Integrated default parity vs the full retile fallback: max_abs=0 for
  `(1,32,1504,64)`, `(1,32,8784,128)`, and `(1,32,35136,128)`.
- Integrated timing on M1 Max:
  `(1,32,8784,128)` stock 212.915 ms, compact 198.355 ms, retile 196.568 ms;
  `(1,32,35136,128)` stock 3411.369 ms, compact 3051.386 ms, retile
  3055.026 ms.

Lean-source follow-up (2026-06-01):

- The default was narrowed again to a BF16-only lean subset: 207-line header
  plus 207-line body.  It keeps only the LTX-2.3 no-mask D64/D128 path and
  routes FP16/unsupported calls back to stock MLX.
- The Python launcher first stopped eager-loading the older compact source and
  full vendored fallback.  MLX attribution now lives in
  `STEEL_ATTENTION_LICENSE.md` rather than runtime Python constants.
- Lean smoke vs the post-fp64-RoPE retile baseline was bit-exact for all saved
  sidecars: stage-1/stage-2/final video and audio latents plus text
  conditioning (`compared=22 exact=22`).  Wall time was in the same noise band
  as retile/compact: lean 8m55.7s, retile 8m55.6s, compact 8m58.4s.

Live-path simplification follow-up (2026-06-01):

- Removed `LTX_STEEL_ATTN_IMPL` from the runtime dispatcher and archived the
  compact/vendor resources outside the package.  The live kernel path is now
  lean-only; use git history or `archive/steel_attention/` for old parity
  archaeology.

Lean-kernel shave probes (2026-06-01):

- All probes below were in-memory `mx.fast.metal_kernel` variants against the
  live lean Metal source, with repo files untouched unless a change survived.
- D128 stage-1 `(1,32,8784,128)`, warmup=4, interleaved iters=16:
  `no_pre_q`, `no_final_bar`, `no_both_bar`, `skip_first_k_bar`,
  `fused_exp_sum`, `scalar_state`, and `scalar_fused` were exact vs the live
  kernel.  The only sub-1% apparent win was `no_both_bar` at `0.9983x`, and it
  did not survive larger stage-2 checking.  Fused/scalar softmax state variants
  were slower (`1.011x`-`1.021x`).
- D128 stage-2 `(1,32,35136,128)`, warmup=1, interleaved iters=3:
  `no_both_bar` was exact but slower (`1.0047x`); `fused_exp_sum` was exact but
  slower (`1.0159x`).  Reject both for the real wall-time shape.
- D64 no-mask shapes were exact for the same variants.  Results were neutral:
  self `1504` stayed ~`1.000x`; cross-length `1504->8784` and `8784->1504`
  showed at most ~1% wins/losses depending on direction, not enough to justify
  added branches in the lean source.
- BK sweep: `BK=16` was slightly different math (`max_abs=0.000244141`) and
  slower on D128 stage-1 (`1.0717x`).  `BK=64` exceeded M1 Max threadgroup
  memory (`35840 > 32768`).  Keep `BK=32`.
- Vectorized BF16 load variants: `vec2_load` and `vec4_load` were exact.
  Stage-1 D128 showed a tiny noisy `vec2_load` apparent win (`0.9964x`), but
  stage-2 D128 rejected it (`vec2_load=1.0044x`, `vec4_load=1.0109x`).
- Removing simdgroup execution barriers around the QK/PV MMA loops was exact
  but slower/noisy on D128 stage-1 (`1.004x`-`1.036x`).  Keep the barriers; they
  are cheap scheduling guardrails even though they are not threadgroup memory
  barriers.
- Literalizing the scale expression was exact and neutral (`0.9999x`).
  Replacing final divide with reciprocal-multiply drifted (`max_abs=0.000244141`)
  for only ~`0.16%`, so keep the divide.
- Writing output directly as physical `(B,H,L,D)` was exact, but the full caller
  path was slower (`1.0124x`) because the current physical `(B,L,H,D)` write plus
  return transpose matches the caller's immediate transpose/reshape layout trick.

Conclusion for this first shave batch: the lean STEEL kernel is not just "lean
in the advertising sense".  Simple local rewrites inside the existing loop were
noise or slower.  The next useful search direction was a structural tail
specialization rather than another scalar expression tweak.

Follow-up investigation (2026-06-01):

- `mx.compile` around reshape -> STEEL SDPA -> reshape still helps.  On the
  actual 576x320 stage-2 video self-attention length (`T=16380`, D128),
  compiled core was exact and measured `716.8 ms` vs inline `728.5 ms`
  (`0.984x`), so keep the wrapper.
- Full video self-attention projection-layout retry at `T=16380` was exact but
  negative: q/k/v pretranspose `1.0027x`, to_out pretranspose `1.0061x`,
  gate-logits pretranspose `1.0352x`, all attention projections `1.0441x`.
  Keep attention layout default empty.
- Forced-eval synthetic self-attention breakdown at `T=16380`:
  Q/K/V projections ~21%, STEEL SDPA core ~70%, to_out ~7%, RoPE ~2%, qk norm
  and gate work tiny.  This says remaining self-attention wins must mostly come
  from SDPA itself, not Python dispatch or pointwise glue.
- Single wide self-attention QKV GEMM (`4096 -> 12288`) was exact and won the
  projection-only slice by ~1.9%, but the win collapsed after qk norm, RoPE, and
  SDPA (`0.9984x` through the core).  Not worth the loader/cache complexity.
- Stage-2 FF at `T=16380` confirmed the current default: project_out-only
  pretranspose `563.7 ms`, both project_in+project_out pretranspose `574.5 ms`,
  no pretranspose `673.5 ms`, compiled FF chain `565.6 ms`.  Keep project_out
  pretranspose only.
- Stage-2 FF fused-kernel upper bound: project_in `272.7 ms`, project_out
  `278.3 ms`, GELU-only hidden pass `22.9 ms`, stock FF `563.7 ms`.  A heroic
  fused FF kernel that only deletes the hidden GELU pass is bounded around 4%
  per video FF call, much less at whole-denoise scale.

Integrated stage-2 primitive attribution (2026-06-01):

- Command shape: saved 576x320 stage-1 latents, `--bench-mode 2`,
  `--profile-transformer-steps 2`, representative blocks
  `0,8,16,24,32,40,47`.  Block 47's `entry sync` is not a real block-47 cost;
  it flushes deferred work from unprofiled blocks 41-46.  Use the clean blocks
  `0,8,16,24,32,40` for percentages.
- Approximate share of the stage-2 transformer/denoise step:

  | Bucket | Share |
  | --- | ---: |
  | Video self-attn SDPA | 29.2% |
  | Video FF GEMMs | 27.9% |
  | Video self-attn Q/K/V/out projections | 13.4% |
  | Video text cross-attn | 11.3% |
  | Audio-video cross-attn | 11.1% |
  | Video self-attn RoPE | 3.7% |
  | Video FF GELU | 1.7% |
  | Audio FF | 1.2% |
  | Misc / rounding | 0.4% |

- Rolled up: video self-attention is about 46.7%, video feed-forward about
  29.7%, text plus A/V cross-attention about 22.5%, and everything else about
  1%.  Remaining same-math wins need to come mostly from video SDPA and
  GEMM-heavy FF/projection work.  RoPE is measurable but too small to carry a
  major whole-run win by itself.

Split-K-tail follow-up (2026-06-01):

- The real 576x320 stage-2 token count is `T=16380`, which is `4` short of a
  `64` Q-tile multiple and `28` into the final `BK=32` K tile.  Neighbor timing
  showed the Q tail is basically free (`L=16352`, AlignK=true, was ~0.4% off
  aligned per L^2), while the K tail carried a visible tax (`L=16380` was ~3%
  off aligned per L^2).
- Patched the lean body so the full K tiles run through a branch-free loop and
  the one partial K tile is handled by an explicit epilogue.  This keeps exact
  online-softmax math while avoiding `kb == NK_aligned` checks and tail masks in
  every full K tile.
- Old-vs-new Metal body parity was exact (`max_abs=0`) for D128 stage1
  `8784`, D128 stage2 `16380`, aligned neighbor `16384`, D64 self `1504`, and
  D64 cross-length `1504->8784`.
- Targeted old-vs-new timing after patch:
  D128 stage1 `211.5 -> 208.2 ms` (`0.984x` median),
  D128 stage2 `695.6 -> 685.9 ms` (`0.986x` median, `0.971x` mean),
  D64 `1504->8784` neutral (`17.35 -> 17.35 ms` median).
  Paired stage-2 harness bench-mode on the saved 576x320 kitten latents measured
  old source `99.982s` vs patched source `99.006s` for stage-2 step 2.
  This is a real same-math win, but still modest at whole-run scale.
- Keep caveat: this is shape-dependent.  The K-tail split helps when the
  attention key length is not a multiple of `BK=32`; for distilled stage-2
  video self-attention that length is `latent_frames * (width / 32) *
  (height / 32)`.  At 30s/24fps (`721` frames -> `91` latent frames),
  576x320 gives `T=16380` (`T % 32 = 28`) and 768x448 gives `T=30576`
  (`T % 32 = 16`), so both keep a partial K tile.  1024x576 gives
  `T=52416` (`T % 32 = 0`), so the expensive stage-2 path should be neutral
  rather than faster.  We keep the split because it is exact, auditable, and
  exposes the tail/alignment surface for common unaligned shapes; it should not
  be sold as a universal whole-run win.
- Rejected Q-tail follow-up: splitting `T=16380` into a `16320` aligned-Q prefix
  plus a `60`-query tail was exact (`max_abs=0`) but slower.  Direct Metal body
  timing measured baseline `667.8 ms` median vs Q-split `701.2 ms` median
  (`1.05x`).  The extra kernel launch/concat costs more than removing the
  Q-tail branch, matching the earlier neighbor result that Q-tail was basically
  free.  Do not port a two-call Q split.
- Rejected QK prologue tweak: using `simdgroup_multiply` for the first
  head-dimension slice after `Stile.clear()` looked like a D128 win, but that
  result was measured against a source variant already missing the full-loop
  K/V tile advances.  After restoring the advances and anchoring to the
  last-known-good split-K source, the prologue was exact but not faster:
  stage1 D128 split-good `44.9 ms` vs prologue `46.3 ms`, stage2 D128
  split-good `680.4 ms` vs prologue `683.7 ms`.  The prologue was removed; keep
  the simpler `dd=0..TD` multiply-accumulate loop.
- Possible next exact-kernel option: a standalone Metal/MLX extension wrapper,
  not a raw PyObjC dispatch.  The only credible upside is compiler/control-plane
  control that `mx.fast.metal_kernel` does not expose: explicit
  `max_total_threads_per_threadgroup(256)`, a tiny fixed params surface instead
  of generated shape/stride ABI loads, and precompiled/specialized entry points
  for D128 aligned-K, D128 tail-K, and D64.  The risk is equally real: if this
  leaves MLX's command stream or adds synchronization/buffer-lifetime glue per
  call, command-buffer overhead can erase the gain.  Treat it as a bounded
  wrapper experiment for kernel attributes and fixed params, not as a generic
  rewrite.
- Shallow wrapper-adjacent probe: MLX's generator confirms there is no public
  `mx.fast.metal_kernel` hook for kernel attributes; it always emits
  `[[kernel]] void custom_kernel_*` and only auto-adds thread-position
  attributes referenced in the body.  A fixed-stride/fixed-shape-body variant
  that replaced `Q_shape/K_shape` and `*_strides` ABI loads with scalar
  `qL/kL` inputs plus hardcoded production transpose strides was exact
  (`max_abs=0`) but negative on D128: stage1 `1.002x`, stage2 `1.004x`.
  D64 `1504` won (`0.972x`), but that is not the wall-time path.  Do not port
  fixed strides through `mx.fast.metal_kernel`; if revisited, it belongs in a
  real wrapper/extension test with kernel attributes too.
- Smoke-test correction: a cleanup pass briefly removed the `loader_k.next()`
  and `loader_v.next()` advances from the branch-free full-K loop, mistaking
  them for dead epilogue increments.  That was wrong: without those advances
  every full K/V tile rereads the first block, so the 8+3 smoke diverges from
  stage 1 onward.  The slow `kitten_smoke_qk_mul_prologue_20260601_003004` run
  exposed it: text sidecars were exact, but stage-1/final latents were not, and
  denoise was slower (stage1 `155.8s`, stage2 `303.1s`) than the prior exact
  split-K smoke (`150.3s`, `283.7s`).  Restoring the advances makes the current
  source exact again versus the pre-cleanup split-K source for D128 video
  self-attention and D64 audio/cross-modal shapes; D128 vs stock MLX SDPA is
  back to tiny BF16-level delta (`max_abs <= 4.9e-4` in the probe).  Do not
  remove the full-loop advances; only post-epilogue advances are dead.
- Kernel-parity rule from that miss: validating new source only against the
  immediately previous source can prove "same as broken".  For future STEEL
  edits, use three anchors before trusting parity: immediate previous source,
  last-known-good source/saved sidecar, and stock MLX SDPA on real hot-path
  shapes.  Full 8+3 smoke NPZ comparison remains the final guard.

Validation:

- Small parity:
  - D128 self: `max_abs=0.000488`
  - D64 self: `max_abs=0.000000`
  - D64 cross-length A2V/V2A smoke: `max_abs=0.000000`
- Integrated `_attention_core` synthetic:
  - stage1 D128: stock `230.3 ms`, steel `199.6 ms`
  - stage2 D128: stock `3730.2 ms`, steel `3129.2 ms`

Production-ish smoke, 576x320, 721 frames, distilled 8+3,
BF16, audio on, seed 42:

| Run | Total | Stage 1 denoise | Stage 2 denoise | Denoise total |
|---|---:|---:|---:|---:|
| fresh stock | 562.6s | 155.1s | 310.2s | 465.4s |
| D128 only | 546.5s | 152.7s | 292.0s | 444.6s |
| D64 default | 535.7s | 151.5s | 286.9s | 438.4s |

The D64 microbench was neutral to slightly slower in isolation, but the
integrated run won: stage 2 improved another 5.1s over D128-only and
23.4s over stock.  Probe for the D64-default run:

```text
hit_d128          4
hit_d64           6
fallback          2
fallback reason: mask
```

Remaining uncovered attention is masked text cross-attention; leave it on
stock MLX unless a future masked wrapper is explicitly benchmarked.  This
result supersedes the 2026-05-18 "custom FlashAttention-2 Metal kernel"
abandonment note for this narrow no-mask STEEL-retile path.

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

- (a) **DEAD — no INT8 hardware path exists on M1.**  Earlier text said
  "a from-scratch native-INT8 kernel using M1's `dot4I8Packed` intrinsics
  (~21 TFlops/s theoretical INT8 peak)" could maybe win.  That claim was
  wrong; verified empirically 2026-05-23 via
  `scripts/bench_int8_alu.py`.  Findings:
  * **Metal Shading Language has no `dot(char4, char4)` intrinsic** —
    compile error.  Anyone wanting packed INT8 dot must manually unroll
    4 scalar mul+add ops.
  * **`simdgroup_matrix<T>` (M1's tensor accelerator) accepts only
    `half`/`bfloat`/`float`, not `char`** — per Apple's MSL spec section
    2.4.  No INT8 path through the GEMM accelerator.
  * **Microbench result** (32k threads × 1M MACs each on M1 Max):
    FP16 scalar fma = 0.76 TOps/s, INT8 scalar mac = 0.63 TOps/s, INT8
    4-way unrolled "dot" = 0.80 TOps/s (which is 4 ops/iter at 3.8×
    slower per iter than FP16 single MAC — per-MAC throughput equal,
    no advantage).
  * **`mpp::tensor_ops::matmul2d` (Metal 4 cooperative-tensor with
    INT8 support) is M5+ only** — confirmed via Cider
    (https://github.com/Mininglamp-AI/cider), explicitly skips C++
    build on M4 and below.
  Net: M1's INT8 lives on the same scalar ALU as FP16/INT32.  No path
  to INT8 > BF16 throughput exists on Apple7 silicon.
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
