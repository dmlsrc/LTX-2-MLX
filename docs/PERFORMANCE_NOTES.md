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

### 2026-05-17: Microbench results for FF / attention candidate optimizations `[OPEN, ready to promote]`

Concrete numbers from `scripts/bench_ff_microbench.py` — clean run with
50 iters, 10-iter warmup, `bench-process-watch.sh` killing background
macOS GPU agents (`mediaanalysisd`, `photoanalysisd`), Claude Code +
other Metal-using apps closed.  BF16, M1 Max, shape T=8784 D=4096 H=32
D_head=128 FF_inner=16384.  Override the back-of-envelope estimates
from the earlier research scan.

| Bench | mean / call | p99 | notes |
|---|---|---|---|
| `nn.gelu_approx` at (1, 8784, 16384) BF16 | **10.43 ms** | 10.49 ms | very tight tail |
| `x * 1.0` (pure bandwidth reference) | 1.82 ms | 1.91 ms | |
| 3× separate Q/K/V addmm (current) | **111.75 ms** | 113.18 ms | |
| 1× packed Q/K/V addmm + split | **113.10 ms (-1.2 % SLOWER)** | 113.49 ms | |
| FF chain naive (eval per op) | 309.73 ms | 311.56 ms | |
| FF chain lazy (production path) | **309.16 ms** | 310.76 ms | |
| FF chain `mx.compile` wrapped | **309.51 ms (-0.11 % vs lazy)** | 311.80 ms | both within noise |
| SDPA at (1, 32, 8784, 128) BF16 isolated | **212.23 ms** | 213.92 ms | |
| SDPA at same shape from 05-17 per-call probe | 230 ms | — | live pipeline |

**SDPA T-alignment sweep (50 iters, clean run with bench-process-watch):**

| T | aligned to | mean ms | norm (ns/T²) | vs T=8784 | viable padding target? |
|---|---|---|---|---|---|
| 8704 | 128 | 208.36 | 2.7503 | -1.82 % | ❌ truncates |
| 8736 | 32 | 209.42 | 2.7441 | -1.32 % | ❌ truncates |
| 8768 | 64 | 210.89 | 2.7432 | -0.63 % | ❌ truncates |
| **8784** | **16 only** | **212.23** | **2.7505** | **baseline** | n/a |
| 8800 | 32 | 212.63 | 2.7457 | +0.19 % | ✅ +16 — slower |
| 8832 | 128 | 214.27 | 2.7469 | +0.96 % | ✅ +48 — slower |
| 8896 | 64 | 217.19 | 2.7445 | +2.34 % | ✅ +112 — slower |
| 8960 | 128 | 221.05 | 2.7535 | +4.16 % | ✅ +176 — slower |
| 9216 | 128 | 236.52 | 2.7847 | +11.45 % | ✅ +432 — much slower |

**SDPA scales nearly perfectly O(T²) across this range** — normalized cost
spread is only 1.5 % across all 9 T values, and our T=8784 carries only a
0.27 % per-tile-alignment penalty vs the best neighbor (T=8768).  The
MLX `sdpa_full` kernel handles non-aligned T well.  No cliff to dodge.

**Methodology note:** the bench-process-watch loop dropped p99 tails
dramatically — GELU p99 went from 21 ms (noisy run) to 10.5 ms (clean
run), and all other p99s collapsed to within ~1 % of mean.  Means
shifted only slightly (GELU -14 %, Q/K/V -6 %, FF/SDPA <1 %) but the
tighter distribution makes the comparisons more trustworthy.

**Key empirical findings:**

1. **GELU is compute-bound, not bandwidth-bound.**  5.7× the cost of pure
   `x * 1.0` copy at the same shape.  The tanh + multiplies dominate.
   Per-step ceiling for GELU-pass elimination via custom matmul-epilogue
   kernel: **~500 ms / step = 1.10 % of 45.5 s/step.**  Higher than the
   research agent's 0.7 % estimate (which assumed bandwidth-bound GELU),
   but still small.
2. **`mx.compile` does NOT fuse matmul+activation.**  Lazy vs compiled
   differ by 0.11 % over 50 iters — both within noise.  Confirms MLX
   source inspection (`is_fusable()` in `mlx/compile.cpp` excludes
   matmul).  This kills the "wrap FF in mx.compile" idea.
3. **Q+K+V fusion is empirically a regression** at our shape.  Packed
   matmul is 1.2 % slower than 3 separate matmuls — the bandwidth saving
   on reading `x` once (~0.5 ms at 300 GB/s) is dominated by the larger
   output's worse tile alignment + the post-matmul split overhead.  This
   also explains why the earlier `self_qkv:pack` / `kv:pack` experiments
   in `PERFORMANCE.md`'s archive were "neutral-to-tiny".
4. **SDPA tile-floor confirmed.**  212 ms isolated vs 230 ms in
   production-context — the 8 % gap is plausible surrounding-op
   interference (other dispatches competing for GPU, cache state).  The
   tile-floor analysis holds.
5. **FF chain is matmul-bound by a huge margin.**  309 ms total, of which
   ~10 ms (3.4 %) is GELU and ~299 ms (96.6 %) is the two matmuls.  Eval
   barriers between ops add only 0.6 ms — MLX's lazy graph isn't doing
   meaningful fusion magic in the production path either.
6. **Bench vs production FF discrepancy.**  Single FF call in isolation
   = 309 ms → 48 blocks × 309 ms = **14.8 s/step**.  Production sync-mode
   says video_ff = 18.7 s/step.  ~26 % higher in sync-mode production —
   explained by (a) ~13 % signpost/eval-barrier overhead distributed
   across phases (per the 05-17 follow-up's sync-vs-non-sync analysis)
   and (b) weight cache / GPU contention with surrounding ops.
7. **SDPA T-padding is not a lever.**  Sweeping T from 8704 to 9216
   shows our T=8784 sits in the middle of a nearly-flat normalized-cost
   distribution (spread only 1.5 % across the range).  T=8784 carries
   only a 0.27 % per-tile-alignment penalty vs the best neighbor.  Every
   padding-up target is slower (best at T=8800 is +0.19 % = +0.4 ms/call,
   T=9216 is +11.45 %).  MLX `sdpa_full` handles non-aligned T well.

**Revised brutal-efficiency ranking (post-microbench):**

| Lever | Win | Effort | Verdict |
|---|---|---|---|
| Q+K+V fusion | -1 % (regression) | n/a | **DEAD** |
| `mx.compile` around FF | 0 % | n/a | **DEAD** |
| GELU-into-matmul fusion (custom Metal kernel) | ~1.3 % | 1-2 weeks | Small reward for the effort |
| C++ MLX patch adding `TransformGELU` epilogue | ~1.3 % | 1-2 days + MLX rebuild | Same ceiling, less code |
| `mxfp8 project_out 32-47` (already implemented) | ~5 % | hours | **Best BF16-quality trade currently** |
| All-layer `mxfp8 project_out` | ~13 % (draft mode) | hours | Visibly different |
| Custom FlashAttention-2 Metal kernel | ~10 % | 2-4 weeks | Biggest BF16-pure ceiling, hardest |
| **Stop here** | 0 | 0 | **45.5 s/it IS the BF16 floor on M1 Max for this shape** |

**To promote to `PERFORMANCE.md`:**

- Update the "Active brutal-efficiency targets" section in the TL;DR to
  reflect this ranking.  Specifically:
  - Drop Q+K+V fusion as an actionable target (move to "tested neutral or
    removed" matrix row, citing this scratchpad entry).
  - Drop "wrap FF in mx.compile" if it appears anywhere (it doesn't, but
    the framing in the existing matrix row could be updated).
  - Reframe v_ff_inner as "BF16-pure ceiling is ~1.3 %; biggest realistic
    BF16 lever is custom FA-2 kernel; biggest near-term lever is
    mxfp8 project_out quant".
- Move this entry to `[PROMOTED]` once that's done.

### 2026-05-17: Pretranspose default cleanup -- project_out is the only real win `[PROMOTED]`

The `--video-ff-layout project_in,project_out:pretranspose` +
`--video-attn-layout to_q,to_k,to_v,to_out:pretranspose` default stack
was inherited from the early "more pretranspose = more win" framing,
but the isolated BF16 matmul microbench
(`scripts/bench_ff_microbench.py bf16_layout`, clean 50-iter run at
`$SHARED_TEMP_DIR/trace_analysis/bf16_layout_clean.log`) shows the win
is concentrated at a single shape:

| Shape | Naive (no layout) | Pretranspose | Δ | TFlops/s naive → pre |
|---|---|---|---|---|
| FF.project_in (4096→16384) | 147.29 ms | 151.03 ms | **+2.5 %** (regression) | 8.00 → 7.81 |
| **FF.project_out (16384→4096)** | **228.15 ms** | **148.28 ms** | **−35.0 %** (huge) | **5.17 → 7.95** |
| attn.to_q/k/v (4096→4096) | 36.84 ms | 37.22 ms | +1.0 % (tied) | 8.00 → 7.92 |
| attn.to_out (4096→4096) | 36.91 ms | 37.13 ms | +0.6 % (tied) | 7.99 → 7.94 |

**Root cause:** every BF16 matmul reaches ~8 TFlops/s (~77 % of M1 Max
BF16 peak) EXCEPT naive `project_out`, which falls off a kernel-
selection cliff to 5.17 TFlops/s.  The implicit `mx.addmm(b, x, W.T)`
transpose-then-matmul for (K=16384, N=4096) picks a worse
`steel_gemm_*_nt` variant than the explicit `mx.contiguous(W.T)` +
matmul path.  Pretranspose's "production layout win" is entirely
this cliff rescue; other shapes already achieve ~8 TFlops without
pretranspose.

**Memory implications:** none.  The runtime implementation
(`pretranspose_project_*`) materializes the transposed weight then
drops the original (`del self.project_out.weight`).  One weight per
linear at steady state in either orientation.  The on-disk cache
file size is also identical.  See `PERFORMANCE.md`'s archive note:
"current path materializes each transposed weight layer by layer and
drops the original weight immediately afterward... stable around 44GB
process memory."

**Defaults changed in `scripts/generate.py`:**

- `DEFAULT_VIDEO_FF_LAYOUT_SPECS`:
  `(project_in:pretranspose, project_out:pretranspose)` →
  `(project_out:pretranspose,)`.  Help string updated to explain.
- `DEFAULT_VIDEO_ATTN_LAYOUT_SPECS`:
  `(to_out, to_q, to_k, to_v:pretranspose)` → `()` (empty, all off).
  Help string updated.

The flags themselves and the underlying transform code are
**unchanged** -- only the default tuple is shorter.  Users can opt
back into the historical behavior with
`--video-ff-layout project_in:pretranspose,project_out:pretranspose
--video-attn-layout to_out:pretranspose,to_q:pretranspose,...`.

**Expected production impact:** ~6-15 s saved on bakery wall (~0.4-1 %
of 24m 40s).  Tiny but free.  Cache hash changes for both flag
families, so a fresh cache will be built on first run with the new
defaults.

**Verified end-to-end** via kitten smoke (one-stage 288×512, 721
frames, 8 distilled steps):

| Metric | Old defaults | New defaults | Δ |
|---|---|---|---|
| Per-step | 75.9 s/it | **75.4 s/it** | **−0.7%** |
| Denoise total | 10m 07s | **10m 03s** | **−4 s (−0.6%)** |
| Total wall | 11m 34s | **11m 30s** | **−4 s (−0.6%)** |
| Transformer cache TENSOR COUNT (metadata) | 4186 (1344 "layout") | 4186 (96 "layout") | same total; layout-tensor metadata down 14× |
| Transformer cache FILE SIZE on disk | ~38.0 GB | ~38.0 GB | **essentially unchanged** (same weights, just in different orientation) |

Speed delta is within run-to-run noise — at minimum a clean no-
regression result, possibly a small real win.  **The cache file size
does NOT change** — pretranspose stores `mx.contiguous(W.T)` instead
of `W`, same number of bytes either way.  The "1344 → 96 layout
tensors" is metadata about which tensors got the pretranspose
treatment; it's not a disk-size win.  The real benefits of the trim
are: (a) no regression, (b) no false invariant (we no longer carry
defaults that the microbench shows are no-ops), and (c) honest
metadata about what's actually pretransposed.

**Side effect: audio defaults also trimmed.**  When
`DEFAULT_VIDEO_FF_LAYOUT_SPECS` and `DEFAULT_VIDEO_ATTN_LAYOUT_SPECS`
were changed, audio inherited via the shared cache-build path.  See
"Audio pretranspose microbench" entry in Archive — verified safe at
bakery scale; small-T workloads not re-verified.

**Note for next investigator:** the audio FF / audio attn / video→audio
pretranspose stack (`LTX_DISABLE_AUDIO_PRETRANSPOSE=1` to opt out) was
NOT touched.  Original measurement was −11 % at small T (256×256×25)
pre-AdaLN-fix.  Audio microbench at bakery T=502 (see next entry below)
shows audio pretranspose is **neutral within noise across all four
projections** — no cliff to rescue (unlike video project_out), no per-
matmul win.  Recommendation: keep audio defaults as-is because the
historical 11 % win at small T (where dispatch overhead dominates per-
call matmul time) is plausibly real even if it doesn't show up at
bakery scale.  Zero risk to leaving the defaults on (microbench shows
neutral, not regression).

### 2026-05-17: FlashAttention-2 custom Metal kernel for attn_sdpa `[OPEN, low priority]`

Carried over from previous note.  Per the microbench above, SDPA is at the
expected tile-floor (213 ms isolated, 230 ms live).  Custom FA-2 Metal port
is the biggest remaining BF16-pure ceiling (~10 % of step) but multi-week
effort with high risk.  Revisit only after the mxfp8 quant lever has been
deployed and quality validated.

**Open questions:**

- Is there an existing FA-2 Metal port we could lift?  Manual GitHub
  search needed (WebSearch was sandbox-blocked in earlier research).
- What's the worst-case latency of a NAIVE Metal SDPA kernel at our shape
  (T=8784, H=32, D=128)?  Just to know how much headroom Apple's
  hand-tuned `sdpa_full` has over a naive baseline — if it's only 2×,
  hand-rolling a kernel that BEATS it is unlikely.

---

## Promoted

(Entries moved to `PERFORMANCE.md` — kept here as stubs for cross-reference.)

_(empty)_

---

## Archive

(Abandoned investigations kept for negative-result evidence.)

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

The `mx.quantized_matmul` kernel achieves only ~22% of M1 Max INT8
peak (~21 TFlops/s) — well below `steel_gemm`'s ~74% of BF16 peak.
The "bandwidth savings" hypothesis (smaller quantized weights →
fewer bytes read) does not pay off because at our matmul shapes we
are **compute-bound, not bandwidth-bound** — every quantized matmul
replaces a fast steel_gemm with a slower quantized_matmul, and the
dequant overhead is additive.

**Why the historical "mxfp8 ~13% draft mode" was real**: it was
measured against a pre-AdaLN-fix BF16 baseline of 77.8 s/it
(FP32-polluted SDPA path).  The race that mxfp8 was winning was
broken-BF16 vs quant.  Post-AdaLN-fix BF16 is 45.5 s/it (-42%) but
the mxfp8 path got essentially no benefit (it doesn't hit the same
SDPA path).  So the relative ordering flipped.

**Verdict:** mxfp8 is not a draft-mode lever on this hardware/MLX
version.  Until either (a) Apple ships a faster `mx.quantized_matmul`
kernel that closes the gap to `steel_gemm`, or (b) you move to M3+
NAX where fp8 hardware support might bypass the dequant bottleneck,
quantization will always be a net regression at our shapes.  CLI
flags (`--video-ff-quantize`, `--transformer-cache-quantize`) and
their underlying code paths are kept intact for future research and
because the cache infrastructure is reusable for other quant modes
that might land later.

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
floor on M1 Max non-NAX; only a custom Metal kernel or M3+ hardware
will change it.
