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
| SDPA at (1, 32, 8784, 128) BF16 isolated | **212.38 ms** | 214.42 ms | |
| SDPA at same shape from 05-17 per-call probe | 230 ms | — | live pipeline |

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
