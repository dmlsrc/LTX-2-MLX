#!/bin/bash
# A/B/C/D wall-time bench: baseline BF16 vs mxfp8 draft-mode variants.
#
# Four variants run sequentially (no GPU contention) at the same workload:
#   A. baseline (no quant flag) -- reference (production default)
#   B. in-memory FF quant (--video-ff-quantize project_out:mxfp8
#                          --video-ff-layout off)
#                          -- quantizes only FF project_out after BF16 load;
#                             must disable pretranspose layout (CLI rejects combo)
#   C. cache quant mxfp8-blocks (--transformer-cache-quantize mxfp8-blocks)
#                          -- pre-quantized cache for heavy attention + FF
#                             linears; full-resident (no streaming); auto-
#                             disables same-math layouts
#   D. cache quant + pretranspose (--transformer-cache-quantize
#                                  mxfp8-blocks-pretranspose)
#                          -- same as C but packs weight.T BEFORE quantizing
#                             (does the pretranspose layout win stack on quant?)
#
# NOTE on streaming: variants C and D do NOT pass --stream-transformer.
# Block streaming is a constrained-memory mode with known latency overhead
# (~70s/it for r4, ~55s/it for r16 on quiet system, all at the bakery shape).
# Mixing it in would conflate the quant measurement with streaming overhead.
# If you want to specifically test quant+streaming, add --stream-transformer
# via an EXTRA_FLAGS env var (not currently wired -- ask if you need it).
#
# Output: per-step trajectory + total wall comparison.
# Sidecar .npz latents are saved per run so quality diffs are reproducible.
#
# Required env (no defaults -- set per your machine):
#   LTX_REPO          Absolute path to LTX-2-MLX repo root.
#
# Optional env (sane defaults):
#   LTX_VENV_BIN      Path to LTX venv activate script.  If unset, assumes
#                     `python` is on PATH.
#   STEPS             Stop after N stage-1 steps (default 8 = full stage 1
#                     only, ~16 min total for both variants).  STEPS=4 for
#                     a quick sanity check (~10 min).  Unset stays at 8;
#                     to run the FULL bakery (stage 1 + stage 2, ~50 min
#                     total) pass STEPS=99 -- the stop hook will never fire.
#   PREWARM           1 (default) = do a STEPS=1 untimed pre-warm before
#                     each variant's measured run, so the transformer cache
#                     is built and Metal kernels are JIT'd before timing
#                     starts.  Adds ~1-2 min per variant but means the
#                     total-wall delta reflects denoise-only differences,
#                     not cache-build cost.  PREWARM=0 to skip (faster but
#                     mxfp8 variant's wall will include the one-time
#                     cache-build cost of ~25-60s on first run).
#   SKIP_BASELINE     1 to skip variant A.  Useful when you've already
#                     measured baseline at this STEPS/seed.
#   SKIP_B            1 to skip variant B (in-memory FF quant).
#   SKIP_C            1 to skip variant C (cache quant mxfp8-blocks).
#   SKIP_D            1 to skip variant D (cache quant pretranspose).
#   AB_OUTDIR         Output dir (default $SHARED_TEMP_DIR/trace_analysis/
#                     mxfp8_<ts> when SHARED_TEMP_DIR is set, else
#                     ${TMPDIR:-/tmp}/mxfp8_<ts>).
#   SEED              Seed (default 124, the bakery reference).
#   HEIGHT, WIDTH     Output res (default 576, 1024).
#   NUM_FRAMES        Frame count (default 481, the bakery reference).
#
# Usage:
#   export LTX_REPO=/path/to/LTX-2-MLX
#   export LTX_VENV_BIN=/path/to/venv/bin/activate
#   bash scripts/bench_mxfp8_draft.sh            # 8 steps, ~16 min total
#   STEPS=4 bash scripts/bench_mxfp8_draft.sh    # 4 steps, ~10 min total
#   STEPS=99 bash scripts/bench_mxfp8_draft.sh   # full bakery, ~50 min total

set -u

: "${LTX_REPO:?LTX_REPO must be set to the LTX-2-MLX repo root}"
if [ ! -d "$LTX_REPO" ]; then
    echo "ERROR: LTX_REPO does not exist: $LTX_REPO" >&2
    exit 1
fi

STEPS="${STEPS:-8}"
SEED="${SEED:-124}"
HEIGHT="${HEIGHT:-576}"
WIDTH="${WIDTH:-1024}"
NUM_FRAMES="${NUM_FRAMES:-481}"
LTX_VENV_BIN="${LTX_VENV_BIN:-}"
PREWARM="${PREWARM:-1}"

# Default output dir: prefer $SHARED_TEMP_DIR/trace_analysis/mxfp8_<ts>
# (shared, persistent), fall back to ${TMPDIR:-/tmp}/mxfp8_<ts>.
if [ -z "${AB_OUTDIR:-}" ]; then
    if [ -n "${SHARED_TEMP_DIR:-}" ]; then
        AB_OUTDIR="${SHARED_TEMP_DIR}/trace_analysis/mxfp8_$(date +%Y%m%d_%H%M%S)"
    else
        AB_OUTDIR="${TMPDIR:-/tmp}/mxfp8_$(date +%Y%m%d_%H%M%S)"
    fi
fi

mkdir -p "$AB_OUTDIR"
chmod g+w "$AB_OUTDIR" 2>/dev/null || true

# ----- prompt: canonical BAKERY from docs/TEST_PROMPTS.md -----
BAKERY='Live-action footage shot on a consumer mirrorless camera inside a quiet neighborhood bakery in soft morning light. A flour-dusted wooden table fills the foreground. On the left, a young baker in a blue apron kneads dough slowly. On the right, an older baker places four golden croissants onto a metal tray. The camera begins close on the dough, slowly pulls back to reveal both bakers, then gently pans right toward the croissants. The young baker looks at the tray and says, "Four for the window, two for the regulars." The older baker smiles and replies, "Save the best one for Mrs. Ito." Their voices are close, dry, natural, and clearly recorded. The room is very quiet, with a soft low indoor hush. The dialogue is the main sound. The background stays minimal and smooth. The frame contains only the bakery, the bakers, the table, the tray, and warm morning light. Realistic lip movement, handheld camera with gentle natural sway, warm shadows, detailed textures.'

echo "================================================================"
echo "mxfp8 draft-mode A/B/C/D @ STEPS=$STEPS  PREWARM=$PREWARM"
echo "Output dir: $AB_OUTDIR"
echo "Workload: distilled stage 1, ${WIDTH}x${HEIGHT} -> $((WIDTH/2))x$((HEIGHT/2)) latent, $NUM_FRAMES frames"
echo "Mode: production lazy-graph (no signposts, no eval barriers)"
if [ "$PREWARM" != "0" ]; then
    echo "Pre-warm: ENABLED (STEPS=1 untimed pass per variant -- builds cache + JITs Metal)"
fi
echo "================================================================"
echo

cd "$LTX_REPO" || exit 1
if [ -n "$LTX_VENV_BIN" ]; then
    # shellcheck disable=SC1090
    source "$LTX_VENV_BIN"
fi

run_variant() {
    local label="$1"
    local prefix="$2"
    shift 2
    local extra_flags=("$@")

    # Pre-warm: STEPS=1, untimed, output discarded.  Builds the transformer
    # cache (if missing for this flag combination) + JITs Metal kernels so
    # the timed run reflects denoise-only differences.
    if [ "$PREWARM" != "0" ]; then
        echo "[$label] pre-warming (cache build + Metal JIT) ..."
        LTX_PROFILE_STOP_AFTER_STEPS=1 \
        caffeinate -di python LTX_2_MLX/generate.py "$BAKERY" \
          --pipeline distilled \
          --height "$HEIGHT" --width "$WIDTH" \
          --duration 20 --seed "$SEED" \
          --generate-audio \
          --fast-mode \
          --output-prefix "${prefix}_prewarm" \
          ${extra_flags[@]+"${extra_flags[@]}"} \
          > "$AB_OUTDIR/${prefix}_prewarm.log" 2>&1
        local pre_rc=$?
        if [ $pre_rc -ne 0 ]; then
            echo "[$label] pre-warm failed (rc=$pre_rc) -- see $AB_OUTDIR/${prefix}_prewarm.log"
            return $pre_rc
        fi
    fi

    echo "[$label] starting timed run..."
    local t0=$(date +%s)
    LTX_PROFILE_STOP_AFTER_STEPS="$STEPS" \
    caffeinate -di python LTX_2_MLX/generate.py "$BAKERY" \
      --pipeline distilled \
      --height "$HEIGHT" --width "$WIDTH" \
      --duration 20 --seed "$SEED" \
      --generate-audio \
      --fast-mode \
      --save-all-sidecars \
      --output-prefix "$prefix" \
      ${extra_flags[@]+"${extra_flags[@]}"} \
      > "$AB_OUTDIR/$prefix.log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local wall=$((t1 - t0))
    echo "[$label] done (rc=$rc, wall=${wall}s)"
    echo
    echo "$wall" > "$AB_OUTDIR/$prefix.wall"
    return $rc
}

# A. baseline -- current production default (BF16 + FF pretranspose layout).
# Skip with SKIP_BASELINE=1 if you've already measured it at this STEPS/seed
# (the summary table will note it as "not run").
if [ -z "${SKIP_BASELINE:-}" ]; then
    run_variant "A baseline" "bench_baseline"
else
    echo "[A baseline] SKIPPED (SKIP_BASELINE=$SKIP_BASELINE)"
    echo
fi

# B. all-layer in-memory mxfp8 project_out.  The CLI rejects combining
# --video-ff-quantize with --video-ff-layout (they mutate the same weights),
# so we must pass `--video-ff-layout off` for variant B.  This is what
# in-memory FF quant looks like in practice: you swap the pretranspose
# layout win for the quant win.
if [ -z "${SKIP_B:-}" ]; then
    run_variant "B mxfp8 all-layer" "bench_mxfp8_all" \
        --video-ff-quantize project_out:mxfp8 \
        --video-ff-layout off
else
    echo "[B mxfp8 all-layer] SKIPPED (SKIP_B=$SKIP_B)"
    echo
fi

# C. cache quant mxfp8-blocks (full-resident, NO streaming): rebuilds the
# transformer cache with mxfp8 block-quantized weights for heavy attention
# and FF linears.  Auto-disables same-math layouts on the quantized weights.
# Running full-resident isolates the quant effect from streaming overhead.
if [ -z "${SKIP_C:-}" ]; then
    run_variant "C mxfp8-blocks" "bench_mxfp8_blocks" \
        --transformer-cache-quantize mxfp8-blocks
else
    echo "[C mxfp8-blocks] SKIPPED (SKIP_C=$SKIP_C)"
    echo
fi

# D. cache quant mxfp8-blocks-pretranspose (full-resident, NO streaming):
# same as C but packs weight.T before quantizing and calls quantized_matmul
# with transpose=False.  Probes whether the pretranspose layout win still
# applies on top of quant.
if [ -z "${SKIP_D:-}" ]; then
    run_variant "D mxfp8-blocks-pretranspose" "bench_mxfp8_blocks_pre" \
        --transformer-cache-quantize mxfp8-blocks-pretranspose
else
    echo "[D mxfp8-blocks-pretranspose] SKIPPED (SKIP_D=$SKIP_D)"
    echo
fi

# Summary
python3 - "$AB_OUTDIR" "$STEPS" <<'PYEOF'
import re, sys
from pathlib import Path

outdir = Path(sys.argv[1])
steps = int(sys.argv[2])

variants = [
    ("baseline",        "bench_baseline"),
    ("mxfp8 all-layer", "bench_mxfp8_all"),
    ("mxfp8-blocks",    "bench_mxfp8_blocks"),
    ("mxfp8-blocks-pre","bench_mxfp8_blocks_pre"),
]

def parse_steps(log_path):
    """Extract per-step times from LTX-2-MLX tqdm progress."""
    out = {}
    with open(log_path) as f:
        log = f.read()
    for raw in re.split(r"[\r\n]+", log):
        m = re.search(r"STEP1\s+(\d+m\s*\d+\s*s|\d+\.\d+s|\d+s)\b", raw)
        if m and "s/it" in raw:
            sit = re.search(r"(\d+\.\d+|\d+)\s*s/it", raw)
            step_n = re.search(r"\]\s*(\d+)/(\d+)", raw)
            if sit and step_n:
                out[int(step_n.group(1))] = float(sit.group(1))
    return sorted(out.items())

def hms(s):
    if s < 60: return f"{s:.1f}s"
    m, s = divmod(s, 60)
    return f"{int(m)}m {int(s):02d}s"

def read_wall(prefix):
    p = outdir / f"{prefix}.wall"
    if p.exists():
        return int(p.read_text().strip())
    return -1

print()
print("=" * 72)
print(f"mxfp8 DRAFT-MODE COMPARISON @ {steps} steps")
print("=" * 72)
print()

# Per-variant per-step trajectories
all_steps = {}
walls = {}
for label, prefix in variants:
    log_path = outdir / f"{prefix}.log"
    if not log_path.exists():
        print(f"{label}: log missing ({log_path})")
        continue
    pairs = parse_steps(log_path)
    all_steps[label] = pairs
    walls[label] = read_wall(prefix)

# Total process wall summary
print(f"Total process wall (model load + denoise + teardown):")
baseline_wall = walls.get("baseline", -1)
for label, _ in variants:
    w = walls.get(label, -1)
    if w < 0:
        print(f"  {label:<20s} n/a")
        continue
    note = ""
    if baseline_wall > 0 and label != "baseline":
        delta = w - baseline_wall
        pct = 100 * delta / baseline_wall
        note = f"  Δ {delta:+d}s ({pct:+.1f}% vs baseline)"
    print(f"  {label:<20s} {hms(w)}  ({w}s){note}")
print()

# Per-step trajectory
labels_ordered = [label for label, _ in variants if label in all_steps]
if labels_ordered:
    n_steps = min(len(all_steps[l]) for l in labels_ordered)
    if n_steps > 0:
        header_cols = "step | " + " | ".join(f"{l:>20s}" for l in labels_ordered)
        print(f"Per-step denoise time:")
        print(f"  {header_cols}")
        print(f"  {'-'*4} | " + " | ".join("-"*20 for _ in labels_ordered))
        baseline_sum = 0.0
        sums = {l: 0.0 for l in labels_ordered}
        for i in range(n_steps):
            row_parts = []
            for l in labels_ordered:
                _, t = all_steps[l][i]
                sums[l] += t
                if l == "baseline":
                    row_parts.append(f"{t:>10.2f}s          ")
                else:
                    bt = all_steps["baseline"][i][1] if "baseline" in labels_ordered else None
                    if bt is not None:
                        delta_pct = 100 * (t - bt) / bt
                        row_parts.append(f"{t:>10.2f}s ({delta_pct:>+5.1f}%)")
                    else:
                        row_parts.append(f"{t:>10.2f}s          ")
            step_n = all_steps[labels_ordered[0]][i][0]
            print(f"  {step_n:>4} | " + " | ".join(row_parts))
        # Sum row
        sum_row = []
        for l in labels_ordered:
            s = sums[l]
            if l == "baseline":
                sum_row.append(f"{s:>10.2f}s          ")
            else:
                bs = sums.get("baseline", 0)
                if bs > 0:
                    sum_row.append(f"{s:>10.2f}s ({100*(s-bs)/bs:>+5.1f}%)")
                else:
                    sum_row.append(f"{s:>10.2f}s          ")
        print(f"  {'sum':>4} | " + " | ".join(sum_row))

print()
print(f"Log files in: {outdir}")
print(f"Quality A/B: .npz latent sidecars saved per run for cosine-sim diffing.")
print("=" * 72)
PYEOF
