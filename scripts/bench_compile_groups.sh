#!/bin/bash
# A/B sweep: LTX_COMPILE_BLOCK_GROUPS at N in {off, 1, 2, 3, 4}.
#
# PERFORMANCE.md currently records:
#   "N=4 at small T neutral; N=48 at bakery neutral (18m 29s vs 18m 41s)"
# but has no data at N=1, N=2, N=3 — the question is whether one of those
# is meaningfully faster (sweet spot) or whether the curve is truly flat.
#
# Historical note: pre-AdaLN/RoPE-dtype-cast (2026-05-17), compile-group>4
# caused Metal watchdog hangs.  Since the BF16-cast fix the hangs no longer
# reproduce — empirically N=20 STEPS=2 ran to completion with no abort and
# no desktop-interactivity regression.  Feel free to extend the sweep above
# N=4 if a lower-N value is suspicious.  This is an end-to-end A/B (not a
# microbench) because the lever's effect depends on the real block
# structure — SDPA, AdaLN, residual gates — which a synthetic block stack
# can't faithfully replicate.
#
# Required environment variables (no defaults — set per your machine):
#   LTX_REPO                Absolute path to LTX-2-MLX repo root.
#   LTX_DEFAULT_WEIGHTS_PATH   Distilled checkpoint path (matches the
#                              same env var used by generate.py).
#
# Optional environment variables (sane defaults):
#   LTX_VENV_BIN      Path to LTX-2-MLX venv activate script.
#   STEPS             Stage-1 steps to capture per run (default 2 —
#                     step-1 warmup is heavy, so STEPS=2 yields 1
#                     steady-state step per run).  STEPS=3 adds one
#                     steady-state step (mean over 2 instead of 1) but
#                     adds ~45 s per run.
#   COMPILE_GROUPS            Comma-separated group sizes (default "off,4" —
#                     minimum-information sweep: production baseline vs
#                     the previously-reported "neutral at N=4" point.
#                     Add ",2" or ",1,2,3" to fill in middle values if
#                     the off-vs-N=4 comparison surprises).  "off" runs
#                     without LTX_COMPILE_BLOCK_GROUPS set at all.
#   COMPILE_GROUPS_OUTDIR  Output dir.  Default:
#                     $SHARED_TEMP_DIR/trace_analysis/compile_groups_<ts>
#                     if SHARED_TEMP_DIR is set, else
#                     ${TMPDIR:-/tmp}/compile_groups_<ts>.
#   SEED              Seed (default 124, matches bench_ab_wall_time.sh).
#   HEIGHT, WIDTH     Output res (default 576, 1024 — bakery shape).
#   PROMPT            Prompt (default the bakery reference).
#
# Per-run wall is roughly model load (~30 s) + warmup step (~50 s) +
# steady steps (~45 s each).  Defaults => ~2 runs × ~2 min ≈ ~12 min total.
# Live tqdm output is teed to stdout AND the log so you can watch progress.
#
# Usage:
#   export LTX_REPO=/Users/Shared/huggingface/lib/LTX-2-MLX
#   bash scripts/bench_compile_groups.sh                          # off,4 ≈ 12 min
#   COMPILE_GROUPS=off,1,2,3,4 bash scripts/bench_compile_groups.sh       # full sweep ≈ 30 min
#   STEPS=3 COMPILE_GROUPS=off,2,4 bash scripts/bench_compile_groups.sh   # mid set, tighter ≈ 25 min
#
# Output: per-run log files in $COMPILE_GROUPS_OUTDIR/group_<N>.log
# plus a summary table comparing mean s/it across group sizes.

set -u

# ----- required config -----
: "${LTX_REPO:?LTX_REPO must be set to the LTX-2-MLX repo root}"
: "${LTX_DEFAULT_WEIGHTS_PATH:?LTX_DEFAULT_WEIGHTS_PATH must be set}"

if [ ! -d "$LTX_REPO" ]; then
    echo "ERROR: LTX_REPO does not exist: $LTX_REPO" >&2
    exit 1
fi
if [ ! -f "$LTX_DEFAULT_WEIGHTS_PATH" ]; then
    echo "ERROR: weights not found: $LTX_DEFAULT_WEIGHTS_PATH" >&2
    exit 1
fi

# ----- optional config -----
LTX_VENV_BIN="${LTX_VENV_BIN:-}"
STEPS="${STEPS:-2}"
COMPILE_GROUPS="${COMPILE_GROUPS:-off,4}"
SEED="${SEED:-124}"
HEIGHT="${HEIGHT:-576}"
WIDTH="${WIDTH:-1024}"
PROMPT="${PROMPT:-A baker carefully placing fresh croissants in a wicker basket, warm morning light streaming through a Parisian patisserie window, steam rising gently from the pastries, cinematic shallow depth of field}"

TS=$(date +%Y%m%d_%H%M%S)
SHARED_BASE="${SHARED_TEMP_DIR:-${TMPDIR:-/tmp}}"
COMPILE_GROUPS_OUTDIR="${COMPILE_GROUPS_OUTDIR:-$SHARED_BASE/trace_analysis/compile_groups_$TS}"
mkdir -p "$COMPILE_GROUPS_OUTDIR"

echo "================================================================"
echo "LTX_COMPILE_BLOCK_GROUPS sweep"
echo "================================================================"
echo "  STEPS=$STEPS  COMPILE_GROUPS=$COMPILE_GROUPS  SEED=$SEED  res=${HEIGHT}x${WIDTH}"
echo "  outdir=$COMPILE_GROUPS_OUTDIR"
echo "  weights=$LTX_DEFAULT_WEIGHTS_PATH"
echo

IFS=',' read -ra GROUP_LIST <<< "$COMPILE_GROUPS"

run_one() {
    local group="$1"
    local log="$COMPILE_GROUPS_OUTDIR/group_${group}.log"
    local label="N=$group"
    local env_prefix=""

    if [ "$group" = "off" ]; then
        label="off (no LTX_COMPILE_BLOCK_GROUPS)"
        # explicitly unset to avoid inheriting a stray value
        env_prefix="env -u LTX_COMPILE_BLOCK_GROUPS"
    else
        env_prefix="env LTX_COMPILE_BLOCK_GROUPS=$group"
    fi

    echo
    echo "============================================================"
    echo " $label  (log: $log)"
    echo "============================================================"
    local t0
    t0=$(date +%s)
    # Tee stdout+stderr so progress (tqdm STEP1 line) is visible live AND
    # captured to disk for the summary parser.  PIPESTATUS preserves the
    # python process's exit code despite the pipe.
    # shellcheck disable=SC2086
    $env_prefix \
        LTX_PROFILE_STOP_AFTER_STEPS="$STEPS" \
        caffeinate -di python scripts/generate.py "$PROMPT" \
            --pipeline distilled \
            --height "$HEIGHT" --width "$WIDTH" \
            --duration 20 --seed "$SEED" \
            --generate-audio \
            --fast-mode \
            --output-prefix "compile_groups_${group}" \
            2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local t1
    t1=$(date +%s)
    local wall=$((t1 - t0))
    if [ "$rc" -ne 0 ]; then
        echo "  => rc=$rc wall=${wall}s (FAILED — see $log)"
    else
        echo "  => rc=0 wall=${wall}s"
    fi
}

cd "$LTX_REPO" || exit 1
if [ -n "$LTX_VENV_BIN" ]; then
    # shellcheck disable=SC1090
    source "$LTX_VENV_BIN"
fi

for g in "${GROUP_LIST[@]}"; do
    run_one "$g"
done

echo
echo "================================================================"
echo "Summary"
echo "================================================================"

python3 - "$COMPILE_GROUPS_OUTDIR" "$COMPILE_GROUPS" "$STEPS" <<'PYEOF'
import re
import sys
from pathlib import Path

outdir = Path(sys.argv[1])
groups = sys.argv[2].split(",")
steps = int(sys.argv[3])

def parse_step_times(log_text: str):
    """Extract per-step s/it from tqdm 'STEP1 ... | NN.Ns/it' lines."""
    per_step = {}
    for raw in re.split(r"[\r\n]+", log_text):
        if "STEP1" not in raw or "s/it" not in raw:
            continue
        m_step = re.search(r"\]\s*(\d+)/(\d+)", raw)
        m_sit = re.search(r"(\d+\.\d+|\d+)\s*s/it", raw)
        if m_step and m_sit:
            cur = int(m_step.group(1))
            per_step[cur] = float(m_sit.group(1))
    return [per_step[k] for k in sorted(per_step)]

rows = []
for g in groups:
    log = outdir / f"group_{g}.log"
    if not log.exists():
        rows.append((g, None, None, None, "no log"))
        continue
    text = log.read_text(errors="replace")
    times = parse_step_times(text)
    if not times:
        rows.append((g, None, None, None, "no step times parsed"))
        continue
    # Skip step 1 (one-time compile / warmup cost) when computing mean
    steady = times[1:] if len(times) > 1 else times
    mean = sum(steady) / len(steady)
    rows.append((g, times[0], mean, len(times), ""))

# Headline table
print(f"  {'group':<10s}  {'step1_sit':>10s}  {'steady_mean_sit':>16s}  "
      f"{'n_steps':>8s}  {'vs off':>10s}  notes")
print(f"  {'-'*10}  {'-'*10}  {'-'*16}  {'-'*8}  {'-'*10}  -----")

off_mean = None
for g, t1, mean, n, note in rows:
    if g == "off" and mean is not None:
        off_mean = mean

for g, t1, mean, n, note in rows:
    if mean is None:
        print(f"  {g:<10s}  {'n/a':>10s}  {'n/a':>16s}  {'n/a':>8s}  "
              f"{'n/a':>10s}  {note}")
        continue
    vs_off = f"{100*(mean-off_mean)/off_mean:+.2f}%" if off_mean else "—"
    print(f"  {g:<10s}  {t1:>9.2f}s  {mean:>15.2f}s  "
          f"{n:>8d}  {vs_off:>10s}  {note}")

print()
print("  Read:")
print("    * step1_sit = first step wall (includes mx.compile trace cost)")
print("    * steady_mean_sit = mean over steps 2..N (steady-state per-step)")
print("    * vs off = (group_N steady) vs (off steady), positive = SLOWER")
print()
print("  Note: empirically (2026-05-18) all of off / N=4 / N=20 / N=48 landed")
print("  within ~1.3 % of each other = run-to-run noise.  mx.compile over the")
print("  LTX block stack appears to be a no-op at bakery scale; this sweep is")
print("  here for regression detection if MLX upstream changes the picture.")
PYEOF

echo
echo "Logs: $COMPILE_GROUPS_OUTDIR"
