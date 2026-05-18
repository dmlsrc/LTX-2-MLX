#!/bin/bash
# A/B wall-time comparison: LTX-2-MLX vs mlx-video at the same workload.
#
# Production lazy-graph mode (no signposts, no eval barriers) — the closest
# we can get to "what the user actually experiences". Both projects run the
# same workload: distilled stage 1 at 288x512 latent. mlx-video's distilled
# halves resolution for stage 1 automatically; LTX-2-MLX two-stage does the
# same. Both stop via LTX_PROFILE_STOP_AFTER_STEPS so we never enter stage 2.
#
# Run with Claude Code (or any other GPU-using UI) closed to avoid contention.
#
# Required environment variables (no defaults — set per your machine):
#   LTX_REPO          Absolute path to LTX-2-MLX repo root.
#   MLXV_REPO         Absolute path to mlx-video repo root.
#   MLXV_MODEL_REPO   Absolute path to a local mlx-video-format LTX-2.3 model
#                     directory (contains transformer/, text_encoder/,
#                     tokenizer/, vae/, audio_vae/, vocoder/, and
#                     ltx-2.3-spatial-upscaler-x2-*.safetensors at root).
#
# Optional environment variables (sane defaults):
#   LTX_VENV_BIN      Path to the LTX-2-MLX venv's activate script. If
#                     unset, assumes `python` is on PATH and works for LTX.
#   MLXV_VENV_BIN     Path to the mlx-video venv's activate script. If
#                     unset, assumes `python` is on PATH and works for mlx-video.
#   STEPS             Stage-1 step count (default 4 -> ~10 min total).
#                     STEPS=2 for quick smoke (~6 min, step-1 warmup heavy).
#                     STEPS=8 for full stage-1 (~16 min).
#   AB_OUTDIR         Output dir for logs (default ${TMPDIR:-/tmp}/ab_<ts>).
#   SEED              Seed for both runs (default 124).
#   HEIGHT, WIDTH     Output res (default 576, 1024 — distilled stage 1
#                     runs at half = 288x512 latent).
#   NUM_FRAMES        Frame count (default 481, the bakery reference).
#
# Usage:
#   export LTX_REPO=/path/to/LTX-2-MLX
#   export MLXV_REPO=/path/to/mlx-video
#   export MLXV_MODEL_REPO=/path/to/local/mlx-video-format-model-dir
#   bash scripts/bench_ab_wall_time.sh                # 4 steps, ~10 min
#   STEPS=8 bash scripts/bench_ab_wall_time.sh        # full stage-1
#
# Output: a comparison summary printed to stdout, plus per-side log files
# in $AB_OUTDIR/{ltx,mlxv}.log. The summary parses per-step times from each
# project's progress output:
#   - LTX-2-MLX:  tqdm-style "STEP1 1m 17s ... | 75.9s/it"
#   - mlx-video:  "[mlx-video step 1/8: 53.84s]"

set -u

# ----- required config -----
: "${LTX_REPO:?LTX_REPO must be set to the LTX-2-MLX repo root}"
: "${MLXV_REPO:?MLXV_REPO must be set to the mlx-video repo root}"
: "${MLXV_MODEL_REPO:?MLXV_MODEL_REPO must be set to a local mlx-video-format model dir}"

for p in "$LTX_REPO" "$MLXV_REPO" "$MLXV_MODEL_REPO"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: directory does not exist: $p" >&2
        exit 1
    fi
done

# ----- optional config -----
STEPS="${STEPS:-4}"
SEED="${SEED:-124}"
HEIGHT="${HEIGHT:-576}"
WIDTH="${WIDTH:-1024}"
NUM_FRAMES="${NUM_FRAMES:-481}"
AB_OUTDIR="${AB_OUTDIR:-${TMPDIR:-/tmp}/ab_$(date +%Y%m%d_%H%M%S)}"
LTX_VENV_BIN="${LTX_VENV_BIN:-}"
MLXV_VENV_BIN="${MLXV_VENV_BIN:-}"

mkdir -p "$AB_OUTDIR"

# ----- prompt: the canonical BAKERY from docs/TEST_PROMPTS.md -----
# Pulled into the script so it's self-contained; if you need to change it,
# update docs/TEST_PROMPTS.md and copy the new value here.
BAKERY='Live-action footage shot on a consumer mirrorless camera inside a quiet neighborhood bakery in soft morning light. A flour-dusted wooden table fills the foreground. On the left, a young baker in a blue apron kneads dough slowly. On the right, an older baker places four golden croissants onto a metal tray. The camera begins close on the dough, slowly pulls back to reveal both bakers, then gently pans right toward the croissants. The young baker looks at the tray and says, "Four for the window, two for the regulars." The older baker smiles and replies, "Save the best one for Mrs. Ito." Their voices are close, dry, natural, and clearly recorded. The room is very quiet, with a soft low indoor hush. The dialogue is the main sound. The background stays minimal and smooth. The frame contains only the bakery, the bakers, the table, the tray, and warm morning light. Realistic lip movement, handheld camera with gentle natural sway, warm shadows, detailed textures.'

echo "================================================================"
echo "A/B wall-time comparison @ STEPS=$STEPS"
echo "Output dir: $AB_OUTDIR"
echo "Workload: distilled stage 1, ${WIDTH}x${HEIGHT} -> $(($WIDTH/2))x$(($HEIGHT/2)) latent, $NUM_FRAMES frames"
echo "Mode: production lazy-graph (no signposts, no eval barriers)"
echo "================================================================"
echo

# ============================================================
# SIDE A: LTX-2-MLX
# ============================================================
echo "[A] LTX-2-MLX starting..."
cd "$LTX_REPO" || exit 1
if [ -n "$LTX_VENV_BIN" ]; then
    # shellcheck disable=SC1090
    source "$LTX_VENV_BIN"
fi

LTX_T0=$(date +%s)
LTX_PROFILE_STOP_AFTER_STEPS="$STEPS" \
caffeinate -di python scripts/generate.py "$BAKERY" \
  --pipeline distilled \
  --height "$HEIGHT" --width "$WIDTH" \
  --duration 20 --seed "$SEED" \
  --generate-audio \
  --fast-mode \
  --output-prefix ab_ltx \
  > "$AB_OUTDIR/ltx.log" 2>&1
LTX_RC=$?
LTX_T1=$(date +%s)
LTX_WALL=$((LTX_T1 - LTX_T0))
echo "[A] LTX-2-MLX done (rc=$LTX_RC, wall=${LTX_WALL}s)"
echo

# ============================================================
# SIDE B: mlx-video
# ============================================================
echo "[B] mlx-video starting..."
cd "$MLXV_REPO" || exit 1
if [ -n "$MLXV_VENV_BIN" ]; then
    # shellcheck disable=SC1090
    source "$MLXV_VENV_BIN"
fi

MLXV_T0=$(date +%s)
LTX_PROFILE_STOP_AFTER_STEPS="$STEPS" \
caffeinate -di python -m mlx_video.models.ltx_2.generate \
  --prompt "$BAKERY" \
  --pipeline distilled \
  --model-repo "$MLXV_MODEL_REPO" \
  --height "$HEIGHT" --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" --seed "$SEED" \
  --output-path "$AB_OUTDIR/mlxv.mp4" \
  > "$AB_OUTDIR/mlxv.log" 2>&1
MLXV_RC=$?
MLXV_T1=$(date +%s)
MLXV_WALL=$((MLXV_T1 - MLXV_T0))
echo "[B] mlx-video done (rc=$MLXV_RC, wall=${MLXV_WALL}s)"
echo

# ============================================================
# Summary
# ============================================================
python3 - "$AB_OUTDIR/ltx.log" "$AB_OUTDIR/mlxv.log" "$LTX_WALL" "$MLXV_WALL" "$STEPS" <<'PYEOF'
import re, sys
ltx_log_path, mlxv_log_path, ltx_wall, mlxv_wall, steps = sys.argv[1:]
ltx_wall = int(ltx_wall)
mlxv_wall = int(mlxv_wall)
steps = int(steps)

def parse_ltx(log):
    # LTX prints a tqdm-style line that updates per step.
    # Final state has: "| STEP1 1m 17s | RUN 10m 07s | ETA 0m 00s | 75.9s/it"
    # The progress writer rewrites the same line, but splitting on
    # \r and \n catches all intermediate states.
    out = []
    for raw in re.split(r"[\r\n]+", log):
        m = re.search(r"STEP1\s+(\d+m\s*\d+\s*s|\d+\.\d+s|\d+s)\b", raw)
        if m and "s/it" in raw:
            sit = re.search(r"(\d+\.\d+|\d+)\s*s/it", raw)
            step_n = re.search(r"\]\s*(\d+)/(\d+)", raw)
            if sit and step_n:
                cur, total = int(step_n.group(1)), int(step_n.group(2))
                out.append((cur, float(sit.group(1))))
    return out

def parse_mlxv(log):
    # "  [mlx-video step 1/8: 53.84s]"
    out = []
    for m in re.finditer(r"\[mlx-video step (\d+)/\d+:\s*(\d+\.\d+)s\]", log):
        out.append((int(m.group(1)), float(m.group(2))))
    return out

def hms(s):
    if s < 60: return f"{s:.1f}s"
    m, s = divmod(s, 60)
    return f"{int(m)}m {int(s):02d}s"

with open(ltx_log_path) as f:
    ltx_log = f.read()
with open(mlxv_log_path) as f:
    mlxv_log = f.read()

ltx_steps = parse_ltx(ltx_log)
mlxv_steps = parse_mlxv(mlxv_log)

# Deduplicate ltx_steps by step number (tqdm rewrites same line)
ltx_per_step = {}
for n, t in ltx_steps:
    ltx_per_step[n] = t  # last reading per step
ltx_pairs = sorted(ltx_per_step.items())

print()
print("=" * 64)
print(f"A/B WALL-TIME COMPARISON @ {steps} steps")
print("=" * 64)
print()
print(f"Total process wall (includes model load + denoise + teardown):")
print(f"  LTX-2-MLX:  {hms(ltx_wall)}  ({ltx_wall}s)")
print(f"  mlx-video:  {hms(mlxv_wall)}  ({mlxv_wall}s)")
print(f"  Delta:      {mlxv_wall - ltx_wall:+d}s "
      f"({100*(mlxv_wall - ltx_wall)/ltx_wall:+.1f}% mlxv vs LTX)")
print()
print("Per-step denoise time (clean GPU-only signal):")
print(f"  {'step':>4} | {'LTX':>10} | {'mlxv':>10} | {'delta':>10}")
print(f"  {'----':>4} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
n_steps = min(len(ltx_pairs), len(mlxv_steps))
ltx_total_step = 0
mlxv_total_step = 0
for i in range(n_steps):
    ltx_n, ltx_t = ltx_pairs[i]
    mlxv_n, mlxv_t = mlxv_steps[i]
    ltx_total_step += ltx_t
    mlxv_total_step += mlxv_t
    print(f"  {ltx_n:>4} | {ltx_t:>8.2f}s | {mlxv_t:>8.2f}s | {mlxv_t - ltx_t:>+8.2f}s")
if n_steps:
    print(f"  {'sum':>4} | {ltx_total_step:>8.2f}s | {mlxv_total_step:>8.2f}s | "
          f"{mlxv_total_step - ltx_total_step:>+8.2f}s")
    if ltx_total_step > 0:
        print(f"  {'%':>4} | {'':>9} | {'':>9} | "
              f"{100*(mlxv_total_step - ltx_total_step)/ltx_total_step:>+8.2f}%")
print()
print(f"Log files:")
print(f"  {ltx_log_path}")
print(f"  {mlxv_log_path}")
print("=" * 64)
PYEOF
