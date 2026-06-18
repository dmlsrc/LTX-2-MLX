#!/usr/bin/env bash
# profile.sh — wrap a Python command with macOS's built-in `sample` profiler.
#
# Works without sudo on processes you own.  Output is a text call tree, less
# Python-aware than tools like py-spy but sufficient for "where is the time
# going" questions and always available on macOS.
#
# (A py-spy backend used to live here, but py-spy on macOS requires root for
# SIP-gated task_for_pid and we don't have a clean workaround.  Removed; use
# `sample` or run a real profiler under Instruments for deeper Metal traces.)
#
# Usage:
#   scripts/profile.sh <output> <duration_sec> -- <command...>
#
#   output         .txt call tree file
#   duration_sec   How many seconds of samples to capture
#   command...     The command to launch and profile
#
# Example:
#   scripts/profile.sh /tmp/smoke.txt 60 -- \
#     python LTX_2_MLX/generate.py "test" --pipeline distilled \
#       --height 256 --width 256 --duration 1 --fps 24 --seed 124
#
# View results:
#   less /tmp/smoke.txt

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -2
  exit 0
fi

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <output> <duration_sec> -- <command...>" >&2
  exit 2
fi

OUTPUT="$1"
DURATION="$2"
SEP="$3"
shift 3

if [[ "${SEP}" != "--" ]]; then
  echo "error: missing '--' separator before the command" >&2
  exit 2
fi

if [[ -z "${1:-}" ]]; then
  echo "error: no command given after '--'" >&2
  exit 2
fi

if ! command -v sample >/dev/null 2>&1; then
  echo "error: macOS 'sample' not found (this script is macOS-only)" >&2
  exit 1
fi

echo "[sample] launching: $*"
# Launch target in background, capture its PID, sample it, then wait.
"$@" &
TARGET_PID=$!
echo "[sample] target pid ${TARGET_PID}; sampling for ${DURATION}s -> ${OUTPUT}"

# Don't leave the target running on Ctrl-C; don't kill it on natural exit.
cleanup() {
  if kill -0 "${TARGET_PID}" 2>/dev/null; then
    echo "[sample] target still running (pid ${TARGET_PID})"
    echo "[sample] leave it alone or kill manually:  kill ${TARGET_PID}"
  fi
}
trap cleanup EXIT

# Give the process ~1s to start its main loop, then sample.
sleep 1
if ! kill -0 "${TARGET_PID}" 2>/dev/null; then
  echo "error: target process exited before sampling started" >&2
  wait "${TARGET_PID}" 2>/dev/null
  exit 1
fi

sample "${TARGET_PID}" "${DURATION}" -file "${OUTPUT}" -fullPaths || true
echo "[sample] done -> ${OUTPUT}"

# Don't wait on the target; the profile is the artifact we wanted.
# If you want the target to complete, run it separately or wait manually.
