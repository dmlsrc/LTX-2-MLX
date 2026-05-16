#!/usr/bin/env bash
# bench-quiet.sh — kill GPU-competing background processes for a bench window.
#
# Targets ML/indexing user agents on macOS that wake up on idle and use the
# GPU (mediaanalysisd, photoanalysisd, spotlightknowledged).  System
# Integrity Protection blocks `launchctl bootout` of system-installed agents
# even with sudo, so the only option is to kill the running processes.
#
# The launch agents themselves stay bootstrapped, so macOS may respawn them
# on the next trigger event (idle, file change, etc).  During an active bench
# where the GPU is hammered they tend to stay quiet, but for a long bench you
# can just re-run `kill` as needed.  There's no state to restore — re-enable
# happens automatically when launchd's triggers fire.
#
# Run as your normal user (no sudo).  pkill of your own processes doesn't
# need root.
#
# Usage:
#   scripts/bench-quiet.sh status
#   scripts/bench-quiet.sh kill

set -euo pipefail

PROCS=(
  mediaanalysisd
  mediaanalysisd-access
  photoanalysisd
  spotlightknowledged
)

UID_=$(id -u)
USER_=$(id -un)

cmd_status() {
  echo "user: ${USER_} (uid ${UID_})"
  local running=0
  for proc in "${PROCS[@]}"; do
    if pgrep -xu "${UID_}" "${proc}" >/dev/null 2>&1; then
      pid=$(pgrep -xu "${UID_}" "${proc}" | head -1)
      printf "  %-30s pid %s\n" "${proc}" "${pid}"
      running=1
    else
      printf "  %-30s (not running)\n" "${proc}"
    fi
  done
  if [[ ${running} -eq 0 ]]; then
    echo "(nothing to kill)"
  fi
}

cmd_kill() {
  local killed=0
  for proc in "${PROCS[@]}"; do
    if pgrep -xu "${UID_}" "${proc}" >/dev/null 2>&1; then
      echo "kill ${proc}"
      pkill -9 -xu "${UID_}" "${proc}" || true
      killed=1
    fi
  done
  if [[ ${killed} -eq 0 ]]; then
    echo "(nothing to kill)"
  fi
}

case "${1:-}" in
  status) cmd_status ;;
  kill)   cmd_kill ;;
  *)
    cat >&2 <<EOF
usage: $0 {status|kill}

  status  Show which target processes are currently running.
  kill    SIGKILL the target processes.  May be re-run during a long bench
          if any have respawned.
EOF
    exit 1
    ;;
esac
