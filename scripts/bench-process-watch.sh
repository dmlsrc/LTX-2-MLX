#!/usr/bin/env bash
# bench-process-watch.sh
#
# Kill only the main GPU-competing media/photo analysis agents.
# This does NOT disable launchd jobs. It only kills live processes.
#
# Usage:
#   scripts/bench-process-watch.sh
#   INTERVAL=5 scripts/bench-process-watch.sh

set -euo pipefail

UID_="$(id -u)"
USER_="$(id -un)"
INTERVAL="${INTERVAL:-2}"

PROCS=(
  mediaanalysisd
  mediaanalysisd-access
  photoanalysisd
)

kill_once() {
  local proc pids

  for proc in "${PROCS[@]}"; do
    pids="$(pgrep -xu "${UID_}" "${proc}" 2>/dev/null || true)"

    if [[ -n "${pids}" ]]; then
      printf '[%s] kill %-24s %s\n' "$(date '+%H:%M:%S')" "${proc}" "${pids//$'\n'/ }"
      pkill -9 -xu "${UID_}" "${proc}" 2>/dev/null || true
    fi
  done
}

echo "current user: ${USER_} uid ${UID_}"
echo "interval: ${INTERVAL}s"
echo "watching: ${PROCS[*]}"
echo "Press Ctrl-C to stop."
echo

while :; do
  kill_once
  sleep "${INTERVAL}"
done
