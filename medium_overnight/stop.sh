#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run="results/esa/medium_overnight/run_20261122"
mkdir -p "$run"
touch "$run/stop.request"
if [[ -f "$run/pid" ]]; then
  pid="$(cat "$run/pid")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    echo "Requested graceful Medium stop, supervisor PID=$pid"
  fi
else echo 'No Medium supervisor PID file found.'
fi
