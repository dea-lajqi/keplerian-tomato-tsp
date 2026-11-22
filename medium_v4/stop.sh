#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rundir="results/esa/medium_v4/run_01"
mkdir -p "$rundir"
touch "$rundir/stop.request"
if [ -f "$rundir/pid" ]; then
  pid="$(cat "$rundir/pid")"
  kill -TERM "$pid" 2>/dev/null || true
  echo "Graceful stop requested for PID $pid"
else
  echo "No active V4 PID file; stop request saved"
fi
