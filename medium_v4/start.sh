#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/miniforge3/envs/kttsp-medium/bin/python"
[ -x "$PY" ] || { echo "Missing kttsp-medium environment: $PY"; exit 1; }
run="results/esa/medium_v4/run_01"
mkdir -p "$run"
if [ -f "$run/pid" ]; then
 pid="$(cat "$run/pid")"
 if kill -0 "$pid" 2>/dev/null; then echo "V4 already running PID $pid; not launching twice"; exit 1; fi
fi
nohup "$PY" -u -B medium_v4/driver.py run --directory "$run" --minutes 90 --seed 20261122 > "$run/campaign.log" 2>&1 < /dev/null &
pid=$!
sleep 2
if kill -0 "$pid" 2>/dev/null; then
 echo "MEDIUM V4 started PID $pid. Monitor: bash medium_v4/status.sh"
else
 echo "MEDIUM V4 failed to start. See: tail -n 40 $run/campaign.log"
 exit 1
fi
