#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run="results/esa/medium_overnight/run_20261122"
mkdir -p "$run"
py="$HOME/miniforge3/envs/kttsp-medium/bin/python"
if [[ ! -x "$py" ]]; then echo "Python environment not found: $py"; exit 1; fi
if [[ -f "$run/pid" ]] && kill -0 "$(cat "$run/pid")" 2>/dev/null; then
  echo "Medium supervisor already active: PID $(cat "$run/pid")"; exit 1
fi
# Prevent two launch scripts running simultaneously.
exec 9>"$run/launch.lock"
if ! flock -n 9; then echo 'Another Medium launch is starting'; exit 1; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp
nohup setsid -f nice -n 12 "$py" -u -B medium_overnight/campaign.py run \
  --directory "$run" --hours 8 --workers 2 --seed 20261122 \
  </dev/null >> "$run/campaign.log" 2>&1
for attempt in {1..15}; do
  if [[ -f "$run/pid" ]] && kill -0 "$(cat "$run/pid")" 2>/dev/null; then break; fi
  sleep 1
done
if [[ -f "$run/pid" ]] && kill -0 "$(cat "$run/pid")" 2>/dev/null; then
  echo "Medium process STARTED: PID $(cat "$run/pid")"
  echo "Follow output: tail -f $run/campaign.log"
else
  echo 'Failed to launch; inspect the log:'
  tail -n 30 "$run/campaign.log"
  exit 1
fi
