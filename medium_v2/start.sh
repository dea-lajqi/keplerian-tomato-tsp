#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DIR="results/esa/medium_v2/run_01"
mkdir -p "$DIR"
if [[ -f "$DIR/pid" ]]; then
 OLD_PID="$(cat "$DIR/pid")"
 if kill -0 "$OLD_PID" 2>/dev/null; then
  echo "PID $OLD_PID appears to exist. Check before starting a duplicate.";exit 1
 fi
fi
rm -f "$DIR/stop.request"
PY="$HOME/miniforge3/envs/kttsp-medium/bin/python"
if [[ ! -x "$PY" ]]; then echo "Missing $PY";exit 1;fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup nice -n 12 "$PY" -u -B medium_v2/run.py run --directory "$DIR" --hours 0.5 --beam 10 > "$DIR/campaign.log" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$DIR/pid"
sleep 1
if kill -0 "$pid" 2>/dev/null;then
 echo "MEDIUM V2 started PID $pid. Log: tail -f $DIR/campaign.log"
else
 echo "MEDIUM V2 exited immediately. Check: tail -n 40 $DIR/campaign.log"
fi
