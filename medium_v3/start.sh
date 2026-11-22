#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" - <<'PY'
import sys,pykep
if not hasattr(pykep.planet,'keplerian'):
 raise SystemExit('ERROR: activate kttsp-medium with PyKEP 2.x before starting')
print('Python:',sys.executable)
PY
run=results/esa/medium_v3/run_01
mkdir -p "$run"
if [ -f "$run/pid" ]; then
 pid="$(cat "$run/pid")"
 if kill -0 "$pid" 2>/dev/null; then
  echo "ALREADY RUNNING PID $pid; do not launch a duplicate"
  exit 1
 fi
fi
if [ -f "$run/stop.request" ]; then rm "$run/stop.request"; fi
nohup "$PYTHON_BIN" -u -B medium_v3/run.py run --directory "$run" --minutes 45 >> "$run/campaign.log" 2>&1 </dev/null &
PID=$!
echo "MEDIUM V3 launch requested: PID $PID"
sleep 2
if kill -0 "$PID" 2>/dev/null; then
 echo "Process still alive. Monitor: bash medium_v3/status.sh"
else
 echo "Process exited early. Check: tail -n 30 '$run/campaign.log'"
 exit 1
fi
