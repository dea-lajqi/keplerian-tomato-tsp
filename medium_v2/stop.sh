#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/results/esa/medium_v2/run_01"
touch "$DIR/stop.request"
if [[ -f "$DIR/pid" ]];then
 pid="$(cat "$DIR/pid")"
 kill -TERM "$pid" 2>/dev/null || true
fi
echo "Safe-stop requested; checkpoint is written before exit."
