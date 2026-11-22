#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run=results/esa/medium_v3/run_01
mkdir -p "$run"
touch "$run/stop.request"
echo 'Graceful stop requested; checkpoint will be saved.'
