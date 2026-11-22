#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$HOME/miniforge3/envs/kttsp-medium/bin/python" medium_v4/driver.py status --directory results/esa/medium_v4/run_01
