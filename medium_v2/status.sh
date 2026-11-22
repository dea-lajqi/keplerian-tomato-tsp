#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_v2/run.py status
