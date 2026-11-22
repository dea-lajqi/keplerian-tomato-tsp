#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -B medium_v3/run.py status
