#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"$HOME/miniforge3/envs/kttsp-medium/bin/python" -B medium_overnight/campaign.py status
