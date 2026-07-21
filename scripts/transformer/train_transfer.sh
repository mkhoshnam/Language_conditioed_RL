#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python3 -u scripts/transformer/train.py --config configs/transformer/transfer.json "$@"
