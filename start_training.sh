#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source franka_rl_env/bin/activate
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python3 -u real_franka_pick_place/train.py "$@"

