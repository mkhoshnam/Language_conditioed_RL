import runpy
import sys
from pathlib import Path

import full_drawer_task  # noqa: F401  register local tasks

ISAACLAB = Path("/raid/home/than/Mohammad/isaaclab_ws/IsaacLab")

candidates = [
    ISAACLAB / "scripts/reinforcement_learning/rsl_rl/train.py",
    ISAACLAB / "source/standalone/workflows/rsl_rl/train.py",
]

for p in candidates:
    if p.exists():
        sys.path.insert(0, str(p.parent))
        runpy.run_path(str(p), run_name="__main__")
        break
else:
    raise FileNotFoundError(f"Could not find official rsl_rl train.py in: {candidates}")
