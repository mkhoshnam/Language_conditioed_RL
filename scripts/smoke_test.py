import os
import sys
from pathlib import Path

import imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import RealFrankaPickPlaceEnv


RENDER_DIR = PROJECT_ROOT / "renders"
os.makedirs(RENDER_DIR, exist_ok=True)


def main():
    env = RealFrankaPickPlaceEnv(render_mode="rgb_array")
    rng = np.random.default_rng(7)
    obs, info = env.reset(seed=7)
    print(f"obs shape: {obs.shape}")
    print(f"goal: {info['language_goal']}")

    total_reward = 0.0
    last_info = {}
    for _ in range(80):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    fixed = env.render("fixed_scene")
    wrist = env.render("wrist_camera")
    imageio.imwrite(os.path.join(RENDER_DIR, "fixed_scene.png"), fixed)
    imageio.imwrite(os.path.join(RENDER_DIR, "wrist_camera.png"), wrist)
    imageio.imwrite(os.path.join(RENDER_DIR, "both_cameras.png"), np.concatenate([fixed, wrist], axis=1))
    env.close()

    print(f"smoke reward: {total_reward:.2f}")
    print(
        "last distances: "
        f"reach={last_info.get('reach_dist', float('nan')) * 100:.1f}cm "
        f"approach={last_info.get('approach_dist', float('nan')) * 100:.1f}cm "
        f"place={last_info.get('place_dist', float('nan')) * 100:.1f}cm"
    )
    print(f"saved renders in {RENDER_DIR}")


if __name__ == "__main__":
    main()
