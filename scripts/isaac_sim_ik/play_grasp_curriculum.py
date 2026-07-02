# Play Franka grasp/place curriculum checkpoint.

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

parser = argparse.ArgumentParser(description="Play Franka grasp/place curriculum policy.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--stage", type=str, default="full", choices=["approach", "grasp", "lift", "place", "full"])
parser.add_argument("--controller", type=str, default="ik", choices=["ik", "joint"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

if args_cli.controller == "ik":
    from language_conditioned_rl.isaac_sim_ik.franka_grasp_ik_env import (  # noqa: E402
        FrankaGraspCurriculumEnv,
        FrankaGraspCurriculumEnvCfg,
    )
else:
    from language_conditioned_rl.isaac_sim_ik.franka_grasp_curriculum_env import (  # noqa: E402
        FrankaGraspCurriculumEnv,
        FrankaGraspCurriculumEnvCfg,
    )
from language_conditioned_rl.isaac_sim_ik.rsl_rl_grasp_curriculum_cfg import (  # noqa: E402
    FrankaGraspCurriculumPPORunnerCfg,
)


def main():
    env_cfg = FrankaGraspCurriculumEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.curriculum_stage = args_cli.stage

    env = FrankaGraspCurriculumEnv(env_cfg, render_mode="human")
    device = env.device
    env = RslRlVecEnvWrapper(env)

    agent_cfg = FrankaGraspCurriculumPPORunnerCfg()
    agent_cfg.device = device
    cfg_dict = agent_cfg.to_dict()
    cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})

    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=device)
    runner.load(args_cli.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    out = env.get_observations()
    obs = out[0] if isinstance(out, tuple) else out
    for _ in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(obs)
        step_out = env.step(actions)
        obs = step_out[0]

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
