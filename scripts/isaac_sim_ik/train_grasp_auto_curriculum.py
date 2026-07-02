# Auto-curriculum training for Franka grasp/place.
# Uses ONE Isaac Lab env only. No second validation env, because DirectRLEnv
# allows only one SimulationContext per process.

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

parser = argparse.ArgumentParser(description="Train Franka grasp/place with automatic curriculum.")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--experiment_name", type=str, default="grasp_auto_curriculum")
parser.add_argument("--controller", type=str, default="ik", choices=["ik", "joint"],
                    help="ik = task-space differential IK (recommended), joint = raw joint deltas")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaacsim.core.utils.torch.transformations import tf_vector  # noqa: E402

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


STAGES = ["approach", "grasp", "lift", "place", "full"]
THRESHOLDS = {
    "approach": 0.75,
    "grasp": 0.45,
    "lift": 0.25,
    "place": 0.20,
}

# Reverse curriculum: fraction of TRAINING resets that start already grasping
# the cube (fingers closed on it, cube in the grasp frame). Lets lift/place
# learn from solved-grasp states while grasp itself is still being learned.
# Evaluation always uses 0 (see evaluate_on_same_env).
PREGRASP_RATIO = {
    "approach": 0.0,
    "grasp": 0.25,   # some episodes show what a held cube feels like
    "lift": 0.6,
    "place": 0.5,
    "full": 0.25,
}


def unwrap_obs(out):
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out["policy"]
    return out


def compute_stage_success(raw_env, stage: str):
    raw_env._compute_intermediate_values()

    grasp_pos = raw_env.robot_grasp_pos
    cube = raw_env.cube_pos_w

    xy_dist = torch.norm(grasp_pos[:, :2] - cube[:, :2], dim=-1)
    z_above = grasp_pos[:, 2] - cube[:, 2]

    approach = tf_vector(raw_env.robot_grasp_rot, raw_env.gripper_forward_axis)
    downness = torch.clamp(-approach[:, 2], 0.0, 1.0)

    finger_pos = raw_env._robot.data.joint_pos[:, raw_env._finger_dof_idx].mean(dim=-1)
    finger_open = torch.clamp(finger_pos / 0.04, 0.0, 1.0)
    finger_closing = torch.clamp((0.04 - finger_pos) / 0.04, 0.0, 1.0)

    xy_align = 1.0 - torch.tanh(xy_dist / 0.05)
    z_at = 1.0 - torch.tanh(torch.abs(z_above) / 0.05)
    down_f = torch.clamp((downness - 0.45) / 0.55, 0.0, 1.0)
    readiness = xy_align * z_at * down_f

    cube_lift = torch.clamp(cube[:, 2] - raw_env.cfg.cube_rest_z, min=0.0)
    d_place = torch.norm(cube - raw_env.place_target_pos, dim=-1)

    if stage == "approach":
        return (xy_dist < 0.07) & (downness > 0.55) & (finger_open > 0.5)

    if stage == "grasp":
        return (readiness > 0.35) & (finger_closing > 0.55)

    if stage == "lift":
        return cube_lift > raw_env.cfg.lifted_height

    if stage in ["place", "full"]:
        return (d_place < raw_env.cfg.place_success_dist) & (cube_lift > raw_env.cfg.lifted_height)

    raise ValueError(stage)


@torch.no_grad()
def evaluate_on_same_env(raw_env, vec_env, runner, device, stage: str, steps: int):
    raw_env.cfg.curriculum_stage = stage

    # Evaluation must NOT use pre-grasped initial states, otherwise lift/place
    # success is trivially inflated. Restore the training ratio afterwards.
    train_pregrasp = float(getattr(raw_env.cfg, "pregrasp_ratio", 0.0))
    raw_env.cfg.pregrasp_ratio = 0.0

    # evaluate for at least one full episode, but stop BEFORE the synchronized
    # truncation+reset so post-eval state still reflects the policy, not a reset
    steps = max(steps, int(raw_env.max_episode_length) - 2)

    try:
        out = vec_env.reset()
        obs = unwrap_obs(out)
    except Exception:
        obs = unwrap_obs(vec_env.get_observations())

    policy = runner.get_inference_policy(device=device)
    success = torch.zeros(raw_env.num_envs, dtype=torch.bool, device=device)
    hold_count = torch.zeros(raw_env.num_envs, dtype=torch.long, device=device)
    hold_required = 10

    # metrics accumulated over the WHOLE rollout. (The old code called
    # get_stage_metrics() after eval ended on the exact truncation step, so it
    # always measured the freshly reset state -- identical numbers every time.)
    metric_sums: dict[str, float] = {}
    metric_n = 0

    for _ in range(steps):
        actions = policy(obs)
        step_out = vec_env.step(actions)
        obs = unwrap_obs(step_out[0])

        if hasattr(raw_env, "get_stage_success"):
            ok = raw_env.get_stage_success(stage)
        else:
            ok = compute_stage_success(raw_env, stage)

        hold_count = torch.where(ok, hold_count + 1, torch.zeros_like(hold_count))
        success |= hold_count >= hold_required

        if hasattr(raw_env, "get_stage_metrics"):
            m = raw_env.get_stage_metrics()
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v
            metric_n += 1

    raw_env.cfg.pregrasp_ratio = train_pregrasp

    metrics = {k: v / max(metric_n, 1) for k, v in metric_sums.items()}
    return success.float().mean().item(), metrics


def main():
    env_cfg = FrankaGraspCurriculumEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.curriculum_stage = "approach"

    print("=" * 80)
    print("[TASK] Franka grasp AUTO curriculum")
    print("[TASK] stages: approach -> grasp -> lift -> place -> full")
    print("[TASK] validation uses SAME env; no second SimulationContext")
    print("=" * 80)

    raw_env = FrankaGraspCurriculumEnv(env_cfg, render_mode=None)
    device = raw_env.device
    vec_env = RslRlVecEnvWrapper(raw_env)

    agent_cfg = FrankaGraspCurriculumPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = device
    agent_cfg.experiment_name = args_cli.experiment_name

    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    cfg_dict = agent_cfg.to_dict()
    cfg_dict.setdefault("obs_groups", {"actor": ["policy"], "critic": ["policy"]})

    runner = OnPolicyRunner(vec_env, cfg_dict, log_dir=log_dir, device=device)

    stage_idx = 0
    current_stage = STAGES[stage_idx]
    raw_env.cfg.curriculum_stage = current_stage

    total_iter = 0
    first_learn = True

    while total_iter < args_cli.max_iterations:
        current_stage = STAGES[stage_idx]
        raw_env.cfg.curriculum_stage = current_stage
        raw_env.cfg.pregrasp_ratio = PREGRASP_RATIO.get(current_stage, 0.0)

        chunk = min(args_cli.eval_every, args_cli.max_iterations - total_iter)

        print("\n" + "=" * 80)
        print(f"[TRAIN] iter {total_iter}->{total_iter + chunk} | stage={current_stage}")
        print("=" * 80)

        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first_learn)
        first_learn = False
        total_iter += chunk

        success, metrics = evaluate_on_same_env(
            raw_env=raw_env,
            vec_env=vec_env,
            runner=runner,
            device=device,
            stage=current_stage,
            steps=args_cli.eval_steps,
        )

        threshold = THRESHOLDS.get(current_stage, 1.0)
        print(f"[VALID] iter={total_iter} stage={current_stage} success={success:.3f} threshold={threshold:.3f}")
        print("[METRICS] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        ckpt_path = os.path.join(log_dir, f"model_auto_{current_stage}_{total_iter}.pt")
        runner.save(ckpt_path)
        print(f"[SAVE] {ckpt_path}")

        if current_stage != "full" and success >= threshold:
            next_stage = STAGES[stage_idx + 1]
            print(f"[AUTO] ADVANCE {current_stage} -> {next_stage}")
            stage_idx += 1
            raw_env.cfg.curriculum_stage = next_stage

    final_path = os.path.join(log_dir, f"model_auto_final_{total_iter}.pt")
    runner.save(final_path)
    print(f"[DONE] final checkpoint: {final_path}")
    raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
