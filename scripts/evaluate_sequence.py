import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import (
    BLOCK_NAMES,
    TARGET_NAMES,
    CAMERAS,
    GRIPPER_OPEN_CTRL,
    STAGE_PLACE,
    RealFrankaPickPlaceEnv,
)
from language_conditioned_rl.llm_parser import parse_command
from language_conditioned_rl.ppo import PPO


DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "checkpoints", "ppo_real_franka_best_place_success.pt"
)
BASE_CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
PLACE_CKPT = os.environ.get("PLACE_CKPT", BASE_CKPT)
STACK_CKPT = os.environ.get("STACK_CKPT", BASE_CKPT)

SEQUENCE_COMMANDS = os.environ.get(
    "SEQUENCE_COMMANDS",
    "put the red block on the yellow plate ;; put the green block in the cyan bowl",
)
COMMANDS = [c.strip() for c in SEQUENCE_COMMANDS.split(";;") if c.strip()]

N_SEQUENCES = int(os.environ.get("N_SEQUENCES", 30))
VIDEO_SEQUENCES = int(os.environ.get("VIDEO_SEQUENCES", 1))
STEPS_PER_COMMAND = int(os.environ.get("STEPS_PER_COMMAND", 280))
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", os.path.join(PROJECT_ROOT, "same_scene_sequence.mp4")
)
CAMERA = os.environ.get("CAMERA", "fixed_scene")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETERMINISTIC = os.environ.get("DETERMINISTIC", "0") == "1"


def put_text(img, text):
    import cv2

    out = img.copy()
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_frame(env, goal):
    if CAMERA == "both":
        fixed = env.render("fixed_scene")
        wrist = env.render("wrist_camera")
        frame = np.concatenate([fixed, wrist], axis=1)
    else:
        if CAMERA not in CAMERAS:
            raise ValueError(f"CAMERA must be one of {CAMERAS} or 'both', got {CAMERA!r}")
        frame = env.render(CAMERA)
    return put_text(frame, f"goal: {goal}")


def set_task_by_index(env, task_index):
    env.fixed_task_index = int(task_index)
    env._set_task()
    reset_task_progress(env)


def reset_task_progress(env):
    mujoco.mj_forward(env.model, env.data)

    ee = env.data.site_xpos[env._ee_site_id].copy()
    block = env._block_pos()
    target = env._destination_pos()

    env._prev_reach_dist = float(np.linalg.norm(block - ee))
    env._prev_approach_dist = float(np.linalg.norm(env._approach_pos(block) - ee))
    env._best_approach_dist = env._prev_approach_dist
    env._prev_place_dist = float(np.linalg.norm((target - block)[:2]))
    env._prev_lift_height = 0.0
    env._prev_height_above_dest = float(block[2] - env._destination_height())
    env._prev_ee_z = float(ee[2])
    env._prev_settle_score = 0.0
    env._prev_released = False
    env._post_release_steps = 0
    env._dest_xy_at_grasp = None
    env._step_count = 0
    env._success_hold = 0
    env._release_gate_latch = 0
    env._ever_grasped = False
    env._ever_lifted = False
    env._max_lift_height = 0.0


def reset_robot_home_keep_scene(env):
    block_qpos = {}
    block_qvel = {}
    for name in BLOCK_NAMES:
        qpos_addr = env._block_qpos_addr[name]
        qvel_addr = env._block_qvel_addr[name]
        block_qpos[name] = env.data.qpos[qpos_addr:qpos_addr + 7].copy()
        block_qvel[name] = np.zeros(6, dtype=np.float64)

    target_pos = {}
    for name in TARGET_NAMES:
        body_id = env._target_body_ids[name]
        target_pos[name] = env.model.body_pos[body_id].copy()

    if env._home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(env.model, env.data, env._home_key_id)
    else:
        mujoco.mj_resetData(env.model, env.data)

    for name in BLOCK_NAMES:
        qpos_addr = env._block_qpos_addr[name]
        qvel_addr = env._block_qvel_addr[name]
        env.data.qpos[qpos_addr:qpos_addr + 7] = block_qpos[name]
        env.data.qvel[qvel_addr:qvel_addr + 6] = block_qvel[name]

    for name in TARGET_NAMES:
        body_id = env._target_body_ids[name]
        env.model.body_pos[body_id] = target_pos[name]

    env.data.qvel[:] = 0.0
    env._arm_ctrl_target = env.data.qpos[env._act_qpos_addr[:7]].copy()
    env._gripper_ctrl_target = GRIPPER_OPEN_CTRL
    env.data.ctrl[:7] = env._arm_ctrl_target
    env.data.ctrl[7] = env._gripper_ctrl_target

    mujoco.mj_forward(env.model, env.data)
    reset_task_progress(env)


def load_agent(path, env):
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        device=DEVICE,
    )
    agent.load(path)
    return agent


def apply_agent_extra(env, agent):
    if "curriculum_dist" in agent.extra:
        env.curriculum_dist = float(agent.extra["curriculum_dist"])
    if "lift_goal_height" in agent.extra:
        env.curriculum_lift_height = float(agent.extra["lift_goal_height"])
    if "success_radius" in agent.extra:
        env.success_radius = float(agent.extra["success_radius"])


def run_one_command(env, agents, command, parsed_command, record_frames):
    task_index = int(parsed_command["task_index"])
    skill = parsed_command.get("skill", "place")
    agent = agents[skill]
    apply_agent_extra(env, agent)
    set_task_by_index(env, task_index)
    obs = env._get_obs()

    success = False
    for step_i in range(STEPS_PER_COMMAND):
        obs_norm = agent.obs_rms.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            if DETERMINISTIC:
                act_t, _ = agent.policy.deterministic_act(obs_t)
            else:
                act_t, _, _ = agent.policy.act(obs_t)

        act = act_t.cpu().numpy()
        obs, _, terminated, truncated, info = env.step(act)

        if record_frames is not None:
            record_frames.append(render_frame(env, command))

        if terminated:
            success = True
            break
        if truncated:
            break

    return success


def main():
    parsed = [parse_command(cmd) for cmd in COMMANDS]

    print("Parsed sequence:")
    for i, item in enumerate(parsed, 1):
        print(f"  {i}. {item}")

    env = RealFrankaPickPlaceEnv(render_mode="rgb_array")
    env.task_stage = STAGE_PLACE

    agents = {
        "place": load_agent(PLACE_CKPT, env),
        "stack": load_agent(STACK_CKPT, env),
    }
    first_skill = parsed[0].get("skill", "place") if parsed else "place"
    print(f"Place checkpoint: {PLACE_CKPT}")
    print(f"Stack checkpoint: {STACK_CKPT}")

    writer = imageio.get_writer(VIDEO_PATH, fps=35) if VIDEO_SEQUENCES > 0 else None
    saved = 0
    full_successes = 0

    try:
        for seq_i in range(N_SEQUENCES):
            apply_agent_extra(env, agents[first_skill])
            env.reset(options={"task_index": int(parsed[0]["task_index"])})
            env.task_stage = STAGE_PLACE

            frames = []
            sequence_ok = True

            for cmd_i, item in enumerate(parsed):
                command = COMMANDS[cmd_i]
                ok = run_one_command(
                    env,
                    agents,
                    command,
                    item,
                    frames if saved < VIDEO_SEQUENCES else None,
                )

                print(
                    f"Sequence {seq_i + 1}, command {cmd_i + 1}/{len(parsed)}: "
                    f"{command!r} -> success={ok}"
                )

                if not ok:
                    sequence_ok = False
                    break

                if cmd_i < len(parsed) - 1:
                    reset_robot_home_keep_scene(env)

            if sequence_ok:
                full_successes += 1
                if writer is not None and saved < VIDEO_SEQUENCES:
                    for frame in frames:
                        writer.append_data(frame)
                    saved += 1
                    print(f"Saved successful same-scene sequence {saved}/{VIDEO_SEQUENCES}")

            if saved >= VIDEO_SEQUENCES:
                break

    finally:
        if writer is not None:
            writer.close()
        env.close()

    print(f"\nFull sequence success: {full_successes}/{N_SEQUENCES}")
    print(f"Video saved to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
