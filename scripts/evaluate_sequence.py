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
    TASKS,
    CAMERAS,
    GRIPPER_OPEN_CTRL,
    STAGE_PLACE,
    RealFrankaPickPlaceEnv,
    BLOCK_X_RANGE,
    BLOCK_Y_RANGE,
    TARGET_X_RANGE,
    TARGET_Y_RANGE,
)
from language_conditioned_rl.llm_parser import parse_command
from language_conditioned_rl.ppo import PPO


DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "GOOD_78_best_place_success.pt"
CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT

SEQUENCE_COMMANDS = os.environ.get(
    "SEQUENCE_COMMANDS",
    "put the red block on the yellow plate ;; put the green block in the cyan bowl",
)
COMMANDS = [c.strip() for c in SEQUENCE_COMMANDS.split(";;") if c.strip()]

N_SEQUENCES = int(os.environ.get("N_SEQUENCES", 30))
VIDEO_SEQUENCES = int(os.environ.get("VIDEO_SEQUENCES", 1))
STEPS_PER_COMMAND = int(os.environ.get("STEPS_PER_COMMAND", 280))
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", str(PROJECT_ROOT / "same_scene_sequence.mp4")
)
CAMERA = os.environ.get("CAMERA", "fixed_scene")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    target = env._target_pos()

    env._prev_reach_dist = float(np.linalg.norm(block - ee))
    env._prev_approach_dist = float(np.linalg.norm(env._approach_pos(block) - ee))
    env._best_approach_dist = env._prev_approach_dist
    env._prev_place_dist = float(np.linalg.norm((target - block)[:2]))
    env._prev_lift_height = 0.0
    env._prev_ee_z = float(ee[2])
    env._prev_settle_score = 0.0
    env._step_count = 0
    env._success_hold = 0
    env._ever_grasped = False
    env._ever_lifted = False
    env._max_lift_height = 0.0


def setup_sequence_scene(env, parsed_tasks):
    used = []
    block_xy = {}

    for name in BLOCK_NAMES:
        xy = env._sample_xy(used, BLOCK_X_RANGE, BLOCK_Y_RANGE, min_sep=0.080)
        used.append(xy)
        block_xy[name] = xy
        env._set_block_pose(name, xy)

    placed_targets = set()
    for item in parsed_tasks:
        block = item["block"]
        target = item["target"]

        if target in placed_targets:
            continue

        target_xy = env._sample_target_xy(block_xy[block], used)
        used.append(target_xy)
        env._set_target_pose(target, target_xy)
        placed_targets.add(target)

    for name in TARGET_NAMES:
        if name in placed_targets:
            continue
        xy = env._sample_xy(used, TARGET_X_RANGE, TARGET_Y_RANGE, min_sep=0.095)
        used.append(xy)
        env._set_target_pose(name, xy)

    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    reset_task_progress(env)


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


def move_robot_home_keep_scene(env, frames=None, goal="returning home", steps=140):
    if env._home_key_id >= 0:
        home_qpos = env.model.key_qpos[env._home_key_id].copy()
        home_arm = home_qpos[env._act_qpos_addr[:7]].copy()
    else:
        home_arm = env.data.qpos[env._act_qpos_addr[:7]].copy()

    env._gripper_ctrl_target = GRIPPER_OPEN_CTRL
    env.data.ctrl[7] = env._gripper_ctrl_target

    for _ in range(steps):
        q_arm = env.data.qpos[env._act_qpos_addr[:7]].copy()
        diff = home_arm - q_arm

        step = np.clip(diff, -0.025, 0.025)
        env._arm_ctrl_target = q_arm + step
        env.data.ctrl[:7] = env._arm_ctrl_target
        env.data.ctrl[7] = GRIPPER_OPEN_CTRL

        mujoco.mj_step(env.model, env.data)

        if frames is not None:
            frames.append(render_frame(env, goal))

    mujoco.mj_forward(env.model, env.data)
    reset_task_progress(env)


def run_one_command(env, agent, command, task_index, record_frames):
    set_task_by_index(env, task_index)
    obs = env._get_obs()

    success = False
    for step_i in range(STEPS_PER_COMMAND):
        obs_norm = agent.obs_rms.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
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

    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        device=DEVICE,
    )
    agent.load(CKPT)

    if "curriculum_dist" in agent.extra:
        env.curriculum_dist = float(agent.extra["curriculum_dist"])
    if "lift_goal_height" in agent.extra:
        env.curriculum_lift_height = float(agent.extra["lift_goal_height"])
    if "success_radius" in agent.extra:
        env.success_radius = float(agent.extra["success_radius"])

    writer = imageio.get_writer(VIDEO_PATH, fps=35) if VIDEO_SEQUENCES > 0 else None
    saved = 0
    full_successes = 0

    try:
        for seq_i in range(N_SEQUENCES):
            env.reset()
            env.task_stage = STAGE_PLACE
            setup_sequence_scene(env, parsed)

            frames = []
            sequence_ok = True

            for cmd_i, item in enumerate(parsed):
                command = COMMANDS[cmd_i]
                ok = run_one_command(
                    env,
                    agent,
                    command,
                    int(item["task_index"]),
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
                    move_robot_home_keep_scene(
                        env,
                        frames if saved < VIDEO_SEQUENCES else None,
                        goal="returning to home",
                    )

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
