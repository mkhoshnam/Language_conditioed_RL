import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import CAMERAS, RealFrankaPickPlaceEnv, STAGE_NAMES, STAGE_PLACE
from language_conditioned_rl.ppo import PPO


DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "ppo_real_franka_best_place_success.pt"
CKPT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
N_EPISODES = int(os.environ.get("N_EPISODES", 20))
VIDEO_EPISODES = int(os.environ.get("VIDEO_EPISODES", 3))
VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", str(PROJECT_ROOT / "eval_real_franka.mp4")
)
CAMERA = os.environ.get("CAMERA", "fixed_scene")
TASK_INDEX = os.environ.get("TASK_INDEX")
COMMAND = os.environ.get("COMMAND")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _stage_from_env():
    value = os.environ.get("EVAL_STAGE", "place").lower()
    if value.isdigit():
        stage = int(value)
        if 0 <= stage < len(STAGE_NAMES):
            return stage
    if value in STAGE_NAMES:
        return STAGE_NAMES.index(value)
    return STAGE_PLACE


def _put_text(img, lines):
    import cv2

    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 28
    return out


def _render_frame(env, info=None, step_i=None, goal=None):
    if CAMERA == "both":
        fixed = env.render("fixed_scene")
        wrist = env.render("wrist_camera")
        frame = np.concatenate([fixed, wrist], axis=1)
    else:
        if CAMERA not in CAMERAS:
            raise ValueError(f"CAMERA must be one of {CAMERAS} or 'both', got {CAMERA!r}")
        frame = env.render(CAMERA)

    if info is not None:
        lines = [
            f"goal: {goal}",
        ]
        frame = _put_text(frame, lines)

    return frame


def evaluate():
    fixed_task = int(TASK_INDEX) if TASK_INDEX is not None else None
    parsed_command = None

    if COMMAND and fixed_task is None:
        from language_conditioned_rl.llm_parser import parse_command

        parsed_command = parse_command(COMMAND)
        fixed_task = int(parsed_command["task_index"])
        print(f"Parsed command: {parsed_command}")

    env = RealFrankaPickPlaceEnv(
        render_mode="rgb_array" if VIDEO_EPISODES > 0 else None,
        fixed_task_index=fixed_task,
    )
    env.task_stage = _stage_from_env()
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

    print(f"Loaded {CKPT}")
    print(
        f"Eval stage: {STAGE_NAMES[env.task_stage]} | "
        f"curriculum distance: {env.curriculum_dist:.2f}m | "
        f"lift goal: {env.curriculum_lift_height * 100:.1f}cm"
    )
    if fixed_task is not None:
        print(f"Fixed task index: {fixed_task}")

    writer = None
    successes = []
    grasped = []
    secure_grasped = []
    lifted = []
    released = []
    open_near = []
    drop_far = []
    held = []
    max_gopen = []
    ee_down = []
    place_ee_down = []
    at_dest = []
    max_lifts = []
    best_places = []
    best_settles = []
    goals = []
    saved_video_episodes = 0

    try:
        if VIDEO_EPISODES > 0:
            writer = imageio.get_writer(VIDEO_PATH, fps=35)
            print(
                f"Recording first {min(VIDEO_EPISODES, N_EPISODES)} episode(s) "
                f"with camera={CAMERA} -> {VIDEO_PATH}"
            )

        for ep in range(N_EPISODES):
            obs, reset_info = env.reset()
            ep_success = False
            ep_grasped = False
            ep_secure = False
            ep_lifted = False
            ep_released = False
            ep_open_near = False
            ep_drop_far = False
            ep_held = False
            ep_max_gopen = 0.0
            ep_ee_down = -1.0
            ep_place_ee_down = 0.0
            ep_at_dest = False
            ep_max_lift = 0.0
            ep_best_place = np.inf
            ep_best_settle = 0.0
            goals.append(COMMAND if COMMAND else reset_info.get("language_goal", env.language_goal))
            episode_frames = []

            for _ in range(280):
                obs_norm = agent.obs_rms.normalize(obs)
                obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    act_t, _, _ = agent.policy.act(obs_t)
                act = act_t.cpu().numpy()

                obs, _, terminated, truncated, info = env.step(act)
                ep_success = ep_success or info["success"]
                ep_grasped = ep_grasped or bool(info["grasped"])
                ep_secure = ep_secure or bool(info["secure_grasped"])
                ep_lifted = ep_lifted or bool(info["lifted"])
                ep_released = ep_released or bool(info["released"])
                ep_open_near = ep_open_near or bool(info["open_near_target"])
                ep_drop_far = ep_drop_far or bool(info["dropped_far"])
                ep_held = ep_held or bool(info["held_like"])
                ep_max_gopen = max(ep_max_gopen, info["post_lift_gripper_open"])
                ep_ee_down = max(ep_ee_down, info["ee_z_down"])
                ep_place_ee_down = max(ep_place_ee_down, info["place_ee_down_score"])
                ep_at_dest = ep_at_dest or bool(
                    info.get("cube_at_dest", info["cube_on_table"])
                )
                ep_max_lift = max(ep_max_lift, info["max_lift_height"])
                ep_best_place = min(ep_best_place, info["place_dist"])
                ep_best_settle = max(ep_best_settle, info["settle_score"])

                if writer is not None and saved_video_episodes < VIDEO_EPISODES:
                    episode_frames.append(_render_frame(env, info=info, step_i=_, goal=goals[-1]))

                if terminated or truncated:
                    break

            if writer is not None and ep_success and saved_video_episodes < VIDEO_EPISODES:
                for frame in episode_frames:
                    writer.append_data(frame)
                saved_video_episodes += 1
                print(f"  Saved successful video episode {saved_video_episodes}/{VIDEO_EPISODES}")

            successes.append(ep_success)
            grasped.append(ep_grasped)
            secure_grasped.append(ep_secure)
            lifted.append(ep_lifted)
            released.append(ep_released)
            open_near.append(ep_open_near)
            drop_far.append(ep_drop_far)
            held.append(ep_held)
            max_gopen.append(ep_max_gopen)
            ee_down.append(ep_ee_down)
            place_ee_down.append(ep_place_ee_down)
            at_dest.append(ep_at_dest)
            max_lifts.append(ep_max_lift)
            best_places.append(ep_best_place)
            best_settles.append(ep_best_settle)
            print(
                f"  Episode {ep + 1}: success={ep_success} "
                f"goal={goals[-1]!r} grasped={ep_grasped} secure={ep_secure} "
                f"lifted={ep_lifted} released={ep_released} at_dest={ep_at_dest} "
                f"open_near={ep_open_near} drop_far={ep_drop_far} "
                f"held={ep_held} gopen={ep_max_gopen:.2f} "
                f"ee_down={ep_place_ee_down:.2f} "
                f"max_lift={ep_max_lift:.3f}m best_place={ep_best_place:.3f}m "
                f"settle={ep_best_settle:.2f}"
            )
    finally:
        if writer is not None:
            writer.close()
        env.close()

    print(f"\nSuccess rate: {np.mean(successes):.0%}")
    print(f"Grasp rate: {np.mean(grasped):.0%}")
    print(f"Secure grasp rate: {np.mean(secure_grasped):.0%}")
    print(f"Lift rate: {np.mean(lifted):.0%}")
    print(f"Release rate: {np.mean(released):.0%}")
    print(f"Open-near-target rate: {np.mean(open_near):.0%}")
    print(f"Far-drop rate: {np.mean(drop_far):.0%}")
    print(f"Held-like rate: {np.mean(held):.0%}")
    print(f"Mean max gripper-open: {np.mean(max_gopen):.2f}")
    print(f"Mean best EE-down near target: {np.mean(place_ee_down):.2f}")
    print(f"At-destination-height rate: {np.mean(at_dest):.0%}")
    print(f"Mean max lift: {np.mean(max_lifts) * 100:.1f} cm")
    print(f"Mean best place dist: {np.mean(best_places) * 100:.1f} cm")
    print(f"Mean best settle score: {np.mean(best_settles):.2f}")


if __name__ == "__main__":
    evaluate()
