import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import (
    APPROACH_RADIUS,
    LIFT_CURRICULUM_START,
    LIFT_HEIGHT,
    RealFrankaPickPlaceEnv,
    STAGE_GRASP,
    STAGE_LIFT,
    STAGE_NAMES,
    STAGE_PLACE,
    STAGE_REACH,
    STAGE_TRANSPORT,
    TRANSPORT_RADIUS,
)
from language_conditioned_rl.ppo import PPO


TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 12_000_000))
N_STEPS = int(os.environ.get("N_STEPS", 4096))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", 5))
SAVE_INTERVAL = int(os.environ.get("SAVE_INTERVAL", 25))
ROLLING_WINDOW = int(os.environ.get("ROLLING_WINDOW", 50))
MIN_BEST_WINDOW = int(os.environ.get("MIN_BEST_WINDOW", 20))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
RESUME_CKPT = os.environ.get("RESUME_CKPT")
LR = float(os.environ.get("LR", 1.5e-4))
PLACE_LR = float(os.environ.get("PLACE_LR", LR * 0.35))
ENT_COEF = float(os.environ.get("ENT_COEF", 0.004))
PLACE_ENT_COEF = float(os.environ.get("PLACE_ENT_COEF", ENT_COEF * 0.35))
PLACE_CLIP_EPS = float(os.environ.get("PLACE_CLIP_EPS", 0.10))

BEST_STAGE_CKPT = CKPT_DIR / "ppo_real_franka_best_stage.pt"
BEST_PLACE_SUCCESS_CKPT = CKPT_DIR / "ppo_real_franka_best_place_success.pt"
BEST_PLACE_HARD_CKPT = CKPT_DIR / "ppo_real_franka_best_place_hard.pt"
BEST_SETTLE_CKPT = CKPT_DIR / "ppo_real_franka_best_settle.pt"

CURRICULUM_START = float(os.environ.get("CURRICULUM_START", 0.10))
CURRICULUM_MAX = float(os.environ.get("CURRICULUM_MAX", 0.34))
CURRICULUM_STEP = float(os.environ.get("CURRICULUM_STEP", 0.02))
CURRICULUM_THRESH = float(os.environ.get("CURRICULUM_THRESH", 0.70))
CURRICULUM_PLACE_DIST = float(os.environ.get("CURRICULUM_PLACE_DIST", 0.040))
CURRICULUM_COOLDOWN_UPDATES = int(os.environ.get("CURRICULUM_COOLDOWN_UPDATES", 40))
SUCCESS_RADIUS_START = float(os.environ.get("SUCCESS_RADIUS_START", 0.075))
SUCCESS_RADIUS_MIN = float(os.environ.get("SUCCESS_RADIUS_MIN", 0.060))
SUCCESS_RADIUS_STEP = float(os.environ.get("SUCCESS_RADIUS_STEP", 0.0025))
SUCCESS_RADIUS_TIGHTEN_SUCC = float(os.environ.get("SUCCESS_RADIUS_TIGHTEN_SUCC", 0.50))
STAGE_COOLDOWN_UPDATES = int(os.environ.get("STAGE_COOLDOWN_UPDATES", 10))
REACH_TO_GRASP_SUCCESS = float(os.environ.get("REACH_TO_GRASP_SUCCESS", 0.80))
REACH_TO_GRASP_DIST = APPROACH_RADIUS + 0.010
GRASP_TO_LIFT_SUCCESS = float(os.environ.get("GRASP_TO_LIFT_SUCCESS", 0.78))
LIFT_TO_PLACE_SUCCESS = float(os.environ.get("LIFT_TO_PLACE_SUCCESS", 0.82))
LIFT_CURRICULUM_STEP = float(os.environ.get("LIFT_CURRICULUM_STEP", 0.010))
LIFT_CURRICULUM_SUCCESS = float(os.environ.get("LIFT_CURRICULUM_SUCCESS", 0.82))
LIFT_TO_PLACE_SECURE = float(os.environ.get("LIFT_TO_PLACE_SECURE", 0.88))
LIFT_TO_PLACE_MARGIN = float(os.environ.get("LIFT_TO_PLACE_MARGIN", 0.010))
TRANSPORT_TO_PLACE_SUCCESS = float(os.environ.get("TRANSPORT_TO_PLACE_SUCCESS", 0.65))
TRANSPORT_TO_PLACE_DIST = float(os.environ.get("TRANSPORT_TO_PLACE_DIST", TRANSPORT_RADIUS + 0.010))
TRANSPORT_MAX_DROPFAR = float(os.environ.get("TRANSPORT_MAX_DROPFAR", 0.18))


def _mean(values, window, default=0.0):
    if not values or window <= 0:
        return default
    return float(np.mean(values[-window:]))


def train():
    env = RealFrankaPickPlaceEnv()
    env.curriculum_dist = CURRICULUM_START
    env.curriculum_lift_height = LIFT_CURRICULUM_START
    env.success_radius = SUCCESS_RADIUS_START
    task_stage = STAGE_REACH
    env.task_stage = task_stage

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=LR,
        n_steps=N_STEPS,
        n_epochs=6,
        batch_size=256,
        clip_eps=0.15,
        vf_coef=0.5,
        ent_coef=ENT_COEF,
        gamma=0.99,
        lam=0.95,
        device=DEVICE,
    )

    def set_optimizer_lr(lr):
        for group in agent.optim.param_groups:
            group["lr"] = lr

    def set_transport_ppo():
        set_optimizer_lr(PLACE_LR)
        agent.ent_coef = PLACE_ENT_COEF
        agent.clip_eps = PLACE_CLIP_EPS

    global_step = 0
    update_num = 0
    last_curriculum_update = 0
    last_stage_update = 0
    best_stage_score = (-1, -1.0, -1.0, -1.0, -1e9)
    best_place_success_score = (-1.0, -1.0, -1e9, -1e9)
    best_place_hard_score = (-1.0, -1.0, -1e9, -1e9)
    best_settle_score = (-1.0, -1.0, -1.0, -1e9)

    if RESUME_CKPT:
        agent.load(RESUME_CKPT)
        agent.policy.train()
        global_step = int(agent.extra.get("global_step", 0))
        update_num = int(agent.extra.get("update_num", 0))
        env.curriculum_dist = float(agent.extra.get("curriculum_dist", env.curriculum_dist))
        env.curriculum_lift_height = float(
            agent.extra.get("lift_goal_height", env.curriculum_lift_height)
        )
        env.success_radius = float(agent.extra.get("success_radius", env.success_radius))
        task_stage = int(agent.extra.get("task_stage", task_stage))
        env.task_stage = task_stage
        if task_stage >= STAGE_PLACE:
            set_transport_ppo()
        last_stage_update = update_num
        last_curriculum_update = update_num
        best_stage_score = tuple(agent.extra.get("best_stage_score", best_stage_score))
        best_place_success_score = tuple(
            agent.extra.get("best_place_success_score", best_place_success_score)
        )
        best_place_hard_score = tuple(
            agent.extra.get("best_place_hard_score", best_place_hard_score)
        )
        best_settle_score = tuple(agent.extra.get("best_settle_score", best_settle_score))

    print(f"Real Franka pick-place PPO on {DEVICE} | obs={obs_dim} act={act_dim}")
    print("Scene: calvin_franka_scene/calvin_scene.xml | actions: EE delta + wrist rotation + gripper")
    print(f"Total steps: {TOTAL_STEPS:,} | Steps/update: {N_STEPS}")
    print(
        f"LR {LR:g} | PlaceLR {PLACE_LR:g} | "
        f"EntCoef {ENT_COEF:g} | PlaceEnt {PLACE_ENT_COEF:g} | "
        f"PPO clip 0.15 | PlaceClip {PLACE_CLIP_EPS:g} | epochs 6"
    )
    if RESUME_CKPT:
        print(
            f"Resumed {RESUME_CKPT} | stage={STAGE_NAMES[task_stage]} | "
            f"steps={global_step:,} | update={update_num}"
        )
    print("-" * 118)

    def checkpoint_extra(kind, extra_metrics=None):
        extra = {
            "checkpoint_kind": kind,
            "global_step": global_step,
            "update_num": update_num,
            "curriculum_dist": env.curriculum_dist,
            "lift_goal_height": env.curriculum_lift_height,
            "success_radius": env.success_radius,
            "task_stage": task_stage,
            "stage_name": STAGE_NAMES[task_stage],
            "best_stage_score": list(best_stage_score),
            "best_place_success_score": list(best_place_success_score),
            "best_place_hard_score": list(best_place_hard_score),
            "best_settle_score": list(best_settle_score),
        }
        if extra_metrics:
            extra.update(extra_metrics)
        return extra

    def save_checkpoint(path, kind, extra_metrics=None):
        agent.save(path, extra=checkpoint_extra(kind, extra_metrics))
        print(f"  Saved {path}")

    def reset_episode_trackers():
        return {
            "ret": 0.0,
            "len": 0,
            "best_reach": np.inf,
            "best_approach": np.inf,
            "best_place": np.inf,
            "grasped": 0.0,
            "secure": 0.0,
            "lifted": 0.0,
            "max_lift": 0.0,
            "task_success": 0.0,
            "unstable": 0.0,
            "released": 0.0,
            "open_near": 0.0,
            "drop_far": 0.0,
            "held": 0.0,
            "max_gopen": 0.0,
            "gate": 0.0,
            "guard": 0.0,
            "over_lift": 0.0,
            "ee_down": -1.0,
            "place_ee_down": 0.0,
            "on_table": 0.0,
            "best_settle": 0.0,
        }

    def clear_history():
        ep_rets.clear()
        ep_lens.clear()
        ep_best_reaches.clear()
        ep_best_approaches.clear()
        ep_best_places.clear()
        ep_grasped_flags.clear()
        ep_secure_flags.clear()
        ep_lifted_flags.clear()
        ep_max_lifts.clear()
        ep_task_successes.clear()
        ep_unstables.clear()
        ep_released_flags.clear()
        ep_open_near_flags.clear()
        ep_drop_far_flags.clear()
        ep_held_flags.clear()
        ep_max_gopens.clear()
        ep_gate_flags.clear()
        ep_guard_flags.clear()
        ep_over_lifts.clear()
        ep_ee_down_scores.clear()
        ep_place_ee_down_scores.clear()
        ep_on_table_flags.clear()
        ep_best_settles.clear()
        successes.clear()

    obs, _ = env.reset()
    ep = reset_episode_trackers()
    ep_rets = []
    ep_lens = []
    ep_best_reaches = []
    ep_best_approaches = []
    ep_best_places = []
    ep_grasped_flags = []
    ep_secure_flags = []
    ep_lifted_flags = []
    ep_max_lifts = []
    ep_task_successes = []
    ep_unstables = []
    ep_released_flags = []
    ep_open_near_flags = []
    ep_drop_far_flags = []
    ep_held_flags = []
    ep_max_gopens = []
    ep_gate_flags = []
    ep_guard_flags = []
    ep_over_lifts = []
    ep_ee_down_scores = []
    ep_place_ee_down_scores = []
    ep_on_table_flags = []
    ep_best_settles = []
    successes = []
    session_start_step = global_step
    t0 = time.time()

    while global_step < TOTAL_STEPS:
        env.task_stage = task_stage
        act, logp, val = agent.select_action(obs)
        next_obs, rew, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        agent.store(obs, act, logp, rew, val, float(done))
        obs = next_obs
        ep["ret"] += rew
        ep["len"] += 1
        ep["best_reach"] = min(ep["best_reach"], info["reach_dist"])
        ep["best_approach"] = min(ep["best_approach"], info["approach_dist"])
        ep["best_place"] = min(ep["best_place"], info["place_dist"])
        ep["grasped"] = max(ep["grasped"], info["grasped"])
        ep["secure"] = max(ep["secure"], info["secure_grasped"])
        ep["lifted"] = max(ep["lifted"], info["lifted"])
        ep["max_lift"] = max(ep["max_lift"], info["max_lift_height"])
        ep["task_success"] = max(ep["task_success"], float(info["task_success"]))
        ep["unstable"] = max(ep["unstable"], info["unstable"])
        ep["released"] = max(ep["released"], info["released"])
        ep["open_near"] = max(ep["open_near"], info["open_near_target"])
        ep["drop_far"] = max(ep["drop_far"], info["dropped_far"])
        ep["held"] = max(ep["held"], info["held_like"])
        ep["max_gopen"] = max(ep["max_gopen"], info["post_lift_gripper_open"])
        ep["gate"] = max(ep["gate"], info["gripper_gate_closed"])
        ep["guard"] = max(ep["guard"], info["carry_guard"])
        ep["over_lift"] = max(ep["over_lift"], info["over_lift"])
        ep["ee_down"] = max(ep["ee_down"], info["ee_z_down"])
        ep["place_ee_down"] = max(ep["place_ee_down"], info["place_ee_down_score"])
        ep["on_table"] = max(ep["on_table"], info["cube_on_table"])
        ep["best_settle"] = max(ep["best_settle"], info["settle_score"])
        global_step += 1

        if done:
            ep_rets.append(ep["ret"])
            ep_lens.append(ep["len"])
            ep_best_reaches.append(ep["best_reach"])
            ep_best_approaches.append(ep["best_approach"])
            ep_best_places.append(ep["best_place"])
            ep_grasped_flags.append(ep["grasped"])
            ep_secure_flags.append(ep["secure"])
            ep_lifted_flags.append(ep["lifted"])
            ep_max_lifts.append(ep["max_lift"])
            ep_task_successes.append(ep["task_success"])
            ep_unstables.append(ep["unstable"])
            ep_released_flags.append(ep["released"])
            ep_open_near_flags.append(ep["open_near"])
            ep_drop_far_flags.append(ep["drop_far"])
            ep_held_flags.append(ep["held"])
            ep_max_gopens.append(ep["max_gopen"])
            ep_gate_flags.append(ep["gate"])
            ep_guard_flags.append(ep["guard"])
            ep_over_lifts.append(ep["over_lift"])
            ep_ee_down_scores.append(ep["ee_down"])
            ep_place_ee_down_scores.append(ep["place_ee_down"])
            ep_on_table_flags.append(ep["on_table"])
            ep_best_settles.append(ep["best_settle"])
            successes.append(float(info["success"]))
            ep = reset_episode_trackers()
            obs, _ = env.reset()

        if agent.buffer.full():
            metrics = agent.update(obs)
            update_num += 1

            if update_num % LOG_INTERVAL == 0:
                window = min(ROLLING_WINDOW, len(successes))
                mean_ret = _mean(ep_rets, window)
                mean_len = _mean(ep_lens, window)
                mean_suc = _mean(successes, window)
                mean_task_suc = _mean(ep_task_successes, window)
                mean_grasp = _mean(ep_grasped_flags, window)
                mean_secure = _mean(ep_secure_flags, window)
                mean_lift = _mean(ep_lifted_flags, window)
                mean_max_lift = _mean(ep_max_lifts, window)
                mean_unstable = _mean(ep_unstables, window)
                mean_released = _mean(ep_released_flags, window)
                mean_open_near = _mean(ep_open_near_flags, window)
                mean_drop_far = _mean(ep_drop_far_flags, window)
                mean_held = _mean(ep_held_flags, window)
                mean_gopen = _mean(ep_max_gopens, window)
                mean_gate = _mean(ep_gate_flags, window)
                mean_guard = _mean(ep_guard_flags, window)
                mean_over_lift = _mean(ep_over_lifts, window)
                mean_ee_down = _mean(ep_ee_down_scores, window)
                mean_place_ee_down = _mean(ep_place_ee_down_scores, window)
                mean_on_table = _mean(ep_on_table_flags, window)
                mean_settle = _mean(ep_best_settles, window)
                mean_reach = _mean(ep_best_reaches, window, np.nan)
                mean_approach = _mean(ep_best_approaches, window, np.nan)
                mean_place = _mean(ep_best_places, window, np.nan)
                sps = (global_step - session_start_step) / max(1e-6, time.time() - t0)
                print(
                    f"Update {update_num:4d} | "
                    f"Steps {global_step:>10,} | "
                    f"Stage {STAGE_NAMES[task_stage]:>5} | "
                    f"Return {mean_ret:8.2f} | "
                    f"Success {mean_suc:6.2%} | "
                    f"Grasp {mean_grasp:6.2%} | "
                    f"Secure {mean_secure:6.2%} | "
                    f"Lift {mean_lift:6.2%} | "
                    f"Place {mean_task_suc:6.2%} | "
                    f"MaxLift {mean_max_lift * 100:4.1f}cm | "
                    f"Len {mean_len:5.1f} | "
                    f"BestReach {mean_reach * 100:5.1f}cm | "
                    f"BestApproach {mean_approach * 100:5.1f}cm | "
                    f"BestPlace {mean_place * 100:5.1f}cm | "
                    f"Release {mean_released:5.1%} | "
                    f"OpenNear {mean_open_near:5.1%} | "
                    f"DropFar {mean_drop_far:5.1%} | "
                    f"Held {mean_held:5.1%} | "
                    f"GOpen {mean_gopen:4.2f} | "
                    f"Gate {mean_gate:5.1%} | "
                    f"Guard {mean_guard:5.1%} | "
                    f"OverLift {mean_over_lift * 100:4.1f}cm | "
                    f"EEDown {mean_place_ee_down:4.2f} | "
                    f"Table {mean_on_table:5.1%} | "
                    f"Settle {mean_settle:4.2f} | "
                    f"Unstable {mean_unstable:5.1%} | "
                    f"Loss {metrics['loss_total']:.4f} | "
                    f"CurrDist {env.curriculum_dist:.2f} | "
                    f"LiftGoal {env.curriculum_lift_height * 100:4.1f}cm | "
                    f"SPS {sps:.0f}"
                )

                if window >= MIN_BEST_WINDOW:
                    stage_score = (
                        int(task_stage),
                        mean_suc,
                        mean_task_suc,
                        -mean_unstable,
                        mean_ret,
                    )
                    if stage_score > best_stage_score:
                        best_stage_score = stage_score
                        save_checkpoint(
                            BEST_STAGE_CKPT,
                            "best_stage",
                            {
                                "rolling_success": mean_suc,
                                "rolling_place_success": mean_task_suc,
                                "rolling_return": mean_ret,
                                "rolling_unstable": mean_unstable,
                                "rolling_best_reach": mean_reach,
                                "rolling_best_approach": mean_approach,
                                "rolling_best_place": mean_place,
                                "rolling_released": mean_released,
                                "rolling_open_near": mean_open_near,
                                "rolling_drop_far": mean_drop_far,
                                "rolling_held": mean_held,
                                "rolling_gripper_open": mean_gopen,
                                "rolling_ee_down": mean_ee_down,
                                "rolling_place_ee_down": mean_place_ee_down,
                                "rolling_on_table": mean_on_table,
                                "rolling_settle": mean_settle,
                            },
                        )
                        print(
                            "  >>> New best stage checkpoint: "
                            f"stage={STAGE_NAMES[task_stage]} success={mean_suc:.2%}"
                        )

                    if task_stage == STAGE_PLACE:
                        place_success_score = (
                            mean_task_suc,
                            env.curriculum_dist,
                            -mean_place,
                            -mean_unstable,
                        )
                        if place_success_score > best_place_success_score:
                            best_place_success_score = place_success_score
                            save_checkpoint(
                                BEST_PLACE_SUCCESS_CKPT,
                                "best_place_success",
                                {
                                    "rolling_place_success": mean_task_suc,
                                    "rolling_return": mean_ret,
                                    "rolling_unstable": mean_unstable,
                                    "rolling_best_place": mean_place,
                                    "rolling_max_lift": mean_max_lift,
                                    "rolling_released": mean_released,
                                    "rolling_open_near": mean_open_near,
                                    "rolling_drop_far": mean_drop_far,
                                    "rolling_held": mean_held,
                                    "rolling_gripper_open": mean_gopen,
                                    "rolling_ee_down": mean_ee_down,
                                    "rolling_place_ee_down": mean_place_ee_down,
                                    "rolling_on_table": mean_on_table,
                                    "rolling_settle": mean_settle,
                                },
                            )
                            print(
                                "  >>> New best place-success checkpoint: "
                                f"success={mean_task_suc:.2%} "
                                f"dist={env.curriculum_dist:.2f}m "
                                f"best_place={mean_place * 100:.1f}cm"
                            )

                        place_hard_score = (
                            env.curriculum_dist,
                            mean_task_suc,
                            -mean_place,
                            -mean_unstable,
                        )
                        if place_hard_score > best_place_hard_score:
                            best_place_hard_score = place_hard_score
                            save_checkpoint(
                                BEST_PLACE_HARD_CKPT,
                                "best_place_hard",
                                {
                                    "rolling_place_success": mean_task_suc,
                                    "rolling_return": mean_ret,
                                    "rolling_unstable": mean_unstable,
                                    "rolling_best_place": mean_place,
                                    "rolling_max_lift": mean_max_lift,
                                    "rolling_released": mean_released,
                                    "rolling_open_near": mean_open_near,
                                    "rolling_drop_far": mean_drop_far,
                                    "rolling_held": mean_held,
                                    "rolling_gripper_open": mean_gopen,
                                    "rolling_ee_down": mean_ee_down,
                                    "rolling_place_ee_down": mean_place_ee_down,
                                    "rolling_on_table": mean_on_table,
                                    "rolling_settle": mean_settle,
                                },
                            )
                            print(
                                "  >>> New best hard-place checkpoint: "
                                f"dist={env.curriculum_dist:.2f}m "
                                f"success={mean_task_suc:.2%} "
                                f"best_place={mean_place * 100:.1f}cm"
                            )

                        settle_score = (
                            mean_settle,
                            mean_task_suc,
                            env.curriculum_dist,
                            -mean_unstable,
                        )
                        if settle_score > best_settle_score:
                            best_settle_score = settle_score
                            save_checkpoint(
                                BEST_SETTLE_CKPT,
                                "best_settle",
                                {
                                    "rolling_place_success": mean_task_suc,
                                    "rolling_return": mean_ret,
                                    "rolling_unstable": mean_unstable,
                                    "rolling_best_place": mean_place,
                                    "rolling_settle": mean_settle,
                                    "rolling_released": mean_released,
                                    "rolling_open_near": mean_open_near,
                                    "rolling_drop_far": mean_drop_far,
                                    "rolling_held": mean_held,
                                    "rolling_gripper_open": mean_gopen,
                                    "rolling_ee_down": mean_ee_down,
                                    "rolling_place_ee_down": mean_place_ee_down,
                                    "rolling_on_table": mean_on_table,
                                },
                            )
                            print(
                                "  >>> New best settle checkpoint: "
                                f"settle={mean_settle:.2f} success={mean_task_suc:.2%}"
                            )

                cooldown_elapsed = (
                    update_num - last_curriculum_update >= CURRICULUM_COOLDOWN_UPDATES
                )
                stage_cooldown_elapsed = (
                    update_num - last_stage_update >= STAGE_COOLDOWN_UPDATES
                )
                stage_changed = False
                if (
                    task_stage == STAGE_REACH
                    and window >= ROLLING_WINDOW
                    and stage_cooldown_elapsed
                    and mean_suc >= REACH_TO_GRASP_SUCCESS
                    and mean_approach <= REACH_TO_GRASP_DIST
                ):
                    task_stage = STAGE_GRASP
                    env.task_stage = task_stage
                    last_stage_update = update_num
                    stage_changed = True
                    print("  >>> Stage advanced to grasp: contact objective enabled")
                elif (
                    task_stage == STAGE_GRASP
                    and window >= ROLLING_WINDOW
                    and stage_cooldown_elapsed
                    and mean_suc >= GRASP_TO_LIFT_SUCCESS
                    and mean_grasp >= GRASP_TO_LIFT_SUCCESS
                ):
                    task_stage = STAGE_LIFT
                    env.task_stage = task_stage
                    env.curriculum_lift_height = LIFT_CURRICULUM_START
                    last_stage_update = update_num
                    stage_changed = True
                    print(
                        "  >>> Stage advanced to lift: "
                        f"lift target starts at {env.curriculum_lift_height * 100:.1f}cm"
                    )
                elif (
                    task_stage == STAGE_LIFT
                    and window >= ROLLING_WINDOW
                    and stage_cooldown_elapsed
                    and mean_suc >= LIFT_TO_PLACE_SUCCESS
                    and mean_lift >= LIFT_TO_PLACE_SUCCESS
                ):
                    if env.curriculum_lift_height < LIFT_HEIGHT - 1e-6:
                        env.curriculum_lift_height = min(
                            env.curriculum_lift_height + LIFT_CURRICULUM_STEP,
                            LIFT_HEIGHT,
                        )
                        last_stage_update = update_num
                        stage_changed = True
                        print(
                            "  >>> Lift curriculum advanced to "
                            f"{env.curriculum_lift_height * 100:.1f}cm"
                        )
                    elif mean_suc >= LIFT_CURRICULUM_SUCCESS and mean_max_lift >= LIFT_HEIGHT:
                        if (
                            mean_secure >= LIFT_TO_PLACE_SECURE
                            and mean_lift >= LIFT_TO_PLACE_SUCCESS
                            and mean_max_lift >= LIFT_HEIGHT + LIFT_TO_PLACE_MARGIN
                        ):
                            task_stage = STAGE_TRANSPORT
                            env.task_stage = task_stage
                            last_stage_update = update_num
                            stage_changed = True
                            pre_transport = os.path.join(
                                CKPT_DIR, "ppo_real_franka_pre_transport.pt"
                            )
                            save_checkpoint(pre_transport, "pre_transport")
                            print(
                                "  >>> Stage advanced to transport: "
                                "carry-to-target objective enabled"
                            )
                        else:
                            print(
                                "  --- Transport held: waiting for stronger lift "
                                f"(secure {mean_secure:.2%}, lift {mean_lift:.2%}, "
                                f"max_lift {mean_max_lift * 100:.1f}cm)"
                            )
                elif (
                    task_stage == STAGE_TRANSPORT
                    and window >= ROLLING_WINDOW
                    and stage_cooldown_elapsed
                    and mean_suc >= TRANSPORT_TO_PLACE_SUCCESS
                    and mean_place <= TRANSPORT_TO_PLACE_DIST
                    and mean_drop_far <= TRANSPORT_MAX_DROPFAR
                    and mean_lift >= 0.65
                    and mean_held >= 0.85
                ):
                    task_stage = STAGE_PLACE
                    env.task_stage = task_stage
                    set_transport_ppo()
                    last_stage_update = update_num
                    stage_changed = True
                    pre_release = os.path.join(CKPT_DIR, "ppo_real_franka_pre_release.pt")
                    save_checkpoint(pre_release, "pre_release")
                    print(
                        "  >>> Stage advanced to place: "
                        "release-and-settle objective enabled"
                    )
                elif (
                    task_stage == STAGE_PLACE
                    and window >= ROLLING_WINDOW
                    and mean_task_suc >= CURRICULUM_THRESH
                    and mean_place <= CURRICULUM_PLACE_DIST
                    and mean_unstable <= 0.05
                    and cooldown_elapsed
                    and env.curriculum_dist < CURRICULUM_MAX
                ):
                    env.curriculum_dist = min(
                        env.curriculum_dist + CURRICULUM_STEP, CURRICULUM_MAX
                    )
                    last_curriculum_update = update_num
                    print(f"  >>> Place curriculum advanced to {env.curriculum_dist:.2f}m")

                if (
                    task_stage == STAGE_PLACE
                    and window >= ROLLING_WINDOW
                    and mean_task_suc >= SUCCESS_RADIUS_TIGHTEN_SUCC
                    and mean_place <= env.success_radius - 0.005
                    and env.success_radius > SUCCESS_RADIUS_MIN
                ):
                    env.success_radius = max(
                        env.success_radius - SUCCESS_RADIUS_STEP,
                        SUCCESS_RADIUS_MIN,
                    )
                    print(
                        "  >>> Success radius tightened to "
                        f"{env.success_radius * 100:.1f}cm"
                    )

                if stage_changed:
                    clear_history()
                    ep = reset_episode_trackers()
                    obs, _ = env.reset()

            if update_num % SAVE_INTERVAL == 0:
                ckpt = os.path.join(CKPT_DIR, f"ppo_real_franka_{update_num}.pt")
                save_checkpoint(ckpt, "periodic")

    final_path = os.path.join(CKPT_DIR, "ppo_real_franka_final.pt")
    save_checkpoint(final_path, "final")
    env.close()
    print(f"Training complete. Saved {final_path}")


if __name__ == "__main__":
    train()
