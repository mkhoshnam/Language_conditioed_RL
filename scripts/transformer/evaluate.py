#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import RealFrankaPickPlaceEnv, STAGE_PLACE
from language_conditioned_rl.llm_parser import parse_command
from language_conditioned_rl.task_config import TASKS
from language_conditioned_rl.transformer.config import ModelConfig, PPOConfig
from language_conditioned_rl.transformer.observation import LanguageFrankaEnv
from language_conditioned_rl.transformer.ppo import TransformerPPO


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a true-language transformer policy")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--command", default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--paraphrases", action="store_true")
    parser.add_argument("--intervention", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    payload = torch.load(args.checkpoint, map_location="cpu")
    model_config = ModelConfig(**payload["model_config"])
    ppo_config = PPOConfig(**payload["ppo_config"])
    agent = TransformerPPO(model_config, ppo_config, device=device)
    extra = agent.load(args.checkpoint)

    fixed_task = args.task_index
    if args.command:
        fixed_task = int(parse_command(args.command)["task_index"])
    base_env = RealFrankaPickPlaceEnv(fixed_task_index=fixed_task)
    base_env.task_stage = STAGE_PLACE
    saved_curriculum = extra.get("curriculum", {})
    if isinstance(saved_curriculum, dict):
        base_env.curriculum_dist = float(
            saved_curriculum.get("curriculum_dist", base_env.curriculum_dist)
        )
        base_env.curriculum_lift_height = float(
            saved_curriculum.get("curriculum_lift_height", base_env.curriculum_lift_height)
        )
        base_env.stack_task_fraction = saved_curriculum.get("stack_task_fraction", 0.5)
    env = LanguageFrankaEnv(
        env=base_env,
        paraphrase_probability=1.0 if args.paraphrases else 0.0,
        seed=17,
    )

    results = []
    try:
        for episode in range(args.episodes):
            state, command, info = env.reset(task_index=fixed_task)
            if args.command:
                command = args.command
                env.command = command
            success = False
            metrics = {"grasp": 0.0, "lift": 0.0, "release": 0.0}
            for _ in range(280):
                action, _, _, _ = agent.select_action(
                    state, command, update_normalizer=False, deterministic=True
                )
                state, _, terminated, truncated, step_info = env.step(action)
                metrics["grasp"] = max(metrics["grasp"], float(step_info["grasped"]))
                metrics["lift"] = max(metrics["lift"], float(step_info["lifted"]))
                metrics["release"] = max(metrics["release"], float(step_info["released"]))
                success = success or bool(step_info["success"])
                if terminated or truncated:
                    break
            results.append((success, metrics, str(info["skill"]), command))
            print(f"Episode {episode + 1:3d}: success={success} command={command!r}")

        print("\nEvaluation summary")
        print(f"Success: {np.mean([item[0] for item in results]):.1%}")
        print(f"Grasp:   {np.mean([item[1]['grasp'] for item in results]):.1%}")
        print(f"Lift:    {np.mean([item[1]['lift'] for item in results]):.1%}")
        print(f"Release: {np.mean([item[1]['release'] for item in results]):.1%}")

        if args.intervention:
            state, _, info = env.reset(task_index=fixed_task or 0)
            first_index = int(info["task_index"])
            first_skill = TASKS[first_index][2]
            second_index = next(
                index
                for index, task in enumerate(TASKS)
                if index != first_index and task[2] == first_skill
            )
            first_command = TASKS[first_index][3]
            second_command = TASKS[second_index][3]
            first_action, _, _, _ = agent.select_action(
                state, first_command, update_normalizer=False, deterministic=True
            )
            second_action, _, _, _ = agent.select_action(
                state, second_command, update_normalizer=False, deterministic=True
            )
            print("\nSame-scene command intervention")
            print(f"A: {first_command!r}")
            print(f"B: {second_command!r}")
            print(f"Action L2 difference: {np.linalg.norm(first_action - second_action):.6f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
