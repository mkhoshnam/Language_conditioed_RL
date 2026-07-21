#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import RealFrankaPickPlaceEnv
from language_conditioned_rl.transformer.config import ExperimentConfig
from language_conditioned_rl.transformer.curriculum import CurriculumManager, EpisodeSummary
from language_conditioned_rl.transformer.observation import LanguageFrankaEnv
from language_conditioned_rl.transformer.ppo import TransformerPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train true-language transformer PPO")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "transformer" / "place.json",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override session steps")
    parser.add_argument(
        "--text-backend",
        choices=("pretrained", "hash"),
        default=None,
        help="hash is intended only for offline smoke tests",
    )
    return parser.parse_args()


@dataclass
class EpisodeTracker:
    episode_return: float = 0.0
    length: int = 0
    grasped: float = 0.0
    lifted: float = 0.0
    released: float = 0.0
    stable: float = 0.0
    best_reach: float = float("inf")
    best_place: float = float("inf")
    skill: str = "place"

    def step(self, reward: float, info: dict[str, object]) -> None:
        self.episode_return += reward
        self.length += 1
        self.grasped = max(self.grasped, float(info["grasped"]))
        self.lifted = max(self.lifted, float(info["lifted"]))
        self.released = max(self.released, float(info["released"]))
        self.stable = max(self.stable, float(info["settle_score"]))
        self.best_reach = min(self.best_reach, float(info["reach_dist"]))
        self.best_place = min(self.best_place, float(info["place_dist"]))
        self.skill = str(info["skill"])

    def finish(self, info: dict[str, object]) -> EpisodeSummary:
        return EpisodeSummary(
            stage=int(info["stage"]),
            skill=self.skill,
            success=float(info["success"]),
            task_success=float(info["task_success"]),
            grasped=self.grasped,
            lifted=self.lifted,
            released=self.released,
            stable=self.stable,
            best_reach=self.best_reach,
            best_place=self.best_place,
            episode_return=self.episode_return,
            length=self.length,
        )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.load(args.config)
    if args.resume is not None:
        config.resume_checkpoint = str(args.resume)
    if args.device is not None:
        config.device = args.device
    if args.steps is not None:
        config.total_steps = args.steps
    if args.text_backend is not None:
        config.model.text_backend = args.text_backend
    config.validate()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config.device)

    base_env = RealFrankaPickPlaceEnv()
    env = LanguageFrankaEnv(
        env=base_env,
        paraphrase_probability=config.curriculum.paraphrase_probability,
        seed=config.seed,
        task_indices=config.train_task_indices,
    )
    curriculum = CurriculumManager(config.curriculum, base_env, seed=config.seed)
    agent = TransformerPPO(
        config.model,
        config.ppo,
        device=device,
        action_dim=env.action_space.shape[0],
        transfer=config.curriculum.phase == "transfer",
    )

    global_step = 0
    update = 0
    if config.resume_checkpoint:
        resume_path = Path(config.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        extra = agent.load(
            resume_path,
            load_optimizer=config.curriculum.phase != "transfer",
        )
        global_step = int(extra.get("global_step", 0))
        update = int(extra.get("update", 0))
        saved_curriculum = extra.get("curriculum")
        if isinstance(saved_curriculum, dict):
            curriculum.load_state_dict(
                saved_curriculum,
                preserve_phase=config.curriculum.phase == "transfer",
            )
        print(f"Loaded checkpoint: {resume_path}")

    checkpoint_dir = PROJECT_ROOT / config.checkpoint_dir
    log_dir = PROJECT_ROOT / config.log_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{config.name}.jsonl"
    best_path = checkpoint_dir / f"{config.name}_best.pt"
    final_path = checkpoint_dir / f"{config.name}_final.pt"
    trainable, total = agent.policy.parameter_count()
    print(
        f"True-language Transformer PPO | phase={config.curriculum.phase} device={device} "
        f"trainable={trainable:,}/{total:,} text={config.model.text_model_name}"
    )
    print(
        f"Session steps={config.total_steps:,} rollout={config.ppo.rollout_steps} "
        f"stage={curriculum.stage_name} stack_fraction={base_env.stack_task_fraction:.0%}"
    )

    state, command, _ = env.reset(seed=config.seed)
    base_env.task_stage = curriculum.sample_stage()
    tracker = EpisodeTracker()
    session_step = 0
    start_time = time.time()
    best_score = -float("inf")

    def checkpoint_extra() -> dict[str, object]:
        return {
            "experiment": config.to_dict(),
            "global_step": global_step,
            "update": update,
            "curriculum": curriculum.state_dict(),
        }

    try:
        while session_step < config.total_steps:
            action, log_probability, value, normalized = agent.select_action(
                state, command, update_normalizer=True
            )
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            stored_reward = reward
            timeout = truncated and not terminated and not bool(info.get("unstable", 0.0))
            if timeout:
                stored_reward += config.ppo.gamma * agent.value(next_state, command)
            agent.store(
                normalized,
                command,
                action,
                log_probability,
                stored_reward,
                value,
                done,
                str(info["skill"]),
            )
            tracker.step(reward, info)
            state = next_state
            session_step += 1
            global_step += 1

            if done:
                curriculum.record(tracker.finish(info))
                tracker = EpisodeTracker()
                state, command, _ = env.reset()
                base_env.task_stage = curriculum.sample_stage()

            if agent.buffer.full:
                losses = agent.update(state, command)
                update += 1
                change = curriculum.maybe_advance(update)
                if change:
                    print(f"  Curriculum: {change}")

                if update % config.log_interval_updates == 0:
                    rolling = curriculum.metrics()
                    elapsed = max(time.time() - start_time, 1.0e-6)
                    record = {
                        "global_step": global_step,
                        "session_step": session_step,
                        "update": update,
                        "stage": curriculum.stage_name,
                        "stack_fraction": base_env.stack_task_fraction,
                        "curriculum_distance": base_env.curriculum_dist,
                        "steps_per_second": session_step / elapsed,
                        **rolling,
                        **losses,
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=True) + "\n")
                    print(
                        f"Update {update:5d} | steps {global_step:>10,} | "
                        f"stage {curriculum.stage_name:>9} | return {rolling['return']:7.2f} | "
                        f"success {rolling['success']:6.1%} | place {rolling['place_success']:6.1%} | "
                        f"stack {rolling['stack_success']:6.1%} | grasp {rolling['grasp']:6.1%} | "
                        f"lift {rolling['lift']:6.1%} | KL {losses['kl']:.4f} | "
                        f"stack mix {base_env.stack_task_fraction:.0%}"
                    )
                    stack_score = rolling["stack_success"]
                    if np.isnan(stack_score):
                        stack_score = 0.0
                    place_score = rolling["place_success"]
                    if np.isnan(place_score):
                        place_score = 0.0
                    score = (
                        rolling["success"]
                        if config.curriculum.phase == "place"
                        else stack_score + 0.5 * place_score
                    )
                    if score > best_score and rolling["episodes"] >= 10:
                        best_score = score
                        agent.save(best_path, checkpoint_extra())
                        print(f"  Saved best checkpoint: {best_path}")

                if update % config.save_interval_updates == 0:
                    periodic = checkpoint_dir / f"{config.name}_update_{update:05d}.pt"
                    agent.save(periodic, checkpoint_extra())
                    print(f"  Saved periodic checkpoint: {periodic}")
    finally:
        agent.save(final_path, checkpoint_extra())
        env.close()
        print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
