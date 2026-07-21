#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from language_conditioned_rl.env import RealFrankaPickPlaceEnv
from language_conditioned_rl.transformer.config import ModelConfig, PPOConfig
from language_conditioned_rl.transformer.observation import LanguageFrankaEnv
from language_conditioned_rl.transformer.ppo import TransformerPPO


def main() -> None:
    model = ModelConfig(
        text_backend="hash",
        d_model=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        state_mlp_hidden=64,
        actor_hidden=64,
        dropout=0.0,
    )
    ppo = PPOConfig(rollout_steps=16, epochs=1, batch_size=8)
    env = LanguageFrankaEnv(RealFrankaPickPlaceEnv(), paraphrase_probability=1.0, seed=3)
    agent = TransformerPPO(model, ppo, device="cpu")
    state, command, _ = env.reset(seed=3)
    total_reward = 0.0
    try:
        for _ in range(ppo.rollout_steps):
            action, log_probability, value, normalized = agent.select_action(state, command)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(
                normalized,
                command,
                action,
                log_probability,
                reward,
                value,
                done,
                str(info["skill"]),
            )
            total_reward += reward
            state = next_state
            if done:
                state, command, _ = env.reset()
        metrics = agent.update(state, command)
    finally:
        env.close()
    print(f"command: {command}")
    print(f"rollout reward: {total_reward:.3f}")
    print(f"PPO metrics: {metrics}")
    print("transformer smoke test passed")


if __name__ == "__main__":
    main()
