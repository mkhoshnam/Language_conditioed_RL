from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from language_conditioned_rl.transformer.config import ModelConfig, PPOConfig
from language_conditioned_rl.transformer.language import FrozenTextEncoder
from language_conditioned_rl.transformer.model import TransformerActorCritic
from language_conditioned_rl.transformer.observation import (
    SemanticNormalizer,
    SemanticState,
    index_state_batch,
    stack_states,
)


class RolloutBuffer:
    def __init__(self, capacity: int, action_dim: int):
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.clear()

    def clear(self) -> None:
        self.states: list[SemanticState] = []
        self.commands: list[str] = []
        self.actions: list[np.ndarray] = []
        self.log_probabilities: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[float] = []
        self.skills: list[int] = []

    @property
    def full(self) -> bool:
        return len(self.states) >= self.capacity

    def add(
        self,
        state: SemanticState,
        command: str,
        action: np.ndarray,
        log_probability: float,
        reward: float,
        value: float,
        done: bool,
        skill: str,
    ) -> None:
        if self.full:
            raise RuntimeError("rollout buffer is already full")
        self.states.append(state)
        self.commands.append(command)
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probabilities.append(float(log_probability))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))
        self.skills.append(int(skill == "stack"))

    def advantages_and_returns(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), device=device)
        next_value = torch.tensor(last_value, dtype=torch.float32, device=device)
        for index in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + gamma * next_value * nonterminal - values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[index] = gae
            next_value = values[index]
        return advantages, advantages + values


class TransformerPPO:
    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        device: str | torch.device,
        action_dim: int = 7,
        transfer: bool = False,
    ):
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.device = torch.device(device)
        self.text_encoder = FrozenTextEncoder(
            model_name=model_config.text_model_name,
            device=self.device,
            max_tokens=model_config.max_language_tokens,
            backend=model_config.text_backend,
        )
        self.policy = TransformerActorCritic(
            model_config,
            text_hidden_size=self.text_encoder.hidden_size,
            action_dim=action_dim,
        ).to(self.device)
        self.normalizer = SemanticNormalizer()
        self.buffer = RolloutBuffer(ppo_config.rollout_steps, action_dim)
        self.extra: dict[str, object] = {}
        self._build_optimizer(transfer=transfer)

    def _build_optimizer(self, transfer: bool) -> None:
        base_lr = (
            self.ppo_config.transfer_learning_rate
            if transfer
            else self.ppo_config.learning_rate
        )
        projection_parameters = list(self.policy.language_projection.parameters())
        projection_ids = {id(parameter) for parameter in projection_parameters}
        remaining = [
            parameter
            for parameter in self.policy.parameters()
            if id(parameter) not in projection_ids and parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": remaining, "lr": base_lr},
                {
                    "params": projection_parameters,
                    "lr": min(base_lr, self.ppo_config.text_projection_learning_rate),
                },
            ],
            eps=1.0e-5,
            weight_decay=self.ppo_config.weight_decay,
        )

    def normalize(self, state: SemanticState, update: bool = False) -> SemanticState:
        if update:
            self.normalizer.update(state)
        return self.normalizer.normalize(state)

    def select_action(
        self,
        state: SemanticState,
        command: str,
        update_normalizer: bool = True,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float, SemanticState]:
        normalized = self.normalize(state, update=update_normalizer)
        state_batch = stack_states([normalized], self.device)
        language = self.text_encoder.encode([command])
        self.policy.eval()
        with torch.inference_mode():
            if deterministic:
                action, value = self.policy.deterministic_act(state_batch, language)
                log_probability = torch.zeros(1, device=self.device)
            else:
                action, log_probability, value = self.policy.act(state_batch, language)
        return (
            action[0].cpu().numpy(),
            float(log_probability[0].item()),
            float(value[0].item()),
            normalized,
        )

    def value(self, state: SemanticState, command: str) -> float:
        normalized = self.normalize(state, update=False)
        state_batch = stack_states([normalized], self.device)
        language = self.text_encoder.encode([command])
        self.policy.eval()
        with torch.inference_mode():
            _, _, value = self.policy(state_batch, language)
        return float(value[0].item())

    def store(
        self,
        normalized_state: SemanticState,
        command: str,
        action: np.ndarray,
        log_probability: float,
        reward: float,
        value: float,
        done: bool,
        skill: str,
    ) -> None:
        self.buffer.add(
            normalized_state,
            command,
            action,
            log_probability,
            reward,
            value,
            done,
            skill,
        )

    @staticmethod
    def _normalize_advantages_by_skill(
        advantages: torch.Tensor, skills: torch.Tensor
    ) -> torch.Tensor:
        output = advantages.clone()
        for skill_index in (0, 1):
            mask = skills == skill_index
            if mask.any():
                values = advantages[mask]
                output[mask] = (values - values.mean()) / (
                    values.std(unbiased=False) + 1.0e-8
                )
        return output

    def update(self, last_state: SemanticState, last_command: str) -> dict[str, float]:
        if not self.buffer.full:
            raise RuntimeError("PPO update requires a full rollout")
        config = self.ppo_config
        last_value = self.value(last_state, last_command)
        advantages, returns = self.buffer.advantages_and_returns(
            last_value, config.gamma, config.gae_lambda, self.device
        )
        skills = torch.tensor(self.buffer.skills, dtype=torch.long, device=self.device)
        advantages = self._normalize_advantages_by_skill(advantages, skills)

        state_batch = stack_states(self.buffer.states, self.device)
        actions = torch.as_tensor(np.stack(self.buffer.actions), device=self.device)
        old_log_probabilities = torch.tensor(
            self.buffer.log_probabilities, dtype=torch.float32, device=self.device
        )
        count = len(self.buffer.states)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        updates = 0

        self.policy.train()
        stop_early = False
        for _ in range(config.epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, config.batch_size):
                indices = permutation[start : start + config.batch_size]
                commands = [self.buffer.commands[index] for index in indices.cpu().tolist()]
                language = self.text_encoder.encode(commands)
                new_log_probability, values, entropy = self.policy.evaluate_actions(
                    index_state_batch(state_batch, indices), language, actions[indices]
                )
                log_ratio = new_log_probability - old_log_probabilities[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[indices]
                clipped = ratio.clamp(
                    1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (returns[indices] - values).square().mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.max_grad_norm
                )
                self.optimizer.step()

                approximate_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_mean.item())
                metrics["kl"] += approximate_kl
                updates += 1
                if config.target_kl > 0 and approximate_kl > config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.buffer.clear()
        for key in metrics:
            metrics[key] /= max(updates, 1)
        metrics["epochs_stopped_early"] = float(stop_early)
        return metrics

    def save(self, path: str | Path, extra: dict[str, object] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "architecture": "true_language_transformer_ppo",
            "model_config": asdict(self.model_config),
            "ppo_config": asdict(self.ppo_config),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, destination)

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict[str, object]:
        payload = torch.load(path, map_location=self.device)
        if payload.get("architecture") != "true_language_transformer_ppo":
            raise ValueError("checkpoint is not a true-language transformer PPO model")
        saved_model = ModelConfig(**payload["model_config"])
        if asdict(saved_model) != asdict(self.model_config):
            raise ValueError("checkpoint model configuration does not match this agent")
        self.policy.load_state_dict(payload["policy"])
        self.normalizer.load_state_dict(payload["normalizer"])
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.extra = dict(payload.get("extra", {}))
        return self.extra
