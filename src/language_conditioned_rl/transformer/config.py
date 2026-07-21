from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_backend: str = "pretrained"
    max_language_tokens: int = 32
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    state_mlp_hidden: int = 256
    actor_hidden: int = 256
    min_log_std: float = -5.0
    max_log_std: float = 0.0

    def validate(self) -> None:
        if self.text_backend not in {"pretrained", "hash"}:
            raise ValueError("text_backend must be 'pretrained' or 'hash'")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1 or self.max_language_tokens < 4:
            raise ValueError("n_layers and max_language_tokens are too small")


@dataclass
class PPOConfig:
    learning_rate: float = 1.5e-4
    transfer_learning_rate: float = 5.0e-5
    text_projection_learning_rate: float = 5.0e-5
    rollout_steps: int = 4096
    epochs: int = 6
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.004
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    weight_decay: float = 1.0e-4

    def validate(self) -> None:
        if self.rollout_steps < self.batch_size:
            raise ValueError("rollout_steps must be >= batch_size")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("invalid gamma or gae_lambda")


@dataclass
class CurriculumConfig:
    phase: str = "place"
    rolling_episodes: int = 50
    minimum_episodes: int = 30
    cooldown_updates: int = 10
    earlier_stage_probability: float = 0.20
    stage_success_threshold: float = 0.75
    transport_success_threshold: float = 0.60
    curriculum_distance_start: float = 0.10
    curriculum_distance_max: float = 0.34
    curriculum_distance_step: float = 0.02
    lift_height_start: float = 0.02
    lift_height_step: float = 0.01
    stack_fraction_start: float = 0.10
    stack_fraction_max: float = 0.50
    stack_fraction_step: float = 0.05
    stack_success_threshold: float = 0.25
    place_retention_threshold: float = 0.55
    shared_grasp_threshold: float = 0.65
    shared_lift_threshold: float = 0.60
    paraphrase_probability: float = 0.35

    def validate(self) -> None:
        if self.phase not in {"place", "transfer", "joint"}:
            raise ValueError("phase must be place, transfer, or joint")
        for value in (
            self.stack_fraction_start,
            self.stack_fraction_max,
            self.paraphrase_probability,
            self.earlier_stage_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("curriculum probabilities must be in [0, 1]")


@dataclass
class ExperimentConfig:
    name: str = "transformer_place"
    seed: int = 7
    total_steps: int = 12_000_000
    device: str = "auto"
    checkpoint_dir: str = "checkpoints/transformer"
    log_dir: str = "runs/transformer"
    log_interval_updates: int = 5
    save_interval_updates: int = 25
    resume_checkpoint: str | None = None
    train_task_indices: list[int] | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def validate(self) -> None:
        self.model.validate()
        self.ppo.validate()
        self.curriculum.validate()
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.curriculum.phase == "transfer" and not self.resume_checkpoint:
            raise ValueError("transfer phase requires resume_checkpoint")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        values = dict(raw)
        values["model"] = ModelConfig(**values.get("model", {}))
        values["ppo"] = PPOConfig(**values.get("ppo", {}))
        values["curriculum"] = CurriculumConfig(**values.get("curriculum", {}))
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
