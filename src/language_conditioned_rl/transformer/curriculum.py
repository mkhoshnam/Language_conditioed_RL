from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from language_conditioned_rl.env import (
    LIFT_HEIGHT,
    STAGE_GRASP,
    STAGE_LIFT,
    STAGE_NAMES,
    STAGE_PLACE,
    STAGE_REACH,
    STAGE_TRANSPORT,
)
from language_conditioned_rl.transformer.config import CurriculumConfig


@dataclass
class EpisodeSummary:
    stage: int
    skill: str
    success: float
    task_success: float
    grasped: float
    lifted: float
    released: float
    stable: float
    best_reach: float
    best_place: float
    episode_return: float
    length: int


class CurriculumManager:
    """Place skill curriculum followed by guarded place-to-stack transfer."""

    def __init__(self, config: CurriculumConfig, env, seed: int = 7):
        self.config = config
        self.env = env
        self.history: deque[EpisodeSummary] = deque(maxlen=config.rolling_episodes)
        self.last_change_update = 0
        self.rng = np.random.default_rng(seed)
        if config.phase == "place":
            self.stage = STAGE_REACH
            env.stack_task_fraction = 0.0
        else:
            self.stage = STAGE_PLACE
            env.stack_task_fraction = (
                config.stack_fraction_start if config.phase == "transfer" else 0.50
            )
        env.task_stage = self.stage
        env.curriculum_dist = config.curriculum_distance_start
        env.curriculum_lift_height = config.lift_height_start

    @staticmethod
    def _mean(items: list[float], default: float = 0.0) -> float:
        return float(np.mean(items)) if items else default

    def record(self, episode: EpisodeSummary) -> None:
        self.history.append(episode)

    def metrics(self) -> dict[str, float]:
        episodes = list(self.history)
        current = [episode for episode in episodes if episode.stage == self.stage]
        place = [
            episode
            for episode in episodes
            if episode.skill == "place" and episode.stage == STAGE_PLACE
        ]
        stack = [
            episode
            for episode in episodes
            if episode.skill == "stack" and episode.stage == STAGE_PLACE
        ]
        return {
            "episodes": float(len(episodes)),
            "return": self._mean([episode.episode_return for episode in episodes]),
            "success": self._mean([episode.success for episode in current]),
            "place_success": self._mean([episode.success for episode in place], np.nan),
            "stack_success": self._mean([episode.success for episode in stack], np.nan),
            "grasp": self._mean([episode.grasped for episode in current]),
            "lift": self._mean([episode.lifted for episode in current]),
            "release": self._mean([episode.released for episode in current]),
            "stable": self._mean([episode.stable for episode in current]),
            "best_reach": self._mean([episode.best_reach for episode in current], np.nan),
            "best_place": self._mean([episode.best_place for episode in current], np.nan),
            "length": self._mean([float(episode.length) for episode in episodes]),
        }

    def sample_stage(self) -> int:
        """Mix earlier objectives into place training to reduce forgetting."""
        if (
            self.config.phase != "place"
            or self.stage == STAGE_REACH
            or self.rng.random() >= self.config.earlier_stage_probability
        ):
            return self.stage
        earlier = list(range(STAGE_REACH, self.stage))
        return int(earlier[int(self.rng.integers(0, len(earlier)))])

    def maybe_advance(self, update: int) -> str | None:
        config = self.config
        if len(self.history) < config.minimum_episodes:
            return None
        if update - self.last_change_update < config.cooldown_updates:
            return None
        metrics = self.metrics()
        message = None

        if config.phase == "place":
            threshold = config.stage_success_threshold
            if self.stage == STAGE_REACH and metrics["success"] >= threshold:
                self.stage = STAGE_GRASP
                message = "advanced place curriculum to grasp"
            elif self.stage == STAGE_GRASP and metrics["success"] >= threshold and metrics["grasp"] >= threshold:
                self.stage = STAGE_LIFT
                message = "advanced place curriculum to lift"
            elif self.stage == STAGE_LIFT and metrics["success"] >= threshold:
                if self.env.curriculum_lift_height < LIFT_HEIGHT - 1.0e-6:
                    self.env.curriculum_lift_height = min(
                        LIFT_HEIGHT,
                        self.env.curriculum_lift_height + config.lift_height_step,
                    )
                    message = f"raised lift goal to {self.env.curriculum_lift_height:.3f} m"
                elif metrics["lift"] >= threshold:
                    self.stage = STAGE_TRANSPORT
                    message = "advanced place curriculum to transport"
            elif (
                self.stage == STAGE_TRANSPORT
                and metrics["success"] >= config.transport_success_threshold
                and metrics["lift"] >= config.shared_lift_threshold
            ):
                self.stage = STAGE_PLACE
                message = "advanced place curriculum to release and settle"
            elif (
                self.stage == STAGE_PLACE
                and metrics["place_success"] >= threshold
                and self.env.curriculum_dist < config.curriculum_distance_max
            ):
                self.env.curriculum_dist = min(
                    config.curriculum_distance_max,
                    self.env.curriculum_dist + config.curriculum_distance_step,
                )
                message = f"expanded scene curriculum to {self.env.curriculum_dist:.2f} m"
        elif config.phase == "transfer":
            place_retained = (
                np.isnan(metrics["place_success"])
                or metrics["place_success"] >= config.place_retention_threshold
            )
            stack_ready = (
                not np.isnan(metrics["stack_success"])
                and metrics["stack_success"] >= config.stack_success_threshold
            )
            shared_ready = (
                metrics["grasp"] >= config.shared_grasp_threshold
                and metrics["lift"] >= config.shared_lift_threshold
            )
            if (
                place_retained
                and stack_ready
                and shared_ready
                and self.env.stack_task_fraction < config.stack_fraction_max
            ):
                self.env.stack_task_fraction = min(
                    config.stack_fraction_max,
                    self.env.stack_task_fraction + config.stack_fraction_step,
                )
                message = f"increased stack training fraction to {self.env.stack_task_fraction:.0%}"

        if message:
            self.last_change_update = update
            self.history.clear()
        return message

    def state_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "last_change_update": self.last_change_update,
            "curriculum_dist": self.env.curriculum_dist,
            "curriculum_lift_height": self.env.curriculum_lift_height,
            "stack_task_fraction": self.env.stack_task_fraction,
            "history": [asdict(episode) for episode in self.history],
        }

    def load_state_dict(self, values: dict[str, object], preserve_phase: bool = False) -> None:
        if not preserve_phase:
            self.stage = int(values.get("stage", self.stage))
            self.env.stack_task_fraction = float(
                values.get("stack_task_fraction", self.env.stack_task_fraction)
            )
        self.last_change_update = int(values.get("last_change_update", 0))
        self.env.curriculum_dist = float(
            values.get("curriculum_dist", self.env.curriculum_dist)
        )
        self.env.curriculum_lift_height = float(
            values.get("curriculum_lift_height", self.env.curriculum_lift_height)
        )
        self.env.task_stage = self.stage
        self.history.clear()
        for item in values.get("history", []):
            self.history.append(EpisodeSummary(**item))

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]
