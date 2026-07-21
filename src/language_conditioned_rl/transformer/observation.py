from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from language_conditioned_rl.env import RealFrankaPickPlaceEnv
from language_conditioned_rl.task_config import (
    BLOCK_NAMES,
    PLACE_TASK_INDICES,
    STACK_TASK_INDICES,
    TARGET_NAMES,
    TASKS,
)
from language_conditioned_rl.transformer.language import CommandSampler


STATE_DIMS = {
    "robot": 21,
    "end_effector": 12,
    "gripper": 7,
    "blocks": 18,
    "targets": 8,
}


@dataclass
class SemanticState:
    """Task-agnostic structured scene state consumed by the actor."""

    robot: np.ndarray
    end_effector: np.ndarray
    gripper: np.ndarray
    blocks: np.ndarray
    targets: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "robot": self.robot,
            "end_effector": self.end_effector,
            "gripper": self.gripper,
            "blocks": self.blocks,
            "targets": self.targets,
        }


class SemanticNormalizer:
    """Independent running statistics for each semantic token type."""

    def __init__(self, epsilon: float = 1.0e-4):
        self.mean = {name: np.zeros(dim, dtype=np.float64) for name, dim in STATE_DIMS.items()}
        self.var = {name: np.ones(dim, dtype=np.float64) for name, dim in STATE_DIMS.items()}
        self.count = {name: float(epsilon) for name in STATE_DIMS}

    @staticmethod
    def _rows(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        return array.reshape(-1, array.shape[-1])

    def update(self, state: SemanticState) -> None:
        for name, value in state.as_dict().items():
            rows = self._rows(value)
            batch_mean = rows.mean(axis=0)
            batch_var = rows.var(axis=0)
            batch_count = rows.shape[0]
            delta = batch_mean - self.mean[name]
            total = self.count[name] + batch_count
            self.mean[name] += delta * batch_count / total
            self.var[name] = (
                self.var[name] * self.count[name]
                + batch_var * batch_count
                + delta**2 * self.count[name] * batch_count / total
            ) / total
            self.count[name] = total

    def normalize(self, state: SemanticState, clip: float = 10.0) -> SemanticState:
        values = {}
        for name, value in state.as_dict().items():
            normalized = (np.asarray(value) - self.mean[name]) / np.sqrt(self.var[name] + 1.0e-8)
            values[name] = np.clip(normalized, -clip, clip).astype(np.float32)
        return SemanticState(**values)

    def state_dict(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "mean": self.mean[name].tolist(),
                "var": self.var[name].tolist(),
                "count": self.count[name],
            }
            for name in STATE_DIMS
        }

    def load_state_dict(self, values: dict[str, dict[str, object]]) -> None:
        for name in STATE_DIMS:
            self.mean[name] = np.asarray(values[name]["mean"], dtype=np.float64)
            self.var[name] = np.asarray(values[name]["var"], dtype=np.float64)
            self.count[name] = float(values[name]["count"])


def stack_states(states: Iterable[SemanticState], device: torch.device | str) -> dict[str, torch.Tensor]:
    items = list(states)
    if not items:
        raise ValueError("cannot stack an empty state list")
    return {
        name: torch.as_tensor(
            np.stack([getattr(state, name) for state in items]),
            dtype=torch.float32,
            device=device,
        )
        for name in STATE_DIMS
    }


def index_state_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in batch.items()}


class LanguageFrankaEnv:
    """Language/semantic adapter around the existing MuJoCo reward environment.

    The base observation contains privileged selected-source, destination, stage,
    and skill fields. They are deliberately discarded here. Reward calculation
    can still use them internally, as proposed in the design document.
    """

    def __init__(
        self,
        env: RealFrankaPickPlaceEnv | None = None,
        paraphrase_probability: float = 0.35,
        seed: int = 7,
        task_indices: list[int] | tuple[int, ...] | None = None,
        render_mode: str | None = None,
    ):
        self.env = env or RealFrankaPickPlaceEnv(render_mode=render_mode)
        self.command_sampler = CommandSampler(paraphrase_probability, seed)
        self.rng = np.random.default_rng(seed)
        self.task_indices = tuple(task_indices) if task_indices is not None else None
        self.command = ""
        self.task_index = 0

    @property
    def action_space(self):
        return self.env.action_space

    def _contact_flags(self, block_name: str) -> tuple[float, float]:
        block_body = self.env._block_body_ids[block_name]
        left_body = self.env._finger_body_ids["left_finger"]
        right_body = self.env._finger_body_ids["right_finger"]
        left = right = False
        for index in range(self.env.data.ncon):
            contact = self.env.data.contact[index]
            bodies = {
                int(self.env.model.geom_bodyid[contact.geom1]),
                int(self.env.model.geom_bodyid[contact.geom2]),
            }
            if block_body in bodies:
                left = left or left_body in bodies
                right = right or right_body in bodies
        return float(left), float(right)

    def semantic_state(self) -> SemanticState:
        env = self.env
        q = env.data.qpos[env._act_qpos_addr]
        dq = env.data.qvel[env._act_qvel_addr]
        ctrl_q = np.concatenate(
            [env._arm_ctrl_target, np.full(2, env._gripper_q_target(), dtype=np.float64)]
        )
        ctrl_error = ctrl_q - q

        robot = np.concatenate([q[:7], dq[:7], ctrl_error[:7]])
        end_effector = np.concatenate(
            [env.data.site_xpos[env._ee_site_id], env.data.site_xmat[env._ee_site_id]]
        )
        gripper = np.concatenate(
            [q[7:9], dq[7:9], ctrl_error[7:9], np.array([env._close_fraction()])]
        )

        blocks = []
        for block_index, name in enumerate(BLOCK_NAMES):
            qpos_address = env._block_qpos_addr[name]
            identity = np.eye(len(BLOCK_NAMES), dtype=np.float64)[block_index]
            blocks.append(
                np.concatenate(
                    [
                        env.data.qpos[qpos_address : qpos_address + 3],
                        env.data.qpos[qpos_address + 3 : qpos_address + 7],
                        env.data.cvel[env._block_body_ids[name]],
                        identity,
                        np.asarray(self._contact_flags(name)),
                    ]
                )
            )

        targets = []
        for target_index, name in enumerate(TARGET_NAMES):
            identity = np.eye(len(TARGET_NAMES), dtype=np.float64)[target_index]
            targets.append(
                np.concatenate(
                    [
                        env.data.xpos[env._target_body_ids[name]],
                        identity,
                        np.array([float("bowl" in name)]),
                    ]
                )
            )

        state = SemanticState(
            robot=robot.astype(np.float32),
            end_effector=end_effector.astype(np.float32),
            gripper=gripper.astype(np.float32),
            blocks=np.asarray(blocks, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
        )
        for name, expected in STATE_DIMS.items():
            if getattr(state, name).shape[-1] != expected:
                raise RuntimeError(f"{name} feature dimension changed unexpectedly")
        return state

    def _sample_task_index(self) -> int | None:
        if self.task_indices is None:
            return None
        place = [index for index in self.task_indices if index in PLACE_TASK_INDICES]
        stack = [index for index in self.task_indices if index in STACK_TASK_INDICES]
        fraction = self.env.stack_task_fraction
        if fraction is None:
            candidates = list(self.task_indices)
        elif stack and (not place or self.rng.random() < fraction):
            candidates = stack
        else:
            candidates = place or stack
        return int(candidates[int(self.rng.integers(0, len(candidates)))])

    def reset(self, *, seed: int | None = None, task_index: int | None = None):
        sampled = task_index if task_index is not None else self._sample_task_index()
        options = {"task_index": int(sampled)} if sampled is not None else None
        _, info = self.env.reset(seed=seed, options=options)
        task_tuple = (
            self.env.selected_block,
            self.env.destination_block if self.env.skill == "stack" else self.env.selected_target,
            self.env.skill,
        )
        self.task_index = next(
            index for index, task in enumerate(TASKS) if task[:3] == task_tuple
        )
        self.command = self.command_sampler.sample(self.task_index)
        info = dict(info, command=self.command, task_index=self.task_index)
        return self.semantic_state(), self.command, info

    def step(self, action: np.ndarray):
        _, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info, command=self.command, task_index=self.task_index)
        return self.semantic_state(), reward, terminated, truncated, info

    def render(self, camera: str = "fixed_scene"):
        return self.env.render(camera)

    def close(self) -> None:
        self.env.close()
