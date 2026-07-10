"""Language-conditioned reinforcement learning for Franka placing and stacking."""

from typing import TYPE_CHECKING

__all__ = ["PPO", "RealFrankaPickPlaceEnv"]

if TYPE_CHECKING:
    from language_conditioned_rl.env import RealFrankaPickPlaceEnv
    from language_conditioned_rl.ppo import PPO


def __getattr__(name):
    if name == "PPO":
        from language_conditioned_rl.ppo import PPO

        return PPO
    if name == "RealFrankaPickPlaceEnv":
        from language_conditioned_rl.env import RealFrankaPickPlaceEnv

        return RealFrankaPickPlaceEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
