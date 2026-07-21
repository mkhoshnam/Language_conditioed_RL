"""True token-level language-conditioned transformer PPO.

This package is intentionally separate from the original structured-task PPO.
The actor consumes raw language and task-agnostic semantic scene tokens; task
labels remain private to the environment reward and curriculum.
"""

from language_conditioned_rl.transformer.config import ExperimentConfig
from language_conditioned_rl.transformer.model import TransformerActorCritic
from language_conditioned_rl.transformer.observation import LanguageFrankaEnv
from language_conditioned_rl.transformer.ppo import TransformerPPO

__all__ = [
    "ExperimentConfig",
    "LanguageFrankaEnv",
    "TransformerActorCritic",
    "TransformerPPO",
]
