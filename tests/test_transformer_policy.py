import unittest

import numpy as np
import torch

from language_conditioned_rl.transformer.config import ModelConfig
from language_conditioned_rl.transformer.language import FrozenTextEncoder
from language_conditioned_rl.transformer.model import TransformerActorCritic
from language_conditioned_rl.transformer.observation import SemanticState, stack_states


def sample_state(batch_size=2):
    rng = np.random.default_rng(9)
    states = []
    for _ in range(batch_size):
        states.append(
            SemanticState(
                robot=rng.normal(size=21).astype(np.float32),
                end_effector=rng.normal(size=12).astype(np.float32),
                gripper=rng.normal(size=7).astype(np.float32),
                blocks=rng.normal(size=(3, 18)).astype(np.float32),
                targets=rng.normal(size=(4, 8)).astype(np.float32),
            )
        )
    return states


class TransformerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            text_backend="hash",
            d_model=64,
            n_layers=2,
            n_heads=4,
            dim_feedforward=128,
            state_mlp_hidden=64,
            actor_hidden=64,
            dropout=0.0,
        )
        self.encoder = FrozenTextEncoder("offline", "cpu", backend="hash")
        self.policy = TransformerActorCritic(
            self.config, self.encoder.hidden_size, action_dim=7
        ).eval()

    def test_policy_shapes_and_bounded_actions(self):
        state = stack_states(sample_state(), "cpu")
        language = self.encoder.encode(
            ["put the red block in the yellow plate", "stack blue on green"]
        )
        with torch.no_grad():
            action, log_probability, value = self.policy.act(state, language)
        self.assertEqual(action.shape, (2, 7))
        self.assertEqual(log_probability.shape, (2,))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.all(action.abs() <= 1.0))

    def test_same_scene_different_commands_change_readout(self):
        one_state = sample_state(batch_size=1)[0]
        state = stack_states([one_state, one_state], "cpu")
        language = self.encoder.encode(
            [
                "put the red block in the yellow plate",
                "put the blue block in the purple plate",
            ]
        )
        with torch.no_grad():
            mean, _, _ = self.policy(state, language)
        self.assertGreater(torch.linalg.vector_norm(mean[0] - mean[1]).item(), 1.0e-8)

    def test_checkpoint_architecture_has_separate_readouts(self):
        self.assertEqual(tuple(self.policy.readout_tokens.shape), (1, 2, 64))
        self.assertIsNot(self.policy.actor, self.policy.value)

    def test_state_schema_contains_no_privileged_task_fields(self):
        fields = set(SemanticState.__dataclass_fields__)
        self.assertEqual(
            fields,
            {"robot", "end_effector", "gripper", "blocks", "targets"},
        )
        self.assertTrue(
            fields.isdisjoint(
                {"task_id", "selected_block", "selected_target", "skill", "stage"}
            )
        )


if __name__ == "__main__":
    unittest.main()
