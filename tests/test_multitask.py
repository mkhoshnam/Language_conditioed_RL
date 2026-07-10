import unittest

import numpy as np
import torch

from language_conditioned_rl.llm_parser import rule_fallback
from language_conditioned_rl.ppo import ActorCritic, RunningMeanStd
from language_conditioned_rl.task_config import (
    PLACE_TASK_INDICES,
    STACK_TASK_INDICES,
    TASKS,
)


class TaskConfigTests(unittest.TestCase):
    def test_task_catalog_contains_place_and_stack_goals(self):
        self.assertEqual(len(TASKS), 18)
        self.assertEqual(len(PLACE_TASK_INDICES), 12)
        self.assertEqual(len(STACK_TASK_INDICES), 6)
        for index in STACK_TASK_INDICES:
            source, destination, skill, _ = TASKS[index]
            self.assertEqual(skill, "stack")
            self.assertNotEqual(source, destination)

    def test_rule_parser_handles_both_skills(self):
        place = rule_fallback("put the red cube in the yellow plate")
        stack = rule_fallback("stack the green block on the blue block")

        self.assertEqual(place["skill"], "place")
        self.assertEqual(place["destination"], "yellow_plate")
        self.assertEqual(stack["skill"], "stack")
        self.assertEqual(stack["block"], "green_block")
        self.assertEqual(stack["destination"], "blue_block")


class PolicyRoutingTests(unittest.TestCase):
    def test_normalization_preserves_skill_one_hot(self):
        rms = RunningMeanStd((8,))
        rms.mean[:] = 3.0
        obs = np.array([0, 1, 2, 3, 4, 0, 1, 0.5], dtype=np.float32)

        normalized = rms.normalize(obs)

        np.testing.assert_array_equal(normalized[-3:-1], obs[-3:-1])

    def test_actor_critic_routes_to_selected_skill_head(self):
        policy = ActorCritic(obs_dim=8, act_dim=2, hidden=4)
        with torch.no_grad():
            for head in policy.actor_mean:
                head.weight.zero_()
            for head in policy.critic:
                head.weight.zero_()
            policy.actor_mean[0].bias.fill_(0.1)
            policy.actor_mean[1].bias.fill_(0.2)
            policy.critic[0].bias.fill_(1.0)
            policy.critic[1].bias.fill_(2.0)

        place_obs = torch.zeros(8)
        place_obs[-3] = 1.0
        stack_obs = torch.zeros(8)
        stack_obs[-2] = 1.0

        place_mean, _, place_value = policy(place_obs)
        stack_mean, _, stack_value = policy(stack_obs)

        torch.testing.assert_close(place_mean, torch.full((2,), 0.1))
        torch.testing.assert_close(stack_mean, torch.full((2,), 0.2))
        torch.testing.assert_close(place_value, torch.tensor(1.0))
        torch.testing.assert_close(stack_value, torch.tensor(2.0))


if __name__ == "__main__":
    unittest.main()
