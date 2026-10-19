import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.env import BasketballEnv
from rl_playground.envs.basketball.evaluate import evaluate
from rl_playground.envs.basketball.policy import PolicyNetwork, load_policy, save_policy
from rl_playground.envs.basketball.train import simulate_shot, update_policy


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(config.TORCH_THREADS)

    def test_sampled_and_mean_actions_inside_space(self):
        policy = PolicyNetwork()
        with BasketballEnv() as env:
            observation, _ = env.reset(seed=1)
            for seed in range(30):
                self.assertTrue(
                    env.action_space.contains(
                        policy.choose_action(observation, seed=seed)
                    )
                )
            self.assertTrue(
                env.action_space.contains(
                    policy.choose_action(observation, greedy=True)
                )
            )

    def test_action_sampling_reproducible_and_does_not_change_caller_rng(self):
        policy = PolicyNetwork()
        with BasketballEnv() as env:
            observation, _ = env.reset(seed=9)
        before = torch.get_rng_state()
        first = policy.choose_action(observation, seed=24)
        np.testing.assert_array_equal(first, policy.choose_action(observation, seed=24))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_checkpoint_roundtrip_preserves_architecture_settings_and_actions(self):
        settings = replace(config.ENVIRONMENT, max_wind=1.4)
        policy = PolicyNetwork((16, 8), settings)
        with BasketballEnv(settings=settings) as env:
            observation, _ = env.reset(seed=3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roundtrip.pt"
            save_policy(policy, path)
            loaded = load_policy(path)
            self.assertEqual(loaded.hidden_sizes, (16, 8))
            self.assertEqual(loaded.environment_settings, settings)
            np.testing.assert_array_equal(
                policy.choose_action(observation, greedy=True),
                loaded.choose_action(observation, greedy=True),
            )
            with self.assertRaises(FileExistsError):
                save_policy(policy, path)

    def test_update_changes_parameters_without_nan(self):
        torch.manual_seed(4)
        policy = PolicyNetwork()
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.LEARNING_RATE)
        with BasketballEnv() as env:
            observations = [env.reset(seed=i)[0] for i in range(8)]
        actions = [
            policy.choose_action(obs, seed=i) for i, obs in enumerate(observations)
        ]
        before = [p.detach().clone() for p in policy.parameters()]
        loss = update_policy(
            policy, optimizer, observations, actions, [-1, 1, 2, -1, 10, 0, -1, 3]
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
        )
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))

    def test_single_shot_batch_and_equal_rewards_are_finite(self):
        policy = PolicyNetwork()
        optimizer = torch.optim.Adam(policy.parameters())
        with BasketballEnv() as env:
            observation, _ = env.reset(seed=1)
        action = policy.choose_action(observation, seed=1)
        for count in (1, 4):
            loss = update_policy(
                policy,
                optimizer,
                [observation] * count,
                [action] * count,
                [-1.0] * count,
            )
            self.assertTrue(np.isfinite(loss))

    def test_sequential_and_parallel_shots_match_in_order(self):
        tasks = [
            (config.ENVIRONMENT, seed, np.array([0.45 + seed * 0.05, 0.65], np.float32))
            for seed in range(4)
        ]
        sequential = list(map(simulate_shot, tasks))
        with ProcessPoolExecutor(
            2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            parallel = list(pool.map(simulate_shot, tasks))
        self.assertEqual(sequential, parallel)

    def test_held_out_evaluation_is_repeatable(self):
        policy = PolicyNetwork()
        self.assertEqual(
            evaluate(policy, episodes=8, seed=100),
            evaluate(policy, episodes=8, seed=100),
        )
        demo = evaluate(demo=True, episodes=20, seed=100)
        self.assertEqual(demo["success_rate"], 1)


if __name__ == "__main__":
    unittest.main()
