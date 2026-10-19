import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import unittest
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from rl_playground.envs import basketball
from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.demo import demo_action
from rl_playground.envs.basketball.env import BasketballEnv


class EnvironmentTests(unittest.TestCase):
    def test_registered_gym_contract_and_render_modes(self):
        with gym.make(basketball.ENV_ID) as env:
            check_env(env.unwrapped, skip_render_check=False)

    def test_same_seed_reproduces_whole_shot(self):
        with BasketballEnv() as first, BasketballEnv() as second:
            a, _ = first.reset(seed=16)
            b, _ = second.reset(seed=16)
            np.testing.assert_array_equal(a, b)
            outcome_a = first.step([0.55, 0.7])
            outcome_b = second.step([0.55, 0.7])
            self.assertEqual(outcome_a[1:], outcome_b[1:])
            np.testing.assert_array_equal(first.trajectory, second.trajectory)

    def test_wind_constant_in_flight_and_changes_on_reset(self):
        with BasketballEnv() as env:
            _, start = env.reset(seed=7)
            _, _, _, _, end = env.step([0.4, 0.6])
            self.assertEqual(start["wind"], end["wind"])
            values = [env.reset()[1]["wind"] for _ in range(10)]
            self.assertEqual(len(set(values)), len(values))
            self.assertTrue(
                all(abs(value) <= env.settings.max_wind for value in values)
            )

    def test_action_bounds_map_to_documented_units(self):
        with BasketballEnv() as env:
            self.assertEqual(env.decode_action([0, 0]), (5.0, 15.0))
            self.assertEqual(env.decode_action([1, 1]), (15.0, 80.0))
            for bad in ([1.1, 0.5], [0.5], [np.nan, 0], [0, np.inf]):
                with self.assertRaises(ValueError):
                    env.decode_action(bad)

    def test_reset_required_before_each_shot(self):
        with BasketballEnv() as env:
            with self.assertRaises(gym.error.ResetNeeded):
                env.step([0, 0])
            env.reset(seed=1)
            env.step([0, 0])
            with self.assertRaises(gym.error.ResetNeeded):
                env.step([0, 0])

    def test_controlled_wind_and_zero_wind_setting(self):
        with BasketballEnv() as env:
            observation, info = env.reset(options={"wind": -1.25})
            self.assertEqual(info["wind"], -1.25)
            self.assertAlmostEqual(observation[-1], -0.625)
            for bad in (3, np.nan, np.inf):
                with self.assertRaises(ValueError):
                    env.reset(options={"wind": bad})
        with BasketballEnv(settings=replace(config.ENVIRONMENT, max_wind=0)) as env:
            observation, info = env.reset(seed=1)
            self.assertEqual(info["wind"], 0)
            self.assertEqual(observation[-1], 0)

    def test_ballistic_positions_match_equations_before_contact(self):
        with BasketballEnv() as env:
            env.reset(options={"wind": 1.5})
            speed, angle = env.decode_action([0.5, 0.5])
            env.step([0.5, 0.5])
            for index in (1, 10, 30):
                t = index * env.settings.physics_dt
                radians = np.deg2rad(angle)
                expected = [
                    env.settings.launcher_x
                    + speed * np.cos(radians) * t
                    + 0.75 * t * t,
                    env.settings.launcher_y
                    + speed * np.sin(radians) * t
                    - 0.5 * env.settings.gravity * t * t,
                ]
                np.testing.assert_allclose(env.trajectory[index], expected, atol=1e-10)

    def test_demo_scores_across_wind_range(self):
        with BasketballEnv() as env:
            for wind in np.linspace(-2, 2, 21):
                with self.subTest(wind=wind):
                    observation, _ = env.reset(options={"wind": float(wind)})
                    _, reward, terminated, truncated, info = env.step(
                        demo_action(observation)
                    )
                    self.assertTrue(info["scored"])
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertGreater(reward, config.ENVIRONMENT.score_bonus)
                    self.assertLess(env.velocity[1], 0)

    def test_low_power_miss_terminates_on_ground(self):
        with BasketballEnv() as env:
            env.reset(options={"wind": 0})
            _, reward, terminated, truncated, info = env.step([0, 0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["scored"])
            self.assertEqual(info["termination_reason"], "ground")
            self.assertLess(reward, 0)

    def test_time_limit_is_truncation(self):
        with BasketballEnv(
            settings=replace(config.ENVIRONMENT, max_flight_seconds=0.1)
        ) as env:
            env.reset(seed=3)
            _, _, terminated, truncated, info = env.step([0.5, 0.7])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["termination_reason"], "time_limit")
            self.assertAlmostEqual(info["flight_time"], 0.1)

    def test_upward_pass_through_hoop_does_not_score(self):
        settings = replace(config.ENVIRONMENT, launcher_x=8.9, max_flight_seconds=0.2)
        with BasketballEnv(settings=settings) as env:
            env.reset(options={"wind": 0})
            _, _, terminated, truncated, info = env.step([1, 1])
            self.assertGreater(env.ball[1], settings.hoop_y)
            self.assertGreater(env.velocity[1], 0)
            self.assertFalse(info["scored"])
            self.assertFalse(terminated)
            self.assertTrue(truncated)

    def test_out_of_bounds_shot_terminates(self):
        with BasketballEnv() as env:
            env.reset(options={"wind": 2})
            _, _, terminated, truncated, info = env.step([1, 0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["termination_reason"], "out_of_bounds")

    def test_rim_reflects_incoming_velocity(self):
        with BasketballEnv() as env:
            env.reset(seed=1)
            s = env.settings
            edge = s.hoop_x - s.hoop_width / 2
            env.ball[:] = [edge, s.hoop_y + 0.1]
            env.velocity[:] = [0, -4]
            env._collide(env.ball.copy())
            self.assertGreater(env.velocity[1], 0)
            self.assertAlmostEqual(env.velocity[1], 4 * s.restitution)
            self.assertAlmostEqual(env.ball[1], s.hoop_y + s.ball_radius + s.rim_radius)

    def test_backboard_uses_swept_contact(self):
        with BasketballEnv() as env:
            env.reset(seed=1)
            s = env.settings
            board = s.hoop_x + s.backboard_offset
            previous = np.array([board - 0.3, s.hoop_y + 0.6])
            env.ball[:] = [board + 0.05, s.hoop_y + 0.6]
            env.velocity[:] = [8, 0]
            env._collide(previous)
            self.assertLess(env.velocity[0], 0)
            self.assertAlmostEqual(env.ball[0], board - s.ball_radius)

    def test_rgb_frame(self):
        with gym.make(basketball.ENV_ID, render_mode="rgb_array") as env:
            env.reset(seed=1)
            frame = env.render()
            self.assertEqual(
                frame.shape, (config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3)
            )
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(float(frame.std()), 10)

    def test_random_shots_are_finite_and_always_end(self):
        with BasketballEnv() as env:
            env.action_space.seed(51)
            for seed in range(50):
                env.reset(seed=seed)
                observation, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                self.assertTrue(terminated or truncated)
                self.assertTrue(np.isfinite(reward))
                self.assertTrue(env.observation_space.contains(observation))
                self.assertTrue(np.isfinite(info["closest_distance"]))

    def test_invalid_settings_rejected(self):
        for changes in (
            {"physics_dt": 0},
            {"hoop_width": 0.1},
            {"gravity": np.nan},
            {"max_wind": -1},
            {"max_angle": 95},
            {"hoop_x": 15},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                BasketballEnv(settings=replace(config.ENVIRONMENT, **changes))


if __name__ == "__main__":
    unittest.main()
