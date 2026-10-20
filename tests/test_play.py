"""Exercise the real manual loop and renderer with scripted mouse and keyboard events."""

import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from rl_playground.envs.basketball.env import BasketballEnv
from rl_playground.envs.basketball.play import main


class PlayTests(unittest.TestCase):
    def test_mouse_aim_and_hold_release_launch_two_shots(self):
        env = BasketballEnv(render_mode="human")
        frames = iter(
            [
                (0, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]),
                (1, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]),
                (1.75, []),  # Holding alone must not launch.
                (1.75, [pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE)]),
                (3, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)]),
                (4.5, [pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE)]),
            ]
        )
        now = [0]

        def get_events(event_types=None):
            if event_types is not None:  # Renderer close checks during flight.
                return []
            try:
                now[0], events = next(frames)
                return events
            except StopIteration:
                self.fail("Two press/hold/release cycles should complete two shots.")

        with (
            patch("sys.argv", ["main.py", "--shots", "2"]),
            patch("rl_playground.envs.basketball.play.gym.make", return_value=env),
            patch("pygame.event.get", side_effect=get_events),
            patch("pygame.mouse.get_pos", return_value=(600, 250)),
            patch("time.monotonic", side_effect=lambda: now[0]),
            patch("pygame.time.Clock"),
            patch("builtins.print"),
            patch.object(env, "reset", wraps=env.reset) as resets,
            patch.object(env, "step", wraps=env.step) as steps,
        ):
            main()
        self.assertEqual(steps.call_count, 2)
        self.assertAlmostEqual(float(steps.call_args_list[0].args[0][0]), 0.5)
        self.assertAlmostEqual(float(steps.call_args_list[1].args[0][0]), 1.0)
        s = env.settings
        # Court projection: the default launcher is at (267.5, 514).
        angle = math.degrees(math.atan2(514 - 250, 600 - 267.5))
        expected = (angle - s.min_angle) / (s.max_angle - s.min_angle)
        self.assertAlmostEqual(
            float(steps.call_args_list[0].args[0][1]), expected, places=5
        )
        self.assertEqual(resets.call_count, 3)
        self.assertEqual(resets.call_args_list[1].kwargs, {})
        self.assertEqual(resets.call_args_list[2].kwargs, {"seed": 43})

    def test_window_close_during_flight_truncates(self):
        with BasketballEnv(render_mode="human") as env:
            env.reset(seed=1)
            close_event = pygame.event.Event(pygame.QUIT)
            with patch("pygame.event.get", return_value=[close_event]):
                _, _, terminated, truncated, info = env.step([0.5, 0.6])
            self.assertTrue(env.window_closed)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["termination_reason"], "window_closed")


if __name__ == "__main__":
    unittest.main()
