"""Exercise the real manual loop and renderer with scripted keyboard events."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from rl_playground.envs.basketball.env import BasketballEnv
from rl_playground.envs.basketball.play import main


class PlayTests(unittest.TestCase):
    def test_manual_new_wind_and_next_shot_allow_aiming(self):
        env = BasketballEnv(render_mode="human")
        # R draws fresh wind; first Space fires; second Space prepares only;
        # third Space fires. Two shots must require exactly these three Spaces.
        events = iter([pygame.K_r, pygame.K_SPACE, pygame.K_SPACE, pygame.K_SPACE])

        def get_events(event_types=None):
            if event_types is not None:  # Renderer checks close events during flights.
                return []
            try:
                return [pygame.event.Event(pygame.KEYDOWN, key=next(events))]
            except StopIteration:
                self.fail(
                    "Manual loop failed to complete two shots with the expected keys."
                )

        with (
            patch("sys.argv", ["main.py", "--shots", "2"]),
            patch("rl_playground.envs.basketball.play.gym.make", return_value=env),
            patch("pygame.event.get", side_effect=get_events),
            patch("pygame.time.Clock"),
            patch("builtins.print"),
            patch.object(env, "reset", wraps=env.reset) as resets,
            patch.object(env, "step", wraps=env.step) as steps,
        ):
            main()
        self.assertEqual(steps.call_count, 2)
        self.assertEqual(resets.call_count, 3)
        self.assertEqual(resets.call_args_list[1].kwargs, {})

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
