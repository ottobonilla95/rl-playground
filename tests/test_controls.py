"""Manual controls must choose exactly the action shown when Space is released."""

import math
import os
import unittest
from dataclasses import replace

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.controls import ManualControls, court_coordinates
from rl_playground.envs.basketball.env import BasketballEnv


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.env = BasketballEnv()
        self.env.reset(seed=42)
        self.controls = ManualControls()
        self.controls.cancel(self.env)

    def pointer_at_angle(self, degrees):
        s = self.env.settings
        left, floor, scale = court_coordinates(
            s, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        radians = math.radians(degrees)
        return (
            left + s.launcher_x * scale + 100 * math.cos(radians),
            floor - s.launcher_y * scale - 100 * math.sin(radians),
        )

    def update(self, events=(), now=0, angle=45):
        return self.controls.update(
            self.env, events, self.pointer_at_angle(angle), now, next_seed=43
        )

    def space(self, kind, **kwargs):
        return pygame.event.Event(kind, key=pygame.K_SPACE, **kwargs)

    def test_mouse_tracks_angle_and_clamps_to_environment_limits(self):
        self.env = BasketballEnv(
            settings=replace(config.ENVIRONMENT, min_angle=20, max_angle=70)
        )
        self.env.reset(seed=1)
        for degrees, expected in ((45, 0.5), (0, 0), (90, 1), (170, 1), (-45, 0)):
            self.update(angle=degrees)
            self.assertAlmostEqual(float(self.env.aim[1]), expected)
        self.update(angle=45)
        s = self.env.settings
        left, floor, scale = court_coordinates(
            s, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        self.controls.update(
            self.env,
            [],
            (left + s.launcher_x * scale, floor - s.launcher_y * scale),
            0,
            next_seed=43,
        )
        self.assertAlmostEqual(float(self.env.aim[1]), 0.5)

    def test_hold_charges_without_shooting_and_release_uses_current_mouse(self):
        self.assertIsNone(self.update([self.space(pygame.KEYDOWN)], now=10))
        self.assertIsNone(self.update(now=10.75))
        self.assertAlmostEqual(float(self.env.aim[0]), 0.5)
        # OS key repeat must not restart charging.
        self.update([self.space(pygame.KEYDOWN, repeat=True)], now=10.75)
        action = self.update([self.space(pygame.KEYUP)], now=10.75, angle=60)
        self.assertAlmostEqual(float(action[0]), 0.5)
        self.assertAlmostEqual(float(action[1]), (60 - 15) / 65)
        self.assertFalse(self.env.charging)
        self.assertIsNone(self.update([self.space(pygame.KEYUP)], now=11))

    def test_full_charge_stays_capped_until_release_and_wind_is_fixed(self):
        wind = self.env.wind
        self.update([self.space(pygame.KEYDOWN)], now=0)
        self.assertIsNone(self.update(now=10))
        self.assertEqual(self.env.aim[0], 1)
        self.assertTrue(self.env.charging)
        action = self.update([self.space(pygame.KEYUP)], now=11)
        _, _, _, _, info = self.env.step(action)
        self.assertEqual(info["wind"], wind)
        self.assertEqual(action[0], 1)

    def test_reset_and_focus_loss_cancel_charging(self):
        for event in (
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r),
            pygame.event.Event(pygame.WINDOWFOCUSLOST),
        ):
            self.update([self.space(pygame.KEYDOWN)], now=0)
            self.update([event], now=0.5)
            self.assertFalse(self.env.charging)
            self.assertEqual(self.env.aim[0], 0)
            self.assertIsNone(self.update([self.space(pygame.KEYUP)], now=1))

    def test_quick_tap_fires_at_minimum_power(self):
        action = self.update([self.space(pygame.KEYDOWN), self.space(pygame.KEYUP)])
        self.assertEqual(action[0], 0)
        speed, _ = self.env.decode_action(action)
        self.assertEqual(speed, self.env.settings.min_speed)


if __name__ == "__main__":
    unittest.main()
