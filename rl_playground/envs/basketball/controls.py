"""Mouse aiming and hold-to-charge controls for any basketball player."""

import math

import numpy as np
import pygame

from rl_playground.envs.basketball import config


def court_coordinates(settings, size):
    """The shared court transform used by both drawing and mouse aiming."""
    width, height = size
    floor = height - 155
    scale = min(
        (width - 120) / settings.world_width, (floor - 150) / settings.world_height
    )
    left = (width - settings.world_width * scale) / 2
    return left, floor, scale


class ManualControls:
    def __init__(self, charge_seconds=config.CHARGE_SECONDS):
        if not math.isfinite(charge_seconds) or charge_seconds <= 0:
            raise ValueError("Charge duration must be positive and finite.")
        self.charge_seconds = charge_seconds
        self.started_at = None

    def cancel(self, base):
        self.started_at = None
        base.charging = False
        base.aim[0] = 0

    def update(self, env, events, mouse_position, now, next_seed):
        """Update the live aim; return an action only on a charged Space release.

        A Space press after a result prepares the next shot and starts charging.
        Reset/focus loss cancels a charge so a later release cannot fire it.
        """
        base = env.unwrapped
        s = base.settings
        left, floor, scale = court_coordinates(
            s, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        dx = mouse_position[0] - (left + s.launcher_x * scale)
        dy = (floor - s.launcher_y * scale) - mouse_position[1]
        if dx or dy:  # Keep the previous angle when the pointer is on the launcher.
            degrees = math.degrees(math.atan2(dy, dx))
            base.aim[1] = np.clip(
                (degrees - s.min_angle) / (s.max_angle - s.min_angle), 0, 1
            )

        for event in events:
            if event.type == pygame.WINDOWFOCUSLOST:
                self.cancel(base)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.cancel(base)
                env.reset()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.started_at is None and not getattr(event, "repeat", False):
                    if not base._ready:
                        self.cancel(base)
                        env.reset(seed=next_seed)
                    self.started_at = now
                    base.charging = True
            elif (
                event.type == pygame.KEYUP
                and event.key == pygame.K_SPACE
                and self.started_at is not None
                and base._ready
            ):
                self._charge(base, now)
                action = base.aim.copy()
                self.started_at = None
                base.charging = False
                return action

        if self.started_at is not None:
            self._charge(base, now)
        return None

    def _charge(self, base, now):
        base.aim[0] = np.clip((now - self.started_at) / self.charge_seconds, 0, 1)
