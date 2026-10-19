"""The simulation: no neural networks and no display required for training."""

import math
from dataclasses import asdict
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_playground.envs.basketball import config


class BasketballEnv(gym.Env):
    """One action [power, angle] in [0, 1] launches one complete shot.

    Observations are [launcher_x / width, launcher_y / height, hoop_x / width,
    hoop_y / height, wind / max_wind]. This is a one-decision episode: the ball
    cannot be steered after release. Wind acts as constant horizontal acceleration.
    """

    metadata: ClassVar[dict] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": config.FPS,
    }

    def __init__(self, render_mode=None, settings=None):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if isinstance(settings, dict):
            settings = config.EnvironmentSettings(**settings)
        self.settings = settings or config.ENVIRONMENT
        self._validate_settings()
        self.render_mode = render_mode
        self.action_space = spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self._renderer = None
        self.window_closed = False
        self._ready = False
        self.mode_label = "GYMNASIUM"
        self.attempts = self.baskets = 0
        self.aim = np.array([config.INITIAL_POWER, config.INITIAL_ANGLE], np.float32)

    def _validate_settings(self):
        s = self.settings
        if not all(math.isfinite(v) for v in asdict(s).values()):
            raise ValueError("Environment settings must be finite.")
        if (
            min(
                s.world_width,
                s.world_height,
                s.gravity,
                s.physics_dt,
                s.max_flight_seconds,
                s.ball_radius,
                s.rim_radius,
            )
            <= 0
        ):
            raise ValueError(
                "Dimensions, gravity, radii, and time settings must be positive."
            )
        if not 0 <= s.max_wind or not 0 <= s.restitution <= 1:
            raise ValueError(
                "Wind must be nonnegative and restitution must be in [0, 1]."
            )
        if not 0 < s.min_speed < s.max_speed or not 0 < s.min_angle < s.max_angle < 90:
            raise ValueError(
                "Speed/angle ranges must increase; angles must be inside (0, 90)."
            )
        if s.hoop_width <= 2 * (s.ball_radius + s.rim_radius):
            raise ValueError("The opening must be wide enough for the ball.")
        if not (
            s.ball_radius < s.launcher_x < s.hoop_x < s.world_width
            and s.ball_radius < s.launcher_y < s.world_height
            and s.ball_radius < s.hoop_y < s.world_height
        ):
            raise ValueError(
                "Launcher and hoop must be inside the world, with the hoop to the right."
            )

    def observation(self):
        s = self.settings
        return np.array(
            [
                s.launcher_x / s.world_width,
                s.launcher_y / s.world_height,
                s.hoop_x / s.world_width,
                s.hoop_y / s.world_height,
                self.wind / s.max_wind if s.max_wind else 0.0,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.settings
        options = options or {}
        if "wind" in options:
            wind = float(options["wind"])
            if not math.isfinite(wind) or abs(wind) > s.max_wind:
                raise ValueError(
                    f"Wind must be finite and inside [-{s.max_wind}, {s.max_wind}]."
                )
            self.wind = wind
        else:
            self.wind = float(self.np_random.uniform(-s.max_wind, s.max_wind))
        self.ball = np.array([s.launcher_x, s.launcher_y], dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.trajectory = [tuple(self.ball)]
        self.flight_time = 0.0
        self.result = "READY"
        self._ready = True
        if self.render_mode == "human":
            self.render()
        return self.observation(), {"wind": self.wind}

    def decode_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(
                "Action must contain two finite numbers [power, angle] inside [0, 1]."
            )
        s = self.settings
        speed = s.min_speed + float(action[0]) * (s.max_speed - s.min_speed)
        angle = s.min_angle + float(action[1]) * (s.max_angle - s.min_angle)
        return speed, angle

    def _collide(self, previous):
        """Resolve the two round rim edges and a vertical, two-sided backboard."""
        s = self.settings
        for x in (s.hoop_x - s.hoop_width / 2, s.hoop_x + s.hoop_width / 2):
            offset = self.ball - np.array([x, s.hoop_y])
            distance = float(np.linalg.norm(offset))
            radius = s.ball_radius + s.rim_radius
            if distance < radius:
                normal = offset / distance if distance > 1e-12 else np.array([0.0, 1.0])
                self.ball = np.array([x, s.hoop_y]) + normal * radius
                normal_speed = float(self.velocity @ normal)
                if normal_speed < 0:
                    self.velocity -= (1 + s.restitution) * normal_speed * normal

        board = s.hoop_x + s.backboard_offset
        # Swept plane checks prevent fast balls passing through the board.
        for side in (-1, 1):
            plane = board + side * s.ball_radius
            if side * (previous[0] - plane) >= 0 > side * (self.ball[0] - plane):
                fraction = (plane - previous[0]) / (self.ball[0] - previous[0])
                crossing_y = previous[1] + fraction * (self.ball[1] - previous[1])
                if (
                    s.hoop_y - s.backboard_below - s.ball_radius
                    <= crossing_y
                    <= s.hoop_y + s.backboard_above + s.ball_radius
                ):
                    self.ball[0] = plane
                    self.velocity[0] *= -s.restitution
                    break

    def step(self, action):
        if not self._ready:
            raise gym.error.ResetNeeded("Call reset() before each shot.")
        speed, angle = self.decode_action(action)
        self.aim = np.asarray(action, dtype=np.float32).copy()
        s = self.settings
        radians = math.radians(angle)
        self.velocity[:] = (speed * math.cos(radians), speed * math.sin(radians))
        acceleration = np.array([self.wind, -s.gravity])
        self.result = "IN FLIGHT"
        closest = math.inf
        crossing_error = None
        scored = False
        reason = "time_limit"
        terminated = False
        steps = math.ceil(s.max_flight_seconds / s.physics_dt)
        render_stride = max(1, round(1 / (config.FPS * s.physics_dt)))
        for tick in range(steps):
            previous = self.ball.copy()
            dt = min(s.physics_dt, s.max_flight_seconds - self.flight_time)
            self.ball += self.velocity * dt + 0.5 * acceleration * dt * dt
            self.velocity += acceleration * dt
            self.flight_time += dt
            self._collide(previous)
            self.trajectory.append(tuple(self.ball))
            # Reward near misses on the descending part of the arc, not an upward fly-by.
            if self.velocity[1] < 0:
                closest = min(
                    closest,
                    math.hypot(self.ball[0] - s.hoop_x, self.ball[1] - s.hoop_y),
                )
                if previous[1] > s.hoop_y >= self.ball[1]:
                    fraction = (previous[1] - s.hoop_y) / (previous[1] - self.ball[1])
                    crossing_x = previous[0] + fraction * (self.ball[0] - previous[0])
                    error = abs(float(crossing_x) - s.hoop_x)
                    crossing_error = (
                        error if crossing_error is None else min(crossing_error, error)
                    )
                    clearance = s.hoop_width / 2 - s.ball_radius - s.rim_radius
                    if error < clearance:
                        scored, terminated, reason = True, True, "basket"
            if scored:
                break
            if self.ball[1] <= s.ball_radius:
                self.ball[1] = s.ball_radius
                terminated, reason = True, "ground"
                break
            if not (
                -1 <= self.ball[0] <= s.world_width + 1
                and self.ball[1] <= s.world_height + 4
            ):
                terminated, reason = True, "out_of_bounds"
                break
            if self.render_mode == "human" and tick % render_stride == 0:
                self.render()
                if self.window_closed:
                    reason = "window_closed"
                    break

        if not math.isfinite(closest):
            closest = math.hypot(self.ball[0] - s.hoop_x, self.ball[1] - s.hoop_y)
        reward = 2 * math.exp(-s.miss_distance_scale * closest) - 1
        if scored:
            reward += s.score_bonus
        self._ready = False
        self.attempts += 1
        self.baskets += int(scored)
        self.result = "BASKET!" if scored else "MISS" if terminated else "SHOT ENDED"
        info = {
            "scored": scored,
            "wind": self.wind,
            "closest_distance": closest,
            "crossing_error": crossing_error,
            "speed": speed,
            "angle_degrees": angle,
            "flight_time": self.flight_time,
            "termination_reason": reason,
        }
        if self.render_mode == "human" and not self.window_closed:
            self.render()
        return self.observation(), float(reward), terminated, not terminated, info

    def render(self):
        if self.render_mode is None:
            return None
        if not hasattr(self, "ball"):
            raise gym.error.ResetNeeded("Call reset() before render().")
        if self._renderer is None:
            from rl_playground.envs.basketball.rendering import Renderer

            self._renderer = Renderer(self.render_mode)
        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
