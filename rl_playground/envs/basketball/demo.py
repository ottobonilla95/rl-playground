"""A hardcoded ballistic controller. It contains no learned weights."""

import math

import numpy as np

from rl_playground.envs.basketball import config


def demo_action(observation, settings=None):
    """Pick a flight time, then solve x(t) and y(t) for the launch velocity.

    The controller sees exactly the observation that the RL policy sees. It knows
    the physics settings, as a hand-written controller would, and aims at the rim
    centre on the descending arc. No simulated search or training is involved.
    """
    s = settings or config.ENVIRONMENT
    launch_x, launch_y, hoop_x, hoop_y, wind = np.asarray(observation, dtype=float)
    dx = (hoop_x - launch_x) * s.world_width
    dy = (hoop_y - launch_y) * s.world_height
    wind *= s.max_wind
    candidates = np.linspace(
        config.DEMO_MIN_FLIGHT_TIME,
        config.DEMO_MAX_FLIGHT_TIME,
        config.DEMO_CANDIDATE_COUNT,
    )
    candidates = sorted(candidates, key=lambda t: abs(t - config.DEMO_FLIGHT_TIME))
    for time in candidates:
        vx = dx / time - 0.5 * wind * time
        vy = dy / time + 0.5 * s.gravity * time
        speed, angle = math.hypot(vx, vy), math.degrees(math.atan2(vy, vx))
        if (
            s.min_speed <= speed <= s.max_speed
            and s.min_angle <= angle <= s.max_angle
            and vy - s.gravity * time < 0
        ):
            return np.array(
                [
                    (speed - s.min_speed) / (s.max_speed - s.min_speed),
                    (angle - s.min_angle) / (s.max_angle - s.min_angle),
                ],
                np.float32,
            )
    raise ValueError(
        "No valid ballistic shot for these settings; widen the speed/angle ranges."
    )
