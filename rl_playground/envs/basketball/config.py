"""All editable environment, display, and experiment defaults live here."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentSettings:
    # Metres, seconds, and metres/second. Positive X is right; positive Y is up.
    world_width: float = 12.0
    world_height: float = 7.0
    launcher_x: float = 1.5
    launcher_y: float = 1.4
    hoop_x: float = 9.2
    hoop_y: float = 3.05
    # A generous opening keeps this first environment approachable.
    hoop_width: float = 0.8
    ball_radius: float = 0.12
    rim_radius: float = 0.035
    backboard_offset: float = 0.53
    backboard_below: float = 0.15
    backboard_above: float = 1.15
    gravity: float = 9.81
    max_wind: float = 2.0  # Horizontal acceleration, NOT wind speed.
    min_speed: float = 5.0
    max_speed: float = 15.0
    min_angle: float = 15.0
    max_angle: float = 80.0
    physics_dt: float = 1 / 240
    max_flight_seconds: float = 6.0
    restitution: float = 0.65
    score_bonus: float = 10.0
    miss_distance_scale: float = 1.5


ENVIRONMENT = EnvironmentSettings()
WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 760
FPS = 60
INITIAL_POWER = 0.5
INITIAL_ANGLE = 0.65
CONTROL_RATE = 0.35  # Normalized action units/second while a key is held.
AUTO_SHOT_PAUSE = 0.8
DEMO_FLIGHT_TIME = 1.6
DEMO_CANDIDATE_COUNT = 241
DEMO_MIN_FLIGHT_TIME = 0.8
DEMO_MAX_FLIGHT_TIME = 2.8
DEMO_SHOTS = 20

HIDDEN_SIZES = (64, 64)
EPISODES = 5000
BATCH_SIZE = 64
WORKERS = 1
LEARNING_RATE = 0.003
ENTROPY_COEFFICIENT = 0.001
MAX_GRAD_NORM = 1.0
NORMALIZATION_EPSILON = 1e-8
TORCH_THREADS = 1
SEED = 42
EVALUATION_EPISODES = 100
EVALUATION_SEED = 100_000
OUTPUT = Path("models/policy.pt")
INPUT_SIZE = 5
ACTION_SIZE = 2
