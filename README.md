# RL Playground

A reusable library for your own Gymnasium environments. Each world lives in its
own module, with its physics, settings, playable mode, demo, and learning examples.
This project is independent of rocket-land and has its own Git repository.

**Basketball Launcher is the first environment**, under
`rl_playground.envs.basketball`. More environments can be added alongside it; they
are not implemented yet.

```text
rl_playground/
  __init__.py              # Register the library's environments
  envs/
    basketball/            # First environment module
      env.py               # Physics, spaces, scoring
      config.py            # Basketball-specific settings
      rendering.py         # Basketball court and controls
      demo.py              # Hardcoded controller
      play.py              # Manual, demo, and neural play
      policy.py            # Example learned controller
      train.py             # Example RL trainer
      evaluate.py          # Held-out evaluation
```

## Basketball Launcher

Read the wind, choose launch power and angle, and try to score. Play by hand,
watch a hardcoded physics demo, or train a neural policy.

Wind changes on every new attempt and stays constant throughout that shot. An
arrow, moving wind marks, and a number show its direction and strength. Here wind
is a simplified **horizontal acceleration**, measured in m/s², rather than a
simulation of airflow. Gravity, round rim edges, and backboard bounces affect the ball.

## Start playing

Install [Git](https://git-scm.com/downloads) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/ottobonilla95/rl-playground.git
cd rl-playground
uv sync --locked --extra train
uv run main.py
```

| Control | Action |
| --- | --- |
| Move mouse | Aim toward the pointer (within the allowed launch angles) |
| Hold Space | Charge power from minimum to maximum in 1.5 seconds |
| Release Space | Launch with the current angle and charged power |
| R | Prepare a new shot with fresh wind |
| Escape / close window | Quit |

Power, angle, and a live charge bar remain visible at the bottom. Move the mouse
to aim, hold Space to build power, then release to shoot. Full power stays capped
until you release. After a result, the next Space press prepares fresh wind and
begins charging the next shot; you can keep aiming while holding it. R draws new
wind and cancels charging; switching windows cancels charging too. Wind stays
constant throughout each shot. Set `CHARGE_SECONDS` in the basketball config to
change the charging time. The dotted trail shows where the ball actually flew;
it does not predict the answer.

```sh
uv run main.py --demo                   # Physics controller; no training
uv run main.py --nn                     # Untrained network
uv run main.py --demo --headless --shots 20
```

The demo solves projectile equations in a function. It knows the wind and gravity;
it is a reference controller, not a trained agent. A graphical desktop is required
for windowed play. The headless commands work without a display.

## Import your environment elsewhere

Install the library into another Python project:

```sh
uv add git+https://github.com/ottobonilla95/rl-playground.git
```

For a local editable checkout, use `uv add --editable /path/to/rl-playground` instead.

Then use the same Gymnasium interface as LunarLander:

```python
import gymnasium as gym
import rl_playground  # Registers the available environments

env = gym.make("BasketballLauncher-v0", render_mode="human")
observation, info = env.reset(seed=42)
action = env.action_space.sample()  # [power, angle], each between 0 and 1
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

You can also import the basketball module directly:

```python
from rl_playground.envs.basketball import BasketballEnv

with BasketballEnv() as env:
    observation, info = env.reset(seed=42)
```

The package follows Gymnasium's [custom environment interface](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
Omit `render_mode` for fast, headless training; `render_mode="rgb_array"` returns
an RGB NumPy image. The environment itself does not require PyTorch. Install the
`train` extra only for this project's neural policy and training/chart commands.
Basketball's installed commands are `basketball-play`, `basketball-train`, and
`basketball-evaluate`. The root `main.py`, `train.py`, `policy.py`, and `evaluate.py`
are convenience shortcuts for this first environment.

## Train, then watch the exact saved model

A short execution check (not a test of learning quality):

```sh
uv run train.py --episodes 16 --batch-size 8 --workers 2 --eval-episodes 8 --output models/agent-smoke.pt
uv run main.py --nn --model models/agent-smoke.pt --greedy
```

For a longer experiment when you are ready:

```sh
uv run train.py --episodes 5000 --output models/experiment-01.pt
uv run main.py --nn --model models/experiment-01.pt --greedy
uv run evaluate.py --model models/experiment-01.pt --episodes 100
```

Every training command starts a **fresh network**. Choose a new output basename
each run: existing outputs are refused. Training writes `.pt` weights and architecture,
`.csv` batch metrics, `.png` charts, and `.json` settings, command, code identity, and
initial/final evaluation. It saves the final network, not the best intermediate one.
The model retains the environment settings used for training when viewed/evaluated.

`--greedy` uses the two Beta distributions' **means** for repeatable shots; omitting
it samples actions. Batch training reward and held-out evaluation are different
measurements. A decreasing policy loss is not required for better shooting.
No long experiment or successful trained policy is included by default.

## Customize and understand

Edit **[rl_playground/envs/basketball/config.py](rl_playground/envs/basketball/config.py)** for geometry,
wind limits, power/angle ranges, physics, network size, and training defaults.
CLI flags override a single run; see `uv run train.py --help`.

| Part | File |
| --- | --- |
| Library registration | `rl_playground/__init__.py` |
| Basketball registration | `rl_playground/envs/basketball/__init__.py` |
| Physics, spaces, rewards | `rl_playground/envs/basketball/env.py` |
| Court, ball, hoop, wind, HUD | `rl_playground/envs/basketball/rendering.py` |
| Manual/demo/neural modes | `rl_playground/envs/basketball/play.py` |
| Hardcoded controller | `rl_playground/envs/basketball/demo.py` |
| Continuous neural policy/checkpoints | `rl_playground/envs/basketball/policy.py` |
| REINFORCE trainer | `rl_playground/envs/basketball/train.py` |

Read **[the learning guide](docs/learning-guide.md)** for the action/observation
contract, scoring, physics assumptions, and training math. This first version
uses a fixed launcher and hoop with a generous opening; changing geometry creates
a different task. There is no mid-flight control, spin, drag, or 3D perspective.

## Validate

```sh
uv sync --locked --extra train
uv run python -m unittest discover -s tests -v
uv run policy.py
uv run main.py --demo --headless --shots 20
uv run evaluate.py --demo --episodes 100
```

Tests cover the Gymnasium contract, seeding, scoring, collisions, boundaries,
rendering, checkpoints, policy updates, and deterministic worker ordering.
Generated `models/` and `visualizations/` are ignored by Git.

## Add another environment

Create a sibling module such as `rl_playground/envs/your_environment/`, implement
its Gymnasium environment, and register it with a distinct environment ID in that
module's `__init__.py`. Import that module from `rl_playground/__init__.py` so a
single `import rl_playground` registers all available worlds. Keep its settings,
rendering, and examples inside its module; add its tests under `tests/`.
