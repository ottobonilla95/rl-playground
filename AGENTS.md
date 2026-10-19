# RL Playground

This is an independent, educational library of Gymnasium environments. Keep each
environment in its own module under rl_playground/envs/. Basketball is the first
module; do not make its configuration or interfaces global library defaults.
Keep physics, rendering, rule-based demos, and learning policies separate and readable.

- Environment defaults live in each module's config.py; CLI flags override one run.
- Basketball: one action chooses power and angle; one step simulates a complete shot.
- Basketball wind is visible, sampled on reset, and constant throughout that shot.
- Do not leak the demo controller into the learned policy or reward.
- Preserve seeded determinism across worker counts.
- Never overwrite generated models or results without explicit authorization.
- Every training invocation starts fresh. Checkpoints record architecture/settings.
- Do not launch long training without a request. A smoke run checks execution only.

Validation from the repository root:

```sh
uv sync --locked --extra train
uv run python -m unittest discover -s tests -v
uv run main.py --demo --headless --shots 20
uv run train.py --episodes 16 --batch-size 8 --workers 2 --eval-episodes 8 --output models/agent-smoke.pt
```

Choose a new output basename if that smoke path already exists. For rendering tests,
use SDL_VIDEODRIVER=dummy; manually inspect a rendered image too.
