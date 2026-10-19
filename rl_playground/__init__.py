"""RL Playground: reusable Gymnasium environments, each in its own module.

Importing the library registers the available environments with Gymnasium.
Environment-specific physics, configuration, demos, and training stay together.
"""

from rl_playground.envs import basketball

__all__ = ["basketball"]
