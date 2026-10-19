"""Import this package to register BasketballLauncher-v0 with Gymnasium."""

from gymnasium.envs.registration import register, registry

ENV_ID = "BasketballLauncher-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="rl_playground.envs.basketball.env:BasketballEnv")

from rl_playground.envs.basketball.env import BasketballEnv

__all__ = ["ENV_ID", "BasketballEnv"]
