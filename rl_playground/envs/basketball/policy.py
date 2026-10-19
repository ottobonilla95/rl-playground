"""A small continuous-action REINFORCE policy, separate from the physics demo."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Independent

from rl_playground.envs.basketball import config


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=config.HIDDEN_SIZES, environment_settings=None):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.environment_settings = environment_settings or config.ENVIRONMENT
        layers, previous = [], config.INPUT_SIZE
        for width in self.hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        layers.append(nn.Linear(previous, config.ACTION_SIZE * 2))
        self.network = nn.Sequential(*layers)

    def forward(self, observations):
        # Each action has its own Beta distribution on [0, 1]. Parameters > 1
        # avoid singular densities at the edges and allow a single interior peak.
        concentrations = nn.functional.softplus(self.network(observations)) + 1.0
        alpha, beta = concentrations.chunk(2, dim=-1)
        return Independent(Beta(alpha, beta), 1)

    @torch.no_grad()
    def choose_action(self, observation, *, greedy=False, seed=None):
        # A separate seed per shot keeps results stable across worker counts.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            distribution = self(torch.as_tensor(observation, dtype=torch.float32))
            action = distribution.mean if greedy else distribution.sample()
        return action.numpy().astype(np.float32)


def save_policy(policy, path, *, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "env_id": "BasketballLauncher-v0",
        "hidden_sizes": list(policy.hidden_sizes),
        "activation": "tanh",
        "action_distribution": "independent_beta",
        "state_dict": policy.state_dict(),
        "environment_settings": asdict(policy.environment_settings),
        "metadata": metadata or {},
    }
    with path.open("xb") as stream:  # Never silently replace a trained model.
        torch.save(payload, stream)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("env_id") != "BasketballLauncher-v0"
        or checkpoint.get("activation") != "tanh"
        or checkpoint.get("action_distribution") != "independent_beta"
    ):
        raise ValueError("This is not a supported BasketballLauncher checkpoint.")
    settings = config.EnvironmentSettings(**checkpoint["environment_settings"])
    policy = PolicyNetwork(checkpoint["hidden_sizes"], settings)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def main():
    from rl_playground.envs.basketball.env import BasketballEnv

    torch.manual_seed(config.SEED)
    policy = PolicyNetwork()
    with BasketballEnv() as env:
        observation, _ = env.reset(seed=config.SEED)
        action = policy.choose_action(observation, seed=config.SEED)
        _, reward, _, _, info = env.step(action)
    print(f"Observation: {observation}")
    print(f"Untrained action [power, angle]: {action}")
    print(f"Shot: {info} | reward {reward:.3f}")


if __name__ == "__main__":
    main()
