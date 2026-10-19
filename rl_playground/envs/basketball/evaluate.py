"""Separate, deterministic evaluation on held-out wind seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.demo import demo_action
from rl_playground.envs.basketball.env import BasketballEnv


def evaluate(
    policy=None,
    *,
    episodes=config.EVALUATION_EPISODES,
    seed=config.EVALUATION_SEED,
    demo=False,
    settings=None,
):
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive.")
    settings = settings or (
        policy.environment_settings if policy is not None else config.ENVIRONMENT
    )
    rewards, baskets = [], 0
    with BasketballEnv(settings=settings) as env:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            action = (
                demo_action(observation, settings)
                if demo
                else policy.choose_action(observation, greedy=True)
            )
            _, reward, _, _, info = env.step(action)
            rewards.append(reward)
            baskets += int(info["scored"])
    return {
        "episodes": episodes,
        "seed": seed,
        "baskets": baskets,
        "success_rate": baskets / episodes,
        "mean_reward": float(np.mean(rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=Path)
    mode.add_argument("--demo", action="store_true")
    parser.add_argument("--episodes", type=int, default=config.EVALUATION_EPISODES)
    parser.add_argument("--seed", type=int, default=config.EVALUATION_SEED)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    policy = None
    if args.model:
        import torch

        from rl_playground.envs.basketball.policy import load_policy

        torch.set_num_threads(config.TORCH_THREADS)
        policy = load_policy(args.model)
    print(
        json.dumps(
            evaluate(policy, episodes=args.episodes, seed=args.seed, demo=args.demo),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
