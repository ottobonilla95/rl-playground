"""Train from scratch with one policy update per batch of complete shots."""

import argparse
import csv
import hashlib
import json
import multiprocessing
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.env import BasketballEnv
from rl_playground.envs.basketball.evaluate import evaluate
from rl_playground.envs.basketball.policy import PolicyNetwork, save_policy


def simulate_shot(task):
    """Workers simulate only; all action sampling and learning stay in the parent."""
    settings, seed, action = task
    with BasketballEnv(settings=settings) as env:
        env.reset(seed=seed)
        _, reward, _, _, info = env.step(action)
    return reward, info["scored"]


def update_policy(policy, optimizer, observations, actions, rewards, *, baseline=0.0):
    states = torch.as_tensor(np.array(observations), dtype=torch.float32)
    chosen = torch.as_tensor(np.array(actions), dtype=torch.float32)
    returns = torch.as_tensor(rewards, dtype=torch.float32)
    # Every episode contains one decision, so its return is simply its reward.
    # A leave-one-out baseline excludes a shot's own reward from its baseline.
    if len(rewards) > 1:
        other_mean = (returns.sum() - returns) / (len(rewards) - 1)
        advantages = returns - other_mean
        advantages /= returns.std(unbiased=False).clamp_min(
            config.NORMALIZATION_EPSILON
        )
    else:
        advantages = returns - baseline
    distribution = policy(states)
    loss = -(distribution.log_prob(chosen) * advantages).mean()
    loss -= config.ENTROPY_COEFFICIENT * distribution.entropy().mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), config.MAX_GRAD_NORM)
    optimizer.step()
    return float(loss.detach())


def code_identity():
    package_root = Path(__file__).resolve().parents[2]
    root = package_root.parent

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    # A hash identifies even the initial uncommitted source used for a run.
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return {
        "revision": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
        "source_sha256": digest.hexdigest(),
    }


def plot_training(csv_path, png_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    episodes = [int(row["episodes"]) for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for axis, key, label in zip(
        axes,
        ("mean_reward", "success_rate", "loss"),
        ("Training batch reward", "Training batch basket rate", "Policy loss"),
    ):
        axis.plot(episodes, [float(row[key]) for row in rows], color="#df782e")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Training shots")
    figure.suptitle("Basketball Launcher | batch collection metrics")
    with png_path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=config.EPISODES)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.WORKERS)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--eval-episodes", type=int, default=config.EVALUATION_EPISODES)
    parser.add_argument("--output", type=Path, default=config.OUTPUT)
    args = parser.parse_args()
    if min(args.episodes, args.batch_size, args.workers, args.eval_episodes) <= 0:
        parser.error("Episode counts, batch size, and workers must be positive.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if args.seed < 0 or args.output.suffix != ".pt":
        parser.error("Seed must be nonnegative and output must end with .pt")
    # Evaluation is disjoint from every training seed, including very long runs.
    eval_seed = max(config.EVALUATION_SEED, args.seed + args.episodes)
    paths = {
        ext: args.output.with_suffix(ext) for ext in (".pt", ".csv", ".json", ".png")
    }
    for path in paths.values():
        if path.exists():
            parser.error(
                f"Output already exists: {path}. Choose a new --output basename."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.TORCH_THREADS)
    torch.manual_seed(args.seed)
    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    workers = min(args.workers, args.batch_size, args.episodes)
    metadata = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "code": code_identity(),
        "arguments": {**vars(args), "output": str(args.output)},
        "effective_settings": {
            "environment": asdict(policy.environment_settings),
            "hidden_sizes": list(policy.hidden_sizes),
            "activation": "tanh",
            "action_distribution": "independent_beta",
            "workers": workers,
            "entropy_coefficient": config.ENTROPY_COEFFICIENT,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "normalization_epsilon": config.NORMALIZATION_EPSILON,
            "torch_threads": config.TORCH_THREADS,
            "evaluation_seed": eval_seed,
        },
        "versions": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }
    import gymnasium

    metadata["versions"]["gymnasium"] = gymnasium.__version__
    with paths[".json"].open("x") as stream:
        json.dump(metadata, stream, indent=2)
    initial = evaluate(policy, episodes=args.eval_episodes, seed=eval_seed)
    metadata["initial_evaluation"] = initial
    print(f"Initial held-out mean-action evaluation: {initial}")
    print(f"Collecting {args.batch_size} shots per update with {workers} worker(s).")
    completed, baseline = 0, 0.0
    pool = (
        ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        with (
            paths[".csv"].open("x", newline="") as stream,
            BasketballEnv(settings=policy.environment_settings) as env,
        ):
            writer = csv.writer(stream)
            writer.writerow(["episodes", "mean_reward", "success_rate", "loss"])
            for start in range(0, args.episodes, args.batch_size):
                stop = min(start + args.batch_size, args.episodes)
                seeds = list(range(args.seed + start, args.seed + stop))
                observations = [env.reset(seed=seed)[0] for seed in seeds]
                actions = [
                    policy.choose_action(obs, seed=seed)
                    for obs, seed in zip(observations, seeds)
                ]
                tasks = [
                    (policy.environment_settings, seed, action)
                    for seed, action in zip(seeds, actions)
                ]
                outcomes = (
                    list(pool.map(simulate_shot, tasks))
                    if pool
                    else list(map(simulate_shot, tasks))
                )
                rewards, scores = zip(*outcomes)
                loss = update_policy(
                    policy, optimizer, observations, actions, rewards, baseline=baseline
                )
                baseline = float(np.mean(rewards))
                completed = stop
                rate = float(np.mean(scores))
                writer.writerow([completed, baseline, rate, loss])
                stream.flush()
                print(
                    f"Shots {completed}/{args.episodes} | reward {baseline:.3f} | baskets {rate:.1%} | loss {loss:.4f}"
                )
        metadata["status"] = "complete"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        print("Interrupted; saving the last completed policy update.")
    except Exception as error:
        metadata.update(status="failed", error=str(error), completed_episodes=completed)
        paths[".json"].write_text(json.dumps(metadata, indent=2) + "\n")
        raise
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    metadata["completed_episodes"] = completed
    metadata["final_evaluation"] = evaluate(
        policy, episodes=args.eval_episodes, seed=eval_seed
    )
    metadata["finished_utc"] = datetime.now(UTC).isoformat()
    save_policy(policy, paths[".pt"], metadata=metadata)
    plot_training(paths[".csv"], paths[".png"])
    paths[".json"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Final held-out mean-action evaluation: {metadata['final_evaluation']}")
    print(f"Saved {args.output} and its .csv, .json, and .png companions.")


if __name__ == "__main__":
    main()
