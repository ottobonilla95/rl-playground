"""Mouse aiming and hold-to-charge play, a rule-based demo, or a saved neural policy."""

import argparse
import time

import gymnasium as gym

from rl_playground.envs import (
    basketball,  # Registers the environment, just as a third-party user would.
)
from rl_playground.envs.basketball import config
from rl_playground.envs.basketball.controls import ManualControls
from rl_playground.envs.basketball.demo import demo_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo", action="store_true", help="Watch the hardcoded ballistic controller"
    )
    mode.add_argument(
        "--nn",
        action="store_true",
        help="Watch a neural network (untrained without --model)",
    )
    parser.add_argument(
        "--model", type=str, help="Exact checkpoint path; requires --nn"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Use the policy's mean power/angle"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window; requires --demo or --nn",
    )
    parser.add_argument(
        "--shots",
        type=int,
        help="Stop after this many shots (headless default from config)",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    if args.model and not args.nn:
        parser.error("--model requires --nn")
    if args.headless and not (args.demo or args.nn):
        parser.error("Manual play needs a window. Use --demo or --nn with --headless.")
    if args.shots is not None and args.shots <= 0:
        parser.error("--shots must be positive")
    limit = args.shots or (config.DEMO_SHOTS if args.headless else None)
    policy = None
    if args.nn:
        import torch

        from rl_playground.envs.basketball.policy import PolicyNetwork, load_policy

        torch.set_num_threads(config.TORCH_THREADS)
        torch.manual_seed(args.seed)
        policy = load_policy(args.model) if args.model else PolicyNetwork()
        policy.eval()
    settings = policy.environment_settings if policy is not None else config.ENVIRONMENT
    env = gym.make(
        basketball.ENV_ID,
        render_mode=None if args.headless else "human",
        settings=settings,
    )
    base = env.unwrapped
    base.mode_label = (
        "BALLISTIC DEMO" if args.demo else "NEURAL POLICY" if args.nn else "FREE PLAY"
    )
    controls = ManualControls(charge_seconds=config.CHARGE_SECONDS)
    if not (args.demo or args.nn):
        controls.cancel(base)
    observation, _ = env.reset(seed=args.seed)
    shots = baskets = 0
    next_shot_at = time.monotonic() + config.AUTO_SHOT_PAUSE
    try:
        while not base.window_closed and (limit is None or shots < limit):
            if args.headless:
                shoot = True
            else:
                import pygame

                events = pygame.event.get()
                if any(
                    e.type == pygame.QUIT
                    or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE)
                    for e in events
                ):
                    break
                if args.demo or args.nn:
                    if any(
                        e.type == pygame.KEYDOWN and e.key == pygame.K_r for e in events
                    ):
                        observation, _ = env.reset()
                    shoot = time.monotonic() >= next_shot_at or any(
                        e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE
                        for e in events
                    )
                else:
                    action = controls.update(
                        env,
                        events,
                        pygame.mouse.get_pos(),
                        time.monotonic(),
                        next_seed=args.seed + shots,
                    )
                    shoot = action is not None
                if not shoot:
                    env.render()
                    continue
            if not base._ready:
                observation, _ = env.reset(seed=args.seed + shots)
            if args.demo:
                action = demo_action(observation, base.settings)
            elif policy is not None:
                action = policy.choose_action(
                    observation, greedy=args.greedy, seed=args.seed + shots
                )
            _, reward, _, _, info = env.step(action)
            if not args.headless and not (args.demo or args.nn):
                # Flight animation owns the window until landing. Ignore Space
                # taps from that animation rather than queuing an accidental shot.
                pygame.event.clear([pygame.KEYDOWN, pygame.KEYUP])
            if base.window_closed:
                break
            shots += 1
            baskets += int(info["scored"])
            print(
                f"Shot {shots}: {'BASKET' if info['scored'] else 'miss'} | wind {info['wind']:+.2f} | "
                f"angle {info['angle_degrees']:.1f} | speed {info['speed']:.2f} | reward {reward:.3f}"
            )
            next_shot_at = time.monotonic() + config.AUTO_SHOT_PAUSE
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print(f"{baskets}/{shots} baskets ({100 * baskets / max(shots, 1):.1f}%)")


if __name__ == "__main__":
    main()
