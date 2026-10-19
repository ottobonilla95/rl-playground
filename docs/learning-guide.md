# How the basketball module works

The basketball implementation lives in `rl_playground.envs.basketball`. Other
environments can live alongside it with their own rules and training examples.

## The environment contract

Import `rl_playground`, then call `gym.make("BasketballLauncher-v0")`.
`reset(seed=...)` prepares a new ball and samples the wind uniformly from
`[-max_wind, +max_wind]`. Passing the same seed reproduces the same wind. Calling
`reset()` without a seed advances the environment's random generator.

The five float32 observation values are:

| Index | Meaning | Normalization |
| --- | --- | --- |
| 0 | Launcher X | world width |
| 1 | Launcher Y | world height |
| 2 | Hoop centre X | world width |
| 3 | Hoop centre Y | world height |
| 4 | Signed horizontal wind acceleration | maximum wind magnitude |

The action space is `Box(0, 1, shape=(2,))`:

1. Power maps linearly from `min_speed` to `max_speed`.
2. Angle maps linearly from `min_angle` to `max_angle` degrees above the positive X axis.

Power specifies initial speed, not an ongoing thrust force. Actions outside the
space raise an error rather than silently being clipped. Wind is observable and
does not change during flight. For controlled tests use `reset(options={"wind": 0})`.

**One episode is one shot and contains one decision.** A single `step(action)`
advances the entire flight. In human mode it animates; in headless mode it runs as
fast as possible. The returned observation describes the same launcher, hoop, and
wind; call `reset` before the next shot. This is a contextual bandit formulation
of RL. A flight with many physics frames does not imply many agent decisions.
Compatible continuous-action RL algorithms can use the environment directly.

## Physics and scoring

For speed `v` and angle `theta`:

```text
vx = v * cos(theta)
vy = v * sin(theta)
ax = wind
ay = -gravity

position += velocity * dt + 0.5 * acceleration * dt**2
velocity += acceleration * dt
```

The integrator is exact between collisions for constant acceleration, with small
substeps for contact checks. Two circular rim edges reflect velocity along their
contact normals. A swept vertical-plane test handles backboard contact. The
restitution coefficient removes energy on each bounce. This is a small custom
simulation with approximate contacts, not a full rigid-body basketball simulator.

A basket requires the ball centre to cross rim height **downward**, inside the
opening with enough clearance for its radius and the rim's thickness. Flying up
through the opening does not score. A basket, ground contact, or leaving the
world ends with `terminated=True`; the flight time limit ends with
`truncated=True`. Closing a human window also truncates an in-progress attempt.
No reward is given simply for hitting the board or rim.

The reward is a bonus for scoring plus a smooth near-miss signal:

```text
reward = 2 * exp(-miss_distance_scale * closest_descending_distance) - 1
if scored:
    reward += score_bonus
```

Near misses help exploration before baskets are common. Reward is not itself a
success rate: always inspect `info["scored"]` or the separate evaluation score.
`info` also reports wind, chosen speed/angle, flight time, nearest descending
distance, rim-height crossing error (if any), and the termination reason.

## The hardcoded demo

`demo_action` chooses a suitable flight time `t` and rearranges the same physics:

```text
vx = horizontal_distance / t - 0.5 * wind * t
vy = vertical_distance / t + 0.5 * gravity * t
speed = sqrt(vx**2 + vy**2)
angle = atan2(vy, vx)
```

It checks the allowed speed/angle and descending arrival. This controller is
useful to verify reachability. Its actions are never used to initialize or train
the neural network; there is no imitation learning hidden in the RL trainer.

## The neural policy

The default network is `5 → 64 → 64 → 4`, using Tanh hidden layers. Four outputs
become positive `alpha`/`beta` parameters for two independent Beta distributions:
one for power, one for angle. A Beta distribution lives on `[0, 1]`, which matches
our action bounds without clipping sampled actions.

During training we **sample** from those distributions. During deterministic
evaluation we use their **means**, `alpha / (alpha + beta)`. The `--greedy` flag
means this deterministic mean action; it is not the mathematical mode of a Beta
distribution. The implementation uses PyTorch's documented
[distribution and REINFORCE operations](https://docs.pytorch.org/docs/stable/distributions.html).

## Learning from batches

Every training invocation creates a fresh policy and Adam optimizer:

1. Reset each shot using its own seed, giving a known wind observation.
2. Sample one action per shot using a separate per-shot PyTorch seed.
3. Simulate the complete shots with the same fixed policy.
4. Compute an advantage by subtracting the other shots' mean reward, then scale
   by batch reward standard deviation. A batch of one uses the previous batch's
   mean as its baseline instead.
5. Update the policy once with `loss = -mean(log_prob(action) * advantage)` plus a
   small entropy term that encourages exploration. Clip large gradients.
6. Discard the batch and collect new shots with the updated policy.

There is one action in each episode, so its return equals its reward. A discount
factor would have no effect and is deliberately absent here. Entropy encourages
exploration; it does not guarantee convergence or a particular basket rate.

With `--workers 2`, persistent worker processes simulate shots only. The parent
samples actions, preserves shot ordering, and performs the sole weight update.
Changing worker count keeps the same per-shot seeds and ordered data. The default
is one worker because process startup can dominate this small environment.

## Reading experiments

The CSV/chart contains collection-batch reward, basket rate, and policy loss.
The JSON also records initial and final **mean-action evaluation** on the same
held-out wind seeds, disjoint from the training seed range. Loss values from
different batches need not decrease. Learning quality requires longer tests and
multiple seeds; the 16-shot smoke command only checks that execution works.

Checkpoints contain weights, network architecture, environment settings, and run
metadata. Load them with `load_policy` so viewing and evaluation use the recorded
environment. Training never resumes a checkpoint. Ctrl+C saves the weights from
the last finished update; an unfinished batch is discarded.

Run JSON includes the exact Python invocation, effective settings, package
versions, Git revision/status, and a SHA-256 digest of package source. For an
initial uncommitted repository, the Git revision is null and the source digest
still identifies the implementation. Generated artifacts stay local in `models/`.
