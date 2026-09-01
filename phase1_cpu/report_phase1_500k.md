# Phase 1: One PPO Recipe Across 5 Public Benchmarks (CPU only)

## Headline

A single PPO recipe (identical hyperparameters) scores **well above a random policy on all 5 MinAtar games**, at a fast-demo budget of 500k steps per game, on CPU (Apple M1, no GPU).

## What was built

- **Custom Gymnasium environment** (`envs/gridworld_env.py`, `GridWorld-v0`): the "build an RL environment" artifact. The same recipe learns it to the optimal path (return 0.93, 8-step solution).
- **The one recipe** (`train.py` + `config.yaml`): PPO, a tiny custom CNN feature extractor (`envs/minatar_cnn.py`) for MinAtar's 10x10xC observations, 8 parallel envs, 3 seeds. Reused unchanged across the custom env and all 5 benchmarks.

## Results (500k steps/game, mean of 3 seeds)

| Game | Random | Ours (mean +/- std) | Published DQN | % of DQN |
|---|---|---|---|---|
| Breakout | 0.57 | **13.04 +/- 1.44** | 27.1 | 48% |
| Freeway | 0.13 | **20.77 +/- 1.31** | 55.9 | 37% |
| SpaceInvaders | 3.67 | **49.74 +/- 7.30** | 188.0 | 26% |
| Asterix | 0.33 | **2.40 +/- 0.27** | 13.6 | 18% |
| Seaquest | 0.23 | **1.33 +/- 0.14** | 38.0 | 3.5% |

All 5 clear the random baseline. Per-game learning curves: `results/<game>/curve.png`. Interactive animated replay wired to this data: `docs/replay.html`.

## Compute

CPU only (Apple M1, no CUDA). About 4.5 min per 500k run, 15 runs (5 games x 3 seeds). Environment install, smoke tests, and the full run all completed in one session.

## Honest caveats

- **Fast-demo budget.** The goal was "clearly above random", not matching published DQN. Breakout, Freeway, and SpaceInvaders learn strongly; Asterix and especially Seaquest are weak at 500k (harder games with sparse reward). A ~2M-step push targets these.
- **Two different measurements.** The table above is a 30-episode greedy evaluation of the final model (stable). The animated curves in `replay.html` show the per-checkpoint `EvalCallback` series (20 episodes each, hence noisier). Example: Freeway's last checkpoint eval spikes near 94% of baseline while the stable final-model eval is 37%. Both are honest; the table is the reported score.
- **MinAtar has no episode step limit.** A deterministic policy on Seaquest could produce a non-terminating episode that hangs evaluation. Fixed with `gym.make(env_id, max_episode_steps=1000)`; a stall watchdog caught the original hang.

## Reproduce

```bash
cd phase1_cpu
./run.sh                                  # gated: setup, checks, smoke, full run, export
# or directly:
./.venv/bin/python benchmark.py --budget fast --seeds 0 1 2
```
