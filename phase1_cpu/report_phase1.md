# Phase 1: One PPO Recipe Across 5 Public Benchmarks (CPU only)

## Headline

A single PPO recipe (identical hyperparameters) scores **well above a random policy on all 5 MinAtar games**. At a 2M-step budget on CPU (Apple M1, no GPU), it reaches **44% to 78% of the published DQN baselines on four of the five games**, up sharply from a 500k-step checkpoint.

## What was built

- **Custom Gymnasium environment** (`envs/gridworld_env.py`, `GridWorld-v0`): the "build an RL environment" artifact. The same recipe learns it to the optimal path (return 0.93, 8-step solution).
- **The one recipe** (`train.py` + `config.yaml`): PPO with a tiny custom CNN feature extractor (`envs/minatar_cnn.py`) for MinAtar's 10x10xC observations, 8 parallel envs, 3 seeds. Reused unchanged across the custom env and all 5 benchmarks.

## Results (2M steps/game, mean of 3 seeds)

| Game | Random | Ours (500k) | Ours (2M) | Published DQN | % of DQN |
|---|---|---|---|---|---|
| Breakout | 0.5 | 13.0 | **21.07 +/- 2.58** | 27.1 | 78% |
| Freeway | 0.27 | 20.8 | **24.29 +/- 0.40** | 55.9 | 44% |
| Asterix | 0.37 | 2.4 | **10.31 +/- 2.79** | 13.6 | 76% |
| Seaquest | 0.1 | 1.3 | **6.04 +/- 1.83** | 38.0 | 16% |
| SpaceInvaders | 3.7 | 49.7 | **94.98 +/- 4.88** | 188.0 | 51% |

Every game improved from 500k to 2M; the two weakest games (Asterix, Seaquest) gained the most. Plots: `report/plots/A..E.png` (E = the 500k-vs-2M gain). Interactive replay wired to this data: `docs/replay.html`.

## Compute

CPU only (Apple M1, no CUDA). A 500k fast-demo first (~4.5 min/run), then a 2M push (~13-18 min/run, 15 runs) driven by a resumable auto-relaunch loop with a stall watchdog.

## Honest caveats

- **Still below DQN on the hardest games.** Seaquest (16%) and Freeway (44%) remain well under the published baselines even at 2M, though both improved. Breakout, Asterix, and SpaceInvaders are the strongest.
- **Two different measurements.** The table is a 30-episode greedy evaluation of the final model (stable). The animated curves in `replay.html` show the per-checkpoint `EvalCallback` series (5 to 20 episodes each, noisier at 2M, especially Asterix and SpaceInvaders).
- **MinAtar has no episode step limit.** A deterministic policy on Seaquest could hang evaluation; fixed with `gym.make(env_id, max_episode_steps=1000)`.

## Reproduce

```bash
cd phase1_cpu
./.venv/bin/python benchmark.py --budget fast --seeds 0 1 2        # 500k
./.venv/bin/python benchmark.py --timesteps 2000000 --seeds 0 1 2  # 2M
```
