"""Run the one recipe over all 5 benchmarks x seeds, then emit the results table + curves.

This is the Phase 1 result: results/results.csv, results/summary.md, and a per-game
learning-curve PNG. Auto-selects the MinAtar suite, falling back to classic control.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from envs import load_config, slug, minatar_available
from train import train
from evaluate import run_episodes

COLORS = ["#4f9dff", "#ffb043", "#ff5d8f", "#3ddc97", "#c792ea"]


def load_eval_series(seed_dirs):
    series = []
    for d in seed_dirs:
        npz = os.path.join(d, "evaluations.npz")
        if not os.path.exists(npz):
            continue
        data = np.load(npz)
        series.append((data["timesteps"], data["results"].mean(axis=1)))
    return series


def plot_curve(env_id, seed_dirs, baseline, out_png, color):
    series = load_eval_series(seed_dirs)
    if not series:
        return
    minlen = min(len(ts) for ts, _ in series)
    ts = series[0][0][:minlen]
    mat = np.stack([res[:minlen] for _, res in series], axis=0)
    mean, std = mat.mean(0), mat.std(0)
    plt.figure(figsize=(6, 3.4))
    plt.plot(ts, mean, color=color, lw=2, label="PPO (mean over seeds)")
    plt.fill_between(ts, mean - std, mean + std, color=color, alpha=0.2)
    plt.axhline(baseline, color="#d9a441", ls="--", lw=1.3, label=f"published baseline {baseline}")
    plt.title(env_id)
    plt.xlabel("timesteps")
    plt.ylabel("episode return")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", choices=["smoke", "fast", "full"], default="fast")
    p.add_argument("--timesteps", type=int, default=None, help="override the budget's step count")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--suite", choices=["auto", "minatar", "fallback"], default="auto")
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    cfg = load_config()
    seeds = args.seeds if args.seeds is not None else cfg["run"]["seeds"]
    ts = args.timesteps or {"smoke": 50000, "fast": cfg["run"]["timesteps_fast"],
                            "full": cfg["run"]["timesteps_full"]}[args.budget]

    if args.suite == "minatar" or (args.suite == "auto" and minatar_available()):
        games, suite = cfg["games"], "MinAtar"
    else:
        games, suite = cfg["fallback_games"], "classic-control"
    print(f"[benchmark] suite={suite} budget={args.budget} ({ts} steps) seeds={seeds} games={len(games)}")

    rows = []
    for gi, g in enumerate(games):
        env_id, baseline, color = g["id"], g["baseline"], COLORS[gi % len(COLORS)]
        seed_dirs, our_means = [], []
        for seed in seeds:
            outdir = os.path.join("results", slug(env_id), f"seed{seed}")
            seed_dirs.append(outdir)
            try:
                model_path = os.path.join(outdir, "final_model.zip")
                if os.path.exists(model_path):
                    print(f"  {env_id} seed{seed}: resume (final_model exists, skip train)")
                else:
                    model_path = train(env_id, ts, seed, outdir, cfg)
                om, _ = run_episodes(PPO.load(model_path, device="cpu"), env_id, args.episodes, 123 + seed)
                our_means.append(om)
                print(f"  {env_id} seed{seed}: ours {om:.2f}")
            except Exception as e:
                print(f"  {env_id} seed{seed} FAILED: {type(e).__name__}: {str(e)[:160]}")
        if not our_means:
            continue
        rand_m, _ = run_episodes(None, env_id, args.episodes, 999)
        our_mean, our_std = float(np.mean(our_means)), float(np.std(our_means))
        pct = 100.0 * our_mean / baseline if baseline else float("nan")
        plot_curve(env_id, seed_dirs, baseline, os.path.join("results", slug(env_id), "curve.png"), color)
        rows.append(dict(env_id=env_id, our_mean=round(our_mean, 2), our_std=round(our_std, 2),
                         random=round(rand_m, 2), baseline=baseline, pct_of_baseline=round(pct, 1)))

    os.makedirs("results", exist_ok=True)
    with open("results/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env_id", "our_mean", "our_std", "random", "baseline", "pct_of_baseline"])
        w.writeheader()
        w.writerows(rows)
    with open("results/summary.md", "w") as f:
        f.write(f"# Phase 1 results ({suite}, {args.budget} budget, seeds {seeds})\n\n")
        f.write("| Game | Random | Ours (mean +/- std) | Published baseline | % of baseline |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['env_id']} | {r['random']} | {r['our_mean']} +/- {r['our_std']} | "
                    f"{r['baseline']} | {r['pct_of_baseline']}% |\n")
    print("[benchmark] wrote results/results.csv and results/summary.md")


if __name__ == "__main__":
    main()
