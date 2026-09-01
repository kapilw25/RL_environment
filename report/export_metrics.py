"""Normalize Phase 1 SB3 logs into the tidy metrics.json that feeds docs/replay.html.

Schema (one row per benchmark x step x seed):
  {phase, run, benchmark, step, value, seed, baseline, ghost}

Reads phase1_cpu/results/<slug>/seed<k>/evaluations.npz (written by EvalCallback).
ghost is set to the published baseline as a flat placeholder; swap in real
Open RL Benchmark curves later if desired.
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P1 = os.path.join(ROOT, "phase1_cpu")


def short_name(env_id):
    return env_id.split("/")[-1].replace("-v1", "").replace("-v0", "")


def load_config():
    with open(os.path.join(P1, "config.yaml")) as f:
        return yaml.safe_load(f)


def build_rows(cfg):
    games = cfg["games"] + cfg["fallback_games"]
    collected = []  # (name, baseline, seed, timesteps, values)
    for g in games:
        env_id, baseline = g["id"], float(g["baseline"])
        name = short_name(env_id)
        game_slug = env_id.replace("/", "_")
        for seed_dir in sorted(glob.glob(os.path.join(P1, "results", game_slug, "seed*"))):
            try:
                seed = int(os.path.basename(seed_dir).replace("seed", ""))
            except ValueError:
                continue
            npz = os.path.join(seed_dir, "evaluations.npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            collected.append((name, baseline, seed, d["timesteps"], d["results"].mean(axis=1)))
    if not collected:
        return []
    # Align every series to the shortest length so the D3 aggregation (which reuses
    # one step grid across benchmarks) never indexes past a shorter series.
    min_len = min(len(ts) for _, _, _, ts, _ in collected)
    rows = []
    for name, baseline, seed, ts, vals in collected:
        for step, val in zip(ts[:min_len], vals[:min_len]):
            rows.append(dict(phase=1, run="ppo", benchmark=name, step=int(step),
                             value=round(float(val), 3), seed=seed,
                             baseline=baseline, ghost=baseline))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "metrics.json"))
    args = ap.parse_args()
    cfg = load_config()
    rows = build_rows(cfg)
    with open(args.out, "w") as f:
        json.dump(rows, f)
    benches = sorted(set(r["benchmark"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    print(f"[export] {len(rows)} rows | benchmarks {benches} | seeds {seeds} -> {args.out}")


if __name__ == "__main__":
    main()
