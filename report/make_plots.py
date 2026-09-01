"""Static, human-eyeball plots of score IMPROVEMENT across the 5 MinAtar benchmarks.

Reads the stable 500k backup (phase1_cpu/results_500k) so it never touches an
in-progress run. Writes PNGs to report/plots/:
  A_pct_of_baseline_all5.png  all 5 curves as % of DQN baseline (one view)
  B_per_game_curves.png       per-game score vs random vs DQN (small multiples)
  C_final_bars.png            final score: random vs ours vs DQN (log scale)
  D_improvement_over_random.png  how many times above random, per game
"""
import csv
import glob
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "phase1_cpu")
RESULTS = os.path.join(P1, "results")            # current run (now 2M)
RESULTS_500K = os.path.join(P1, "results_500k")  # frozen 500k, for the 500k-vs-2M comparison
OUT = os.path.join(ROOT, "report", "plots")
os.makedirs(OUT, exist_ok=True)

cfg = yaml.safe_load(open(os.path.join(P1, "config.yaml")))
GAMES = cfg["games"]
COLORS = ["#4f9dff", "#ffb043", "#ff5d8f", "#3ddc97", "#c792ea"]


def short(env_id):
    return env_id.split("/")[-1].replace("-v1", "")


def smooth(y, w=5):
    # windowed mean with a shrinking window at the edges (no zero-padding),
    # so the endpoints are not artificially pulled toward zero.
    if len(y) < 3:
        return np.asarray(y, dtype=float)
    half = min(w, len(y)) // 2
    out = np.empty(len(y), dtype=float)
    for i in range(len(y)):
        out[i] = np.mean(y[max(0, i - half):min(len(y), i + half + 1)])
    return out


def load(env_id):
    slug = env_id.replace("/", "_")
    series = []
    for d in sorted(glob.glob(os.path.join(RESULTS, slug, "seed*"))):
        npz = os.path.join(d, "evaluations.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            series.append((z["timesteps"], z["results"].mean(axis=1)))
    if not series:
        return None
    minlen = min(len(t) for t, _ in series)
    ts = series[0][0][:minlen]
    mat = np.stack([r[:minlen] for _, r in series])
    return ts, mat.mean(0), mat.std(0)


# final stable scores from results.csv (30-episode eval of the final model)
finals, randoms = {}, {}
with open(os.path.join(RESULTS, "results.csv")) as f:
    for row in csv.DictReader(f):
        finals[short(row["env_id"])] = float(row["our_mean"])
        randoms[short(row["env_id"])] = float(row["random"])

# ---- Figure A: % of baseline, all 5 in one view ----
plt.figure(figsize=(9, 5))
for i, g in enumerate(GAMES):
    r = load(g["id"])
    if r is None:
        continue
    ts, mean, std = r
    base = g["baseline"]
    pct, ps = smooth(100 * mean / base), 100 * std / base
    plt.plot(ts, pct, color=COLORS[i], lw=2, label=short(g["id"]))
    plt.fill_between(ts, pct - ps, pct + ps, color=COLORS[i], alpha=0.13)
plt.axhline(100, color="#d9a441", ls="--", lw=1.4, label="published DQN (100%)")
plt.xlabel("training steps")
plt.ylabel("score, % of published DQN baseline")
plt.title("PPO learning curves across 5 MinAtar benchmarks (2M, mean of 3 seeds)")
plt.legend(fontsize=9, ncol=2)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "A_pct_of_baseline_all5.png"), dpi=130)
plt.close()

# ---- Figure B: per-game raw curves (small multiples) ----
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
axs = axs.flatten()
for i, g in enumerate(GAMES):
    ax = axs[i]
    r = load(g["id"])
    if r is None:
        continue
    ts, mean, std = r
    sm = smooth(mean)
    ax.plot(ts, sm, color=COLORS[i], lw=2)
    ax.fill_between(ts, sm - std, sm + std, color=COLORS[i], alpha=0.13)
    ax.axhline(g["baseline"], color="#d9a441", ls="--", lw=1, label=f"DQN {g['baseline']}")
    ax.axhline(g["random"], color="#888", ls=":", lw=1, label=f"random {g['random']}")
    ax.set_title(short(g["id"]))
    ax.set_xlabel("steps")
    ax.set_ylabel("episode return")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
axs[5].axis("off")
fig.suptitle("Per-game learning curves: PPO vs random vs published DQN (2M)", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "B_per_game_curves.png"), dpi=130)
plt.close()

# ---- Figure C: final grouped bars (log scale) ----
names = [short(g["id"]) for g in GAMES]
x = np.arange(len(names))
w = 0.27
plt.figure(figsize=(10, 5))
plt.bar(x - w, [max(randoms[n], 0.05) for n in names], w, label="random", color="#888")
plt.bar(x, [finals[n] for n in names], w, label="PPO (ours, 500k)", color="#4f9dff")
plt.bar(x + w, [g["baseline"] for g in GAMES], w, label="published DQN", color="#d9a441")
plt.yscale("log")
plt.xticks(x, names)
plt.ylabel("episode return (log scale)")
plt.title("Final score: PPO vs random vs published DQN (2M, mean of 3 seeds)")
plt.legend()
plt.grid(alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "C_final_bars.png"), dpi=130)
plt.close()

# ---- Figure D: improvement over random (x times) ----
mult = [finals[n] / max(randoms[n], 1e-6) for n in names]
plt.figure(figsize=(9, 4.5))
bars = plt.bar(names, mult, color=COLORS)
for b, m in zip(bars, mult):
    plt.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}x", ha="center", va="bottom", fontsize=10)
plt.axhline(1, color="#888", ls=":", lw=1, label="random (1x)")
plt.ylabel("score / random score")
plt.title("Improvement over random policy, per benchmark (2M)")
plt.legend()
plt.grid(alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "D_improvement_over_random.png"), dpi=130)
plt.close()

# ---- Figure E: 500k vs 2M (score gain from more training) ----
def read_finals(csv_path):
    fin = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            fin[short(row["env_id"])] = float(row["our_mean"])
    return fin
try:
    f500 = read_finals(os.path.join(RESULTS_500K, "results.csv"))
    f2m = read_finals(os.path.join(RESULTS, "results.csv"))
    xx = np.arange(len(names))
    ww = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(xx - ww / 2, [f500[n] for n in names], ww, label="500k", color="#5b6675")
    plt.bar(xx + ww / 2, [f2m[n] for n in names], ww, label="2M", color="#4f9dff")
    for i, g in enumerate(GAMES):
        plt.plot([i - ww, i + ww], [g["baseline"], g["baseline"]], color="#d9a441", ls="--", lw=1.2)
    plt.yscale("log")
    plt.xticks(xx, names)
    plt.ylabel("episode return (log scale)")
    plt.title("500k vs 2M: score gain from more training (dashed = published DQN)")
    plt.legend()
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "E_500k_vs_2M.png"), dpi=130)
    plt.close()
except Exception as e:
    print("Figure E skipped:", e)

print("wrote 5 plots to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
