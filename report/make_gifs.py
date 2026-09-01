"""Render the two main replay animations as looping GIFs (~10s each).

  report/gifs/learning_curves.gif  the 5 curves growing (% of DQN) over 2M steps
  report/gifs/benchmark_race.gif   a bar-chart race of % of DQN

Reads the current results (2M). PillowWriter saves with loop=0 (infinite loop).
"""
import glob
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "phase1_cpu")
RESULTS = os.path.join(P1, "results")
OUT = os.path.join(ROOT, "report", "gifs")
os.makedirs(OUT, exist_ok=True)

cfg = yaml.safe_load(open(os.path.join(P1, "config.yaml")))
GAMES = cfg["games"]
COLORS = ["#4f9dff", "#ffb043", "#ff5d8f", "#3ddc97", "#c792ea"]
FRAMES, FPS = 100, 10          # 100 frames at 10 fps = 10s per loop
plt.rcParams.update({"figure.facecolor": "#0e1116", "axes.facecolor": "#0e1116",
                     "text.color": "#e6edf3", "axes.labelcolor": "#8b98a5",
                     "xtick.color": "#8b98a5", "ytick.color": "#8b98a5",
                     "axes.edgecolor": "#243040"})


def short(env_id):
    return env_id.split("/")[-1].replace("-v1", "")


def smooth(y, w=9):
    # windowed mean (shrinking window at edges) so the noisy 2M eval series reads
    # as a clean, monotonic-ish animation that clearly completes.
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    h = min(w, len(y)) // 2
    return np.array([y[max(0, i - h):min(len(y), i + h + 1)].mean() for i in range(len(y))])


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
    m = min(len(t) for t, _ in series)
    ts = series[0][0][:m]
    mat = np.stack([r[:m] for _, r in series])
    return ts, mat.mean(0), mat.std(0)


# precompute per-game % of baseline series
DATA = []
for i, g in enumerate(GAMES):
    r = load(g["id"])
    if r is None:
        continue
    ts, mean, std = r
    base = g["baseline"]
    DATA.append(dict(name=short(g["id"]), color=COLORS[i], ts=ts,
                     pct=smooth(100 * mean / base), ps=smooth(100 * std / base), base=base))
N = len(DATA[0]["ts"])
XMAX = float(DATA[0]["ts"][-1])
YMAX = min(210, max(d["pct"].max() + d["ps"].max() for d in DATA) * 1.05)


def k_at(frame):
    return max(2, int(round((frame + 1) / FRAMES * N)))


# ---------- GIF 1: learning curves growing ----------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=85)


def draw_curves(frame):
    ax.clear()
    k = k_at(frame)
    for d in DATA:
        ax.plot(d["ts"][:k], d["pct"][:k], color=d["color"], lw=2, label=d["name"])
        ax.fill_between(d["ts"][:k], d["pct"][:k] - d["ps"][:k], d["pct"][:k] + d["ps"][:k],
                        color=d["color"], alpha=0.13)
    ax.axhline(100, color="#d9a441", ls="--", lw=1.3)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xlabel("training steps")
    ax.set_ylabel("score, % of published DQN")
    ax.set_title("PPO learning curves across 5 MinAtar benchmarks (2M)")
    ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.2)
    ax.grid(alpha=0.15)


anim = FuncAnimation(fig, draw_curves, frames=FRAMES, interval=1000 // FPS)
anim.save(os.path.join(OUT, "learning_curves.gif"), writer=PillowWriter(fps=FPS))
plt.close(fig)
print("wrote learning_curves.gif")

# ---------- GIF 2: benchmark race ----------
fig2, ax2 = plt.subplots(figsize=(8, 4.5), dpi=85)
RMAX = max(120, YMAX)


def draw_race(frame):
    ax2.clear()
    k = k_at(frame)
    rows = sorted(DATA, key=lambda d: d["pct"][k - 1], reverse=True)
    y = np.arange(len(rows))[::-1]
    for yi, d in zip(y, rows):
        val = d["pct"][k - 1]
        ax2.barh(yi, val, color=d["color"], height=0.7)
        ax2.text(val + 2, yi, d["name"] + f"  {val:.0f}%", va="center", fontsize=10, color="#e6edf3")
    ax2.axvline(100, color="#d9a441", ls="--", lw=1.2)
    ax2.set_xlim(0, RMAX)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_yticks([])
    ax2.set_xlabel("score, % of published DQN")
    step = int(DATA[0]["ts"][k - 1])
    ax2.set_title(f"Benchmark race  (step {step:,} / {int(XMAX):,})")
    ax2.grid(alpha=0.15, axis="x")


anim2 = FuncAnimation(fig2, draw_race, frames=FRAMES, interval=1000 // FPS)
anim2.save(os.path.join(OUT, "benchmark_race.gif"), writer=PillowWriter(fps=FPS))
plt.close(fig2)
print("wrote benchmark_race.gif")

for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f)) // 1024
    print(f"  {f}  ({sz} KB)")
