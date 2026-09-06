"""Static, honest Phase-2 result plot: base (pre) vs GRPO (post) per benchmark, with deltas.

A 2-point pre/post result does not animate meaningfully, so we render a static grouped
bar chart instead. Reads docs/metrics_phase2.json, writes docs/P2_pre_post.png (dark theme,
matching replay.html). Run: python report/make_phase2_plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

rows = json.load(open(os.path.join(ROOT, "docs", "metrics_phase2.json")))
if not rows:
    raise SystemExit("metrics_phase2.json is empty (run the eval + export first)")

bench = {}
for r in rows:
    bench.setdefault(r["benchmark"], {})[r["step"]] = r["value"]
order = [b for b in ["GSM8K", "MMLU", "ARC-Challenge", "HellaSwag", "TruthfulQA"] if b in bench]
pre = [bench[b][min(bench[b])] for b in order]
post = [bench[b][max(bench[b])] for b in order]
delta = [po - pr for pr, po in zip(pre, post)]

plt.rcParams.update({
    "figure.facecolor": "#0e1522", "axes.facecolor": "#0e1522", "savefig.facecolor": "#0e1522",
    "text.color": "#e7eefb", "axes.labelcolor": "#e7eefb", "xtick.color": "#8595ad",
    "ytick.color": "#8595ad", "axes.edgecolor": "#1e293b", "font.size": 11,
})
x = np.arange(len(order))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.9))
ax.bar(x - w / 2, pre, w, label="base (pre-RL)", color="#5b6b84")
ax.bar(x + w / 2, post, w, label="GRPO (post)", color="#3b82f6")
for i, dv in enumerate(delta):
    c = "#10b981" if dv > 0.3 else ("#ef4444" if dv < -0.3 else "#8595ad")
    ax.annotate(("+" if dv >= 0 else "") + f"{dv:.1f}", (x[i], max(pre[i], post[i]) + 1.2),
                ha="center", color=c, fontweight="bold", fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=9.5)
ax.set_ylabel("score (%)")
ax.set_ylim(0, max(max(pre), max(post)) + 9)
ax.set_title("Phase 2 (v1): base vs GRPO on 5 public benchmarks", fontweight="bold", pad=26)
ax.text(0.5, 1.045,
        "No material gain. GSM8K regressed 3.1 pts (reward vs eval-format mismatch); the rest are within noise. See plans/v2.md.",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#f6d58a")
ax.legend(frameon=False, loc="upper right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#182338", lw=0.7)
ax.set_axisbelow(True)
out = os.path.join(ROOT, "docs", "P2_pre_post.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out, "| deltas:", dict(zip(order, [round(d, 1) for d in delta])))
