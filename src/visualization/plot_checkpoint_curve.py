#!/usr/bin/env python3
"""One-off plot: zero-shot ADP reduction vs. training checkpoint, for
custom_macbuf and lightweight_cipher, across seeds 7/42/123."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE = Path("/home/digital-2/rl_gnn/vanilla_rl")
STEPS = [5760, 11520, 17280, 23040, 28800, 34560, 40320, 46080, 51840, 57600, 63360, 69120]
BENCHES = ["custom_macbuf", "lightweight_cipher"]
SEEDS = ["seed7", "seed123", "seed42"]

rows = []
for bench in BENCHES:
    for seedname in SEEDS:
        for step in STEPS:
            f = BASE / f"curve_results/curve_{seedname}_{bench}_{step}.json"
            d = json.loads(f.read_text())[bench]
            rows.append({
                "step": step,
                "benchmark": bench,
                "seed": seedname,
                "adp_reduction_pct": d["adp_reduction_pct"],
            })

df = pd.DataFrame(rows)

# Average across the 3 seeds per (benchmark, step), then smooth with a
# centered rolling mean (window=3) over the checkpoint sequence.
avg = (
    df.groupby(["benchmark", "step"])["adp_reduction_pct"]
    .mean()
    .reset_index()
    .sort_values(["benchmark", "step"])
)
avg["smoothed"] = (
    avg.groupby("benchmark")["adp_reduction_pct"]
    .transform(lambda s: s.rolling(window=3, center=True, min_periods=1).mean())
)

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 7))

sns.lineplot(
    data=avg, x="step", y="adp_reduction_pct",
    hue="benchmark", linewidth=1.0, alpha=0.35, legend=False,
    marker="o", markersize=5,
    ax=ax,
)
sns.lineplot(
    data=avg, x="step", y="smoothed",
    hue="benchmark", linewidth=3.0,
    marker="o", markersize=8,
    ax=ax,
)

ax.axhline(0, color="gray", linewidth=1, linestyle=":")
ax.set_xlabel("Training checkpoint (total timesteps)")
ax.set_ylabel("Zero-shot ADP reduction (%)")
ax.set_title("Zero-Shot Generalization Over Training (averaged across seeds 7/42/123)\nfaint = raw 3-seed average, bold = smoothed (rolling mean, window=3)")
ax.legend(loc="lower right", fontsize=11, title=None)

plt.tight_layout()
out = BASE / "visualizations" / "zeroshot_checkpoint_curve.png"
plt.savefig(out, dpi=200)
print(f"Saved -> {out}")
