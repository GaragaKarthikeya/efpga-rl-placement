#!/usr/bin/env python3
"""
Figure 1 (hero): zero-shot scale extrapolation.

The policy is trained only on small fabrics (<=676 tiles; shaded band). With a
single zero-shot rollout per benchmark it then produces ADP reductions on the
six held-out benchmarks, four of which are fabrics up to 2,304 tiles -- ~3.4x
the area and ~5x the logic of anything it trained on. The punchline is visual:
the four large held-out benchmarks sit to the RIGHT of the training band (out
of distribution in size) and ABOVE zero; the two small ones (grey) are in-size
controls.

X = fabric size in grid tiles (log scale). Y = zero-shot ADP reduction vs the
traditional baseline; dots are the three seeds (7/42/123), the short tick is
the median, the thin line spans min-max.
Output: paper/fig_scale.pdf (vector).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from statistics import median as _median
import palette as PAL

RL_BLUE  = PAL.TERRA    # hero accent — large held-out benchmarks
CTRL_GRY = PAL.GRAY     # in-size controls
BAND     = PAL.CREAM    # training-size band
BAND_EDGE= "#d8cdb6"
REF_GRAY = PAL.ZERO

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42,
})

POOL_MIN, POOL_MAX = 36, 676   # fifo 6x6 .. mkPktMerge 26x26

# held-out: name -> (tiles, [seed7, seed42, seed123] zero-shot ADP red %, in_band)
heldout = {
    "lightweight_cipher": (144,  [30.22, 39.73, 30.51], True),   # 12x12
    "custom_macbuf":      (144,  [47.76, 59.53, 59.09], True),   # 12x12 (synthetic)
    "arm_core":           (1225, [6.99, 11.83, 8.52],  False),   # 35x35
    "softmax":            (1681, [5.65, 5.39, 10.77],  False),   # 41x41
    "reduction_layer":    (1764, [28.89, 21.51, 19.49],False),   # 42x42
    "mkDelayWorker32B":   (2304, [62.42, 54.88, 58.88],False),   # 48x48
}

fig, ax = plt.subplots(figsize=(3.4, 2.5))

ax.axvspan(POOL_MIN, POOL_MAX, color=BAND, zorder=0)
ax.axvline(POOL_MAX, color=BAND_EDGE, lw=0.7, zorder=0)
ax.axhline(0, color=REF_GRAY, lw=0.7, ls=(0, (3, 3)), zorder=1)
ax.text(150, 69, "trained here", fontsize=7, color=PAL.SUBINK,
        ha="center", va="center", style="italic")
ax.text(150, 64.5, r"($\leq$676 tiles)", fontsize=6.3, color=PAL.GRAY,
        ha="center", va="center", style="italic")

for name, (tiles, vals, inband) in heldout.items():
    lo, hi, med = min(vals), max(vals), _median(vals)
    c = CTRL_GRY if inband else RL_BLUE
    # min-max whisker (thin)
    ax.plot([tiles, tiles], [lo, hi], color=c, lw=0.8, alpha=0.45, zorder=2,
            solid_capstyle="round")
    # three seed dots (small)
    ax.scatter([tiles] * 3, vals, s=11, color=c, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.3)
    # median tick (short horizontal bar, log-aware half-width)
    hw = tiles * 0.07
    ax.plot([tiles - hw, tiles + hw], [med, med], color=c, lw=1.3, zorder=4,
            solid_capstyle="butt")

# direct labels: (x, y, text, color, ha, va)
lab = [
    (132, 59, "custom_macbuf",      CTRL_GRY, "right", "center"),
    (132, 33, "lightweight",        CTRL_GRY, "right", "center"),
    (132, 29, "cipher",             CTRL_GRY, "right", "center"),
    (1225, 16.5, "arm_core",        RL_BLUE,  "center","bottom"),
    (1880, 6.5, "softmax",          RL_BLUE,  "left",  "center"),
    (1764, 32, "reduction_layer",   RL_BLUE,  "center","bottom"),
    (2304, 67, "mkDelayWorker32B",  RL_BLUE,  "center","bottom"),
]
for x, y, t, c, ha, va in lab:
    ax.text(x, y, t, fontsize=6.6, color=c, ha=ha, va=va)

ax.set_xscale("log")
ax.set_xlim(26, 3600)
ax.set_ylim(-4, 73)
ax.set_xlabel("fabric size (grid tiles, log scale)", fontsize=8.5)
ax.set_ylabel("zero-shot ADP reduction (%)", fontsize=8.5)
ax.set_xticks([100, 1000])
ax.set_xticklabels(["100", "1000"], fontsize=7.5)
ax.set_yticks([0, 20, 40, 60])
ax.tick_params(axis="y", labelsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.3)
out = "paper/fig_scale.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
