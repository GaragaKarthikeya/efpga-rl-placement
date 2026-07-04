#!/usr/bin/env python3
"""
Figure 1 (hero): amortization Pareto for custom_macbuf.

One trained policy reaches a strong layout for a previously-unseen netlist at
near-zero search cost, where the GA pays the full search cost from scratch.
X = VTR oracle calls (log scale), Y = ADP reduction vs traditional baseline.

All three points are measured (baseline ADP = 44,503 on custom_macbuf):
  RL zero-shot   : 1 oracle call,    59.53%  (det_seed42_custom_macbuf.json)
  GA best        : ~935 oracle calls, 68.17% (gen 186, pop 5)
  RL fine-tuned  : 1,575 oracle calls, 74.06% (paper Table 3)

The GA is drawn as a single measured endpoint, not a trajectory: the base GA
cache does not retain per-generation history, so a curve would be schematic.
Output: paper/fig_amortization.pdf (vector).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

# ---- locked visual language (shared across all figures) ------------------
RL_BLUE   = "#1f5fa8"   # RL  = blue
GA_ORANGE = "#e07b2a"   # GA  = orange
REF_GRAY  = "#9a9a9a"   # baselines / reference

# serif to sit inside the acmart body text; fall back gracefully
_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,   # embed real glyphs, not Type-3 (camera-ready safe)
})

# ---- measured data -------------------------------------------------------
zs_calls,  zs_adp  = 1,    59.53      # RL zero-shot
ga_calls,  ga_adp  = 935,  68.17      # GA best
ft_calls,  ft_adp  = 1575, 74.06      # RL fine-tuned

fig, ax = plt.subplots(figsize=(3.35, 2.45))   # single acmart column

# GA reference level: lets both RL points be read against it (adjacency)
ax.axhline(ga_adp, color=GA_ORANGE, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=1)

# amortization path: same policy, zero-shot -> fine-tuned (quiet connector)
ax.annotate("", xy=(ft_calls, ft_adp), xytext=(zs_calls, zs_adp),
            arrowprops=dict(arrowstyle="-|>", color=RL_BLUE, lw=0.9,
                            alpha=0.45, shrinkA=7, shrinkB=7,
                            connectionstyle="arc3,rad=-0.18"), zorder=2)

# GA endpoint — present but quiet
ax.scatter([ga_calls], [ga_adp], s=70, marker="D", facecolor=GA_ORANGE,
           edgecolor="white", linewidth=0.8, zorder=4)

# RL fine-tuned — secondary RL point
ax.scatter([ft_calls], [ft_adp], s=80, marker="o", facecolor="white",
           edgecolor=RL_BLUE, linewidth=1.6, zorder=4)

# RL zero-shot — THE focal point: saturated, largest, against the left wall
ax.scatter([zs_calls], [zs_adp], s=150, marker="o", facecolor=RL_BLUE,
           edgecolor="white", linewidth=1.1, zorder=5)

# ---- direct labels (no legend) -------------------------------------------
ax.annotate("RL zero-shot\n1 call · 59.5%",
            xy=(zs_calls, zs_adp), xytext=(2.1, 55.0),
            fontsize=8.0, color=RL_BLUE, weight="bold", va="top",
            arrowprops=dict(arrowstyle="-", color=RL_BLUE, lw=0.7, alpha=0.6))
ax.annotate("RL fine-tuned\n1,575 calls · 74.1%",
            xy=(ft_calls, ft_adp), xytext=(150, 76.3),
            fontsize=8.0, color=RL_BLUE, va="bottom")
ax.annotate("GA best\n~935 calls · 68.2%",
            xy=(ga_calls, ga_adp), xytext=(70, 63.0),
            fontsize=8.0, color=GA_ORANGE, va="top")

# ---- axes / framing ------------------------------------------------------
ax.set_xscale("log")
ax.set_xlim(0.6, 4000)
ax.set_ylim(52, 79)
ax.set_xlabel("VTR oracle calls (log scale)", fontsize=9)
ax.set_ylabel("ADP reduction vs.\nbaseline (%)", fontsize=9)
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1000"], fontsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.3)
out = "paper/fig_amortization.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
