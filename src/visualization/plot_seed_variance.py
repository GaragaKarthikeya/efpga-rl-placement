#!/usr/bin/env python3
"""
Figure 3 (results + reliability): per-seed ADP reduction, in-pool vs zero-shot.

This figure REPLACES the former per-benchmark result tables. With only three
seeds, mean +/- SD overstates what the data establish, so we show the raw
seed outcomes directly ("show your seeds"): three dots per benchmark, a line
spanning min-max, and a median tick. The eye reads reliability from the line
length: in-pool fitting is fragile on exactly two benchmarks (or1200,
boundtop, shaded), while every held-out zero-shot benchmark is tight.

In-pool values are recovered exactly from the three training logs via
reduction% = (1 - exp(-best_reward))*100 (reward = log(ADP0/ADP)); zero-shot
values are the deterministic per-seed evals (det_seed{7,42,123}_*.json).
Output: paper/fig_seed_variance.pdf (vector).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontManager
from statistics import median as _median
import palette as PAL

# grid (WxH), DSP count, BRAM count  — matches Table 1
BENCH_META = {
    # in-pool
    "fifo":          ( "6×6",    4, 0,  1),
    "ch_intrinsics": ("10×10",  68, 0,  1),
    "diffeq2":       ("14×14",  27, 5,  0),
    "boundtop":      ("13×13",  46, 0,  1),
    "diffeq1":       ("14×14",  37, 5,  0),
    "spree":         ("12×12",  60, 1,  3),
    "mkSMAdapter4B": ("18×18", 156, 0,  5),
    "or1200":        ("25×25", 245, 1,  2),
    "mmc_core":      ("13×13", 128, 0,  1),
    "mkPktMerge":    ("26×26",  29, 0, 15),
    "raygentop":     ("17×17", 121, 6,  1),
    # held-out
    "softmax":           ("41×41", 1227, 8,  0),
    "reduction_layer":   ("42×42",  748, 0, 32),
    "cipher":            ("12×12",   40, 0,  3),
    "macbuf":            ("12×12",   14, 3,  1),
    "mkDelayWorker32B":  ("48×48",  464, 0, 43),
    "arm_core":          ("35×35",  868, 0, 24),
}


RL_BLUE  = PAL.TEAL    # in-pool (trained)
ZS_BLUE  = PAL.TERRA   # held-out (zero-shot)
REF_GRAY = PAL.ZERO
HILITE   = "#ede0c6"   # volatile-benchmark shading (warm sand tint)

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

# name -> [seed7, seed42, seed123]
inpool = {
    "diffeq2": [68.34, 68.24, 69.08], "fifo": [60.20, 60.20, 56.29],
    "diffeq1": [48.45, 55.92, 53.01], "spree": [39.08, 38.89, 39.11],
    "mkPktMerge": [40.19, 37.78, 32.24], "ch_intrinsics": [36.25, 35.08, 36.25],
    "mkSMAdapter4B": [34.90, 35.51, 36.61], "boundtop": [33.84, 50.69, 21.33],
    "mmc_core": [24.42, 25.20, 25.44], "or1200": [5.75, 33.50, 12.48],
    "raygentop": [-2.14, -1.70, -2.20],
}
zeroshot = {
    "mkDelayWorker32B": [62.42, 54.88, 58.88], "macbuf": [47.76, 59.53, 59.09],
    "cipher": [30.22, 39.73, 30.51], "reduction_layer": [28.89, 21.51, 19.49],
    "arm_core": [6.99, 11.83, 8.52], "softmax": [5.65, 5.39, 10.77],
}
VOLATILE = set()  # highlighting removed

def ordered(d):  # by median descending
    return sorted(d.items(), key=lambda kv: _median(kv[1]), reverse=True)

fig, ax = plt.subplots(figsize=(5.5, 4.6))
gap = 2.0
y = 0
rows, ticks, labels = [], [], []
for name, vals in reversed(ordered(zeroshot)):
    rows.append((y, name, vals, ZS_BLUE)); ticks.append(y); labels.append(name); y += 1
zs_top = y - 1
y += gap
inpool_bot = y
for name, vals in reversed(ordered(inpool)):
    rows.append((y, name, vals, RL_BLUE)); ticks.append(y); labels.append(name); y += 1
inpool_top = y - 1

ax.axvline(0, color=REF_GRAY, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=1)

raygentop_y = None
for yy, name, vals, c in rows:
    lo, hi, med = min(vals), max(vals), _median(vals)
    ax.plot([lo, hi], [yy, yy], color=c, lw=0.8, alpha=0.45, zorder=2,
            solid_capstyle="round")
    ax.scatter(vals, [yy] * 3, s=11, color=c, alpha=0.85, zorder=3,
               edgecolor="white", linewidth=0.3)
    ax.plot([med, med], [yy - 0.26, yy + 0.26], color=c, lw=1.3, zorder=4,
            solid_capstyle="butt")
    if name == "raygentop":
        raygentop_y = yy

# vertical tick at x=0 for raygentop to show its data sits left of baseline
if raygentop_y is not None:
    ax.plot([0, 0], [raygentop_y - 0.32, raygentop_y + 0.32],
            color=RL_BLUE, lw=1.4, zorder=5, solid_capstyle="butt", alpha=0.7)

ax.set_yticks(ticks)
ax.set_yticklabels([""] * len(labels))  # blank — we draw our own columns

# --- inline table: benchmark | grid | DSP | BRAM as separate ax.text columns ---
# all x in axes-fraction via get_yaxis_transform() (0=left spine, 1=right spine)
# All columns in plain data coordinates — extend xlim left to make room
# x positions (data coords, negative = left of 0-baseline)
# x positions (data coords, negative = left of 0-baseline)
X_NAME = -82   # benchmark name, left-aligned
X_GRID = -47   # grid size, centered
X_BLOCKS = -23 # (CLB, DSP, BRAM) count, centered
COL_FS = 8.5

for yy, name, _vals, _c in rows:
    grid, clb, dsp, bram = BENCH_META[name]
    bold = name in VOLATILE
    fw = "bold" if bold else "normal"
    display = name.replace("mkDelayWorker32B", "mkDelayW32B").replace("mkSMAdapter4B", "mkSMAdapt4B")
    ax.text(X_NAME, yy, display,                 fontsize=COL_FS, va="center", ha="left",   fontweight=fw, clip_on=False)
    ax.text(X_GRID, yy, grid,                    fontsize=COL_FS, va="center", ha="center", fontweight=fw, clip_on=False)
    # Draw (CLB, DSP, BRAM) as perfectly aligned individual text elements
    # using the default serif font instead of monospace to keep consistency.
    X_L  = -37; X_CLB = -26; X_C1 = -25.5
    X_DSP  = -18; X_C2 = -17.5
    X_BRAM  = -10; X_R = -9.0
    
    ax.text(X_L,    yy, "(", fontsize=COL_FS, va="center", ha="left",  fontweight=fw, clip_on=False)
    ax.text(X_CLB,  yy, str(clb), fontsize=COL_FS, va="center", ha="right", fontweight=fw, clip_on=False)
    ax.text(X_C1,   yy, ",", fontsize=COL_FS, va="center", ha="left",  fontweight=fw, clip_on=False)
    ax.text(X_DSP,  yy, str(dsp), fontsize=COL_FS, va="center", ha="right", fontweight=fw, clip_on=False)
    ax.text(X_C2,   yy, ",", fontsize=COL_FS, va="center", ha="left",  fontweight=fw, clip_on=False)
    ax.text(X_BRAM, yy, str(bram), fontsize=COL_FS, va="center", ha="right", fontweight=fw, clip_on=False)
    ax.text(X_R,    yy, ")", fontsize=COL_FS, va="center", ha="left",  fontweight=fw, clip_on=False)

xr = 87
ax.text(xr, (inpool_bot + inpool_top) / 2, "In-pool\n(trained)",
        fontsize=8, color=RL_BLUE, weight="bold", rotation=270, va="center", ha="center")
ax.text(xr, zs_top / 2, "Held-out\n(zero-shot)",
        fontsize=8, color=ZS_BLUE, weight="bold", rotation=270, va="center", ha="center")

# legend-in-place: what the glyphs mean (one row, top-left empty space)
ax.text(40, inpool_top + 0.85, "dots: 3 seeds (7/42/123)   |: median   line: min–max",
        fontsize=6.6, color=PAL.SUBINK, va="bottom", ha="center")

# column headers aligned to the same x positions as the data columns
HDR_Y = inpool_top + 0.85
HDR_KW = dict(fontsize=8.0, color=PAL.SUBINK, fontweight="bold", va="bottom", clip_on=False)
ax.text(X_NAME, HDR_Y, "Benchmark", ha="left",   **HDR_KW)
ax.text(X_GRID, HDR_Y, "Grid",      ha="center", **HDR_KW)
ax.text(X_BLOCKS, HDR_Y, "(CLB, DSP, BRAM)", ha="center", **HDR_KW)

# line under column headers — spans full table width (X_NAME to right edge)
LINE_Y = inpool_top + 0.55
ax.plot([X_NAME, 87], [LINE_Y, LINE_Y], color=PAL.SUBINK, lw=0.5, alpha=0.5,
        clip_on=False, transform=ax.transData)

# line between in-pool and held-out sections
sep_y = (inpool_bot + zs_top) / 2
ax.axhline(sep_y, color=PAL.SUBINK, lw=0.5, alpha=0.4, ls=(0, (4, 3)), clip_on=False)


ax.set_xlim(-85, 94)
ax.set_ylim(-0.8, inpool_top + 1.8)
ax.set_xlabel("ADP reduction vs. baseline (%), per seed", fontsize=8.3)
ax.set_xticks([0, 20, 40, 60, 80])
ax.tick_params(axis="x", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.09)
out = "paper/fig_seed_variance.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
