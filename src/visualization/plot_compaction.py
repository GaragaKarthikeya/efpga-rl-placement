#!/usr/bin/env python3
"""
Layout figure (the money shot): heterogeneous FPGA fabric, true-scale.

Two real softmax fabrics at the SAME tile scale, bottom-left aligned. Tiles are
inset so the warm background shows through as routing channels -- the FPGA look.
The fabric is heterogeneous: a CLB sea, DSP columns, BRAM columns, and an I/O
ring. The story is visible without reading numbers:

  * the traditional fabric provisions BRAM columns (plum) the netlist never uses;
  * the RL custom fabric strips those columns out and shrinks the array, so it
    is physically smaller -- the dashed ghost is the old footprint, the hatched
    L is the reclaimed area.

Stated once: -22% fabric area (ADP in the caption). Palette is a deliberate
sage / terracotta / plum / sand scheme on warm paper -- not the stock defaults.
Output: paper/fig_layout.pdf (vector). Data: softmax, seed 42.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.font_manager import FontManager

from plot_layout import parse_place, parse_arch, C, _block_height
import palette as PAL

# ---- shared palette (see palette.py) --------------------------------------
CH_BG   = PAL.CHANNEL
INK     = PAL.INK
SUBINK  = PAL.SUBINK
RECLAIM = PAL.RECLAIM
GHOST   = "#8c8475"
P = {
    "void":       CH_BG,
    "io_empty":   PAL.IO_PALE,
    "clb_empty":  PAL.CLB_PALE,    # unused logic site
    "dsp_empty":  PAL.TERRA_PALE,  # DSP column, unoccupied
    "bram_empty": PAL.TEAL_PALE,   # BRAM column, unoccupied
    "clb_used":   PAL.SAGE,        # CLB in use
    "dsp_used":   PAL.TERRA,       # DSP block
    "bram_used":  PAL.TEAL,        # BRAM block
    "io_used":    PAL.SAND,        # I/O pad
    "clb_edge":   PAL.SAGE_E, "dsp_edge": PAL.TERRA_E,
    "bram_edge":  PAL.TEAL_E, "io_edge":  PAL.SAND_E,
    "grid_line":  CH_BG,
}
C.update(P)

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "pdf.fonttype": 42})

TRAD = ("runs/softmax_traditional/softmax.place",
        "runs/softmax_traditional/k6_frac_N10_mem32K_40nm.xml")
RL   = ("runs/best_run_softmaxsoftmax_finetuned_seed42/softmax.place",
        "runs/best_run_softmaxsoftmax_finetuned_seed42/best_baked_layout_softmaxsoftmax_finetuned_seed42.xml")

PAD = 0.17   # half routing-channel width between blocks


def draw_floor(ax, gw, gh, grid, blocks):
    s = 1 - 2 * PAD
    ax.add_patch(mpatches.Rectangle((0, 0), gw, gh, facecolor=CH_BG,
                                    edgecolor="none", zorder=0))
    for x in range(gw):
        for y in range(gh):
            ax.add_patch(mpatches.Rectangle(
                (x + PAD, y + PAD), s, s,
                facecolor=C.get(grid[x][y], C["clb_empty"]),
                edgecolor="none", zorder=1))
    order = sorted(blocks.items(), key=lambda kv: 0 if kv[1]["kind"] == "clb" else 1)
    for _, blk in order:
        x, y, kind = blk["x"], blk["y"], blk["kind"]
        h = _block_height(kind)
        if kind == "dsp":
            fc, ec, lw = C["dsp_used"], C["dsp_edge"], 0.4
        elif kind == "bram":
            fc, ec, lw = C["bram_used"], C["bram_edge"], 0.4
        elif grid[x][y] == "io_empty":
            fc, ec, lw = C["io_used"], C["io_edge"], 0.25
        else:
            fc, ec, lw = C["clb_used"], C["clb_edge"], 0.2
        ax.add_patch(mpatches.Rectangle((x + PAD, y + PAD), s, h - 2 * PAD,
                                        facecolor=fc, edgecolor=ec,
                                        linewidth=lw, zorder=2))


gw1, gh1, b1 = parse_place(Path(TRAD[0])); grid1 = parse_arch(Path(TRAD[1]), gw1, gh1)
gw2, gh2, b2 = parse_place(Path(RL[0]));   grid2 = parse_arch(Path(RL[1]), gw2, gh2)
area_red = 100 * (1 - (gw2 * gh2) / (gw1 * gh1))

M = max(gw1, gh1)
LO, HI = -1.5, M + 1.5

fig, (axL, axM, axR) = plt.subplots(
    1, 3, figsize=(7.0, 3.8), gridspec_kw={"width_ratios": [M, 5.5, M], "wspace": 0.0})

# ---- left: traditional ----
draw_floor(axL, gw1, gh1, grid1, b1)
axL.set_title(f"Traditional   {gw1}×{gh1}", fontsize=10, color=SUBINK, pad=6)

# ---- right: RL, reclaimed L + ghost footprint ----
for (rx, ry, rw, rh) in [(gw2, 0, gw1 - gw2, gh1), (0, gh2, gw2, gh1 - gh2)]:
    axR.add_patch(mpatches.Rectangle((rx, ry), rw, rh, facecolor=RECLAIM,
                                     edgecolor="none", alpha=0.85, hatch="\\\\\\",
                                     linewidth=0, zorder=0.4))
draw_floor(axR, gw2, gh2, grid2, b2)
axR.add_patch(mpatches.Rectangle((0, 0), gw1, gh1, facecolor="none",
                                 edgecolor=GHOST, linewidth=1.0,
                                 linestyle=(0, (5, 3)), zorder=5))
axR.set_title(f"RL-optimized   {gw2}×{gh2}", fontsize=10, color=INK, pad=6)
# single, quiet number — placed in the reclaimed strip
lab = axR.text(gw2 + (gw1 - gw2) / 2, gh1 * 0.5, f"−{area_red:.0f}% area",
               ha="center", va="center", fontsize=9, color="#7a5a2a",
               style="italic", rotation=90, zorder=6)
lab.set_path_effects([pe.withStroke(linewidth=2.2, foreground=RECLAIM)])

for ax in (axL, axR):
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
    ax.set_aspect("equal"); ax.axis("off")

# ---- middle: a thin, text-free transformation arrow ----
axM.set_xlim(0, 1); axM.set_ylim(LO, HI); axM.axis("off")
axM.annotate("", xy=(0.85, M * 0.5), xytext=(0.15, M * 0.5),
             arrowprops=dict(arrowstyle="-|>", color=SUBINK, lw=1.4,
                             shrinkA=0, shrinkB=0))

# ---- legend (heterogeneous fabric; columns vs occupied blocks named consistently) ----
def patch(fc, ec=None, **kw):
    return mpatches.Patch(facecolor=fc, edgecolor=ec if ec else fc, lw=0.4, **kw)
# legend: each resource family shows used (saturated) and unused (pale).
# Unused tiles -- including the CLB holes -- are provisioned-but-unoccupied
# sites that still cost fabric area.
leg = [patch(C["clb_used"], C["clb_edge"]),
       patch(C["dsp_used"], C["dsp_edge"]),
       patch(C["bram_used"], C["bram_edge"]),
       patch(C["io_used"], C["io_edge"]),
       patch(C["clb_empty"], C["clb_edge"]),
       patch(C["dsp_empty"], C["dsp_edge"]),
       patch(C["bram_empty"], C["bram_edge"]),
       patch(RECLAIM, RECLAIM, alpha=0.85, hatch="\\\\\\"),
       mpatches.Patch(facecolor="none", edgecolor=GHOST, linestyle=(0, (5, 3)), lw=1.0)]
labs = ["CLB", "DSP", "BRAM", "I/O", "CLB (unused)", "DSP column (unused)",
        "BRAM column (unused)", "reclaimed area", "traditional footprint"]
fig.legend(leg, labs, loc="lower center", ncol=5, fontsize=7.3, frameon=False,
           handlelength=1.05, columnspacing=1.4, handletextpad=0.45,
           labelcolor=INK, bbox_to_anchor=(0.5, -0.05))

fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.10)
out = "paper/fig_layout.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "| area reduction %.1f%%" % area_red)
