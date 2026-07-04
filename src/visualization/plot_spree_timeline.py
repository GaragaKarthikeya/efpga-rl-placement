#!/usr/bin/env python3
"""
Timeline figure: how the RL agent's spree fabric evolves from the traditional
island-style baseline to its best discovered layout, with REAL VPR
placements at every step (not an architecture-only DSP/BRAM tile map).

One panel per training episode where a new best-so-far ADP reduction was
found (every positive change, in order), using the seed with spree's highest
final ADP reduction (seed 123, 39.11%). Source:
all_layouts_multi_seed_123_multi11_long_seed123.jsonl (per-episode log).

Each RL milestone was re-run through the real VTR flow
(scripts/run_spree_timeline_vtr.py) to get a true .place file showing actual
CLB/DSP/BRAM/IO occupancy, exactly like the traditional baseline. Rendering
reuses parse_place/parse_arch (src/visualization/plot_layout.py) and the
draw_floor look (palette, used-vs-unused shading) from
src/visualization/plot_compaction.py, the script that produced the original
fig_layout.pdf softmax comparison.
Output: paper/fig_spree_timeline.pdf (vector).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines
from matplotlib.patches import ConnectionPatch
from matplotlib.font_manager import FontManager

from plot_layout import parse_place, parse_arch, C, _block_height
import palette as PAL

PROJECT_ROOT = Path(__file__).resolve().parents[2]

JSONL_PATH = PROJECT_ROOT / "all_layouts_multi_seed_123_multi11_long_seed123.jsonl"
TRAD_PLACE = PROJECT_ROOT / "runs/spree_traditional/spree.place"
TRAD_ARCH = PROJECT_ROOT / "runs/spree_traditional/k6_frac_N10_mem32K_40nm.xml"
TIMELINE_DIR = PROJECT_ROOT / "runs/spree_timeline"
BENCHMARK = "spree"

# ── same palette/style as plot_compaction.py (fig_layout.pdf) ──────────────
CH_BG = PAL.CHANNEL
RECLAIM = PAL.RECLAIM
P = {
    "void": CH_BG,
    "io_empty": PAL.IO_PALE,
    "clb_empty": PAL.CLB_PALE,
    "dsp_empty": PAL.TERRA_PALE,
    "bram_empty": PAL.TEAL_PALE,
    "clb_used": PAL.SAGE,
    "dsp_used": PAL.TERRA,
    "bram_used": PAL.TEAL,
    "io_used": PAL.SAND,
    "clb_edge": PAL.SAGE_E, "dsp_edge": PAL.TERRA_E,
    "bram_edge": PAL.TEAL_E, "io_edge": PAL.SAND_E,
    "grid_line": CH_BG,
}
C.update(P)

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "pdf.fonttype": 42})

PAD = 0.17  # half routing-channel width between blocks, matches plot_compaction.py


def draw_floor(ax, gw, gh, grid, blocks):
    s = 1 - 2 * PAD
    ax.add_patch(mpatches.Rectangle((0, 0), gw, gh, facecolor=CH_BG, edgecolor="none", zorder=0))
    for x in range(gw):
        for y in range(gh):
            ax.add_patch(mpatches.Rectangle(
                (x + PAD, y + PAD), s, s,
                facecolor=C.get(grid[x][y], C["clb_empty"]), edgecolor="none", zorder=1))
    order = sorted(blocks.items(), key=lambda kv: 0 if kv[1]["kind"] == "clb" else 1)
    for _, blk in order:
        x, y, kind = blk["x"], blk["y"], blk["kind"]
        h = _block_height(kind)
        if kind == "dsp":
            fc, ec, lw = C["dsp_used"], C["dsp_edge"], 0.35
        elif kind == "bram":
            fc, ec, lw = C["bram_used"], C["bram_edge"], 0.35
        elif grid[x][y] == "io_empty":
            fc, ec, lw = C["io_used"], C["io_edge"], 0.2
        else:
            fc, ec, lw = C["clb_used"], C["clb_edge"], 0.15
        ax.add_patch(mpatches.Rectangle((x + PAD, y + PAD), s, h - 2 * PAD,
                                         facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))


def draw_panel(ax, gw, gh, grid, blocks, title, highlight=False):
    draw_floor(ax, gw, gh, grid, blocks)
    ax.set_xlim(-0.4, gw + 0.4)
    ax.set_ylim(-0.4, gh + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    border_c = PAL.INK if highlight else PAL.SUBINK
    border_lw = 1.3 if highlight else 0.5
    ax.add_patch(mpatches.Rectangle((-0.4, -0.4), gw + 0.8, gh + 0.8, fill=False,
                                     edgecolor=border_c, linewidth=border_lw, zorder=5))
    fw = "bold" if highlight else "normal"
    ax.set_title(title, fontsize=6.6, color=(PAL.INK if highlight else PAL.SUBINK),
                 fontweight=fw, pad=3)


def load_milestones():
    best = -float("inf")
    milestones = []
    with open(JSONL_PATH) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("benchmark_name") != BENCHMARK:
                continue
            if r.get("success") and r["reward"] > best:
                best = r["reward"]
                milestones.append({"reward": best, "reduction": (1 - math.exp(-best)) * 100})
    return milestones


def edge_points(direction):
    """(xyA, xyB) in axes-fraction coords for a ConnectionPatch in `direction`.
    "down" is offset off-center (x=0.82) so it doesn't run through the
    destination panel's title, which sits centered above it."""
    if direction == "right":
        return (1, 0.5), (0, 0.5)
    if direction == "left":
        return (0, 0.5), (1, 0.5)
    return (0.82, 0), (0.82, 1)  # "down"


def main():
    manifest_path = TIMELINE_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run scripts/run_spree_timeline_vtr.py first")
    manifest = json.loads(manifest_path.read_text())

    tgw, tgh, tblocks = parse_place(TRAD_PLACE)
    tgrid = parse_arch(TRAD_ARCH, tgw, tgh)
    trad_panel = (tgw, tgh, tgrid, tblocks, "Traditional\n" + f"{tgw}×{tgh}")

    milestone_panels = []
    for m in manifest:
        if not m.get("ok"):
            print(f"WARNING: {m['tag']} not ok, skipping", file=sys.stderr)
            continue
        run_dir = Path(m["run_dir"])
        place_file = run_dir / f"{BENCHMARK}.place"
        arch_file = Path(m["arch"])
        gw, gh, blocks = parse_place(place_file)
        grid = parse_arch(arch_file, gw, gh)
        label = f"{gw}×{gh}\n{m['reduction']:+.1f}%"
        milestone_panels.append((label, gw, gh, grid, blocks))

    n_ok = len(milestone_panels)
    ncols = 8
    nrows = 2
    fig, axes2d = plt.subplots(nrows, ncols, figsize=(7.1, 1.55 * nrows))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12, wspace=0.35, hspace=-0.05)

    # ── boustrophedon layout: row 0 left->right (Traditional, then RL
    # milestones), row 1 right->left, so the reading order traces the
    # actual chronological search path without the eye jumping back. ──
    draw_panel(axes2d[0, 0], *trad_panel, highlight=False)

    positions = []
    for i in range(n_ok):
        if i < ncols - 1:
            row, col = 0, i + 1
        else:
            j = i - (ncols - 1)
            row, col = 1, ncols - 1 - j
        positions.append((row, col))
        title, gw, gh, grid, blocks = milestone_panels[i]
        draw_panel(axes2d[row, col], gw, gh, grid, blocks, title, highlight=(i == n_ok - 1))

    used = {(0, 0)} | set(positions)
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes2d[r, c].axis("off")

    # vertical dotted line: traditional fabric vs. the start of the RL search
    if positions:
        r1, c1 = positions[0]
        pos0 = axes2d[0, 0].get_position()
        pos1 = axes2d[r1, c1].get_position()
        mid_x = (pos0.x1 + pos1.x0) / 2
        sep = matplotlib.lines.Line2D(
            [mid_x, mid_x], [pos0.y0, pos0.y1], transform=fig.transFigure,
            linestyle=(0, (3, 2)), color=PAL.SUBINK, linewidth=1.1, zorder=10)
        fig.add_artist(sep)

    # progression arrows: best-so-far episode i -> i+1, snaking through the grid
    for i in range(n_ok - 1):
        rA, cA = positions[i]
        rB, cB = positions[i + 1]
        direction = "down" if rA != rB else ("right" if cB > cA else "left")
        pA, pB = edge_points(direction)
        arrow = ConnectionPatch(
            xyA=pA, coordsA="axes fraction", axesA=axes2d[rA, cA],
            xyB=pB, coordsB="axes fraction", axesB=axes2d[rB, cB],
            arrowstyle="-|>", mutation_scale=8, color=PAL.SUBINK,
            linewidth=0.9, shrinkA=1, shrinkB=1, zorder=10)
        fig.add_artist(arrow)

    def patch(fc, ec=None, **kw):
        return mpatches.Patch(facecolor=fc, edgecolor=ec if ec else fc, lw=0.4, **kw)

    leg = [patch(C["clb_used"], C["clb_edge"]),
           patch(C["dsp_used"], C["dsp_edge"]),
           patch(C["bram_used"], C["bram_edge"]),
           patch(C["io_used"], C["io_edge"]),
           patch(C["clb_empty"], C["clb_edge"]),
           patch(C["dsp_empty"], C["dsp_edge"]),
           patch(C["bram_empty"], C["bram_edge"])]
    labs = ["CLB", "DSP", "BRAM", "I/O", "CLB (unused)", "DSP column (unused)", "BRAM column (unused)"]
    fig.legend(leg, labs, loc="lower center", ncol=7, fontsize=6.6, frameon=False,
               handlelength=1.0, columnspacing=1.2, handletextpad=0.4,
               labelcolor=PAL.INK, bbox_to_anchor=(0.5, -0.01))

    out = PROJECT_ROOT / "paper/fig_spree_timeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, "|", 1 + n_ok, f"panels (1 traditional + {n_ok} RL milestones, real VPR placements)")


if __name__ == "__main__":
    main()
