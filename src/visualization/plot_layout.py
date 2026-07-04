#!/usr/bin/env python3
"""
FPGA layout visualizer — reads VTR .place + arch XML and renders a PNG.

Usage (single layout):
    python3 src/visualization/plot_layout.py \
        --place runs/diffeq2_traditional/diffeq2.place \
        --arch  runs/diffeq2_traditional/k6_frac_N10_mem32K_40nm.xml \
        --out   visualizations/trad.png \
        --title "diffeq2 Traditional"

Usage (side-by-side comparison):
    python3 src/visualization/plot_layout.py \
        --compare \
        --place  runs/diffeq2_traditional/diffeq2.place \
        --arch   runs/diffeq2_traditional/k6_frac_N10_mem32K_40nm.xml \
        --place2 runs/best_run_diffeq2/diffeq2.place \
        --arch2  runs/best_run_diffeq2/best_baked_layout_diffeq2.xml \
        --title  "diffeq2 — Traditional" --title2 "diffeq2 — RL Best" \
        --out    visualizations/diffeq2_comparison.png
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontManager

# serif to match the other paper figures (Fig 1, Fig 3) and the acmart body
_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ── Palette (light, professional) ───────────────────────────────────────────
C = {
    # Background fabric
    "void":        "#ffffff",
    "io_empty":    "#eef1f5",
    "clb_empty":   "#f6f8fa",
    "dsp_empty":   "#e3ecfa",   # pale blue  — unoccupied DSP slot
    "bram_empty":  "#fbeaea",   # pale red   — unoccupied BRAM slot

    # Placed / active tiles (muted, aligned to the figure-family palette)
    "io_used":     "#d99a4e",   # soft amber — I/O pad in use
    "clb_used":    "#4f9d69",   # soft green — logic block in use
    "dsp_used":    "#1f5fa8",   # family blue — DSP in use (matches Fig 1)
    "bram_used":   "#c0504d",   # muted red  — BRAM in use

    # Edge colors (darker shade of each fill, for crisp definition)
    "io_edge":     "#a9772f",
    "clb_edge":    "#3a7a4f",
    "dsp_edge":    "#184a82",
    "bram_edge":   "#9a3f3c",

    # Annotation
    "grid_line":   "#d0d7de",
    "bbox":        "#7c3aed",   # purple  — active-logic bounding box
    "text":        "#1f2328",
    "subtext":     "#57606a",
}

DSP_HEIGHT  = 4
BRAM_HEIGHT = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_kind(name: str) -> str:
    """Return 'dsp', 'bram', 'io', or 'clb' from a VPR block name."""
    n = name.lower()
    if n.startswith("$mul") or "mult_36" in n:
        return "dsp"
    if "bram" in n or "mem[" in n or n.startswith("$bram") or "mem~" in n or "dual_port_ram" in n:
        return "bram"
    # IO pads from VPR are on the perimeter; classify anything else as clb
    return "clb"


def _block_height(kind: str) -> int:
    if kind == "dsp":  return DSP_HEIGHT
    if kind == "bram": return BRAM_HEIGHT
    return 1


def parse_place(path: Path) -> tuple[int, int, dict]:
    """Return (grid_w, grid_h, {name: {x, y, kind}})."""
    blocks = {}
    gw = gh = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Array size:"):
                parts = line.split()
                gw, gh = int(parts[2]), int(parts[4])
            if not line or line.startswith("#") or line.startswith("Netlist") or line.startswith("Array"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                name, x, y = parts[0], int(parts[1]), int(parts[2])
                blocks[name] = {"x": x, "y": y, "kind": _block_kind(name)}
            except ValueError:
                pass
    return gw, gh, blocks


def parse_arch(path: Path, gw: int, gh: int) -> list[list[str]]:
    """
    Build a gw×gh grid of tile types from the VPR architecture XML.
    DSP <single> tags also fill the subsequent DSP_HEIGHT-1 rows so the
    background fabric shows the full tall-tile footprint.
    """
    grid = [["clb_empty"] * gh for _ in range(gw)]

    if not path or not path.is_file():
        # Fallback: IO perimeter, CLB interior
        for x in range(gw):
            for y in range(gh):
                if x in (0, gw-1) or y in (0, gh-1):
                    grid[x][y] = "io_empty" if not (x in (0,gw-1) and y in (0,gh-1)) else "void"
                else:
                    grid[x][y] = "clb_empty"
        return grid

    tree = ET.parse(path)
    root = tree.getroot()
    layout = root.find("layout/auto_layout")
    if layout is None:
        layout = root.find("layout/fixed_layout")
    if layout is None:
        return grid

    children = sorted(layout, key=lambda c: int(c.get("priority", "0")))

    TYPE_MAP = {
        "io":      "io_empty",
        "clb":     "clb_empty",
        "mult_36": "dsp_empty",
        "memory":  "bram_empty",
        "EMPTY":   "void",
    }

    def cell(t):
        return TYPE_MAP.get(t, "clb_empty")

    for child in children:
        t = child.get("type", "")
        ct = cell(t)

        if child.tag == "fill":
            for x in range(gw):
                for y in range(gh):
                    grid[x][y] = ct

        elif child.tag == "perimeter":
            for x in range(gw):
                grid[x][0]    = ct
                grid[x][gh-1] = ct
            for y in range(gh):
                grid[0][y]    = ct
                grid[gw-1][y] = ct

        elif child.tag == "corners":
            for x in (0, gw-1):
                for y in (0, gh-1):
                    grid[x][y] = ct

        elif child.tag == "col":
            sx = int(child.get("startx", 0))
            rx = int(child.get("repeatx", gw))
            sy = int(child.get("starty", 0))
            x  = sx
            while x < gw:
                for y in range(sy, gh):
                    grid[x][y] = ct
                if rx <= 0:
                    break
                x += rx

        elif child.tag == "single":
            x = int(child.get("x", 0))
            y = int(child.get("y", 0))
            h = DSP_HEIGHT if t == "mult_36" else (BRAM_HEIGHT if t == "memory" else 1)
            for dy in range(h):
                if x < gw and y + dy < gh:
                    grid[x][y + dy] = ct

    return grid


# ── Rendering ─────────────────────────────────────────────────────────────────

def draw_panel(ax, gw: int, gh: int, grid: list[list[str]],
               blocks: dict, title: str) -> None:

    ax.set_facecolor(C["void"])
    ax.set_xlim(-0.5, gw + 0.5)
    ax.set_ylim(-0.5, gh + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, color=C["text"], fontsize=9.5, fontweight="semibold", pad=7, wrap=True)

    # ── Background fabric (flat, hairline grid) ──────────────────────────────
    # For large fabrics the per-tile grid lines become visual noise; fade them.
    grid_lw = 0.12 if max(gw, gh) > 30 else 0.25
    for x in range(gw):
        for y in range(gh):
            color = C.get(grid[x][y], C["clb_empty"])
            ax.add_patch(mpatches.Rectangle(
                (x, y), 1, 1,
                facecolor=color, edgecolor=C["grid_line"], linewidth=grid_lw, zorder=1
            ))

    # ── Placed blocks (flat tiles, no glyphs) ────────────────────────────────
    bbox_xs, bbox_ys = [], []

    # Draw CLBs first so tall blocks render on top
    draw_order = sorted(blocks.items(), key=lambda kv: 0 if kv[1]["kind"] == "clb" else 1)

    for name, blk in draw_order:
        x, y, kind = blk["x"], blk["y"], blk["kind"]
        h = _block_height(kind)

        if kind == "dsp":
            fc, ec, lw = C["dsp_used"],  C["dsp_edge"],  0.5
        elif kind == "bram":
            fc, ec, lw = C["bram_used"], C["bram_edge"], 0.5
        elif grid[x][y] == "io_empty":
            fc, ec, lw = C["io_used"],   C["io_edge"],   0.3
        else:
            fc, ec, lw = C["clb_used"],  C["clb_edge"],  0.25

        ax.add_patch(mpatches.Rectangle(
            (x, y), 1, h,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
        ))

        # Bounding box only over core logic (not IO)
        if kind not in ("io",) and grid[x][y] != "io_empty":
            bbox_xs += [x, x + 1]
            bbox_ys += [y, y + h]

    # ── Active-logic bounding box ────────────────────────────────────────────
    if bbox_xs and bbox_ys:
        bx0, bx1 = min(bbox_xs), max(bbox_xs)
        by0, by1 = min(bbox_ys), max(bbox_ys)
        pad = 0.3
        ax.add_patch(mpatches.Rectangle(
            (bx0 - pad, by0 - pad), (bx1 - bx0) + 2*pad, (by1 - by0) + 2*pad,
            facecolor="none", edgecolor=C["bbox"],
            linewidth=1.1, linestyle=(0, (4, 3)), zorder=4
        ))
        ax.text((bx0 + bx1) / 2, by1 + pad + 0.5,
                f"{int(bx1-bx0)} × {int(by1-by0)} active",
                color=C["bbox"], fontsize=8, fontweight="semibold",
                ha="center", va="bottom", zorder=4)

    ax.axis("off")


def make_legend():
    """Essentials only — used tile types plus the bounding box."""
    def sw(fc, ec):
        return mpatches.Patch(facecolor=fc, edgecolor=ec, linewidth=0.4)
    return [
        sw(C["clb_used"],  C["clb_edge"]),
        sw(C["dsp_used"],  C["dsp_edge"]),
        sw(C["bram_used"], C["bram_edge"]),
        sw(C["io_used"],   C["io_edge"]),
        mpatches.Patch(facecolor=C["clb_empty"], edgecolor=C["grid_line"], linewidth=0.4),
        mpatches.Patch(facecolor="none", edgecolor=C["bbox"],
                       linestyle=(0, (4, 3)), linewidth=1.1),
    ]


LEGEND_LABELS = ["CLB", "DSP", "BRAM", "I/O pad", "unused tile",
                 "active-logic box"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VTR layout visualizer")
    ap.add_argument("--place",   required=True,  help=".place file (layout 1)")
    ap.add_argument("--arch",    default=None,   help="arch XML  (layout 1)")
    ap.add_argument("--title",   default="FPGA Layout")
    ap.add_argument("--compare", action="store_true", help="Side-by-side comparison mode (2 or 3 panels)")
    ap.add_argument("--place2",  default=None,   help=".place file (layout 2)")
    ap.add_argument("--arch2",   default=None,   help="arch XML  (layout 2)")
    ap.add_argument("--title2",  default="FPGA Layout 2")
    ap.add_argument("--place3",  default=None,   help=".place file (layout 3, optional — enables 3-panel mode)")
    ap.add_argument("--arch3",   default=None,   help="arch XML  (layout 3)")
    ap.add_argument("--title3",  default="FPGA Layout 3")
    ap.add_argument("--out",     required=True,  help="Output PNG path")
    args = ap.parse_args()

    plt.style.use("default")

    panels = [(args.place, args.arch, args.title)]
    if args.compare:
        panels.append((args.place2, args.arch2, args.title2))
        if args.place3:
            panels.append((args.place3, args.arch3, args.title3))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), dpi=300)
    fig.patch.set_facecolor(C["void"])
    if n == 1:
        axes = [axes]

    for ax, (place, arch, title) in zip(axes, panels):
        gw, gh, blks = parse_place(Path(place))
        grid = parse_arch(Path(arch) if arch else None, gw, gh)
        draw_panel(ax, gw, gh, grid, blks, title)

    fig.legend(handles=make_legend(), labels=LEGEND_LABELS,
               loc="lower center", ncol=6, fontsize=8, frameon=False,
               labelcolor=C["text"], handlelength=1.1, columnspacing=1.3,
               handletextpad=0.5, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor=C["void"], bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
