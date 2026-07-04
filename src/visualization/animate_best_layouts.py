#!/usr/bin/env python3
"""
Renders an animated GIF flipbook of the "new best" layout milestones found
during an RL training run — i.e. watching the agent's discovered layouts
compact and improve over the course of training.

Each milestone is re-baked into an architecture XML + VPR constraints and run
through VTR once (to get an accurate final grid size), then rendered with the
same renderer used by plot_layout.py.

Usage:
    python3 src/visualization/animate_best_layouts.py \
        --benchmark diffeq2 \
        --jsonl all_layouts_diffeq2_seed_42_fullscale.jsonl \
        --out visualizations/animations/diffeq2_fullscale.gif
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import VTRPaths, load_env_file
from src.layout.baker import bake_layout
from src.netlist.parser import parse_net_file
from src.visualization.plot_layout import parse_place, parse_arch, draw_panel, C


def find_milestones(jsonl_path: Path) -> list[tuple[int, dict]]:
    """Episodes where a new best (highest-so-far) reward was achieved."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    best = float("-inf")
    milestones = []
    for i, r in enumerate(records, start=1):
        if r.get("success") and r["reward"] > best:
            best = r["reward"]
            milestones.append((i, r))
    return milestones


def get_dsp_block_names(benchmark: str) -> list[str]:
    net_file = PROJECT_ROOT / "runs" / f"{benchmark}_traditional" / f"{benchmark}.net"
    parsed = parse_net_file(net_file)
    dsp_parsed = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "dsp"]
    return [n for n, _ in sorted(dsp_parsed, key=lambda x: x[1], reverse=True)]


def render_milestone(benchmark: str, dsp_block_names: list[str], record: dict,
                      paths: VTRPaths, work_dir: Path):
    arch_path = work_dir / "arch.xml"
    constraints_path = work_dir / "constraints.xml"
    run_dir = work_dir / "vtr_run"

    bake_layout(
        benchmark_name=benchmark,
        dsps=[tuple(d) for d in record["dsps"]],
        mems=[tuple(b) for b in record.get("brams", [])],
        width=16, height=16,
        output_path=str(arch_path),
        aspect_ratio=record["aspect_ratio"],
        block_names=dsp_block_names,
        constraints_output_path=str(constraints_path),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(paths.python), str(paths.flow_script),
           str(PROJECT_ROOT / "benchmarks" / f"{benchmark}.v"), str(arch_path),
           "-temp_dir", str(run_dir),
           "-read_vpr_constraints", str(constraints_path)]
    if paths.has_power_tech:
        cmd += ["-cmos_tech", str(paths.power_tech_file)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    gw, gh, blocks = parse_place(run_dir / f"{benchmark}.place")
    grid = parse_arch(arch_path, gw, gh)
    return gw, gh, grid, blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                     help="Parallel VTR runs (default: all available cores)")
    args = ap.parse_args()

    load_env_file(PROJECT_ROOT / ".env")
    paths = VTRPaths()

    milestones = find_milestones(Path(args.jsonl))
    print(f"Found {len(milestones)} new-best milestones")

    dsp_block_names = get_dsp_block_names(args.benchmark)

    work_root = PROJECT_ROOT / "runs" / f"_anim_tmp_{args.benchmark}"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    out_path = Path(args.out)
    frames_dir = out_path.parent / f"{out_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: run VTR for every milestone first, so we know the largest grid
    # extent across the whole sequence before drawing anything. Different
    # milestones auto-size to very different grids (e.g. 18x15 vs 7x7) — every
    # frame must share one fixed canvas/axis range, or stitching them into a
    # GIF misaligns and "crops" frames against each other.
    # Each milestone's VTR run is independent, so they're farmed out across
    # all available cores instead of running one at a time.
    print(f"Running VTR for {len(milestones)} milestones across {args.workers} workers...")
    rendered = [None] * len(milestones)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(render_milestone, args.benchmark, dsp_block_names, r,
                             paths, work_root / f"m{idx}"): idx
            for idx, (ep, r) in enumerate(milestones)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            ep, r = milestones[idx]
            gw, gh, grid, blocks = future.result()
            rendered[idx] = {"ep": ep, "reward": r["reward"], "gw": gw, "gh": gh,
                              "grid": grid, "blocks": blocks}
            done += 1
            print(f"[{done}/{len(milestones)}] episode {ep}, reward {r['reward']:.4f} done")

    shutil.rmtree(work_root)

    max_gw = max(item["gw"] for item in rendered)
    max_gh = max(item["gh"] for item in rendered)
    pad = 1.5  # room for the axis tick labels / active-box annotation text

    # Pass 2: draw every frame with the same fixed window size, recentered on
    # each grid's own center — so the grid appears to stay in one fixed spot
    # in the middle of the canvas and only its size changes between frames,
    # rather than being pinned to a corner with empty space drifting around it.
    frame_paths = []
    for idx, item in enumerate(rendered):
        gw, gh = item["gw"], item["gh"]
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        fig.patch.set_facecolor(C["void"])
        draw_panel(ax, gw, gh, item["grid"], item["blocks"],
                   f"{args.benchmark} — episode {item['ep']}\nreward {item['reward']:.4f}  "
                   f"(milestone {idx + 1}/{len(rendered)})")
        ax.set_xlim(gw / 2 - max_gw / 2 - pad, gw / 2 + max_gw / 2 + pad)
        ax.set_ylim(gh / 2 - max_gh / 2 - pad, gh / 2 + max_gh / 2 + pad)
        frame_path = frames_dir / f"frame_{idx:03d}.png"
        fig.savefig(frame_path, facecolor=C["void"])
        plt.close(fig)
        frame_paths.append(frame_path)

    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in frame_paths]

    # Crossfade between consecutive milestones instead of a hard cut.
    fade_steps = 6
    hold_ms = 600
    fade_ms = 80
    final_hold_ms = 3000

    sequence, durations = [], []
    for i, img in enumerate(images):
        sequence.append(img)
        durations.append(final_hold_ms if i == len(images) - 1 else hold_ms)
        if i < len(images) - 1:
            nxt = images[i + 1]
            for step in range(1, fade_steps + 1):
                alpha = step / (fade_steps + 1)
                sequence.append(Image.blend(img, nxt, alpha))
                durations.append(fade_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sequence[0].save(out_path, save_all=True, append_images=sequence[1:],
                      duration=durations, loop=0)
    print(f"Saved → {out_path}  ({len(images)} milestones, {len(sequence)} total frames)")


if __name__ == "__main__":
    main()
