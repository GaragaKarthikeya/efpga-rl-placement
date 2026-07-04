#!/usr/bin/env python3
"""
Run real VTR flows for every spree training milestone (seed 123, every
episode that set a new best-so-far ADP reduction), so the timeline figure
can render true VPR placements (real CLB occupancy) instead of
architecture-only DSP/BRAM tile maps.

Reuses the exact bake_layout()/VTRRunner call sequence FPGAEnv uses per
episode (src/env/fpga_env.py:_evaluate_layout), with one difference: results
are kept in runs/spree_timeline/milestone_{i:02d}/ instead of a temp dir, so
the figure script can read the real .place + baked arch XML afterward.

Usage: /home/digital-2/.venv/bin/python3 scripts/run_spree_timeline_vtr.py
"""
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.layout.baker import bake_layout
from src.netlist.parser import parse_net_file
from src.evaluation.vtr_runner import VTRRunner

JSONL_PATH = PROJECT_ROOT / "all_layouts_multi_seed_123_multi11_long_seed123.jsonl"
OUT_DIR = PROJECT_ROOT / "runs" / "spree_timeline"
BENCHMARK = "spree"


def load_block_names():
    net_file = PROJECT_ROOT / "runs" / f"{BENCHMARK}_traditional" / f"{BENCHMARK}.net"
    parsed = parse_net_file(net_file)
    dsp = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "dsp"]
    bram = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "bram"]
    dsp_names = [n for n, _ in sorted(dsp, key=lambda x: x[1], reverse=True)]
    bram_names = [n for n, _ in sorted(bram, key=lambda x: x[1], reverse=True)]
    return dsp_names, bram_names


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
                milestones.append({
                    "reward": best,
                    "reduction": (1 - math.exp(-best)) * 100,
                    "dsps": [tuple(p) for p in r.get("dsps", [])],
                    "brams": [tuple(p) for p in r.get("brams", [])],
                    "ratio": r.get("aspect_ratio", 1.0),
                })
    return milestones


def main():
    dsp_names, bram_names = load_block_names()
    milestones = load_milestones()
    print(f"{len(milestones)} milestones, {len(dsp_names)} DSP names, {len(bram_names)} BRAM names")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = VTRRunner()
    benchmark_file = PROJECT_ROOT / "benchmarks" / f"{BENCHMARK}.v"
    all_block_names = dsp_names + bram_names

    manifest = []
    for i, m in enumerate(milestones):
        tag = f"milestone_{i:02d}"
        run_dir = OUT_DIR / tag
        arch_path = OUT_DIR / f"{tag}_arch.xml"
        constraints_path = OUT_DIR / f"{tag}_constraints.xml"

        place_file = run_dir / f"{BENCHMARK}.place"
        if place_file.is_file():
            print(f"[{i+1}/{len(milestones)}] {tag}: already done, skipping")
            manifest.append({"tag": tag, **m, "run_dir": str(run_dir), "arch": str(arch_path)})
            continue

        result = bake_layout(
            benchmark_name=BENCHMARK,
            dsps=m["dsps"],
            mems=m["brams"],
            width=14, height=14,  # unused: aspect_ratio given -> auto_layout
            output_path=str(arch_path),
            aspect_ratio=m["ratio"],
            block_names=all_block_names,
            constraints_output_path=str(constraints_path),
        )
        if result == -1:
            print(f"[{i+1}/{len(milestones)}] {tag}: bake_layout failed, skipping")
            continue

        print(f"[{i+1}/{len(milestones)}] {tag}: running VTR (ratio={m['ratio']}, "
              f"dsps={m['dsps']}, brams={m['brams']})...")
        rc = runner.run(
            benchmark_file, arch_path, run_dir,
            silent=True, constraints_file=constraints_path, timeout=600,
        )
        ok = rc == 0 and place_file.is_file()
        print(f"    rc={rc} place_ok={ok}")
        manifest.append({"tag": tag, **m, "run_dir": str(run_dir), "arch": str(arch_path), "ok": ok})

    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", OUT_DIR / "manifest.json")


if __name__ == "__main__":
    main()
