#!/usr/bin/env python3
"""
Run one deterministic episode and save the resulting .place + arch .xml
for visualization, by intercepting the cleanup step.

Usage:
    python save_layout_for_viz.py \
        --model_path runs/multi11_long_seed42_v2.zip \
        --benchmark mkDelayWorker32B \
        --universe_benchmarks "fifo,ch_intrinsics,spree,boundtop,mmc_core,diffeq1,diffeq2,raygentop,mkSMAdapter4B,or1200,mkPktMerge,softmax,reduction_layer,robot_rl" \
        --out_dir runs/mkDelayWorker32B_rl_viz_seed42
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_env_file
from src.env.fpga_env import FPGAEnv, build_benchmark_configs, compute_max_dims
from src.training.gnn_extractor import GNNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO
import src.env.fpga_env as _env_mod

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--universe_benchmarks", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    # Monkey-patch _cleanup to copy files before deletion
    original_cleanup = _env_mod.FPGAEnv._cleanup

    def patched_cleanup(arch_file, run_dir, constraints_file=None):
        if not saved:
            place_files = list(run_dir.glob("*.place")) if run_dir.exists() else []
            if place_files:
                shutil.copy(place_files[0], out_dir / place_files[0].name)
                saved["place"] = out_dir / place_files[0].name
                print(f"Saved place → {saved['place']}")
            if arch_file and arch_file.exists():
                dest = out_dir / "baked_arch.xml"
                shutil.copy(arch_file, dest)
                saved["arch"] = dest
                print(f"Saved arch  → {saved['arch']}")
        original_cleanup(arch_file, run_dir, constraints_file)

    _env_mod.FPGAEnv._cleanup = staticmethod(patched_cleanup)

    universe = [b.strip() for b in args.universe_benchmarks.split(",")]
    # Include the eval benchmark in universe for dim computation if not already there
    universe_for_dims = universe if args.benchmark in universe else universe + [args.benchmark]
    max_w, max_h, max_nodes, max_edges = compute_max_dims(universe_for_dims)
    bench_configs = build_benchmark_configs([args.benchmark], max_w, max_h, max_nodes, max_edges,
                                            cache_suffix="_viz_nocache")

    bench_cfg = bench_configs[0]
    env = FPGAEnv(
        benchmark_configs=[bench_cfg],
        max_width=max_w, max_height=max_h,
        max_nodes=max_nodes, max_edges=max_edges,
        vtr_timeout=1200,
    )

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    model = CustomMaskablePPO.load(args.model_path, env=vec_env)

    obs, _ = vec_env.envs[0].env.reset()
    from gymnasium import spaces
    import numpy as np
    done = False
    for _ in range(300):
        mask = vec_env.envs[0].env.get_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = vec_env.envs[0].env.step(int(action))
        if terminated or truncated:
            print(f"Episode done. reward={reward:.5f}")
            print("info:", info)
            break

    if saved:
        print(f"\nFiles saved to {out_dir}")
    else:
        print("WARNING: no place file was captured (episode may not have run VTR)")


if __name__ == "__main__":
    main()
