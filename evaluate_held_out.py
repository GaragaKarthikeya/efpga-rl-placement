#!/usr/bin/env python3
"""
Zero-shot evaluation of a trained multi-benchmark model on benchmarks it
never saw during training.

The model's canvas/graph dims (MAX_WIDTH/MAX_HEIGHT/MAX_NODES/MAX_EDGES) were
fixed at training time from a benchmark universe that's a SUPERSET of what
it actually trained on (see TrainConfig.universe_benchmark_names / the
--universe_benchmarks flag in train.py) specifically so a checkpoint stays
loadable against held-out benchmarks like these. Pass the same
--universe_benchmarks list used at training time, or this will compute
different dims and the load will fail with a shape mismatch.

Usage:
    python evaluate_held_out.py --model_path runs/multi_light_tier_model.zip \\
        --eval_benchmarks softmax,reduction_layer,robot_rl \\
        --universe_benchmarks fifo,ch_intrinsics,spree,boundtop,mmc_core,diffeq1,diffeq2,raygentop,mkSMAdapter4B,or1200,mkPktMerge,softmax,reduction_layer,robot_rl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env.fpga_env import FPGAEnv, build_benchmark_configs, compute_max_dims
from src.training.gnn_extractor import GNNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO
from src.utils.config import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent


def run_episode(env: FPGAEnv, model: CustomMaskablePPO, max_steps: int = 200) -> dict:
    obs, _ = env.reset()
    for _ in range(max_steps):
        mask = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return {"reward": float(reward), **info}
    return {"reward": 0.0, "status": "max_steps_exceeded", "success": False}


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot eval on held-out benchmarks")
    p.add_argument("--model_path", required=True, help="Trained model .zip")
    p.add_argument("--eval_benchmarks", required=True, help="Comma-separated held-out benchmark names")
    p.add_argument("--universe_benchmarks", required=True, help="Comma-separated full universe (must match training)")
    p.add_argument("--vtr_timeout", type=int, default=1200, help="Per-episode VTR timeout (generous for eval)")
    p.add_argument("--out", default=None, help="Optional path to write JSON results")
    p.add_argument("--cache_suffix", default="", help="Suffix for an isolated VTR layout cache DB (use a unique value to force a fresh VTR run, bypassing any existing cached result)")
    args = p.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass

    eval_names = [b.strip() for b in args.eval_benchmarks.split(",") if b.strip()]
    universe_names = [b.strip() for b in args.universe_benchmarks.split(",") if b.strip()]

    max_width, max_height, max_nodes, max_edges = compute_max_dims(universe_names)
    print(f"Universe dims: MAX_WIDTH={max_width} MAX_HEIGHT={max_height} MAX_NODES={max_nodes} MAX_EDGES={max_edges}")

    configs = build_benchmark_configs(eval_names, max_width, max_height, max_nodes, max_edges, cache_suffix=args.cache_suffix)
    env = FPGAEnv(configs, max_width, max_height, max_nodes, max_edges, vtr_timeout=args.vtr_timeout)

    model = CustomMaskablePPO.load(
        args.model_path, env=env, policy_kwargs={"features_extractor_class": GNNFeaturesExtractor}
    )
    print(f"Loaded {args.model_path} -- no shape mismatch against held-out benchmarks: {eval_names}\n")

    results = {}
    for cfg in configs:
        env._active_config = cfg
        # Force this benchmark for the episode (reset() samples uniformly otherwise)
        idx = env.benchmark_configs.index(cfg)
        env.benchmark_configs = [cfg]
        info = run_episode(env, model)
        env.benchmark_configs = configs

        baseline = json.loads((PROJECT_ROOT / "baselines" / f"{cfg.name}_traditional_metric.txt").read_text())
        row = {
            "benchmark": cfg.name,
            "reward": info.get("reward"),
            "success": info.get("success"),
            "status": info.get("status"),
            "wirelength": info.get("wirelength"),
            "delay_ns": info.get("delay_ns"),
            "power_w": info.get("power_w"),
            "routing_area": info.get("routing_area"),
            "baseline": baseline,
        }
        if info.get("success") and info.get("routing_area") not in (None, "?"):
            adp = info["routing_area"] * info["delay_ns"] * info["power_w"]
            base_adp = baseline["routing_area"] * baseline["delay_ns"] * baseline["power_w"]
            row["adp"] = adp
            row["baseline_adp"] = base_adp
            row["adp_reduction_pct"] = (base_adp - adp) / base_adp * 100.0

        results[cfg.name] = row
        print(f"=== {cfg.name} ===")
        print(json.dumps(row, indent=2))
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Saved results -> {args.out}")


if __name__ == "__main__":
    main()
