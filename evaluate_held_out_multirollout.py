#!/usr/bin/env python3
"""
Zero-shot evaluation of a trained multi-benchmark model on held-out
benchmarks, using N stochastic rollouts per benchmark instead of one
deterministic episode (see evaluate_held_out.py for the single-episode
version this is based on).

Why multiple rollouts: a single deterministic episode is one sample from the
policy's behavior on a benchmark it never trained on; it can't distinguish
"this benchmark generalizes reliably" from "this specific greedy rollout
happened to land well/badly." Sampling stochastically (deterministic=False)
and averaging over N episodes gives a per-benchmark mean and spread instead
of one point estimate.

The model's canvas/graph dims (MAX_WIDTH/MAX_HEIGHT/MAX_NODES/MAX_EDGES) were
fixed at training time from a benchmark universe that's a SUPERSET of what
it actually trained on (see TrainConfig.universe_benchmark_names / the
--universe_benchmarks flag in train.py) specifically so a checkpoint stays
loadable against held-out benchmarks like these. Pass the same
--universe_benchmarks list used at training time, or this will compute
different dims and the load will fail with a shape mismatch.

Usage (the standard 6-benchmark held-out set used throughout this project):
    python evaluate_held_out_multirollout.py \\
        --model_path runs/multi11_long_seed42_v2.zip \\
        --eval_benchmarks softmax,reduction_layer,lightweight_cipher,custom_macbuf,mkDelayWorker32B,arm_core \\
        --universe_benchmarks fifo,ch_intrinsics,spree,boundtop,mmc_core,diffeq1,diffeq2,raygentop,mkSMAdapter4B,or1200,mkPktMerge,softmax,reduction_layer,robot_rl \\
        --n_rollouts 10 \\
        --out phaseD_zeroshot6_multirollout_seed42.json
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


def run_episode(env: FPGAEnv, model: CustomMaskablePPO, deterministic: bool, max_steps: int = 200) -> dict:
    obs, _ = env.reset()
    for _ in range(max_steps):
        mask = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            return {"reward": float(reward), **info}
    return {"reward": 0.0, "status": "max_steps_exceeded", "success": False}


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-rollout zero-shot eval on held-out benchmarks")
    p.add_argument("--model_path", required=True, help="Trained model .zip")
    p.add_argument("--eval_benchmarks", required=True, help="Comma-separated held-out benchmark names")
    p.add_argument("--universe_benchmarks", required=True, help="Comma-separated full universe (must match training)")
    p.add_argument("--n_rollouts", type=int, default=10, help="Stochastic rollouts per benchmark")
    p.add_argument("--vtr_timeout", type=int, default=1200, help="Per-episode VTR timeout (generous for eval)")
    p.add_argument("--out", default=None, help="Optional path to write JSON results")
    args = p.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass

    eval_names = [b.strip() for b in args.eval_benchmarks.split(",") if b.strip()]
    universe_names = [b.strip() for b in args.universe_benchmarks.split(",") if b.strip()]

    max_width, max_height, max_nodes, max_edges = compute_max_dims(universe_names)
    print(f"Universe dims: MAX_WIDTH={max_width} MAX_HEIGHT={max_height} MAX_NODES={max_nodes} MAX_EDGES={max_edges}")

    configs = build_benchmark_configs(eval_names, max_width, max_height, max_nodes, max_edges)
    env = FPGAEnv(configs, max_width, max_height, max_nodes, max_edges, vtr_timeout=args.vtr_timeout)

    model = CustomMaskablePPO.load(
        args.model_path, env=env, policy_kwargs={"features_extractor_class": GNNFeaturesExtractor}
    )
    print(f"Loaded {args.model_path} -- no shape mismatch against held-out benchmarks: {eval_names}\n")

    results = {}
    for cfg in configs:
        baseline = json.loads((PROJECT_ROOT / "baselines" / f"{cfg.name}_traditional_metric.txt").read_text())
        base_adp = baseline["routing_area"] * baseline["delay_ns"] * baseline["power_w"]

        # Force this benchmark for every rollout (reset() samples uniformly otherwise)
        env.benchmark_configs = [cfg]

        rollouts = []
        for i in range(args.n_rollouts):
            info = run_episode(env, model, deterministic=False)
            row = {
                "rollout": i,
                "reward": info.get("reward"),
                "success": info.get("success"),
                "status": info.get("status"),
                "wirelength": info.get("wirelength"),
                "delay_ns": info.get("delay_ns"),
                "power_w": info.get("power_w"),
                "routing_area": info.get("routing_area"),
            }
            if info.get("success") and info.get("routing_area") not in (None, "?"):
                adp = info["routing_area"] * info["delay_ns"] * info["power_w"]
                row["adp"] = adp
                row["adp_reduction_pct"] = (base_adp - adp) / base_adp * 100.0
            rollouts.append(row)
            print(f"  [{cfg.name}] rollout {i}: reward={row['reward']:.4f}"
                  + (f"  adp_reduction={row['adp_reduction_pct']:.2f}%" if "adp_reduction_pct" in row else "  (failed)"))

        env.benchmark_configs = configs

        successful = [r["adp_reduction_pct"] for r in rollouts if "adp_reduction_pct" in r]
        avg_adp_reduction_pct = sum(successful) / len(successful) if successful else None
        best_adp_reduction_pct = max(successful) if successful else None
        worst_adp_reduction_pct = min(successful) if successful else None

        results[cfg.name] = {
            "benchmark": cfg.name,
            "baseline": baseline,
            "baseline_adp": base_adp,
            "n_rollouts": args.n_rollouts,
            "n_successful": len(successful),
            "avg_adp_reduction_pct": avg_adp_reduction_pct,
            "best_adp_reduction_pct": best_adp_reduction_pct,
            "worst_adp_reduction_pct": worst_adp_reduction_pct,
            "rollouts": rollouts,
        }
        print(f"=== {cfg.name}: avg over {len(successful)}/{args.n_rollouts} successful rollouts = "
              f"{avg_adp_reduction_pct if avg_adp_reduction_pct is not None else 'N/A'} ===\n")

    print("\n" + "=" * 60)
    print("SUMMARY (average ADP reduction % per benchmark)")
    print("=" * 60)
    overall = []
    for name, row in results.items():
        v = row["avg_adp_reduction_pct"]
        print(f"  {name:22s} {v:+.2f}%" if v is not None else f"  {name:22s} N/A")
        if v is not None:
            overall.append(v)
    if overall:
        print(f"\n  Overall average across {len(overall)} benchmarks: {sum(overall)/len(overall):+.2f}%")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nSaved results -> {args.out}")


if __name__ == "__main__":
    main()
