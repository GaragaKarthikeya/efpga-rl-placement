#!/usr/bin/env python3
"""
Evaluate a trained model (zero additional training) on a different benchmark
to test cross-benchmark transfer. Only valid when the target benchmark's
FPGA grid size and DSP/BRAM counts exactly match what the model was trained
on (same Box/Discrete observation/action space shapes).

Usage:
    python evaluate_transfer.py --model_path runs/diffeq2_fullscale_seed42.zip \
        --benchmark diffeq1 --n_episodes 50 --n_envs 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env.fpga_env import FPGAEnv
from src.training.ppo import CustomMaskablePPO
from src.utils.config import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent


def build_env_kwargs(benchmark_name: str) -> tuple[dict, dict]:
    res_file = PROJECT_ROOT / "baselines" / f"{benchmark_name}_traditional_resources.txt"
    metric_file = PROJECT_ROOT / "baselines" / f"{benchmark_name}_traditional_metric.txt"
    res_data = json.loads(res_file.read_text())
    metric_data = json.loads(metric_file.read_text())

    width, height = res_data["fpga_size"]
    reqs = res_data.get("requirements", {})
    req_dsp = reqs.get("dsp", 0)
    req_bram = reqs.get("bram", 0)

    net_count_data = {}
    net_count_file = PROJECT_ROOT / f"{benchmark_name}_netlist_info.json"
    if net_count_file.is_file():
        for k, v in json.loads(net_count_file.read_text()).items():
            if k.startswith("('"):
                try:
                    net_count_data[eval(k)] = v
                    continue
                except Exception:
                    pass
            net_count_data[k] = v

    dsp_block_names, bram_block_names = [], []
    net_file = PROJECT_ROOT / "runs" / f"{benchmark_name}_traditional" / f"{benchmark_name}.net"
    if net_file.is_file():
        from src.netlist.parser import parse_net_file
        parsed = parse_net_file(net_file)
        dsp_parsed = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "dsp"]
        bram_parsed = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "bram"]
        dsp_block_names = [n for n, _ in sorted(dsp_parsed, key=lambda x: x[1], reverse=True)]
        bram_block_names = [n for n, _ in sorted(bram_parsed, key=lambda x: x[1], reverse=True)]
        if req_dsp and len(dsp_block_names) != req_dsp:
            dsp_block_names = []
        if req_bram and len(bram_block_names) != req_bram:
            bram_block_names = []

    env_kwargs = {
        "benchmark_name": benchmark_name,
        "width": int(width),
        "height": int(height),
        "req_dsp": req_dsp,
        "req_bram": req_bram,
        "traditional_metrics": metric_data,
        "cache_db_path": str(PROJECT_ROOT / "runs" / f"vtr_layout_cache_{benchmark_name}.db"),
        "wl_weight": 0.0,
        "pw_weight": 1.0,
        "dl_weight": 1.0,
        "ar_weight": 1.0,
        "net_count_data": net_count_data,
        "use_net_count_sort": True,
        "dsp_block_names": dsp_block_names or None,
        "bram_block_names": bram_block_names or None,
    }
    return env_kwargs, metric_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--n_envs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    load_env_file(PROJECT_ROOT / ".env")

    env_kwargs, trad_metrics = build_env_kwargs(args.benchmark)
    env = make_vec_env(FPGAEnv, n_envs=args.n_envs, seed=args.seed,
                        vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    print(f"Loading model from {args.model_path} onto {args.benchmark} env "
          f"(obs={env.observation_space.shape}, action={env.action_space.n})")
    model = CustomMaskablePPO.load(args.model_path, env=env)

    obs = env.reset()
    rewards, successes = [], []
    best_reward, best_info = -float("inf"), None

    while len(rewards) < args.n_episodes:
        masks = get_action_masks(env)
        actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, step_rewards, dones, infos = env.step(actions)
        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                if "episode" in info:
                    rewards.append(float(info["episode"]["r"]))
                    successes.append(int(info.get("success", False)))
                    if info.get("success") and info["episode"]["r"] > best_reward:
                        best_reward = float(info["episode"]["r"])
                        best_info = info
                    print(f"  episode {len(rewards)}/{args.n_episodes}: "
                          f"reward={info['episode']['r']:.4f} success={info.get('success')}")

    env.close()

    rewards = rewards[: args.n_episodes]
    print("\n" + "=" * 60)
    print(f"Transfer eval: model trained elsewhere → benchmark={args.benchmark}")
    print(f"Episodes        : {len(rewards)}")
    print(f"Success rate    : {sum(successes)/len(successes):.2%}")
    print(f"Avg reward      : {np.mean(rewards):.5f}")
    print(f"Best reward     : {best_reward:.5f}")
    if best_info:
        print(f"Best wirelength : {best_info.get('wirelength')} (baseline {trad_metrics['wirelength']})")
        print(f"Best delay (ns) : {best_info.get('delay_ns')} (baseline {trad_metrics['delay_ns']})")
        print(f"Best power (W)  : {best_info.get('power_w')} (baseline {trad_metrics['power_w']})")
        print(f"Best routing area: {best_info.get('routing_area')} (baseline {trad_metrics['routing_area']})")
        print(f"Best DSPs       : {best_info.get('placed_dsps')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
