#!/usr/bin/env python3
"""
Zero-shot transfer experiment: take a policy trained on one benchmark and run it
(no further training) on a different benchmark, comparing against a random-valid-action
baseline and the traditional VTR placement.

Usage:
    python eval_zero_shot_transfer.py --model runs/diffeq2_fullscale_seed42.zip \
        --target diffeq1 --n_random 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_env_file
from src.env.fpga_env import FPGAEnv
from src.training.ppo import CustomMaskablePPO


def load_benchmark_env(benchmark: str, cache_db_path: str) -> FPGAEnv:
    res = json.loads((PROJECT_ROOT / "baselines" / f"{benchmark}_traditional_resources.txt").read_text())
    metric = json.loads((PROJECT_ROOT / "baselines" / f"{benchmark}_traditional_metric.txt").read_text())
    width, height = res["fpga_size"]
    req_dsp = res["requirements"]["dsp"]
    req_bram = res["requirements"]["bram"]

    net_count_data = {}
    net_count_file = PROJECT_ROOT / f"{benchmark}_netlist_info.json"
    if net_count_file.is_file():
        for k, v in json.loads(net_count_file.read_text()).items():
            net_count_data[eval(k) if k.startswith("('") else k] = v

    dsp_block_names, bram_block_names = [], []
    net_file = PROJECT_ROOT / "runs" / f"{benchmark}_traditional" / f"{benchmark}.net"
    if net_file.is_file():
        from src.netlist.parser import parse_net_file
        parsed = parse_net_file(net_file)
        dsp_parsed = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "dsp"]
        bram_parsed = [(b.atom_name, b.unique_nets) for b in parsed if b.block_type == "bram"]
        dsp_block_names = [n for n, _ in sorted(dsp_parsed, key=lambda x: x[1], reverse=True)]
        bram_block_names = [n for n, _ in sorted(bram_parsed, key=lambda x: x[1], reverse=True)]
        if len(dsp_block_names) != req_dsp:
            dsp_block_names = []
        if len(bram_block_names) != req_bram:
            bram_block_names = []

    env = FPGAEnv(
        benchmark_name=benchmark,
        width=width, height=height,
        req_dsp=req_dsp, req_bram=req_bram,
        traditional_metrics=metric,
        cache_db_path=cache_db_path,
        wl_weight=0.0, pw_weight=1.0, dl_weight=1.0, ar_weight=1.0,
        net_count_data=net_count_data,
        use_net_count_sort=True,
        dsp_block_names=dsp_block_names or None,
        bram_block_names=bram_block_names or None,
    )
    return env, metric


def run_episode(env: FPGAEnv, policy=None, rng: np.random.Generator = None):
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        mask = env.action_masks()
        if policy is not None:
            action, _ = policy.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
        else:
            valid = np.where(mask)[0]
            action = int(rng.choice(valid))
        obs, reward, done, _, info = env.step(action)
    return reward, info


def adp(area, delay, power):
    return area * delay * power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained model .zip (e.g. trained on diffeq2)")
    ap.add_argument("--target", required=True, help="Target benchmark to transfer to (e.g. diffeq1)")
    ap.add_argument("--n_random", type=int, default=20, help="Number of random-policy trials")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    load_env_file(PROJECT_ROOT / ".env")

    cache_db = str(PROJECT_ROOT / "runs" / f"vtr_layout_cache_{args.target}_zeroshot.db")
    env, baseline_metric = load_benchmark_env(args.target, cache_db)

    print(f"Target benchmark   : {args.target}")
    print(f"Grid               : {env.width}x{env.height}, DSP={env.req_dsp}, BRAM={env.req_bram}")
    print(f"Baseline           : {baseline_metric}")
    print("=" * 70)

    # --- RL zero-shot (deterministic, single rollout) ---
    print(f"\nLoading model from {args.model} (no further training)...")
    model = CustomMaskablePPO.load(args.model, env=env)
    rl_reward, rl_info = run_episode(env, policy=model)
    print(f"[RL zero-shot] reward={rl_reward:.5f}  success={rl_info.get('success')}")
    print(f"  wirelength={rl_info.get('wirelength')}  delay_ns={rl_info.get('delay_ns')}  "
          f"power_w={rl_info.get('power_w')}  routing_area={rl_info.get('routing_area')}")
    print(f"  dsps={rl_info.get('placed_dsps')}  aspect_ratio={rl_info.get('aspect_ratio')}")

    # --- Random-valid-action baseline ---
    print(f"\nRunning {args.n_random} random-valid-action trials...")
    rng = np.random.default_rng(args.seed)
    random_results = []
    for i in range(args.n_random):
        r, info = run_episode(env, policy=None, rng=rng)
        random_results.append((r, info))
        status = "ok" if info.get("success") else "FAIL"
        print(f"  [{i+1:2d}/{args.n_random}] reward={r:.5f}  ({status})")

    succ = [(r, info) for r, info in random_results if info.get("success")]
    print(f"\nRandom: {len(succ)}/{args.n_random} succeeded")

    results = {"target": args.target, "baseline_metric": baseline_metric}

    if succ:
        rewards = [r for r, _ in succ]
        best_r, best_info = max(succ, key=lambda x: x[0])
        results["random"] = {
            "avg_reward": float(np.mean(rewards)),
            "best_reward": float(best_r),
            "best_info": {k: best_info.get(k) for k in
                          ("wirelength", "delay_ns", "power_w", "routing_area", "placed_dsps", "aspect_ratio")},
        }

    results["rl_zero_shot"] = {
        "reward": rl_reward,
        "success": rl_info.get("success"),
        "info": {k: rl_info.get(k) for k in
                  ("wirelength", "delay_ns", "power_w", "routing_area", "placed_dsps", "aspect_ratio")},
    }

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"SUMMARY — zero-shot transfer to {args.target}")
    print("=" * 70)
    bm = baseline_metric
    base_adp = adp(bm["routing_area"], bm["delay_ns"], bm["power_w"])
    print(f"{'':20s} {'area':>12s} {'delay_ns':>10s} {'power_w':>10s} {'ADP':>14s} {'reward':>8s}")
    print(f"{'Traditional':20s} {bm['routing_area']:>12.0f} {bm['delay_ns']:>10.3f} {bm['power_w']:>10.6f} {base_adp:>14.1f} {'--':>8s}")

    if succ:
        bi = results["random"]["best_info"]
        r_adp = adp(bi["routing_area"], bi["delay_ns"], bi["power_w"])
        print(f"{'Random (best of '+str(args.n_random)+')':20s} {bi['routing_area']:>12.0f} {bi['delay_ns']:>10.3f} {bi['power_w']:>10.6f} {r_adp:>14.1f} {results['random']['best_reward']:>8.4f}")
        print(f"{'Random (avg)':20s} {'':>12s} {'':>10s} {'':>10s} {'':>14s} {results['random']['avg_reward']:>8.4f}")

    ri = results["rl_zero_shot"]["info"]
    if rl_info.get("success"):
        rl_adp = adp(ri["routing_area"], ri["delay_ns"], ri["power_w"])
        print(f"{'RL zero-shot':20s} {ri['routing_area']:>12.0f} {ri['delay_ns']:>10.3f} {ri['power_w']:>10.6f} {rl_adp:>14.1f} {rl_reward:>8.4f}")
    else:
        print(f"{'RL zero-shot':20s} FAILED ({rl_info.get('error', rl_info.get('status'))})")

    out_path = PROJECT_ROOT / f"zero_shot_transfer_{args.target}_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nSaved → {out_path.name}")


if __name__ == "__main__":
    main()
