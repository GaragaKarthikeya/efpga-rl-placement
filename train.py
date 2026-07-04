#!/usr/bin/env python3
"""
Vanilla RL training entry point for FPGA placement optimisation.

Usage:
    python train.py --benchmarks diffeq1
    python train.py --benchmarks diffeq1,diffeq2,softmax --seed 1337 --max_episodes 500 --n_envs 8
    python train.py --help
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_env_file
from src.training.trainer import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Vanilla RL — FPGA placement optimisation")
    p.add_argument("--benchmarks",   default="diffeq1",  help="Comma-separated benchmark names actually sampled during training (e.g. diffeq1,diffeq2,softmax)")
    p.add_argument("--universe_benchmarks", default=None, help="Comma-separated benchmark names used to size the shared canvas/graph caps (superset of --benchmarks, e.g. to include held-out benchmarks the model must stay loadable against). Defaults to --benchmarks.")
    p.add_argument("--n_envs",       type=int,           help="Parallel env workers (default: min(8, cpu_count))")
    p.add_argument("--timesteps",    type=int, default=10_000, help="Total training timesteps")
    p.add_argument("--lr",           type=float, default=3e-4,  help="PPO learning rate")
    p.add_argument("--ent_coef",     type=float, default=0.0,   help="Entropy coefficient")
    p.add_argument("--batch_size",   type=int,   default=64,    help="Minibatch size")
    p.add_argument("--n_steps",      type=int,   default=80,    help="Steps per env per rollout")
    p.add_argument("--n_epochs",     type=int,   default=10,    help="PPO epochs per rollout")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed")
    p.add_argument("--max_episodes", type=int,   default=256,   help="Episode budget")
    p.add_argument("--ar_weight",    type=float, default=1.0,   help="Routing-area reward weight")
    p.add_argument("--dl_weight",    type=float, default=1.0,   help="Delay reward weight")
    p.add_argument("--pw_weight",    type=float, default=1.0,   help="Power reward weight")
    p.add_argument("--wl_weight",    type=float, default=0.0,   help="Wirelength reward weight")
    p.add_argument("--save_path",    default=None,              help="Save trained model (.zip)")
    p.add_argument("--load_path",    default=None,              help="Load existing model (.zip)")
    p.add_argument("--log_suffix",   default="",                help="Suffix for output log files")
    p.add_argument("--cache_suffix", default="",                help="Suffix for the VTR layout cache DB filename (isolate cache across side-by-side runs on the same benchmark)")
    p.add_argument("--vtr_timeout",  type=int, default=600,     help="Per-episode VTR subprocess timeout in seconds (lower bounds the stall a degenerate aspect-ratio/placement combo can cause across all parallel workers)")
    p.add_argument("--no_wandb",     action="store_true",       help="Disable W&B logging (on by default)")
    p.add_argument("--wandb_project", default="fpga-placement-gnn", help="W&B project name")
    p.add_argument("--wandb_entity", default=None,              help="W&B entity/team (default: your default entity)")
    args = p.parse_args()

    return TrainConfig(
        benchmark_names=[b.strip() for b in args.benchmarks.split(",") if b.strip()],
        universe_benchmark_names=(
            [b.strip() for b in args.universe_benchmarks.split(",") if b.strip()]
            if args.universe_benchmarks
            else None
        ),
        n_envs=args.n_envs,
        timesteps=args.timesteps,
        lr=args.lr,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        seed=args.seed,
        max_episodes=args.max_episodes,
        wl_weight=args.wl_weight,
        pw_weight=args.pw_weight,
        dl_weight=args.dl_weight,
        ar_weight=args.ar_weight,
        save_path=args.save_path,
        load_path=args.load_path,
        log_suffix=args.log_suffix,
        cache_suffix=args.cache_suffix,
        vtr_timeout=args.vtr_timeout,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass  # optional; paths can be set via env vars

    train(parse_args())
