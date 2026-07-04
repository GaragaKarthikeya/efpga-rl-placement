#!/usr/bin/env python3

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import wandb
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from src.env.fpga_env import FPGAEnv, build_benchmark_configs, compute_max_dims
from src.training.callbacks import BestLayoutCallback
from src.training.gnn_extractor import GNNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    benchmark_names: list[str] = field(default_factory=lambda: ["diffeq1"])
    n_envs: Optional[int] = None
    timesteps: int = 10_000
    lr: float = 3e-4
    ent_coef: float = 0.0
    batch_size: int = 64
    n_steps: int = 80
    n_epochs: int = 10
    gamma: float = 0.99
    seed: int = 42
    max_episodes: int = 256
    ar_weight: float = 1.0
    dl_weight: float = 1.0
    pw_weight: float = 1.0
    wl_weight: float = 0.0
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    log_suffix: str = ""
    cache_suffix: str = ""
    universe_benchmark_names: Optional[list[str]] = None
    vtr_timeout: int = 300
    use_wandb: bool = True
    wandb_project: str = "fpga-placement-gnn"
    wandb_entity: Optional[str] = None


def _bench_label(benchmark_names: list[str]) -> str:
    # W&B's GroupName has a 128-char limit; joining many/long benchmark
    # names (e.g. 13 names -> 137 chars) can exceed it, so collapse to
    # "multiN" beyond a small count instead of listing every name.
    return "-".join(benchmark_names) if len(benchmark_names) <= 3 else f"multi{len(benchmark_names)}"


def _run_name(benchmark_names: list[str], cfg: TrainConfig) -> str:
    return f"{_bench_label(benchmark_names)}_seed{cfg.seed}{cfg.log_suffix}"


def train(cfg: TrainConfig) -> None:
    """Entry point for a full training run across a mix of benchmarks."""
    benchmark_names = [b.removesuffix(".v") for b in cfg.benchmark_names]
    universe_names = (
        [b.removesuffix(".v") for b in cfg.universe_benchmark_names]
        if cfg.universe_benchmark_names
        else benchmark_names
    )

    try:
        max_width, max_height, max_nodes, max_edges = compute_max_dims(universe_names)
        configs = build_benchmark_configs(
            benchmark_names, max_width, max_height, max_nodes, max_edges, cache_suffix=cfg.cache_suffix
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading benchmark configs: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"Benchmarks (train) : {', '.join(benchmark_names)}")
    if universe_names != benchmark_names:
        print(f"Universe (sizing)  : {', '.join(universe_names)}")
    print(f"Shared canvas      : {max_width}x{max_height}  (+2 IO ring per benchmark when baked)")
    print(f"Reduced graph caps : MAX_NODES={max_nodes}  MAX_EDGES={max_edges}")
    for c in configs:
        print(
            f"  {c.name:<16} grid={c.width}x{c.height}  dsp={c.req_dsp} bram={c.req_bram}  "
            f"baseline wl={c.traditional_metrics.get('wirelength')} "
            f"dl={c.traditional_metrics.get('delay_ns')} pw={c.traditional_metrics.get('power_w')}"
        )
    print(f"Reward weights     : wl={cfg.wl_weight} pw={cfg.pw_weight} dl={cfg.dl_weight} ar={cfg.ar_weight}")
    print("=" * 60)

    n_envs = cfg.n_envs or min(8, os.cpu_count() or 1)
    print(f"Parallel workers   : {n_envs}")

    run_name = _run_name(benchmark_names, cfg)
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            group=_bench_label(benchmark_names),
            tags=benchmark_names,
            config={
                **{k: v for k, v in asdict(cfg).items() if k not in ("wandb_project", "wandb_entity")},
                "n_envs": n_envs,
                "max_width": max_width,
                "max_height": max_height,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
            },
            dir=str(PROJECT_ROOT / "runs"),
        )
        print(f"W&B run            : {wandb_run.url}")
        print("=" * 60)

    env_kwargs = {
        "benchmark_configs": configs,
        "max_width": max_width,
        "max_height": max_height,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
        "wl_weight": cfg.wl_weight,
        "pw_weight": cfg.pw_weight,
        "dl_weight": cfg.dl_weight,
        "ar_weight": cfg.ar_weight,
        "vtr_timeout": cfg.vtr_timeout,
    }

    env = make_vec_env(FPGAEnv, n_envs=n_envs, seed=cfg.seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    best_layout_cb = BestLayoutCallback(
        benchmark_configs=configs,
        seed_val=cfg.seed,
        max_episodes=cfg.max_episodes,
        log_suffix=cfg.log_suffix,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, cfg.n_steps * 5),
        save_path=str(PROJECT_ROOT / "runs" / f"checkpoints_seed_{cfg.seed}"),
        name_prefix="multi_model",
    )

    callbacks = [best_layout_cb, checkpoint_cb]
    if wandb_run is not None:
        # No gradient/parameter histogram logging (log=None, gradient_save_freq=0) —
        # per-step histograms add real overhead over a long run and we already log
        # the 7 PPO stability metrics directly via wandb.log() in ppo.py. Model
        # snapshots are saved sparsely (a handful over the whole run) rather than
        # every few rollouts; the local CheckpointCallback above stays frequent
        # since that's cheap disk-only I/O.
        callbacks.append(
            WandbCallback(
                verbose=1,
                model_save_path=str(PROJECT_ROOT / "runs" / f"wandb_models_seed_{cfg.seed}"),
                model_save_freq=max(1, cfg.timesteps // 5),
                gradient_save_freq=0,
                log=None,
            )
        )

    ppo_kwargs = dict(
        learning_rate=cfg.lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        seed=cfg.seed,
        verbose=1,
        policy_kwargs={"features_extractor_class": GNNFeaturesExtractor},
    )

    if cfg.load_path and os.path.exists(cfg.load_path):
        print(f"Loading model from {cfg.load_path}")
        model = CustomMaskablePPO.load(cfg.load_path, env=env, **ppo_kwargs)
    else:
        model = CustomMaskablePPO("MultiInputPolicy", env, **ppo_kwargs)

    print(f"\nStarting training — target timesteps: {cfg.timesteps}")
    print(f"Rollout buffer size: {n_envs * cfg.n_steps} steps per update")
    print("=" * 60)

    try:
        model.learn(total_timesteps=cfg.timesteps, callback=CallbackList(callbacks))
        print("\nTraining complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if cfg.save_path:
            model.save(cfg.save_path)
            print(f"Model saved to {cfg.save_path}")

        _save_all_layouts(best_layout_cb, cfg)
        env.close()
        if wandb_run is not None:
            wandb.finish()


def _save_all_layouts(callback: BestLayoutCallback, cfg: TrainConfig) -> None:
    layouts = callback.all_completed_layouts[: cfg.max_episodes]
    dest = PROJECT_ROOT / f"layouts_multi_seed_{cfg.seed}{cfg.log_suffix}.json"
    try:
        dest.write_text(json.dumps(layouts, indent=4))
        print(f"Saved {len(layouts)} layouts → {dest.name}")
    except Exception as exc:
        print(f"Error saving layouts: {exc}", file=sys.stderr)
