#!/usr/bin/env python3

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback

from src.env.fpga_env import BenchmarkConfig
from src.utils.config import VTRPaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BestLayoutCallback(BaseCallback):
    """
    Tracks and saves the best FPGA layout found during training, per
    benchmark (a single training run mixes episodes across all benchmarks
    in `benchmark_configs`, so "best" is tracked separately for each one
    rather than as one global best across incomparable benchmarks).

    Logs every completed episode to a .jsonl file, saves each benchmark's
    best-ever layout as a baked architecture XML, and prints 2-minute
    progress reports including PPO stability metrics.
    """

    def __init__(
        self,
        benchmark_configs: list[BenchmarkConfig],
        seed_val: int,
        max_episodes: int = 256,
        log_suffix: str = "",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._configs_by_name = {cfg.name: cfg for cfg in benchmark_configs}
        self.seed_val = seed_val
        self.max_episodes = max_episodes
        self.log_suffix = log_suffix

        self.best_reward: dict[str, float] = {}
        self.best_info: dict[str, dict] = {}
        self.start_time = time.time()

        self.episode_rewards: list[float] = []
        self.interval_rewards: list[float] = []
        self.all_completed_layouts: list[dict] = []
        self.last_report_time = time.time()

        # Early-stopping state
        self._patience = 4
        self._min_improvement = 0.001
        self._last_avg_reward = -float("inf")
        self._stagnant_count = 0
        self._episodes_checked = 0

    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for idx, info in enumerate(infos):
            bname = info.get("benchmark_name")

            if "episode" in info:
                ep_rew = float(info["episode"]["r"])
                self.episode_rewards.append(ep_rew)
                self.interval_rewards.append(ep_rew)

                def _to_py(obj):
                    if isinstance(obj, (list, tuple)):
                        return [_to_py(i) for i in obj]
                    return obj.item() if hasattr(obj, "item") else obj

                layout_record = {
                    "benchmark_name": bname,
                    "aspect_ratio": float(info.get("aspect_ratio", 1.0)),
                    "dsps": _to_py(info.get("placed_dsps", [])),
                    "brams": _to_py(info.get("placed_brams", [])),
                    "reward": ep_rew,
                    "wirelength": float(info.get("wirelength", float("inf"))),
                    "delay_ns": float(info.get("delay_ns", float("inf"))),
                    "power_w": float(info.get("power_w", float("inf"))),
                    "routing_area": float(info.get("routing_area")) if info.get("routing_area") not in ("?", None) else float("inf"),
                    "success": int(info.get("success", False)),
                }
                self.all_completed_layouts.append(layout_record)
                self._write_jsonl(layout_record)

                if wandb.run is not None:
                    log_dict = {"episode/reward": ep_rew, "episode/success": int(info.get("success", False))}
                    if bname:
                        log_dict[f"episode_reward/{bname}"] = ep_rew
                    wandb.log(log_dict, step=self.num_timesteps)

            if info.get("success") and bname and rewards[idx] > self.best_reward.get(bname, -float("inf")):
                self.best_reward[bname] = rewards[idx]
                self.best_info[bname] = info
                self._print_new_best(bname, rewards[idx], info)
                self._save_best_layout(bname, info)
                self._log_best_to_wandb(bname, rewards[idx], info)

        if len(self.episode_rewards) >= self.max_episodes:
            print(f"\n[INFO] Reached {len(self.episode_rewards)} episodes. Stopping.")
            return False

        if len(self.episode_rewards) - self._episodes_checked >= 100:
            self._check_convergence()

        if time.time() - self.last_report_time >= 120.0:
            self._print_progress_report()

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_jsonl(self, record: dict) -> None:
        log_path = PROJECT_ROOT / f"all_layouts_multi_seed_{self.seed_val}{self.log_suffix}.jsonl"

        def convert_to_serializable(obj):
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, 'item'):  # numpy types
                return obj.item()
            return obj

        serializable_record = convert_to_serializable(record)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(serializable_record) + "\n")

    def _check_convergence(self) -> None:
        recent = self.episode_rewards[-100:]
        current_avg = sum(recent) / len(recent)

        if current_avg - self._last_avg_reward < self._min_improvement:
            self._stagnant_count += 1
            print(
                f"\n[INFO] Convergence: no improvement ({current_avg:.4f} vs {self._last_avg_reward:.4f}). "
                f"Stagnant {self._stagnant_count}/{self._patience}"
            )
        else:
            print(f"\n[INFO] Convergence: improved ({current_avg:.4f} vs {self._last_avg_reward:.4f}).")
            self._stagnant_count = 0
            self._last_avg_reward = current_avg

        self._episodes_checked = len(self.episode_rewards)

        if wandb.run is not None:
            wandb.log({"convergence/avg_reward_last100": current_avg}, step=self.num_timesteps)

        if self._stagnant_count >= self._patience:
            print(f"\n[INFO] Early stopping at {len(self.episode_rewards)} episodes.")
            return  # caller checks return value; can't stop here directly

    def _print_new_best(self, bname: str, reward: float, info: dict) -> None:
        print("\n" + "=" * 50)
        print(f"[NEW BEST — {bname} — SEED {self.seed_val}]")
        print(f"  Reward       : {reward:.5f}")
        print(f"  Wirelength   : {info.get('wirelength')}")
        print(f"  Delay (ns)   : {info.get('delay_ns')}")
        print(f"  Power (W)    : {info.get('power_w')}")
        print(f"  Grid         : {info.get('grid_W')}x{info.get('grid_H')}")
        print(f"  Routing area : {info.get('routing_area')}")
        print(f"  DSPs         : {info.get('placed_dsps')}")
        print(f"  BRAMs        : {info.get('placed_brams')}")
        print(f"  Cached       : {info.get('cached')}")
        print(f"  Elapsed      : {time.time() - self.start_time:.1f}s")
        print("=" * 50)

    def _print_progress_report(self) -> None:
        now = time.time()
        elapsed_min = (now - self.start_time) / 60.0
        n_interval = len(self.interval_rewards)
        n_total = len(self.episode_rewards)

        print("\n" + "=" * 70)
        print(f"[2-MIN REPORT — SEED {self.seed_val} — {time.strftime('%H:%M:%S')}]")
        print(f"  Elapsed      : {elapsed_min:.1f} min")
        print(f"  This window  : {n_interval} episodes", end="")
        if n_interval:
            print(f", avg reward = {sum(self.interval_rewards)/n_interval:.5f}")
        else:
            print()
        print(f"  Total        : {n_total} episodes", end="")
        if n_total:
            print(f", avg reward = {sum(self.episode_rewards)/n_total:.5f}")
        else:
            print()

        print("-" * 70)
        self._print_stability_metrics()

        print("-" * 70)
        self._print_best_vs_baseline()
        print("=" * 70)

        self.interval_rewards = []
        self.last_report_time = now

    def _print_stability_metrics(self) -> None:
        metrics = getattr(self.model, "latest_metrics", None)
        if metrics:
            print("  PPO Stability Metrics:")
            for k, v in metrics.items():
                print(f"    {k:<35}: {v:.6f}")
        else:
            print("  PPO stability metrics not yet available.")

    def _print_best_vs_baseline(self) -> None:
        if not self.best_reward:
            print("  No successful layout yet.")
            return

        for bname, reward in self.best_reward.items():
            tm = self._configs_by_name[bname].traditional_metrics
            bi = self.best_info[bname]
            print(f"  [{bname}] best reward: {reward:.5f}")
            for metric, label, key in [
                ("wirelength", "Wirelength", "wirelength"),
                ("delay_ns",   "Delay (ns)", "delay_ns"),
                ("power_w",    "Power (W)",  "power_w"),
            ]:
                best_val = bi.get(metric)
                base_val = tm.get(key)
                if best_val is not None and base_val:
                    imp = (base_val - best_val) / base_val * 100.0
                    print(f"    {label:<12}: {best_val}  (baseline {base_val}, {imp:+.2f}%)")

    def _log_best_to_wandb(self, bname: str, reward: float, info: dict) -> None:
        if wandb.run is None:
            return
        tm = self._configs_by_name[bname].traditional_metrics
        log_dict = {f"best_reward/{bname}": reward}
        for metric, key in [("wirelength", "wirelength"), ("delay_ns", "delay_ns"), ("power_w", "power_w")]:
            val = info.get(metric)
            base_val = tm.get(key)
            if val is not None and base_val:
                log_dict[f"best_{metric}/{bname}"] = val
                log_dict[f"best_{metric}_improvement_pct/{bname}"] = (base_val - val) / base_val * 100.0
        wandb.log(log_dict, step=self.num_timesteps)

    def _save_best_layout(self, bname: str, info: dict) -> None:
        try:
            from src.layout.baker import bake_layout

            cfg = self._configs_by_name[bname]
            arch_dest = PROJECT_ROOT / f"best_baked_layout_{bname}{self.log_suffix}.xml"
            constraints_dest = PROJECT_ROOT / f"best_layout_constraints_{bname}{self.log_suffix}.xml"

            all_block_names = cfg.dsp_block_names + cfg.bram_block_names

            bake_layout(
                benchmark_name=bname,
                dsps=info["placed_dsps"],
                mems=info["placed_brams"],
                width=cfg.width + 2,
                height=cfg.height + 2,
                output_path=str(arch_dest),
                aspect_ratio=info.get("aspect_ratio"),
                block_names=all_block_names if all_block_names else None,
                constraints_output_path=str(constraints_dest) if all_block_names else None,
            )

            def make_serializable(obj):
                if isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                return obj

            coords_dest = PROJECT_ROOT / f"best_layout_coordinates_{bname}{self.log_suffix}.txt"
            coords_dest.write_text(
                json.dumps(
                    {
                        "reward": self.best_reward[bname],
                        "wirelength": info.get("wirelength"),
                        "delay_ns": info.get("delay_ns"),
                        "power_w": info.get("power_w"),
                        "routing_area": info.get("routing_area"),
                        "dsps": make_serializable(info["placed_dsps"]),
                        "brams": make_serializable(info["placed_brams"]),
                        "aspect_ratio": info.get("aspect_ratio"),
                        "grid_W": info.get("grid_W"),
                        "grid_H": info.get("grid_H"),
                        "elapsed_s": time.time() - self.start_time,
                    },
                    indent=4,
                )
            )
            print(f"Saved best arch → {arch_dest.name}")
            print(f"Saved best coords → {coords_dest.name}")
            if all_block_names:
                print(f"Saved best constraints → {constraints_dest.name}")

            self._launch_best_vtr_run(bname, info, arch_dest, constraints_dest if all_block_names else None)

        except Exception as exc:
            import sys
            print(f"Error saving best layout for {bname}: {exc}", file=sys.stderr)

    def _launch_best_vtr_run(self, bname: str, info: dict, arch_path: Path, constraints_path: Optional[Path] = None) -> None:
        paths = VTRPaths()
        if not paths.is_flow_available:
            return

        best_run_dir = PROJECT_ROOT / "runs" / f"best_run_{bname}{self.log_suffix}"
        if best_run_dir.exists():
            shutil.rmtree(best_run_dir)
        best_run_dir.mkdir(parents=True, exist_ok=True)

        benchmark_file = PROJECT_ROOT / "benchmarks" / f"{bname}.v"
        cmd = [str(paths.python), str(paths.flow_script), str(benchmark_file), str(arch_path),
               "-temp_dir", str(best_run_dir)]
        if constraints_path and constraints_path.is_file():
            cmd += ["-read_vpr_constraints", str(constraints_path)]
        if paths.has_power_tech:
            cmd += ["-cmos_tech", str(paths.power_tech_file)]

        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Launched background VTR → {best_run_dir.name}")
