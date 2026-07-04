#!/usr/bin/env python3

import json
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.cache import CacheRow, LayoutCache
from src.utils.config import VTRPaths
from src.evaluation.vtr_runner import VTRRunner
from src.netlist.graph_reduction import ReducedGraph, reduce_netlist_graph, pad_graph

# Project root is two levels above this file (src/env/fpga_env.py → /)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# DSP tile occupies 4 rows; BRAM tile occupies 6 rows
_BLOCK_HEIGHT = {1: 4, 2: 6}

# Predefined aspect ratios offered as the first action
ASPECT_RATIOS = [round(0.1 * i, 1) for i in range(1, 21)]  # 0.1 … 2.0

# Sentinel value marking grid cells outside the active benchmark's real
# footprint within the shared padded canvas (distinct from occupancy values
# -1/0/1/2 already in use for occupied-tail/empty/dsp-head/bram-head).
_OUT_OF_CANVAS = -2.0


@dataclass
class BenchmarkConfig:
    """Everything FPGAEnv needs to run episodes for one benchmark, pre-loaded
    and pre-padded to the shared MAX_NODES/MAX_EDGES so reset() is cheap."""

    name: str
    width: int  # core grid width (without +2 IO ring)
    height: int
    req_dsp: int
    req_bram: int
    traditional_metrics: dict
    dsp_block_names: list[str]
    bram_block_names: list[str]
    reduced_graph: ReducedGraph
    padded_node_features: np.ndarray
    padded_edge_index: np.ndarray
    padded_edge_weight: np.ndarray
    cache_db_path: str


def _load_raw_benchmark(benchmark_name: str) -> tuple[dict, dict, list[str], list[str], ReducedGraph]:
    """Load baseline metrics/resources, VPR block-name constraints, and the
    reduced netlist graph for one benchmark. Raises if prerequisite files
    are missing (baselines/, runs/{name}_traditional/{name}.net,
    {name}_netlist_graph.json — the latter two from
    `extract_netlist_info.py --benchmark {name} --include-graph`)."""
    res_file = PROJECT_ROOT / "baselines" / f"{benchmark_name}_traditional_resources.txt"
    metric_file = PROJECT_ROOT / "baselines" / f"{benchmark_name}_traditional_metric.txt"
    res_data = json.loads(res_file.read_text())
    metric_data = json.loads(metric_file.read_text())

    width, height = res_data["fpga_size"]
    reqs = res_data.get("requirements", {})
    req_dsp, req_bram = reqs.get("dsp", 0), reqs.get("bram", 0)

    graph_file = PROJECT_ROOT / f"{benchmark_name}_netlist_graph.json"
    reduced_graph = reduce_netlist_graph(graph_file)
    if reduced_graph.req_dsp != req_dsp or reduced_graph.req_bram != req_bram:
        raise ValueError(
            f"{benchmark_name}: netlist graph has {reduced_graph.req_dsp} DSP / "
            f"{reduced_graph.req_bram} BRAM but resources require {req_dsp} / {req_bram}"
        )

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

    return res_data, metric_data, dsp_block_names, bram_block_names, reduced_graph


def compute_max_dims(benchmark_names: list[str]) -> tuple[int, int, int, int]:
    """Compute MAX_WIDTH/MAX_HEIGHT/MAX_NODES/MAX_EDGES across a benchmark
    universe. Call this once over the full set of benchmarks a trained model
    must ever be loadable against (including ones never actually sampled
    during training) — see `build_benchmark_configs`."""
    raw = {name: _load_raw_benchmark(name) for name in benchmark_names}

    max_width = max(res["fpga_size"][0] for res, _, _, _, _ in raw.values())
    max_height = max(res["fpga_size"][1] for res, _, _, _, _ in raw.values())
    max_nodes = max(g.num_nodes for *_, g in raw.values())
    max_edges = max(g.num_edges for *_, g in raw.values())
    return max_width, max_height, max_nodes, max_edges


def build_benchmark_configs(
    benchmark_names: list[str],
    max_width: int,
    max_height: int,
    max_nodes: int,
    max_edges: int,
    cache_suffix: str = "",
) -> list[BenchmarkConfig]:
    """Load benchmarks and pad them to the given shared MAX_WIDTH/MAX_HEIGHT/
    MAX_NODES/MAX_EDGES. These dims are NOT recomputed from `benchmark_names`
    here — pass dims from `compute_max_dims()` run over the full benchmark
    universe (which may be a superset of `benchmark_names`, e.g. when a
    training run only samples a subset but the model must stay loadable
    against held-out benchmarks too).

    `cache_suffix` isolates the VTR layout cache DB (e.g. for side-by-side
    scratch-vs-fine-tuned runs on the same benchmark, so cache hits from one
    run don't leak into the other's apparent convergence speed)."""
    configs = []
    for name in benchmark_names:
        res_data, metric_data, dsp_names, bram_names, graph = _load_raw_benchmark(name)
        width, height = int(res_data["fpga_size"][0]), int(res_data["fpga_size"][1])
        if width > max_width or height > max_height:
            raise ValueError(
                f"{name}: grid {width}x{height} exceeds max canvas {max_width}x{max_height}"
            )
        node_features, edge_index, edge_weight = pad_graph(graph, max_nodes, max_edges)
        configs.append(
            BenchmarkConfig(
                name=name,
                width=width,
                height=height,
                req_dsp=graph.req_dsp,
                req_bram=graph.req_bram,
                traditional_metrics=metric_data,
                dsp_block_names=dsp_names,
                bram_block_names=bram_names,
                reduced_graph=graph,
                padded_node_features=node_features,
                padded_edge_index=edge_index.astype(np.float32),
                padded_edge_weight=edge_weight,
                cache_db_path=str(PROJECT_ROOT / "runs" / f"vtr_layout_cache_{name}{cache_suffix}.db"),
            )
        )
    return configs


class FPGAEnv(gym.Env):
    """
    Gymnasium environment for heterogeneous FPGA block placement across a
    mix of benchmarks of different grid sizes and DSP/BRAM counts.

    Episode structure
    -----------------
    Step 0          : select an aspect ratio from ASPECT_RATIOS
    Steps 1 … N     : place each BRAM (sorted by nets) then each DSP (sorted by nets)
    Terminal step   : VTR evaluates the layout; reward is returned

    `reset()` samples one BenchmarkConfig uniformly from `benchmark_configs`
    for the episode. Observation/action spaces are fixed across all
    benchmarks (sized to MAX_WIDTH/MAX_HEIGHT/MAX_NODES/MAX_EDGES, the max
    over the whole mix); action masking and the `valid_wh` observation key
    are what tell the policy which subset of the shared canvas is real for
    the active benchmark this episode.

    Observation space : Dict
        grid               — Box(MAX_W, MAX_H, 4): occupancy / block-type hint /
                              aspect ratio / net count, as in the single-benchmark
                              env, with cells outside the active benchmark's real
                              footprint marked with the _OUT_OF_CANVAS sentinel.
        node_features      — Box(MAX_NODES, 4): reduced netlist graph node features
                              (is_dsp, is_bram, is_fabric, net_count_normalized).
        edge_index         — Box(MAX_EDGES, 2): reduced graph edges, -1-padded.
        edge_weight        — Box(MAX_EDGES,): reduced graph edge weights.
        current_block_idx  — Box(1,): reduced-graph node id of the block being
                              placed this step (fabric node id during the
                              aspect-ratio step / after termination).
        valid_wh           — Box(2,): active benchmark's (width, height) normalized
                              by (MAX_WIDTH, MAX_HEIGHT).

    Action space : Discrete(MAX_WIDTH * MAX_HEIGHT), fixed across benchmarks.
        Step 0 maps action index → ASPECT_RATIOS index.
        Other steps decode as x = 1 + action // MAX_HEIGHT, y = 1 + action % MAX_HEIGHT
        — using the fixed MAX_HEIGHT, not the active benchmark's real height, so
        the same action index always means the same (x, y) regardless of which
        benchmark is active. Masking (bounds + overlap, keyed off the active
        benchmark's real width/height) is what restricts validity per benchmark.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark_configs: list[BenchmarkConfig],
        max_width: int,
        max_height: int,
        max_nodes: int,
        max_edges: int,
        wl_weight: float = 0.10,
        pw_weight: float = 0.30,
        dl_weight: float = 0.60,
        ar_weight: float = 0.00,
        vtr_timeout: int = 1200,
    ) -> None:
        super().__init__()

        self.benchmark_configs = benchmark_configs
        self.MAX_WIDTH = max_width
        self.MAX_HEIGHT = max_height
        self.MAX_NODES = max_nodes
        self.MAX_EDGES = max_edges
        # A bad aspect-ratio choice (more likely during early random exploration)
        # can make VPR's auto-layout device-size estimate badly mismatch the
        # actual placement coordinates, causing a single VTR call to thrash for
        # a long time. Since SubprocVecEnv.step() is synchronous, one such call
        # stalls every other parallel worker too — keep this short relative to
        # the benchmark mix's normal eval time so a degenerate episode
        # self-recovers fast instead of eating the full default 1200s.
        self.vtr_timeout = vtr_timeout

        self._caches: dict[str, LayoutCache] = {
            cfg.name: LayoutCache(cfg.cache_db_path) for cfg in benchmark_configs
        }
        self._vtr = VTRRunner(VTRPaths())

        self.wl_weight, self.pw_weight, self.dl_weight, self.ar_weight = (
            wl_weight, pw_weight, dl_weight, ar_weight,
        )

        self.action_space = spaces.Discrete(max_width * max_height)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=_OUT_OF_CANVAS, high=4.0, shape=(max_width, max_height, 4), dtype=np.float32),
            "node_features": spaces.Box(low=-1.0, high=1.0, shape=(max_nodes, 4), dtype=np.float32),
            "edge_index": spaces.Box(low=-1.0, high=float(max_nodes), shape=(max_edges, 2), dtype=np.float32),
            "edge_weight": spaces.Box(low=0.0, high=1.0, shape=(max_edges,), dtype=np.float32),
            "current_block_idx": spaces.Box(low=0.0, high=float(max_nodes), shape=(1,), dtype=np.float32),
            "valid_wh": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        })

        # Mutable episode state (initialised by reset)
        self._active_config: BenchmarkConfig = benchmark_configs[0]
        self._grid = np.zeros((self.MAX_WIDTH, self.MAX_HEIGHT), dtype=np.int8)
        self._placed_dsps: list[tuple[int, int]] = []
        self._placed_brams: list[tuple[int, int]] = []
        self._current_step: int = 0
        self._chosen_aspect_ratio: float = 1.0
        self._blocks_to_place: list[int] = []
        self._block_id_map: dict[int, int] = {}
        self._total_blocks: int = 0

    # ------------------------------------------------------------------
    # Block placement ordering
    # ------------------------------------------------------------------

    def _build_sorted_placement(self, cfg: BenchmarkConfig) -> tuple[list[int], dict[int, int]]:
        """Order: BRAMs (sorted by net count, descending) → DSPs (sorted by
        net count, descending). Net counts come straight from the reduced
        netlist graph already loaded for this benchmark."""
        graph = cfg.reduced_graph
        bram_blocks = sorted(
            ((i, graph.net_count_normalized_for("bram", i)) for i in graph.bram_instance_ids),
            key=lambda x: x[1], reverse=True,
        )
        dsp_blocks = sorted(
            ((i, graph.net_count_normalized_for("dsp", i)) for i in graph.dsp_instance_ids),
            key=lambda x: x[1], reverse=True,
        )

        blocks_to_place = [2] * cfg.req_bram + [1] * cfg.req_dsp
        block_id_map = {}
        for step_idx, (actual_id, _) in enumerate(bram_blocks):
            block_id_map[step_idx] = actual_id
        for step_idx, (actual_id, _) in enumerate(dsp_blocks, start=cfg.req_bram):
            block_id_map[step_idx] = actual_id

        return blocks_to_place, block_id_map

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = int(self.np_random.integers(0, len(self.benchmark_configs)))
        self._active_config = self.benchmark_configs[idx]
        cfg = self._active_config

        self._blocks_to_place, self._block_id_map = self._build_sorted_placement(cfg)
        self._total_blocks = len(self._blocks_to_place)

        self._grid[:] = 0
        self._placed_dsps = []
        self._placed_brams = []
        self._current_step = 0
        self._chosen_aspect_ratio = 1.0
        return self._obs(), {}

    def step(self, action: int):
        cfg = self._active_config

        if self._current_step > self._total_blocks:
            return self._obs(), 0.0, True, False, self._terminal_info(success=False, status="already_completed")

        # --- Aspect-ratio selection (step 0) ---
        if self._current_step == 0:
            if not (0 <= action < len(ASPECT_RATIOS)):
                return self._obs(), -10.0, True, False, {"status": "invalid_aspect_ratio", **self._empty_eval_info()}
            self._chosen_aspect_ratio = ASPECT_RATIOS[action]
            self._current_step += 1
            return self._obs(), 0.0, False, False, {"status": "normal"}

        # --- Block placement (steps 1 … N) ---
        block_type = self._blocks_to_place[self._current_step - 1]
        bh = _BLOCK_HEIGHT[block_type]

        x = 1 + (action // self.MAX_HEIGHT)
        y = 1 + (action % self.MAX_HEIGHT)

        if not (1 <= x <= cfg.width) or not (1 <= y) or (y + bh - 1) > cfg.height:
            return self._obs(), -10.0, True, False, {"status": "out_of_bounds", **self._empty_eval_info()}

        if any(self._grid[x - 1, y - 1 + dy] != 0 for dy in range(bh)):
            return self._obs(), -10.0, True, False, {"status": "overlap", **self._empty_eval_info()}

        for dy in range(bh):
            self._grid[x - 1, y - 1 + dy] = -1
        self._grid[x - 1, y - 1] = block_type

        if block_type == 1:
            self._placed_dsps.append((x, y))
        else:
            self._placed_brams.append((x, y))

        self._current_step += 1

        if self._current_step > self._total_blocks:
            reward, info = self._evaluate_layout(self._chosen_aspect_ratio)
            return self._obs(), reward, True, False, info

        return self._obs(), 0.0, False, False, {"status": "normal"}

    # ------------------------------------------------------------------
    # Action mask (for MaskablePPO)
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        cfg = self._active_config
        mask = np.zeros(self.MAX_WIDTH * self.MAX_HEIGHT, dtype=np.int8)

        if self._current_step == 0:
            mask[: len(ASPECT_RATIOS)] = 1
        elif self._current_step <= self._total_blocks:
            block_type = self._blocks_to_place[self._current_step - 1]
            bh = _BLOCK_HEIGHT[block_type]
            for act in range(self.MAX_WIDTH * self.MAX_HEIGHT):
                x = 1 + (act // self.MAX_HEIGHT)
                y = 1 + (act % self.MAX_HEIGHT)
                if not (1 <= x <= cfg.width):
                    continue
                if y < 1 or (y + bh - 1) > cfg.height:
                    continue
                if any(self._grid[x - 1, y - 1 + dy] != 0 for dy in range(bh)):
                    continue
                mask[act] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask().astype(bool)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, wl: float, pw: float, dl: float, routing_area: float) -> float:
        """Weighted log-ratio reward. Positive = better than baseline."""
        tm = self._active_config.traditional_metrics
        area_norm = max(routing_area, 1e-9) / max(tm.get("routing_area", 1.0), 1e-9)
        delay_norm = max(dl, 1e-9) / max(tm["delay_ns"], 1e-9)
        power_norm = max(pw, 1e-9) / max(tm["power_w"], 1e-9)
        wl_norm = max(wl, 1e-9) / max(tm.get("wirelength", 1.0), 1e-9)

        return float(
            -(
                self.ar_weight * np.log(area_norm)
                + self.dl_weight * np.log(delay_norm)
                + self.pw_weight * np.log(power_norm)
                + self.wl_weight * np.log(wl_norm)
            )
        )

    # ------------------------------------------------------------------
    # Layout evaluation (VTR)
    # ------------------------------------------------------------------

    def _evaluate_layout(self, aspect_ratio: float) -> tuple[float, dict]:
        cfg = self._active_config
        info = self._base_eval_info(aspect_ratio)
        cache_key = self._cache_key(aspect_ratio)
        cache = self._caches[cfg.name]

        cached = cache.get(cache_key)
        if cached is not None:
            info["cached"] = True
            info["success"] = cached.success
            if cached.success:
                self._fill_success_info(info, cached)
                return self._compute_reward(cached.wirelength, cached.power_w, cached.delay_ns, cached.routing_area), info
            return -10.0, info

        worker = uuid.uuid4().hex[:8]
        temp_arch = PROJECT_ROOT / f"temp_arch_{worker}.xml"
        temp_constraints = PROJECT_ROOT / f"temp_constraints_{worker}.xml"
        temp_run_dir = PROJECT_ROOT / "runs" / f"temp_run_{worker}"
        benchmark_file = PROJECT_ROOT / "benchmarks" / f"{cfg.name}.v"

        all_block_names = cfg.dsp_block_names + cfg.bram_block_names

        try:
            from src.layout.baker import bake_layout
            result = bake_layout(
                benchmark_name=cfg.name,
                dsps=self._placed_dsps,
                mems=self._placed_brams,
                width=cfg.width + 2,
                height=cfg.height + 2,
                output_path=str(temp_arch),
                aspect_ratio=aspect_ratio,
                block_names=all_block_names if all_block_names else None,
                constraints_output_path=str(temp_constraints) if all_block_names else None,
            )
            if result == -1:
                cache.put(cache_key, LayoutCache.failure_row())
                info["error"] = "bake_layout: invalid placement"
                return -10.0, info
        except Exception as exc:
            info["error"] = f"bake_layout error: {exc}"
            return -10.0, info

        if not self._vtr.paths.is_flow_available:
            info["error"] = "VTR flow script not found"
            return -10.0, info

        temp_run_dir.mkdir(parents=True, exist_ok=True)
        rc = self._vtr.run(
            benchmark_file, temp_arch, temp_run_dir,
            silent=True,
            constraints_file=temp_constraints if all_block_names else None,
            timeout=self.vtr_timeout,
        )

        vpr_out = temp_run_dir / "vpr.out"
        crit_path = temp_run_dir / "vpr.crit_path.out"
        power_file = temp_run_dir / f"{cfg.name}.power"

        if rc == 0 and vpr_out.is_file():
            metrics = VTRRunner.parse_metrics(vpr_out, crit_path, power_file)
            resources = VTRRunner.parse_resources(vpr_out)

            if metrics.is_complete():
                grid_w = resources.fpga_size[0] + 2
                grid_h = resources.fpga_size[1] + 2
                row = CacheRow(
                    delay_ns=metrics.delay_ns,
                    wirelength=metrics.wirelength,
                    power_w=metrics.power_w,
                    routing_area=metrics.routing_area,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    success=True,
                )
                cache.put(cache_key, row)
                self._fill_success_info(info, row)
                self._cleanup(temp_arch, temp_run_dir, temp_constraints)
                return self._compute_reward(metrics.wirelength, metrics.power_w, metrics.delay_ns, metrics.routing_area), info

        cache.put(cache_key, LayoutCache.failure_row())
        info["error"] = f"VTR failed (rc={rc})"
        self._cleanup(temp_arch, temp_run_dir, temp_constraints)
        return -10.0, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_block_net_count(self) -> float:
        if self._current_step == 0 or self._current_step > self._total_blocks:
            return 0.0
        step_idx = self._current_step - 1
        block_type = self._blocks_to_place[step_idx]
        block_type_str = "dsp" if block_type == 1 else "bram"
        actual_id = self._block_id_map.get(step_idx, step_idx)
        return self._active_config.reduced_graph.net_count_normalized_for(block_type_str, actual_id)

    def _current_reduced_node_id(self) -> int:
        cfg = self._active_config
        if self._current_step == 0 or self._current_step > self._total_blocks:
            return cfg.reduced_graph.fabric_node_id
        step_idx = self._current_step - 1
        block_type = self._blocks_to_place[step_idx]
        block_type_str = "dsp" if block_type == 1 else "bram"
        actual_id = self._block_id_map.get(step_idx, step_idx)
        return cfg.reduced_graph.node_id_for(block_type_str, actual_id)

    def _obs(self) -> dict:
        cfg = self._active_config

        grid = np.full((self.MAX_WIDTH, self.MAX_HEIGHT, 4), _OUT_OF_CANVAS, dtype=np.float32)
        grid[: cfg.width, : cfg.height, 0] = self._grid[: cfg.width, : cfg.height]
        grid[: cfg.width, : cfg.height, 1:4] = 0.0

        if self._current_step == 0:
            grid[: cfg.width, : cfg.height, 1] = 3
            grid[: cfg.width, : cfg.height, 2] = -1.0
            grid[: cfg.width, : cfg.height, 3] = -1.0
        elif self._current_step <= self._total_blocks:
            grid[: cfg.width, : cfg.height, 1] = self._blocks_to_place[self._current_step - 1]
            grid[: cfg.width, : cfg.height, 2] = self._chosen_aspect_ratio
            grid[: cfg.width, : cfg.height, 3] = self._get_current_block_net_count()

        return {
            "grid": grid,
            "node_features": cfg.padded_node_features,
            "edge_index": cfg.padded_edge_index,
            "edge_weight": cfg.padded_edge_weight,
            "current_block_idx": np.array([self._current_reduced_node_id()], dtype=np.float32),
            "valid_wh": np.array(
                [cfg.width / self.MAX_WIDTH, cfg.height / self.MAX_HEIGHT], dtype=np.float32
            ),
        }

    def _cache_key(self, aspect_ratio: float) -> str:
        # Order matters here, not just the set of occupied cells: position j
        # is pinned to the j-th highest-net-count instance (see
        # _build_sorted_placement / the block_names<->coords zip in
        # bake_layout), so two placements using the same cells in a
        # different order are different VPR constraint files and must not
        # collide in the cache.
        return (
            f"dsps:{list(self._placed_dsps)}"
            f"|brams:{list(self._placed_brams)}"
            f"|ratio:{aspect_ratio}"
        )

    def _base_eval_info(self, aspect_ratio: float) -> dict:
        return {
            "success": False,
            "cached": False,
            "benchmark_name": self._active_config.name,
            "placed_dsps": list(self._placed_dsps),
            "placed_brams": list(self._placed_brams),
            "aspect_ratio": aspect_ratio,
            "grid_W": "?",
            "grid_H": "?",
            "grid_clbs": "?",
            "routing_area": "?",
        }

    @staticmethod
    def _fill_success_info(info: dict, row: "CacheRow") -> None:
        info.update(
            {
                "success": True,
                "delay_ns": row.delay_ns,
                "wirelength": row.wirelength,
                "power_w": row.power_w,
                "routing_area": row.routing_area if row.routing_area > 0 else "?",
                "grid_W": row.grid_w if row.grid_w > 0 else "?",
                "grid_H": row.grid_h if row.grid_h > 0 else "?",
                "grid_clbs": (row.grid_w - 2) * (row.grid_h - 2) if row.grid_w > 0 else "?",
            }
        )

    def _terminal_info(self, success: bool, status: str) -> dict:
        return {
            "status": status,
            "benchmark_name": self._active_config.name,
            "placed_dsps": list(self._placed_dsps),
            "placed_brams": list(self._placed_brams),
            "success": success,
            "wirelength": float("inf"),
            "delay_ns": float("inf"),
            "power_w": float("inf"),
            "routing_area": float("inf"),
        }

    @staticmethod
    def _empty_eval_info() -> dict:
        return {
            "placed_dsps": [],
            "placed_brams": [],
            "success": False,
            "wirelength": float("inf"),
            "delay_ns": float("inf"),
            "power_w": float("inf"),
            "routing_area": float("inf"),
        }

    @staticmethod
    def _cleanup(arch_file: Path, run_dir: Path, constraints_file: Optional[Path] = None) -> None:
        try:
            if arch_file.is_file():
                arch_file.unlink()
            if constraints_file is not None and constraints_file.is_file():
                constraints_file.unlink()
            if run_dir.exists():
                shutil.rmtree(run_dir)
        except Exception:
            pass
