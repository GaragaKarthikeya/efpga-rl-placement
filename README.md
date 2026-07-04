# eFPGA RL Placement

RL training pipeline for FPGA placement optimization using VTR
(Verilog-to-Routing). A PPO agent places DSP/BRAM blocks and selects an
aspect ratio on a custom-generated FPGA architecture, evaluated against a
traditional (non-RL) VTR baseline via routing area, delay, and power. A
genetic algorithm baseline (`ga/`) searches the same placement + aspect-ratio
space using the same VTR-based fitness evaluation.

## Setup

Requires a working [VTR (Verilog-to-Routing)](https://github.com/verilog-to-routing/vtr-verilog-to-routing)
installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your VTR paths
```

`.env` must define:
- `VTR_VENV_PATH`: path to the Python venv used to invoke VTR
- `VTR_FLOW_SCRIPT`: path to VTR's `run_vtr_flow.py`
- `VTR_POWER_TECH_FILE`: path to a VTR power tech XML (e.g. `PTM_45nm/45nm.xml`)

Baseline metric/resource files for the 17 benchmarks in `benchmarks/` are
included under `baselines/`; regenerate a given benchmark's baseline with
`run_traditional_flow.py` (see below) if needed.

## Commands

### Training
```bash
python3 train.py --benchmarks diffeq1 --seed 42 --max_episodes 256
python3 train.py --benchmarks diffeq1 --save_path runs/model.zip
python3 train.py --benchmarks diffeq1 --load_path runs/model.zip --timesteps 20000
python3 train.py --benchmarks diffeq1 --ar_weight 0.4 --dl_weight 0.3 --pw_weight 0.3
```

`--benchmarks` accepts a comma-separated list to train a single
benchmark-independent policy across multiple netlists (e.g.
`--benchmarks diffeq1,diffeq2,softmax`). See `train.py --help` for the full
argument list (parallelism, PPO hyperparameters, reward weights, W&B
logging).

Before training on a benchmark for the first time, its netlist graph must
exist: run `run_traditional_flow.py --benchmark <name>` followed by
`extract_netlist_info.py --benchmark <name> --include-graph`.

### Baseline evaluation
```bash
python3 run_traditional_flow.py --benchmark diffeq1
python3 run_traditional_flow.py --benchmark diffeq1 --arch arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml
python3 run_traditional_flow.py --benchmark diffeq1 --no-power
```

### Zero-shot evaluation
```bash
python3 evaluate_held_out.py --model_path runs/model.zip \
    --eval_benchmarks softmax,reduction_layer \
    --universe_benchmarks fifo,ch_intrinsics,spree,boundtop,mmc_core,diffeq1,diffeq2,raygentop,mkSMAdapter4B,or1200,mkPktMerge,softmax,reduction_layer
```

`--universe_benchmarks` must match the superset used to size the model's
canvas/graph dimensions at training time, or loading the checkpoint fails
with a shape mismatch (see `evaluate_held_out.py` module docstring).

### GA baseline
See `ga/README.md`.

## Benchmarks

17 Verilog designs in `benchmarks/`: `arm_core`, `boundtop`, `ch_intrinsics`,
`cipher`, `diffeq1`, `diffeq2`, `fifo`, `macbuf`, `mkDelayWorker32B`,
`mkPktMerge`, `mkSMAdapter4B`, `mmc_core`, `or1200`, `raygentop`,
`reduction_layer`, `softmax`, `spree`. Each has a matching
`baselines/{name}_traditional_metric.txt` (delay/power/routing-area) and
`baselines/{name}_traditional_resources.txt` (FPGA size, DSP/BRAM/CLB
requirements).

## Repository layout

```
train.py                   Training entry point
run_traditional_flow.py    Runs VTR on the default architecture, extracts baseline metrics
evaluate_held_out.py       Zero-shot evaluation of a trained model on held-out benchmarks
extract_netlist_info.py    Extracts net-count data and a netlist graph JSON from a VTR run
ga/                        Standalone GA baseline (ga_search.py) over the same search space
scripts/                   Auxiliary scripts (see scripts/run_spree_timeline_vtr.py docstring)
src/env/                   FPGAEnv (Gymnasium environment)
src/layout/                bake_layout() (DSP/BRAM placement -> VTR architecture XML)
src/evaluation/            VTRRunner (VTR subprocess wrapper, metric/resource parsing)
src/training/              CustomMaskablePPO, TrainConfig/train(), training callbacks
src/netlist/               VTR .net file parsing
src/utils/                 VTRPaths/.env loading, SQLite layout cache
arch/, template/           VTR architecture XML files and Jinja2 templates
benchmarks/, baselines/    Verilog designs and their traditional-flow baselines
```

## Architecture

### High-level workflow
1. **Baseline generation**: `run_traditional_flow.py` runs VTR
   synthesis/placement/routing on the default architecture and extracts
   delay/power/routing-area metrics into `baselines/`.
2. **RL training**: `train.py` creates parallel `FPGAEnv` instances and
   trains `CustomMaskablePPO` to place DSP/BRAM blocks and choose an aspect
   ratio on a custom-generated architecture.
3. **Evaluation**: each episode bakes an architecture XML via `bake_layout()`
   and runs the VTR flow through `VTRRunner`; results are cached in SQLite
   (`LayoutCache`), keyed by `(dsp_coords, bram_coords, aspect_ratio)`.

### `FPGAEnv` (`src/env/fpga_env.py`)
- Observation: `Dict` space —
  - `grid`: `Box(MAX_WIDTH, MAX_HEIGHT, 4)`, the occupancy/placement canvas
  - `node_features`: `Box(MAX_NODES, 4)`, per-node features of the
    benchmark's reduced netlist graph
  - `edge_index`: `Box(MAX_EDGES, 2)`, `edge_weight`: `Box(MAX_EDGES,)` —
    the reduced graph's connectivity, consumed by a GNN feature extractor
    (`src/training/gnn_extractor.py`, referenced by `train.py` and the
    evaluation scripts but not itself documented here)
  - `current_block_idx`: `Box(1,)`, the graph node id of the block being
    placed this step
  - `valid_wh`: `Box(2,)`, the active benchmark's width/height normalized
    against the shared canvas
  - `MAX_WIDTH`/`MAX_HEIGHT`/`MAX_NODES`/`MAX_EDGES` are fixed per training
    run from a benchmark universe (see `compute_max_dims()` /
    `build_benchmark_configs()`), which is why zero-shot evaluation must
    pass the same `--universe_benchmarks` used at training time.
- Action space: `Discrete(MAX_WIDTH * MAX_HEIGHT)`; action 0 selects the
  aspect ratio, subsequent actions place DSPs then BRAMs.
- Episode: select aspect ratio -> place all DSPs -> place all BRAMs -> VTR
  evaluate -> return reward.
- Constructor: `FPGAEnv(benchmark_configs, max_width, max_height, max_nodes,
  max_edges, pw_weight=0.30, dl_weight=0.60, ar_weight=0.00,
  vtr_timeout=1200)`.

### Reward function
Negative log-ratio against the VTR baseline, computed in
`FPGAEnv._compute_reward()`:
```
reward = -(
    ar_weight * log(routing_area / baseline_routing_area) +
    dl_weight * log(delay_ns / baseline_delay_ns) +
    pw_weight * log(power_w / baseline_power_w)
)
```


### Key modules

**`src/utils/`**
- `config.py`: `VTRPaths` (resolves venv/flow/power paths from `.env`),
  `load_env_file()`
- `cache.py`: `LayoutCache` (SQLite wrapper for VTR results), `CacheRow`
  (delay_ns, power_w, routing_area, grid_w, grid_h, success)

**`src/evaluation/`**
- `vtr_runner.py`: `VTRRunner` (runs the VTR flow subprocess via `run()`,
  parses output via `parse_metrics()`/`parse_resources()`), `VTRMetrics`
  (delay_ns, power_w, routing_area), `VTRResources` (fpga_size,
  requirements, limits)

**`src/layout/`**
- `baker.py`: `bake_layout()` — renders DSP/BRAM placement, size, and aspect
  ratio into a Jinja2-templated VTR architecture XML; optionally emits a VPR
  constraints XML pinning specific netlist atoms to coordinates

**`src/training/`**
- `ppo.py`: `CustomMaskablePPO` — extends `sb3_contrib.MaskablePPO` with
  per-update diagnostics (batch_reward_variance, gradient_variance_norm,
  avg_cosine_similarity, global_grad_norm, policy_entropy, value_loss,
  explained_variance)
- `gnn_extractor.py`: `GNNFeaturesExtractor` — SB3 features extractor
  combining a CNN over `grid` with a graph-convolution encoder over
  `node_features`/`edge_index`/`edge_weight`, used as the PPO policy's
  `features_extractor_class`
- `trainer.py`: `TrainConfig` (dataclass of all hyperparameters), `train()`
  (builds envs, runs PPO, attaches callbacks, optional W&B logging)
- `callbacks.py`: `BestLayoutCallback` and related callbacks — track
  best-so-far reward per benchmark, write `best_baked_layout_*.xml`,
  `best_layout_constraints_*.xml`, `best_layout_coordinates_*.txt`, and a
  per-episode `all_layouts_*.jsonl` log

### Adding a new benchmark
1. Place a Verilog file in `benchmarks/{name}.v`.
2. Run `python3 run_traditional_flow.py --benchmark {name}` to generate
   `baselines/{name}_traditional_metric.txt` and
   `baselines/{name}_traditional_resources.txt`.
3. Run `python3 extract_netlist_info.py --benchmark {name} --include-graph`
   to generate `{name}_netlist_info.json` and `{name}_netlist_graph.json`
   (required by `FPGAEnv`, gitignored, regenerated per-machine).
4. Train: `python3 train.py --benchmarks {name}`.

## License

MIT — see `LICENSE`.
