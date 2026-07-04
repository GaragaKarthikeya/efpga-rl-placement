# eFPGA RL Placement

RL training pipeline for FPGA placement optimization using VTR (Verilog-to-Routing).

## Research Motivation

This pipeline exists to support **eFPGA-based logic masking**: embedding a small, custom reconfigurable fabric into an ASIC to implement one critical IP block, so the function isn't visible from the silicon layout/GDSII without the bitstream (a known hardware-security technique against reverse engineering and IP theft). A generic/traditional FPGA architecture is built for arbitrary circuits and is therefore over-provisioned — extra LUTs, routing, DSP/BRAM slots — for any one fixed design that will only ever run a single bitstream. Because the target use case is **one known application on a small embedded fabric** (not general-purpose reconfigurability), a custom-tailored architecture and placement, sized and arranged specifically for that one netlist, is viable and can recover most of the area/delay/power overhead a generic FPGA would otherwise cost.

**This is why Area × Delay × Power (ADP) is the central optimization target throughout this codebase**: it's the size of the "masking tax" — the PPA overhead of implementing the IP on reconfigurable fabric instead of hardening it directly into standard cells. Minimizing ADP is minimizing the cost of the security technique.

**Research lineage:** this work continues the direction of two prior papers using the *same* benchmarks (diffeq1, diffeq2, etc.) and the *same* VTR-based evaluation methodology, but with Genetic Algorithm / NSGA-II search instead of RL:
- *GOLDS: Genetic Algorithm-based Optimization of Custom FPGA Architecture Layout Design for Secure Silicon* (Nandi, Mishra, Rao — GLSVLSI '24) — GA search over DSP/BRAM/CLB placement, fitness = Area×Delay, reports up to ~35% ADP gain over traditional layout on diffeq1.
- *Meta-Heuristic Optimization of Custom Heterogeneous Blocks defined eFPGA Design* (Bhargav, Pradyumna, Rao — VLSID '25) — NSGA-II, multi-objective (delay, power), additionally co-evolves CLB LUT size and custom DSP/BRAM sizing (not just placement), reports up to ~44.6% improvement.

**What this RL-based approach adds that per-instance evolutionary search structurally cannot:** a *learned policy* amortizes across deployments. A GA/NSGA-II run starts its population from scratch for every new IP; our zero-shot transfer experiment (diffeq2-trained policy applied directly to diffeq1, no training) already beat GOLDS' published GA result on diffeq1 (computed on their exact Area×Delay metric) with zero benchmark-specific optimization, and fine-tuning from a pretrained policy converged ~2.4x faster in wall-clock than training from scratch.

## Setup

Requires a working [VTR (Verilog-to-Routing)](https://github.com/verilog-to-routing/vtr-verilog-to-routing) installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your VTR paths
```

`.env` must point to:
- `VTR_VENV_PATH`: path to the Python venv used to run VTR
- `VTR_FLOW_SCRIPT`: path to VTR's `run_vtr_flow.py`
- `VTR_POWER_TECH_FILE`: path to a VTR power tech XML (e.g. `PTM_45nm/45nm.xml`)

Pre-computed baseline metric/resource files for the 17 benchmarks used in the paper are included under `baselines/`.

## Commands

### Training
```bash
python3 train.py --benchmark diffeq1 --seed 42 --max_episodes 256
python3 train.py --benchmark diffeq1 --save_path runs/model.zip
python3 train.py --benchmark diffeq1 --load_path runs/model.zip --timesteps 20000
python3 train.py --benchmark diffeq1 --ar_weight 0.4 --dl_weight 0.3 --pw_weight 0.3
```

### Baseline evaluation
```bash
python3 run_traditional_flow.py --benchmark diffeq1
python3 run_traditional_flow.py --benchmark diffeq1 --arch arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml
python3 run_traditional_flow.py --benchmark diffeq1 --no-power
```

### Benchmarks

17 Verilog designs in `benchmarks/`, drawn from the VTR benchmark suite, Koios, and two small probes authored for this work:
- **In-pool (trained on):** `boundtop`, `ch_intrinsics`, `diffeq1`, `diffeq2`, `fifo`, `mkPktMerge`, `mkSMAdapter4B`, `mmc_core`, `or1200`, `raygentop`, `spree`
- **Held out (zero-shot only):** `custom_macbuf`, `mkDelayWorker32B`, `lightweight_cipher`, `reduction_layer`, `arm_core`, `softmax`

## Architecture

### High-level workflow
1. **Baseline generation**: `run_traditional_flow.py` runs VTR synthesis/placement/routing on the default architecture, extracts delay/power/wirelength/routing-area metrics
2. **RL training**: `train.py` creates parallel `FPGAEnv` instances, trains `CustomMaskablePPO` to place DSP/BRAM blocks on custom-generated architectures
3. **Evaluation**: Each episode runs the VTR flow on the agent-generated architecture XML via `vtr_runner.py`; results cached in SQLite

### Key modules

**`src/utils/`**
- `config.py`: `VTRPaths` (resolves venv/flow/power paths from `.env`), `load_env_file()`
- `cache.py`: `LayoutCache` (SQLite wrapper for VTR results), `CacheRow` (metrics struct)

**`src/evaluation/`**
- `vtr_runner.py`: `VTRRunner` (runs VTR flow subprocess), `VTRMetrics` (delay/wirelength/power/routing-area), `VTRResources` (FPGA size and requirements)

**`src/env/`**
- `fpga_env.py`: `FPGAEnv` — Gymnasium environment for heterogeneous block placement
  - Observation: `Box(W, H, 3)` with occupancy grid, block-type hint, aspect-ratio channel
  - Action space: `Discrete(W * H)`; step 0 selects aspect ratio, steps 1–N place blocks
  - Episode: select aspect ratio → place all DSPs → place all BRAMs → VTR evaluate → return reward

**`src/layout/`**
- `baker.py`: `bake_layout()` — renders DSP/BRAM placement + aspect ratio into a Jinja2-templated VTR architecture XML

**`src/training/`**
- `ppo.py`: `CustomMaskablePPO` — extends `sb3_contrib.MaskablePPO` with per-environment gradient variance diagnostics
- `trainer.py`: `TrainConfig` (dataclass for all hyperparameters), `train()` (orchestrates env creation, model training, callbacks, W&B logging)
- `callbacks.py`: custom callbacks (e.g. `BestLayoutCallback`)

### Reward function
Negative log-ratio against the VTR baseline:
```
reward = -(
    ar_weight * log(routing_area / baseline_area) +
    dl_weight * log(delay / baseline_delay) +
    pw_weight * log(power / baseline_power) +
    wl_weight * log(wirelength / baseline_wirelength)
)
```
Default weights: `ar=0.33, dl=0.33, pw=0.34, wl=0.00` (no wirelength in reward).

### Adding a new benchmark
1. Place a Verilog file in `benchmarks/{name}.v`
2. Run `python3 run_traditional_flow.py --benchmark {name}` to generate baseline files
3. Train: `python3 train.py --benchmark {name}`

## Paper

The ASP-DAC 2027 draft describing this work is in `paper/`.

## License

TODO: add a license before making this repository public.
