# GA Baseline

Genetic algorithm baseline used for the paper's GA-vs-zero-shot comparison
(Section "GA Comparison", Table `ga-vs-zeroshot`). Same H-block-placement +
aspect-ratio search space and ADP fitness function as the RL policy, so the
two are directly comparable on "best layout found."

- `ga_search.py` — `SeededGAAgent`: standalone GA (roulette-wheel selection,
  single-point crossover per block type, per-gene mutation including the
  aspect-ratio gene, early stopping via `--patience`). Fitness is real VTR
  ADP (`routing_area * delay_ns * power_w`), evaluated in parallel across
  the population via `ThreadPoolExecutor`. The initial population can be
  seeded with a known layout (e.g. a policy's best zero-shot layout)
  instead of starting purely random — pass `--num_seeded 0` for a fully
  random baseline run. Uses this repo's `src.layout.baker` and
  `src.evaluation.vtr_runner.VTRRunner` (same evaluation path the RL
  environment uses).
- `generate_ga_layout.py` — bakes one fixed example layout + a fake `.place`
  file into `visualizations/`, used to render the GA-vs-RL layout comparison
  figure.

## Usage

```bash
cd ..  # repo root, so benchmarks/, baselines/, .env resolve correctly
python3 ga/ga_search.py --benchmark softmax --width 41 --height 41 --dsps 8 --brams 0 \
    --seed_layout '{"dsps": [[2,9],[11,4]], "brams": [], "aspect_ratio": 0.9}' --seed 42
```

Progress is logged to `ga_output_<benchmark-or-log_suffix>.txt` in the repo
root; use `--log_suffix` to avoid clobbering when running multiple
seeds/benchmarks in parallel.
