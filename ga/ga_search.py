"""Seeded GA baseline for H-block placement + aspect ratio, used for the
paper's GA-vs-zero-shot comparison. Same search space and ADP fitness
function as the RL policy: roulette-wheel selection, single-point crossover
per block type, per-gene mutation (including the aspect-ratio gene), early
stopping via --patience. The initial population can be seeded with a known
layout (e.g. a policy's best zero-shot layout) instead of starting purely
random.
"""

import json
import random
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.vtr_runner import VTRRunner
from src.layout.baker import bake_layout
from src.utils.config import VTRPaths, load_env_file

SCRIPT_DIR = PROJECT_ROOT


class SeededGAAgent:
    # Discrete aspect-ratio candidates: 0.1, 0.2, ..., 2.0  (width / height)
    ASPECT_RATIO_CANDIDATES = [round(v * 0.1, 1) for v in range(1, 21)]

    def __init__(self, benchmark_name, width, height, req_dsp, req_bram, seed_layout,
                 num_ind=5, num_gen=500, cross_prob=0.87, mut_prob=0.7, patience=50, num_seeded=None):
        self.benchmark_name = benchmark_name
        self.width = width
        self.height = height
        self.req_dsp = req_dsp
        self.req_bram = req_bram
        self.num_ind = num_ind
        self.num_gen = num_gen
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.patience = patience
        self.dsp_height = 4
        self.bram_height = 6
        self.vtr = VTRRunner(VTRPaths())
        self.seed_layout = seed_layout
        # How many of the num_ind population slots get the seed layout
        # verbatim; the rest are freshly randomized. Defaults to all of them.
        self.num_seeded = self.num_ind if num_seeded is None else num_seeded

    def get_random_valid_coordinate(self, current_sol, block_height):
        # Prevent infinite loops if grid is too packed
        for _ in range(50):
            x = random.randint(1, self.width)
            y = random.randint(1, self.height - block_height + 1)

            # Check collision
            collision = False

            for (dx, dy) in current_sol["dsps"]:
                if dx == x:
                    # check vertical overlap
                    if not (y + block_height - 1 < dy or y > dy + self.dsp_height - 1):
                        collision = True
                        break

            if collision: continue

            for (bx, by) in current_sol["brams"]:
                if bx == x:
                    if not (y + block_height - 1 < by or y > by + self.bram_height - 1):
                        collision = True
                        break

            if not collision:
                return (x, y)

        # If we failed to find a spot, return None
        return None

    def resolve_collisions(self, sol):
        resolved_sol = {"dsps": [], "brams": []}

        for dsp in sol["dsps"]:
            x, y = dsp
            collision = False
            for (dx, dy) in resolved_sol["dsps"]:
                if dx == x and not (y + self.dsp_height - 1 < dy or y > dy + self.dsp_height - 1):
                    collision = True
            if not collision:
                resolved_sol["dsps"].append((x, y))
            else:
                new_coord = self.get_random_valid_coordinate(resolved_sol, self.dsp_height)
                if new_coord: resolved_sol["dsps"].append(new_coord)

        for bram in sol["brams"]:
            x, y = bram
            collision = False
            for (dx, dy) in resolved_sol["dsps"]:
                if dx == x and not (y + self.bram_height - 1 < dy or y > dy + self.dsp_height - 1):
                    collision = True
            for (bx, by) in resolved_sol["brams"]:
                if bx == x and not (y + self.bram_height - 1 < by or y > by + self.bram_height - 1):
                    collision = True
            if not collision:
                resolved_sol["brams"].append((x, y))
            else:
                new_coord = self.get_random_valid_coordinate(resolved_sol, self.bram_height)
                if new_coord: resolved_sol["brams"].append(new_coord)

        return resolved_sol

    def generate_random_solution(self):
        # Pick a random aspect ratio for this individual
        aspect_ratio = random.choice(self.ASPECT_RATIO_CANDIDATES)
        sol = {"dsps": [], "brams": [], "aspect_ratio": aspect_ratio}
        for _ in range(self.req_dsp):
            coord = self.get_random_valid_coordinate(sol, self.dsp_height)
            if coord: sol["dsps"].append(coord)
            else: return None
        for _ in range(self.req_bram):
            coord = self.get_random_valid_coordinate(sol, self.bram_height)
            if coord: sol["brams"].append(coord)
            else: return None
        return sol

    def initiate(self):
        population_bag = []
        for _ in range(self.num_seeded):
            population_bag.append({
                "dsps": list(self.seed_layout["dsps"]),
                "brams": list(self.seed_layout["brams"]),
                "aspect_ratio": self.seed_layout["aspect_ratio"],
            })
        while len(population_bag) < self.num_ind:
            sol = None
            while sol is None:
                sol = self.generate_random_solution()
            population_bag.append(sol)
        return population_bag

    def fitness_function(self, sol):
        worker_id = uuid.uuid4().hex[:8]
        temp_arch_file = SCRIPT_DIR / f"ga_arch_{worker_id}.xml"
        temp_run_dir = SCRIPT_DIR / "runs" / f"ga_run_{worker_id}"
        benchmark_file = SCRIPT_DIR / "benchmarks" / f"{self.benchmark_name}.v"

        # We assume sol is always valid (we enforce it during generation/crossover)
        if len(sol["dsps"]) < self.req_dsp or len(sol["brams"]) < self.req_bram:
            return float('inf'), worker_id, None

        # Use the aspect_ratio gene from the solution (defaults to 1.0 for
        # backwards-compatible solutions that pre-date this field)
        aspect_ratio = sol.get("aspect_ratio", 1.0)
        res = bake_layout(
            benchmark_name=self.benchmark_name,
            dsps=sol["dsps"],
            mems=sol["brams"],
            output_path=str(temp_arch_file),
            aspect_ratio=aspect_ratio
        )

        if res == -1: return float('inf'), worker_id, None

        # Run VTR
        if temp_run_dir.exists():
            shutil.rmtree(temp_run_dir)

        retcode = self.vtr.run(benchmark_file, temp_arch_file, temp_run_dir, silent=True)

        # Cleanup arch file
        if temp_arch_file.exists():
            temp_arch_file.unlink()

        if retcode != 0:
            return float('inf'), worker_id, None

        # Extract metrics
        vpr_out_file = temp_run_dir / "vpr.out"
        crit_path_file = temp_run_dir / "vpr.crit_path.out"
        power_file = temp_run_dir / f"{self.benchmark_name}.power"
        metrics_dest = temp_run_dir / "metrics.txt"

        metrics = self.vtr.parse_metrics(vpr_out_file, crit_path_file, power_file, metrics_dest)

        routing_area = metrics.routing_area
        delay = metrics.delay_ns
        power = metrics.power_w

        if routing_area == float('inf') or delay == float('inf') or power == float('inf'):
            return float('inf'), worker_id, None

        # Cleanup run directory to save space (except on failure if we wanted to debug)
        shutil.rmtree(temp_run_dir, ignore_errors=True)

        # Fitness is ADPP. Lower is better.
        adpp = routing_area * delay * power
        metrics_out = {"delay_ns": delay, "power_w": power, "routing_area": routing_area}
        return adpp, worker_id, metrics_out

    def eval_fit_pop(self, population_bag):
        print(f"Evaluating generation of {len(population_bag)} individuals in parallel...")
        # 1. Evaluate all layouts in parallel
        with ThreadPoolExecutor(max_workers=len(population_bag)) as executor:
            results = list(executor.map(self.fitness_function, population_bag))

        fit_vals = [r[0] for r in results]
        run_folders = [r[1] for r in results]
        metrics_list = [r[2] for r in results]

        print(f"Generation Fitness Values: {fit_vals}")

        # Replace inf with a very high number just to allow probability math
        valid_fits = [f for f in fit_vals if f != float('inf')]
        if not valid_fits:
            max_fit = 1000000
        else:
            max_fit = max(valid_fits)

        processed_fits = [f if f != float('inf') else max_fit * 10 for f in fit_vals]
        max_proc_fit = max(processed_fits)

        # 2. Calculate Roulette Wheel Weights (Lower fitness = higher weight)
        weights = [(max_proc_fit - fit) for fit in processed_fits]
        sum_weights = sum(weights)

        # Protect against divide-by-zero if all layouts have identical fitness
        if sum_weights == 0:
            probabilities = [1.0 / len(fit_vals) for _ in fit_vals]
        else:
            probabilities = [w / sum_weights for w in weights]

        return {
            "fit_val": fit_vals,
            "fit_wgt": probabilities,
            "run_folder": run_folders,
            "sol": population_bag,
            "metrics": metrics_list
        }

    def pick(self, population_bag, fit_bag_evals):
        picked_sol = random.choices(
            population=fit_bag_evals["sol"],
            weights=fit_bag_evals["fit_wgt"],
            k=1
        )[0]
        return picked_sol

    def crossover(self, solA, solB):
        # Inherit aspect_ratio from one parent chosen at random
        ar_a = solA.get("aspect_ratio", 1.0)
        ar_b = solB.get("aspect_ratio", 1.0)
        child_ar = random.choice([ar_a, ar_b])

        child = {"dsps": [], "brams": [], "aspect_ratio": child_ar}

        for block_type in ["dsps", "brams"]:
            n = len(solA[block_type])
            if n > 1:
                cut = random.randint(1, n - 1)
                child[block_type] = solA[block_type][:cut] + solB[block_type][cut:]
            else:
                child[block_type] = solA[block_type].copy()

        # Resolve collisions
        child = self.resolve_collisions(child)
        return child

    def mutation(self, sol):
        current_ar = sol.get("aspect_ratio", 1.0)
        mutated_sol = {"dsps": sol["dsps"].copy(), "brams": sol["brams"].copy(), "aspect_ratio": current_ar}

        # Mutate aspect_ratio gene independently
        if random.random() <= self.mut_prob:
            mutated_sol["aspect_ratio"] = random.choice(self.ASPECT_RATIO_CANDIDATES)

        for block_type in ["dsps", "brams"]:
            for i in range(len(mutated_sol[block_type])):
                if random.random() <= self.mut_prob:
                    # Remove the block temporarily to avoid self-collision checking
                    old_coord = mutated_sol[block_type].pop(i)
                    block_height = 4 if block_type == "dsps" else 6
                    new_coord = self.get_random_valid_coordinate(mutated_sol, block_height)

                    if new_coord:
                        mutated_sol[block_type].insert(i, new_coord)
                    else:
                        mutated_sol[block_type].insert(i, old_coord)  # Revert

        return mutated_sol

    def run(self, log_suffix=None):
        suffix = log_suffix if log_suffix is not None else self.benchmark_name
        out_path = SCRIPT_DIR / f"ga_output_{suffix}.txt"

        pop_bag = self.initiate()
        t_con = [0 for i in range(self.num_gen)]
        g = 0
        best_fit_global = float('inf')
        best_sol_global = None
        best_metrics_global = None

        for gen in range(self.num_gen):
            print(f"--- Generation {gen} ---")
            pop_bag_fit = self.eval_fit_pop(pop_bag)

            best_fit = np.min(pop_bag_fit["fit_val"])
            best_fit_index = pop_bag_fit["fit_val"].index(best_fit)
            best_sol = pop_bag_fit["sol"][best_fit_index]
            best_metrics = pop_bag_fit["metrics"][best_fit_index]

            if best_fit < best_fit_global:
                best_fit_global = best_fit
                best_sol_global = best_sol
                best_metrics_global = best_metrics
                print(f"*** NEW BEST GLOBAL FITNESS: {best_fit_global} *** metrics={best_metrics_global}")

            # Save progress (one file per benchmark/run, not shared across parallel runs)
            with open(out_path, "a+") as f:
                ar = best_sol_global.get("aspect_ratio", 1.0) if best_sol_global else "N/A"
                f.write(f"Generation {gen} | Best Fitness: {best_fit_global} | Metrics: {best_metrics_global} | Aspect Ratio: {ar} | Layout: {best_sol_global}\n")

            t_con[g] = best_fit_global

            # Stop early if no improvement
            if g >= self.patience:
                check = 0
                for i in range(g - self.patience, g + 1):
                    if t_con[i] == best_fit_global:
                        check += 1
                if check >= self.patience + 1:
                    print(f"Breaking condition reached: No improvement for {self.patience} generations.")
                    return best_fit_global, best_sol_global, best_metrics_global

            # Create next generation
            new_pop_bag = []
            for i in range(self.num_ind):
                pA = self.pick(pop_bag, pop_bag_fit)
                pB = self.pick(pop_bag, pop_bag_fit)

                new_element = pA
                if random.random() <= self.cross_prob:
                    new_element = self.crossover(pA, pB)

                if random.random() <= self.mut_prob:
                    new_element = self.mutation(new_element)

                # Just in case a layout got totally corrupted, regenerate a valid one
                if len(new_element["dsps"]) < self.req_dsp or len(new_element["brams"]) < self.req_bram:
                    new_element = self.generate_random_solution()

                new_pop_bag.append(new_element)

            pop_bag = new_pop_bag
            g += 1

        print("Completed all generations.")
        return best_fit_global, best_sol_global, best_metrics_global


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="softmax")
    parser.add_argument("--width", type=int, default=41)
    parser.add_argument("--height", type=int, default=41)
    parser.add_argument("--dsps", type=int, default=8)
    parser.add_argument("--brams", type=int, default=0)
    parser.add_argument("--gens", type=int, default=500)
    parser.add_argument("--inds", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed_layout", type=str, required=True,
                         help='JSON dict, e.g. {"dsps": [[2,9],[11,4]], "brams": [], "aspect_ratio": 0.9}')
    parser.add_argument("--num_seeded", type=int, default=None,
                         help="How many of --inds population slots get the seed layout verbatim; rest are random. Default: all of them.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for Python's random module (mutation/crossover/random-individual reproducibility)")
    parser.add_argument("--log_suffix", type=str, default=None, help="Suffix for this run's own ga_output_<suffix>.txt (defaults to --benchmark, so parallel seeds need distinct values to avoid clobbering)")
    args = parser.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    raw = json.loads(args.seed_layout)
    seed_layout = {
        "dsps": [tuple(c) for c in raw["dsps"]],
        "brams": [tuple(c) for c in raw["brams"]],
        "aspect_ratio": raw["aspect_ratio"],
    }

    agent = SeededGAAgent(
        benchmark_name=args.benchmark,
        width=args.width,
        height=args.height,
        req_dsp=args.dsps,
        req_bram=args.brams,
        num_ind=args.inds,
        num_gen=args.gens,
        patience=args.patience,
        seed_layout=seed_layout,
        num_seeded=args.num_seeded,
    )

    best_fit, best_sol, best_metrics = agent.run(log_suffix=args.log_suffix)
    print(f"\nGA Finished! Best Product: {best_fit}")
    print(f"Best Layout: {best_sol}")
    print(f"Best Metrics: {best_metrics}")
