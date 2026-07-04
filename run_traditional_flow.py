#!/usr/bin/env python3
"""
Run the VTR flow on a traditional (non-RL) architecture and extract performance metrics.

Usage:
    python run_traditional_flow.py --benchmark diffeq1
    python run_traditional_flow.py --benchmark diffeq1 --arch arch/k6_frac_N10_mem32K_40nm.xml
    python run_traditional_flow.py --benchmark diffeq1 --no-power
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import VTRPaths, load_env_file
from src.evaluation.vtr_runner import VTRRunner

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARCH = "arch/k6_frac_N10_mem32K_40nm.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VTR flow on a traditional architecture")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g. diffeq1)")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help=f"Architecture XML (default: {DEFAULT_ARCH})")
    parser.add_argument("--no-power", action="store_true", help="Disable power estimation")
    args = parser.parse_args()

    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    benchmark_name = args.benchmark.removesuffix(".v")
    benchmark_file = PROJECT_ROOT / "benchmarks" / f"{benchmark_name}.v"
    arch_file = Path(args.arch).expanduser().resolve()
    output_dir = PROJECT_ROOT / "runs" / f"{benchmark_name}_traditional"

    if not benchmark_file.is_file():
        print(f"Error: Benchmark not found: {benchmark_file}", file=sys.stderr)
        return 1

    if not arch_file.is_file():
        print(f"Error: Architecture file not found: {arch_file}", file=sys.stderr)
        return 1

    runner = VTRRunner(VTRPaths())
    rc = runner.run(benchmark_file, arch_file, output_dir, enable_power=not args.no_power)

    print("-" * 60)
    if rc != 0:
        print(f"VTR flow returned {rc}. Attempting partial metric extraction...", file=sys.stderr)

    vpr_out = output_dir / "vpr.out"
    crit_path = output_dir / "vpr.crit_path.out"
    power_file = output_dir / f"{benchmark_name}.power"

    baselines_dir = PROJECT_ROOT / "baselines"
    baselines_dir.mkdir(exist_ok=True)
    metric_dest = baselines_dir / f"{benchmark_name}_traditional_metric.txt"
    resource_dest = baselines_dir / f"{benchmark_name}_traditional_resources.txt"

    metrics = runner.parse_metrics(vpr_out, crit_path, power_file, dest=metric_dest)
    resources = runner.parse_resources(vpr_out, dest=resource_dest)

    print("-" * 60)
    print("Metrics:")
    print(json.dumps(metrics.to_dict(), indent=4))
    print("\nResources:")
    print(json.dumps(resources.to_dict(), indent=4))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
