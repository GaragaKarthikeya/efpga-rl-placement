#!/usr/bin/env python3

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.config import VTRPaths


@dataclass
class VTRMetrics:
    delay_ns: float = float("inf")
    wirelength: float = float("inf")
    power_w: float = float("inf")
    routing_area: float = float("inf")

    def is_complete(self) -> bool:
        return all(v != float("inf") for v in (self.delay_ns, self.wirelength, self.power_w, self.routing_area))

    def to_dict(self) -> dict:
        return {
            "delay_ns": self.delay_ns,
            "wirelength": self.wirelength,
            "power_w": self.power_w,
            "routing_area": self.routing_area,
        }


@dataclass
class VTRResources:
    fpga_size: list[int] = field(default_factory=lambda: [0, 0])
    requirements: dict = field(default_factory=lambda: {"io": 0, "clb": 0, "dsp": 0, "bram": 0})
    limits: dict = field(default_factory=lambda: {"io": 0, "clb": 0, "dsp": 0, "bram": 0})

    def to_dict(self) -> dict:
        return {
            "fpga_size": self.fpga_size,
            "requirements": self.requirements,
            "limits": self.limits,
        }


class VTRRunner:
    """Wrapper around the VTR flow script for running synthesis, placement and routing."""

    def __init__(self, paths: Optional[VTRPaths] = None) -> None:
        self.paths = paths or VTRPaths()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        benchmark_file: Path,
        arch_file: Path,
        output_dir: Path,
        enable_power: bool = True,
        timeout: int = 1200,
        silent: bool = False,
        constraints_file: Optional[Path] = None,
    ) -> int:
        """Run the VTR flow and return its exit code."""
        if not self.paths.is_flow_available:
            print(f"Error: VTR flow script not found at: {self.paths.flow_script}", file=sys.stderr)
            return 1

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_cmd(benchmark_file, arch_file, output_dir, enable_power, constraints_file)

        if not silent:
            print("Executing VTR flow:")
            print(" ".join(str(c) for c in cmd))
            print("-" * 60)

        stdout = subprocess.DEVNULL if silent else None
        stderr = subprocess.DEVNULL if silent else None

        try:
            result = subprocess.run(cmd, check=False, stdout=stdout, stderr=stderr, timeout=timeout)
            return result.returncode
        except subprocess.TimeoutExpired:
            return -1

    def run_async(
        self,
        benchmark_file: Path,
        arch_file: Path,
        output_dir: Path,
        enable_power: bool = True,
    ) -> subprocess.Popen:
        """Launch VTR in the background and return the Popen handle."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_cmd(benchmark_file, arch_file, output_dir, enable_power)
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ------------------------------------------------------------------
    # Metric / resource parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_metrics(
        vpr_out: Path,
        crit_path: Path,
        power_file: Path,
        dest: Optional[Path] = None,
    ) -> VTRMetrics:
        """Parse VPR output files and return a VTRMetrics instance."""
        m = VTRMetrics()

        if vpr_out.exists():
            content = vpr_out.read_text(errors="ignore")
            wl_match = re.search(
                r"Wire length results.*?Total wirelength:\s+([0-9]+),",
                content,
                re.DOTALL,
            )
            if wl_match:
                m.wirelength = float(wl_match.group(1))

            area_match = re.search(r"Total routing area:\s+([0-9\.\+eE\-]+)", content)
            if area_match:
                m.routing_area = float(area_match.group(1))

        if crit_path.exists():
            content = crit_path.read_text(errors="ignore")
            match = re.search(
                r"Final critical path delay \(least slack\):\s+([0-9.]+)\s+ns", content
            )
            if match:
                m.delay_ns = float(match.group(1))

        if power_file.exists():
            content = power_file.read_text(errors="ignore")
            match = re.search(r"^Total\s+([0-9\.eE\-]+)", content, re.MULTILINE)
            if match:
                m.power_w = float(match.group(1))

        if dest is not None:
            import json
            dest.write_text(json.dumps(m.to_dict(), indent=4))

        return m

    @staticmethod
    def parse_resources(vpr_out: Path, dest: Optional[Path] = None) -> VTRResources:
        """Parse VPR output for FPGA resource usage and limits."""
        res = VTRResources()

        if not vpr_out.exists():
            return res

        content = vpr_out.read_text(errors="ignore")

        size_match = re.search(r"FPGA sized to\s+([0-9]+)\s+x\s+([0-9]+)", content)
        if size_match:
            # subtract IO ring to get core dimensions
            res.fpga_size = [int(size_match.group(1)) - 2, int(size_match.group(2)) - 2]

        label_to_key = {"io": "io", "clb": "clb", "mult_36": "dsp", "memory": "bram"}
        for vpr_label, key in label_to_key.items():
            req = re.search(rf"Netlist\s+([0-9]+)\s+blocks of type: {vpr_label}", content)
            lim = re.search(rf"Architecture\s+([0-9]+)\s+blocks of type: {vpr_label}", content)
            if req:
                res.requirements[key] = int(req.group(1))
            if lim:
                res.limits[key] = int(lim.group(1))

        if dest is not None:
            import json
            dest.write_text(json.dumps(res.to_dict(), indent=4))

        return res

    # ------------------------------------------------------------------

    def _build_cmd(
        self,
        benchmark_file: Path,
        arch_file: Path,
        output_dir: Path,
        enable_power: bool,
        constraints_file: Optional[Path] = None,
    ) -> list:
        cmd = [
            str(self.paths.python),
            str(self.paths.flow_script),
            str(benchmark_file),
            str(arch_file),
            "-temp_dir",
            str(output_dir),
        ]
        if enable_power and self.paths.has_power_tech:
            cmd.extend(["-cmos_tech", str(self.paths.power_tech_file)])
        if constraints_file is not None and constraints_file.is_file():
            cmd.extend(["-read_vpr_constraints", str(constraints_file)])
        return cmd
