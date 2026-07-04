#!/usr/bin/env python3

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(env_file: Path) -> None:
    """Load KEY=VALUE pairs from an .env file into os.environ."""
    if not env_file.is_file():
        raise FileNotFoundError(f"Missing .env file at: {env_file}")
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = os.path.expandvars(value.strip().strip("'\""))


@dataclass
class VTRPaths:
    """Resolved paths to the VTR toolchain components."""

    python: Path = field(init=False)
    flow_script: Path = field(init=False)
    power_tech_file: Path = field(init=False)

    def __post_init__(self) -> None:
        venv = Path(os.environ.get("VTR_VENV_PATH", "/home/digital-2/.venv"))
        candidate = venv / "bin" / "python"
        self.python = candidate if candidate.is_file() else Path(sys.executable)

        self.flow_script = Path(
            os.environ.get(
                "VTR_FLOW_SCRIPT",
                "/home/digital-2/vtr-verilog-to-routing/vtr_flow/scripts/run_vtr_flow.py",
            )
        )
        self.power_tech_file = Path(
            os.environ.get(
                "VTR_POWER_TECH_FILE",
                "/home/digital-2/vtr-verilog-to-routing/vtr_flow/tech/PTM_45nm/45nm.xml",
            )
        )

    @property
    def has_power_tech(self) -> bool:
        return self.power_tech_file.is_file()

    @property
    def is_flow_available(self) -> bool:
        return self.flow_script.is_file()
