#!/usr/bin/env python3
"""
Parse VTR .net XML files to extract per-block net counts.

The .net file is a packed netlist in XML format. Each block instance
(DSP, BRAM, CLB) has input and output ports with connected nets.
This module extracts those nets, filters global/reserved nets,
and counts unique connections per block.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Global and reserved nets to ignore (don't count as real connectivity)
GLOBAL_NETS = {
    "open",
    "vcc",
    "gnd",
    "gnd_net",
    "vcc_net",
    "vss",
    "vsubs",
    "tie0",
    "tie1",
    "const_zero",
    "const_one",
    "unconn",
    "unconnected",
}


@dataclass
class BlockNetInfo:
    """Information about a single block instance and its net connectivity."""

    block_type: str  # 'dsp', 'bram', 'clb', 'io', etc.
    instance_id: int  # Extracted from instance name, e.g., mult_36[5] -> 5
    instance_name: str  # Packed-block instance name, e.g., mult_36[1]
    atom_name: str      # VPR atom name (name= attribute), e.g., $mul~6[0] — use this in VPR constraints
    unique_nets: int  # Count of unique nets connected to this block
    net_names: set[str]  # Set of all connected net names (for debugging)


def _extract_instance_id(instance_str: str) -> Optional[int]:
    """
    Extract numeric ID from instance name.

    Examples:
        "mult_36[0]" -> 0
        "clb[5]" -> 5
        "FPGA_packed_netlist[0]" -> 0 (but we skip this one anyway)

    Returns None if ID cannot be extracted.
    """
    try:
        start = instance_str.rfind("[")
        end = instance_str.rfind("]")
        if start != -1 and end != -1 and start < end:
            return int(instance_str[start + 1 : end])
    except (ValueError, IndexError):
        pass
    return None


def _classify_block_type(instance_str: str) -> str:
    """
    Classify block type based on instance name prefix.

    Returns one of: 'dsp', 'bram', 'clb', 'io', 'lut', 'ble', 'other'
    """
    instance_lower = instance_str.lower()

    if "mult" in instance_lower:
        return "dsp"
    elif "bram" in instance_lower or "ram" in instance_lower or "mem" in instance_lower:
        return "bram"
    elif "clb" in instance_lower:
        return "clb"
    elif "io" in instance_lower or "pad" in instance_lower:
        return "io"
    elif "lut" in instance_lower:
        return "lut"
    elif "ble" in instance_lower:
        return "ble"
    else:
        return "other"


def _parse_nets_from_ports(block_elem: ET.Element) -> set[str]:
    """
    Extract all unique net names from a block's input and output ports.

    Walks through <inputs> and <outputs> tags, finds <port> elements,
    and parses space-separated signal names.

    Filters out global nets defined in GLOBAL_NETS.
    """
    nets = set()

    # Find all port elements (in both inputs and outputs)
    for port in block_elem.findall(".//port"):
        # Port content is space-separated signal names
        port_text = port.text or ""
        signals = port_text.split()

        for signal in signals:
            # Skip empty strings
            if not signal:
                continue

            # Extract base signal name (remove array indices and routing info)
            # Examples: "signal_name" or "signal_name[5]" or "signal.port->routing"
            base_signal = signal.split("[")[0].split(".")[0].split("->")[0]

            # Skip global/reserved nets
            if base_signal.lower() not in GLOBAL_NETS:
                nets.add(base_signal)

    return nets


def parse_net_file(net_path: Path) -> list[BlockNetInfo]:
    """
    Parse a VTR .net XML file and extract block-level net information.

    Only extracts top-level placement blocks (e.g., mult_36[0], not its internal hierarchy).

    Args:
        net_path: Path to the .net file

    Returns:
        List of BlockNetInfo objects, one per block instance.
        Sorted by (block_type, instance_id) for consistency.

    Raises:
        FileNotFoundError: If net_path does not exist
        ET.ParseError: If XML is malformed
    """
    if not net_path.exists():
        raise FileNotFoundError(f"Net file not found: {net_path}")

    tree = ET.parse(net_path)
    root = tree.getroot()

    blocks = []
    seen_block_keys = set()  # Track (block_type, instance_id) to avoid duplicates from hierarchy

    # Prefixes of blocks we consider "top-level" placement blocks
    # Skip their internal hierarchies (e.g., mult_36x36_slice, mult_36x36)
    TOP_LEVEL_PREFIXES = {
        "mult_36",  # DSP
        "bram",
        "memory",  # BRAM — VTR's actual top-level instance name for memory blocks
        "clb",
        "io",
        "lut",
        "ble",
    }

    def is_top_level_instance(instance_str: str) -> bool:
        """Check if this instance is a top-level block (not internal hierarchy)."""
        instance_lower = instance_str.lower()
        for prefix in TOP_LEVEL_PREFIXES:
            if instance_lower.startswith(prefix + "["):
                return True
        return False

    def process_blocks_recursive(elem: ET.Element, depth: int = 0):
        """
        Recursively process block elements.
        Extract only top-level blocks, avoiding internal hierarchy duplicates.
        """
        instance = elem.get("instance")
        if not instance:
            # No instance, process children
            for child in elem.findall("block"):
                process_blocks_recursive(child, depth + 1)
            return

        # Skip FPGA_packed_netlist root
        if "FPGA_packed_netlist" in instance:
            # Process children at next depth
            for child in elem.findall("block"):
                process_blocks_recursive(child, depth + 1)
            return

        # Only process top-level blocks, skip internal hierarchy
        if not is_top_level_instance(instance):
            # Not a top-level block, skip this subtree to avoid internal duplicates
            return

        # Extract ID and classify
        block_id = _extract_instance_id(instance)
        block_type = _classify_block_type(instance)

        if block_id is None:
            return

        # Check if we've already seen this (block_type, instance_id) pair
        block_key = (block_type, block_id)
        if block_key in seen_block_keys:
            return
        seen_block_keys.add(block_key)

        # Parse nets from this block
        nets = _parse_nets_from_ports(elem)

        # Create block info
        atom_name = elem.get("name", instance)  # name= is the VPR atom/block name used in .place and constraints
        if block_type in ["dsp", "bram", "clb", "io", "lut", "ble"] or nets:
            info = BlockNetInfo(
                block_type=block_type,
                instance_id=block_id,
                instance_name=instance,
                atom_name=atom_name,
                unique_nets=len(nets),
                net_names=nets,
            )
            blocks.append(info)

    # Start recursion from root's children
    for child in root.findall("block"):
        process_blocks_recursive(child, depth=0)

    # Sort by type, then by instance ID for consistency
    blocks.sort(key=lambda b: (b.block_type, b.instance_id))

    return blocks


def filter_blocks_by_type(blocks: list[BlockNetInfo], block_type: str) -> list[BlockNetInfo]:
    """
    Filter blocks to only those of a specific type.

    Args:
        blocks: List of BlockNetInfo objects
        block_type: Type to filter for ('dsp', 'bram', 'clb', etc.)

    Returns:
        Filtered list, sorted by instance_id.
    """
    filtered = [b for b in blocks if b.block_type == block_type]
    filtered.sort(key=lambda b: b.instance_id)
    return filtered


def sort_blocks_by_net_count(
    blocks: list[BlockNetInfo], descending: bool = True
) -> list[BlockNetInfo]:
    """
    Sort blocks by their net count.

    Args:
        blocks: List of BlockNetInfo objects
        descending: If True, sort highest net count first (default: True)

    Returns:
        Sorted list. Stable sort preserves instance_id order for ties.
    """
    return sorted(blocks, key=lambda b: b.unique_nets, reverse=descending)
