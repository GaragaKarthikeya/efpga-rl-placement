#!/usr/bin/env python3
"""
High-level netlist analysis and graph generation.

Provides utilities to load parsed netlists, build connectivity graphs,
and generate node/edge representations for GNN training.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.netlist.parser import BlockNetInfo, filter_blocks_by_type, parse_net_file


@dataclass
class NetlistStats:
    """Statistics and block groupings from a parsed netlist."""

    benchmark_name: str
    dsp_blocks: list[BlockNetInfo] = field(default_factory=list)
    bram_blocks: list[BlockNetInfo] = field(default_factory=list)
    clb_blocks: list[BlockNetInfo] = field(default_factory=list)
    other_blocks: list[BlockNetInfo] = field(default_factory=list)

    def total_blocks(self) -> int:
        """Total number of all blocks."""
        return len(self.dsp_blocks) + len(self.bram_blocks) + len(self.clb_blocks) + len(self.other_blocks)

    def summary(self) -> str:
        """Return human-readable summary of netlist statistics."""
        lines = [
            f"Netlist: {self.benchmark_name}",
            f"  DSPs:  {len(self.dsp_blocks)} blocks, "
            f"avg {sum(b.unique_nets for b in self.dsp_blocks) / max(1, len(self.dsp_blocks)):.1f} nets/block",
            f"  BRAMs: {len(self.bram_blocks)} blocks, "
            f"avg {sum(b.unique_nets for b in self.bram_blocks) / max(1, len(self.bram_blocks)):.1f} nets/block",
            f"  CLBs:  {len(self.clb_blocks)} blocks, "
            f"avg {sum(b.unique_nets for b in self.clb_blocks) / max(1, len(self.clb_blocks)):.1f} nets/block",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (net counts only)."""
        result = {}

        for block in self.dsp_blocks:
            result[f"('DSP', {block.instance_id})"] = block.unique_nets

        for block in self.bram_blocks:
            result[f"('BRAM', {block.instance_id})"] = block.unique_nets

        return result


@dataclass
class NetlistGraph:
    """
    Represents the netlist as a graph for GNN training.

    Nodes: Each block instance (DSP, BRAM, CLB, I/O)
    Edges: Weighted connections representing shared nets between blocks
           Edge weight = number of nets shared between two blocks
    """

    nodes: list[dict] = field(default_factory=list)  # List of node feature dicts
    edges: list[dict] = field(default_factory=list)  # List of {src, dst, weight, nets} dicts
    node_id_map: dict[str, int] = field(default_factory=dict)  # Map instance_name -> node_id

    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    def total_edge_weight(self) -> int:
        """Total number of nets across all edges."""
        return sum(e.get("weight", 0) for e in self.edges)


def analyze_netlist(benchmark_name: str, net_file_path: Path) -> NetlistStats:
    """
    Load and analyze a netlist from a .net file.

    Args:
        benchmark_name: Name of the benchmark (for reference)
        net_file_path: Path to the .net file

    Returns:
        NetlistStats object with parsed and grouped blocks.

    Raises:
        FileNotFoundError: If net_file_path does not exist
        ET.ParseError: If XML is malformed
    """
    blocks = parse_net_file(net_file_path)

    stats = NetlistStats(
        benchmark_name=benchmark_name,
        dsp_blocks=filter_blocks_by_type(blocks, "dsp"),
        bram_blocks=filter_blocks_by_type(blocks, "bram"),
        clb_blocks=filter_blocks_by_type(blocks, "clb"),
        other_blocks=[b for b in blocks if b.block_type not in ["dsp", "bram", "clb"]],
    )

    return stats


def build_netlist_graph(
    benchmark_name: str, net_file_path: Path, include_clbs: bool = True, include_ios: bool = True
) -> NetlistGraph:
    """
    Build a graph representation of the netlist for GNN training.

    Nodes represent block instances (DSPs, BRAMs, CLBs, I/Os).
    Edges represent shared nets between blocks, with weight = number of shared nets.

    Args:
        benchmark_name: Name of the benchmark
        net_file_path: Path to the .net file
        include_clbs: If True, include CLB nodes in the graph (default: True)
        include_ios: If True, include I/O pad nodes in the graph (default: True)

    Returns:
        NetlistGraph object ready for GNN training.
    """
    # Parse all blocks first (we need ios even if not graphing them)
    blocks = parse_net_file(net_file_path)
    all_blocks_by_type = {}
    for block in blocks:
        if block.block_type not in all_blocks_by_type:
            all_blocks_by_type[block.block_type] = []
        all_blocks_by_type[block.block_type].append(block)

    stats = analyze_netlist(benchmark_name, net_file_path)

    graph = NetlistGraph()
    node_id = 0

    # Add DSP nodes
    for block in stats.dsp_blocks:
        graph.node_id_map[block.instance_name] = node_id
        graph.nodes.append(
            {
                "id": node_id,
                "instance_name": block.instance_name,
                "type": "dsp",
                "instance_id": block.instance_id,
                "net_count": block.unique_nets,
                "net_count_normalized": min(block.unique_nets / 200.0, 1.0),
            }
        )
        node_id += 1

    # Add BRAM nodes
    for block in stats.bram_blocks:
        graph.node_id_map[block.instance_name] = node_id
        graph.nodes.append(
            {
                "id": node_id,
                "instance_name": block.instance_name,
                "type": "bram",
                "instance_id": block.instance_id,
                "net_count": block.unique_nets,
                "net_count_normalized": min(block.unique_nets / 200.0, 1.0),
            }
        )
        node_id += 1

    # Add CLB nodes if requested
    if include_clbs:
        for block in stats.clb_blocks:
            graph.node_id_map[block.instance_name] = node_id
            graph.nodes.append(
                {
                    "id": node_id,
                    "instance_name": block.instance_name,
                    "type": "clb",
                    "instance_id": block.instance_id,
                    "net_count": block.unique_nets,
                    "net_count_normalized": min(block.unique_nets / 200.0, 1.0),
                }
            )
            node_id += 1

    # Add I/O nodes if requested
    if include_ios:
        io_blocks = all_blocks_by_type.get("io", [])
        for block in io_blocks:
            graph.node_id_map[block.instance_name] = node_id
            graph.nodes.append(
                {
                    "id": node_id,
                    "instance_name": block.instance_name,
                    "type": "io",
                    "instance_id": block.instance_id,
                    "net_count": block.unique_nets,
                    "net_count_normalized": min(block.unique_nets / 200.0, 1.0),
                }
            )
            node_id += 1

    # Build edges with weights
    # Create a mapping: net_name -> [list of blocks that use it]
    net_to_blocks: dict[str, list[BlockNetInfo]] = {}

    all_blocks = stats.dsp_blocks + stats.bram_blocks
    if include_clbs:
        all_blocks += stats.clb_blocks
    if include_ios:
        all_blocks += all_blocks_by_type.get("io", [])

    for block in all_blocks:
        for net_name in block.net_names:
            if net_name not in net_to_blocks:
                net_to_blocks[net_name] = []
            net_to_blocks[net_name].append(block)

    # Build edges with weights: track number of shared nets per edge
    edges_dict: dict[tuple[int, int], dict] = {}  # (src, dst) -> {weight, nets[]}

    for net_name, connected_blocks in net_to_blocks.items():
        if len(connected_blocks) >= 2:
            # Connect all pairs of blocks sharing this net
            for i in range(len(connected_blocks)):
                for j in range(i + 1, len(connected_blocks)):
                    block_i = connected_blocks[i]
                    block_j = connected_blocks[j]

                    # Skip if either block is not in the graph
                    if (
                        block_i.instance_name not in graph.node_id_map
                        or block_j.instance_name not in graph.node_id_map
                    ):
                        continue

                    node_i = graph.node_id_map[block_i.instance_name]
                    node_j = graph.node_id_map[block_j.instance_name]

                    # Store as sorted tuple for consistency
                    edge_key = tuple(sorted([node_i, node_j]))

                    if edge_key not in edges_dict:
                        edges_dict[edge_key] = {"weight": 0, "nets": []}

                    edges_dict[edge_key]["weight"] += 1
                    edges_dict[edge_key]["nets"].append(net_name)

    # Convert to list of edge dicts with src/dst/weight/nets
    graph.edges = [
        {
            "src": edge_key[0],
            "dst": edge_key[1],
            "weight": edge_data["weight"],
            "nets": edge_data["nets"],
        }
        for edge_key, edge_data in sorted(edges_dict.items())
    ]

    return graph


def save_net_count_json(stats: NetlistStats, output_path: Path) -> None:
    """
    Save net counts to JSON file for use in training.

    Output format:
    {
        "('DSP', 0)": 45,
        "('DSP', 1)": 62,
        ...
        "('BRAM', 0)": 128,
        ...
    }

    Args:
        stats: NetlistStats object
        output_path: Path to write JSON file
    """
    data = stats.to_dict()
    output_path.write_text(json.dumps(data, indent=2))


def save_netlist_graph_json(graph: NetlistGraph, output_path: Path) -> None:
    """
    Save netlist graph to JSON for visualization/debugging.

    Includes node features, edges with weights (number of shared nets), and statistics.

    Args:
        graph: NetlistGraph object
        output_path: Path to write JSON file
    """
    data = {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "num_nodes": graph.num_nodes(),
        "num_edges": graph.num_edges(),
        "total_edge_weight": graph.total_edge_weight(),
        "summary": {
            "node_types": {
                "dsp": sum(1 for n in graph.nodes if n["type"] == "dsp"),
                "bram": sum(1 for n in graph.nodes if n["type"] == "bram"),
                "clb": sum(1 for n in graph.nodes if n["type"] == "clb"),
                "io": sum(1 for n in graph.nodes if n["type"] == "io"),
            },
            "description": "Each edge has weight = number of nets connecting the two blocks",
        },
    }
    output_path.write_text(json.dumps(data, indent=2))
