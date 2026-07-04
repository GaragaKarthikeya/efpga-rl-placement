#!/usr/bin/env python3
"""
Collapse a raw per-instance netlist connectivity graph (one node per CLB/DSP/
BRAM/IO atom, as produced by `analyzer.build_netlist_graph`/
`save_netlist_graph_json`) into a small graph the RL policy can actually
consume: one node per DSP/BRAM block being placed, plus a single aggregate
"fabric" node representing everything else (CLB + IO + other).

Raw graphs are dense enough to make this necessary, not just convenient —
diffeq1's raw graph is 300 nodes / 18,781 edges for a benchmark with only 5
placeable blocks; a 1000+ node benchmark like conv_layer would be far worse.
The placement decision only needs how DSP/BRAM blocks relate to each other
and to the surrounding fabric in aggregate.

Node ordering in the reduced graph is fixed per benchmark (DSP by
instance_id ascending, then BRAM by instance_id ascending, then the fabric
node) — independent of episode-to-episode placement order, which is decided
separately by `FPGAEnv._build_sorted_placement`. The env looks up which
reduced-graph node corresponds to "the block being placed this step" via
`ReducedGraph.node_id_for(block_type, instance_id)`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Node feature layout: [is_dsp, is_bram, is_fabric, net_count_normalized]
NUM_NODE_FEATURES = 4

# Edge weights are log-compressed into [0, 1] separately by edge kind, since
# direct DSP/BRAM<->DSP/BRAM weights (raw shared-net count between two
# specific blocks) and DSP/BRAM<->fabric weights (summed shared-net count
# across every individual CLB/IO member a block touches) are different
# quantities on different scales (~1-76 vs ~9-1200 observed) — a single
# shared ceiling either saturates the fabric edges or crushes the direct
# edges near zero. Each ceiling is the observed max across the current
# benchmark set with a small margin; see _check_ceiling for the runtime
# guard against a future benchmark silently exceeding it.
DIRECT_EDGE_CEILING = 80.0
FABRIC_EDGE_CEILING = 1300.0


def _check_ceiling(w: float, ceiling: float, kind: str, benchmark_name: str) -> None:
    """Both ceilings are calibrated to the max observed across the current
    benchmark set, not a structural bound — warn loudly if a benchmark's raw
    weight exceeds it, since past this point the edge silently saturates to
    1.0 again (the exact bug this normalization was introduced to fix)."""
    if w > ceiling:
        import sys
        print(
            f"[Warning] {benchmark_name}: {kind} edge weight {w} exceeds "
            f"ceiling {ceiling} — this edge will saturate to 1.0, losing "
            f"resolution. Consider raising the ceiling.",
            file=sys.stderr,
        )


@dataclass
class ReducedGraph:
    benchmark_name: str
    req_dsp: int
    req_bram: int
    node_features: np.ndarray  # (num_nodes, NUM_NODE_FEATURES) float32
    edge_index: np.ndarray  # (num_edges, 2) int64, symmetrized (both directions present)
    edge_weight: np.ndarray  # (num_edges,) float32, normalized to [0, 1]

    @property
    def num_nodes(self) -> int:
        return self.req_dsp + self.req_bram + 1

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[0]

    @property
    def fabric_node_id(self) -> int:
        return self.req_dsp + self.req_bram

    def node_id_for(self, block_type: str, instance_id: int) -> int:
        """block_type: 'dsp' or 'bram'. `instance_id` is the RAW VTR instance
        id (e.g. mult_36[k]'s k) — NOT assumed to be dense/0-indexed; VTR
        numbers DSP/BRAM instances from a shared counter, so e.g. a
        benchmark with 48 BRAMs can have its first DSP at raw instance_id
        48, not 0. Use `dsp_instance_ids`/`bram_instance_ids` to iterate
        over the actual raw ids for a type, not `range(req_dsp)`."""
        if block_type == "dsp":
            return self._dsp_rank[instance_id]
        if block_type == "bram":
            return self.req_dsp + self._bram_rank[instance_id]
        raise ValueError(f"node_id_for: unknown block_type {block_type!r}")

    def net_count_normalized_for(self, block_type: str, instance_id: int) -> float:
        return float(self.node_features[self.node_id_for(block_type, instance_id), 3])

    @property
    def dsp_instance_ids(self) -> list[int]:
        """Raw VTR instance ids of DSP blocks, ordered by dense rank (0..req_dsp-1)."""
        return sorted(self._dsp_rank, key=self._dsp_rank.get)

    @property
    def bram_instance_ids(self) -> list[int]:
        """Raw VTR instance ids of BRAM blocks, ordered by dense rank (0..req_bram-1)."""
        return sorted(self._bram_rank, key=self._bram_rank.get)

    def __post_init__(self) -> None:
        self._dsp_rank: dict[int, int] = {}
        self._bram_rank: dict[int, int] = {}


def reduce_netlist_graph(graph_json_path: Path) -> ReducedGraph:
    data = json.loads(Path(graph_json_path).read_text())
    nodes = data["nodes"]
    edges = data["edges"]

    benchmark_name = Path(graph_json_path).stem.replace("_netlist_graph", "")

    dsp_nodes = sorted((n for n in nodes if n["type"] == "dsp"), key=lambda n: n["instance_id"])
    bram_nodes = sorted((n for n in nodes if n["type"] == "bram"), key=lambda n: n["instance_id"])
    fabric_raw_ids = {n["id"] for n in nodes if n["type"] not in ("dsp", "bram")}

    req_dsp, req_bram = len(dsp_nodes), len(bram_nodes)
    fabric_id = req_dsp + req_bram

    raw_to_reduced: dict[int, int] = {}
    dsp_rank: dict[int, int] = {}
    bram_rank: dict[int, int] = {}
    for i, n in enumerate(dsp_nodes):
        raw_to_reduced[n["id"]] = i
        dsp_rank[n["instance_id"]] = i
    for i, n in enumerate(bram_nodes):
        raw_to_reduced[n["id"]] = req_dsp + i
        bram_rank[n["instance_id"]] = i
    for raw_id in fabric_raw_ids:
        raw_to_reduced[raw_id] = fabric_id

    node_features = np.zeros((req_dsp + req_bram + 1, NUM_NODE_FEATURES), dtype=np.float32)
    for i, n in enumerate(dsp_nodes):
        node_features[i] = [1.0, 0.0, 0.0, n["net_count_normalized"]]
    for i, n in enumerate(bram_nodes):
        node_features[req_dsp + i] = [0.0, 1.0, 0.0, n["net_count_normalized"]]
    fabric_members = [n for n in nodes if n["type"] not in ("dsp", "bram")]
    fabric_avg_net_count = (
        sum(n["net_count_normalized"] for n in fabric_members) / len(fabric_members)
        if fabric_members else 0.0
    )
    node_features[fabric_id] = [0.0, 0.0, 1.0, fabric_avg_net_count]

    # Aggregate edges: direct DSP/BRAM<->DSP/BRAM edges kept as-is; any edge
    # touching the fabric is summed into one DSP/BRAM<->fabric edge per
    # placeable block; fabric<->fabric edges are dropped (no signal needed
    # about internal CLB/IO connectivity once collapsed to one node).
    pair_weight: dict[tuple[int, int], float] = {}
    for e in edges:
        a, b = raw_to_reduced[e["src"]], raw_to_reduced[e["dst"]]
        if a == fabric_id and b == fabric_id:
            continue
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        pair_weight[key] = pair_weight.get(key, 0.0) + e["weight"]

    edge_index_list: list[tuple[int, int]] = []
    edge_weight_list: list[float] = []
    for (a, b), w in pair_weight.items():
        is_fabric_edge = fabric_id in (a, b)
        ceiling = FABRIC_EDGE_CEILING if is_fabric_edge else DIRECT_EDGE_CEILING
        _check_ceiling(w, ceiling, "fabric" if is_fabric_edge else "direct", benchmark_name)
        w_norm = min(np.log1p(w) / np.log1p(ceiling), 1.0)
        edge_index_list.append((a, b))
        edge_weight_list.append(w_norm)
        edge_index_list.append((b, a))
        edge_weight_list.append(w_norm)

    edge_index = np.array(edge_index_list, dtype=np.int64).reshape(-1, 2)
    edge_weight = np.array(edge_weight_list, dtype=np.float32)

    graph = ReducedGraph(
        benchmark_name=benchmark_name,
        req_dsp=req_dsp,
        req_bram=req_bram,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    graph._dsp_rank = dsp_rank
    graph._bram_rank = bram_rank
    return graph


def pad_graph(
    graph: ReducedGraph, max_nodes: int, max_edges: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad a ReducedGraph's tensors to fixed (max_nodes, max_edges) shapes.

    Padding rows for node_features are all-zero. Padding rows for edge_index
    use -1 sentinels (never a valid node id) so consumers can detect real
    edge count via `edge_index[:, 0] >= 0`.
    """
    if graph.num_nodes > max_nodes:
        raise ValueError(f"{graph.benchmark_name}: {graph.num_nodes} nodes exceeds max_nodes={max_nodes}")
    if graph.num_edges > max_edges:
        raise ValueError(f"{graph.benchmark_name}: {graph.num_edges} edges exceeds max_edges={max_edges}")

    node_features = np.zeros((max_nodes, NUM_NODE_FEATURES), dtype=np.float32)
    node_features[: graph.num_nodes] = graph.node_features

    edge_index = np.full((max_edges, 2), -1, dtype=np.int64)
    edge_index[: graph.num_edges] = graph.edge_index

    edge_weight = np.zeros((max_edges,), dtype=np.float32)
    edge_weight[: graph.num_edges] = graph.edge_weight

    return node_features, edge_index, edge_weight
