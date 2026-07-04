#!/usr/bin/env python3
"""
GNN-based features extractor for the multi-benchmark FPGA placement policy.

Plugs into `MaskableMultiInputActorCriticPolicy` ("MultiInputPolicy") via
`policy_kwargs={"features_extractor_class": GNNFeaturesExtractor, ...}`.
Trained end-to-end with the rest of the actor-critic network under the
standard PPO loss — there is no separate training step for the GNN.

Expects a Dict observation space with these keys (see `FPGAEnv`):
    grid               : Box(MAX_W, MAX_H, 4)
    node_features      : Box(MAX_NODES, NUM_NODE_FEATURES)
    edge_index         : Box(MAX_EDGES, 2)   -- -1 sentinel marks padding
    edge_weight        : Box(MAX_EDGES,)
    current_block_idx  : Box(1,)             -- reduced-graph node id being placed this step
    valid_wh           : Box(2,)             -- active benchmark's real (width, height), normalized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_hidden_dim: int = 64,
        cnn_hidden_dim: int = 64,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        node_feat_dim = observation_space["node_features"].shape[1]
        grid_w, grid_h, grid_c = observation_space["grid"].shape

        self.conv1 = GCNConv(node_feat_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(grid_c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, cnn_hidden_dim),
            nn.ReLU(),
        )

        combined_dim = cnn_hidden_dim + gnn_hidden_dim + gnn_hidden_dim + 2
        self.final = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def _build_batch(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Batch:
        """node_features: (B, MAX_NODES, F); edge_index: (B, MAX_EDGES, 2) float
        (-1 sentinel for padding); edge_weight: (B, MAX_EDGES)."""
        data_list = []
        edge_index = edge_index.long()
        for i in range(node_features.shape[0]):
            nf = node_features[i]
            node_valid = nf[:, :3].sum(dim=-1) > 0
            n_valid = int(node_valid.sum().item())

            ei = edge_index[i]
            edge_valid = ei[:, 0] >= 0
            ei_valid = ei[edge_valid].t().contiguous()
            ew_valid = edge_weight[i][edge_valid]

            data_list.append(Data(x=nf[:n_valid], edge_index=ei_valid, edge_attr=ew_valid))
        return Batch.from_data_list(data_list)

    def forward(self, observations: dict) -> torch.Tensor:
        grid = observations["grid"]
        node_features = observations["node_features"]
        edge_index = observations["edge_index"]
        edge_weight = observations["edge_weight"]
        current_block_idx = observations["current_block_idx"].long().squeeze(-1).clamp(min=0)
        valid_wh = observations["valid_wh"]

        device = grid.device
        batch = self._build_batch(node_features, edge_index, edge_weight).to(device)

        h = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
        h = F.relu(self.conv2(h, batch.edge_index, batch.edge_attr))

        graph_embed = global_mean_pool(h, batch.batch)

        # batch.ptr[i] is the offset of graph i's nodes within the concatenated h
        current_global_idx = batch.ptr[:-1] + current_block_idx
        current_embed = h[current_global_idx]

        grid_chw = grid.permute(0, 3, 1, 2)
        cnn_out = self.cnn(grid_chw)

        combined = torch.cat([cnn_out, graph_embed, current_embed, valid_wh], dim=-1)
        return self.final(combined)
