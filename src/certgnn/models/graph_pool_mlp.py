"""Lightweight graph pooling baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool


class GraphPoolingMLP(nn.Module):
    """Graph-level baseline with node MLP + pooling + classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean_max",
        num_classes: int = 2,
    ):
        super().__init__()
        if pooling not in {"mean", "max", "mean_max"}:
            raise ValueError("pooling must be one of: mean, max, mean_max")

        self.pooling = pooling
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        pooled_dim = hidden_dim * 2 if pooling == "mean_max" else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch):
        x = batch.x.float()
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        encoded = self.node_encoder(x)
        if self.pooling == "mean":
            pooled = global_mean_pool(encoded, batch_index)
        elif self.pooling == "max":
            pooled = global_max_pool(encoded, batch_index)
        else:
            pooled = torch.cat(
                [
                    global_mean_pool(encoded, batch_index),
                    global_max_pool(encoded, batch_index),
                ],
                dim=-1,
            )

        return self.classifier(pooled)
