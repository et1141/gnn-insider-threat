"""GCN + Transformer encoder for activity prediction (anomaly-aware task).

Same graph pipeline as :class:`~certgnn.models.gcn_lstm.GCNLSTMInsiderThreat`:

  1. GCN node embeddings (shared :class:`~certgnn.models.gcn_lstm.GCN`)
  2. ``to_dense_batch`` → padded node sequences per graph
  3. TransformerEncoder over the sequence (padding masked in attention)
  4. Masked mean pool → linear head → activity logits

Defaults are sized for modest GPUs (e.g. 4 GB VRAM); use a smaller
``batch_size`` via CLI when training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from certgnn.models.gcn_lstm import GCN


class ActivityTransformerPredictor(nn.Module):
    """Transformer encoder on padded per-graph node embedding sequences."""

    def __init__(
        self,
        input_dim: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.input_proj = (
            nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node embeddings ``[batch, seq_len, input_dim]``.
            padding_mask: ``True`` at padded positions (invalid nodes), same shape
                as attention ``src_key_padding_mask`` for ``TransformerEncoder``.

        Returns:
            Logits ``[batch, output_dim]``.
        """
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.dropout(x)

        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


class GCNTransformerInsiderThreat(nn.Module):
    """End-to-end GCN + Transformer for insider threat activity prediction."""

    def __init__(
        self,
        num_node_features: int = 54,
        gcn_hidden_dim: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        transformer_num_layers: int = 2,
        dim_feedforward: int = 128,
        num_activity_classes: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gcn = GCN(num_node_features=num_node_features, hidden_dim=gcn_hidden_dim)
        self.predictor = ActivityTransformerPredictor(
            input_dim=gcn_hidden_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=dim_feedforward,
            output_dim=num_activity_classes,
            dropout=dropout,
        )

    def forward(self, batch_data, max_seq_len: int | None = None) -> torch.Tensor:
        node_embeddings = self.gcn(batch_data)
        padded, mask = to_dense_batch(
            node_embeddings,
            batch_data.batch,
            max_num_nodes=max_seq_len,
        )
        # ``mask`` from PyG: True = valid node; Transformer expects True = pad.
        padding_mask = ~mask
        return self.predictor(padded, padding_mask)
