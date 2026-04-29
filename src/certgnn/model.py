"""Graph neural network architecture for insider threat detection.

Paper: GCN + LSTM for activity prediction with anomaly detection.
Architecture:
  1. GCN: 54 node features → 16-dim embeddings (2 layers)
  2. Unbatch & sequence pooling: extract per-graph embeddings from batched output
  3. LSTM: 16 → 32*2 (bidirectional) → 64-dim (2 layers)
  4. Mean pooling over time steps
  5. FC: 64 → 192 activity classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class GCN(nn.Module):
    """Graph Convolutional Network for node embedding.

    Input: batched PyG Data with node features and edge indices
    Output: node embeddings [total_nodes_in_batch, hidden_dim]
    """

    def __init__(self, num_node_features: int = 54, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, data):
        """Forward pass on batched graph data.

        Args:
            data: PyG batch with x [total_nodes, features], edge_index [2, num_edges],
                  batch [total_nodes] (node→graph assignment)

        Returns:
            Node embeddings [total_nodes, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index

        # Layer 1: GCN + ReLU + Dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2: GCN + ReLU
        x = F.relu(self.conv2(x, edge_index))

        return x  # [total_nodes, hidden_dim]


class ActivityPredictor(nn.Module):
    """LSTM-based activity predictor.

    Takes per-graph node embeddings (padded sequences), predicts activity class.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        """Forward pass on padded node embedding sequences.

        Args:
            x: Padded sequences [batch_size, max_seq_len, input_dim]

        Returns:
            Logits [batch_size, output_dim] or probabilities if apply_softmax=True
        """
        # LSTM with zero-initialized hidden states (no state carryover between batches)
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Mean-pool over time steps
        pooled = torch.mean(lstm_out, dim=1)  # [batch, hidden*2]

        # Fully connected layer
        logits = self.fc(pooled)  # [batch, output_dim]

        return logits

    def predict_proba(self, x):
        """Get softmax probabilities instead of logits."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class GCNLSTMInsiderThreat(nn.Module):
    """End-to-end GCN + LSTM model for insider threat detection.

    Takes batched PyG data, produces activity predictions.
    Intended for use with PyTorch Lightning module.
    """

    def __init__(
        self,
        num_node_features: int = 54,
        gcn_hidden_dim: int = 16,
        lstm_hidden_dim: int = 32,
        num_activity_classes: int = 192,
        lstm_num_layers: int = 2,
    ):
        super().__init__()
        self.gcn = GCN(
            num_node_features=num_node_features, hidden_dim=gcn_hidden_dim
        )
        self.predictor = ActivityPredictor(
            input_dim=gcn_hidden_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=num_activity_classes,
            num_layers=lstm_num_layers,
        )

    def forward(self, batch_data, max_seq_len: int | None = None):
        """Forward pass on a batch of graphs.

        Args:
            batch_data: PyG batch with x, edge_index, batch tensor
            max_seq_len: Override max sequence length (default: compute from batch)

        Returns:
            Logits [batch_size, num_activity_classes]
        """
        # GCN: produce node embeddings
        node_embeddings = self.gcn(batch_data)  # [total_nodes, gcn_hidden_dim]

        # Convert ragged node sequences per graph to dense tensor without
        # Python-side loops (better throughput than masking graph-by-graph).
        padded, _ = to_dense_batch(
            node_embeddings,
            batch_data.batch,
            max_num_nodes=max_seq_len,
        )

        # LSTM: process padded sequences
        logits = self.predictor(padded)  # [batch_size, num_activity_classes]

        return logits
