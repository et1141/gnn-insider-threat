from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from certgnn.baseline.lightning import (
    GraphBaselineLightningModule,
    compute_binary_metrics,
)
from certgnn.baseline.model import GraphPoolingMLP
from certgnn.baseline.split import build_user_splits


def _toy_batch() -> Batch:
    graph_1 = Data(
        x=torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y_label=torch.tensor(0),
    )
    graph_2 = Data(
        x=torch.tensor([[0.2, 1.2], [1.1, 0.1]]),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        y_label=torch.tensor(1),
    )
    return Batch.from_data_list([graph_1, graph_2])


def test_user_split_is_deterministic():
    user_ids = [f"user_{idx}" for idx in range(10)]
    user_labels = {user_id: int(idx >= 7) for idx, user_id in enumerate(user_ids)}

    first = build_user_splits(user_ids, user_labels, val_ratio=0.2, test_ratio=0.2, seed=7)
    second = build_user_splits(user_ids, user_labels, val_ratio=0.2, test_ratio=0.2, seed=7)

    assert first == second
    assert sorted(first["train"] + first["val"] + first["test"]) == sorted(user_ids)


def test_graph_pooling_forward_pass():
    batch = _toy_batch()
    model = GraphPoolingMLP(input_dim=2, hidden_dim=8, dropout=0.0, pooling="mean_max")

    logits = model(batch)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_compute_binary_metrics():
    y_true = torch.tensor([0, 0, 1, 1])
    y_prob = torch.tensor([0.1, 0.7, 0.8, 0.9])

    metrics = compute_binary_metrics(y_true, y_prob)

    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.5
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_lightning_training_step_runs():
    batch = _toy_batch()
    module = GraphBaselineLightningModule(
        input_dim=2,
        hidden_dim=8,
        dropout=0.0,
        pooling="mean",
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    loss = module.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
