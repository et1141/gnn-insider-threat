from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from certgnn.lightning import BinaryClassifierLightning, build_lightning_module
from certgnn.lightning.metrics import binary_metrics_from_scores
from certgnn.models import MODEL_REGISTRY, build_model
from certgnn.models.gcn_transformer import GCNTransformerInsiderThreat
from certgnn.models.graph_pool_mlp import GraphPoolingMLP
from certgnn.preprocessing.user_level_split import build_user_splits


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


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def test_model_registry_lists_architectures():
    assert "gcn_lstm" in MODEL_REGISTRY
    assert "gcn_transformer" in MODEL_REGISTRY
    assert "graph_pool_mlp" in MODEL_REGISTRY


def test_build_model_returns_correct_class():
    m1 = build_model("graph_pool_mlp", input_dim=2, hidden_dim=8)
    assert isinstance(m1, GraphPoolingMLP)


def test_build_model_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        build_model("does_not_exist")


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
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


def test_gcn_transformer_forward_pass():
    graph_1 = Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    graph_2 = Data(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    batch = Batch.from_data_list([graph_1, graph_2])
    model = GCNTransformerInsiderThreat(
        num_node_features=4,
        gcn_hidden_dim=8,
        d_model=8,
        nhead=2,
        transformer_num_layers=1,
        dim_feedforward=16,
        num_activity_classes=5,
        dropout=0.0,
    )

    logits = model(batch)

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_binary_metrics_helper():
    """Sanity-check that the unified metric helper produces sensible values
    for the same toy case the previous baseline test used."""
    import numpy as np
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = binary_metrics_from_scores(y_prob, y_true, target_fpr=None)

    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.5
    assert 0.0 <= metrics["roc_auc"] <= 1.0


# ---------------------------------------------------------------------------
# Lightning task wiring
# ---------------------------------------------------------------------------
def test_binary_classifier_training_step_runs():
    batch = _toy_batch()
    module = BinaryClassifierLightning(
        model_name="graph_pool_mlp",
        model_args={"input_dim": 2, "hidden_dim": 8, "dropout": 0.0, "pooling": "mean"},
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler=None,
    )
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_build_lightning_module_dispatches_correctly():
    module = build_lightning_module(
        task="binary",
        model_name="graph_pool_mlp",
        model_args={"input_dim": 2, "hidden_dim": 4},
        scheduler=None,
    )
    assert isinstance(module, BinaryClassifierLightning)
