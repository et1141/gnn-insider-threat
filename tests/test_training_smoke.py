"""End-to-end training smoke tests.

Run a single train+val batch through a real ``pl.Trainer`` for both task/model
combinations, on synthetic graphs (no dataset/DVC dependency). This exercises
the full Lightning loop: forward -> loss -> eval metrics -> optimizer step.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from certgnn.lightning import build_lightning_module


def _synthetic_graphs(n: int, num_features: int, num_classes: int) -> list[Data]:
    graphs = []
    for i in range(n):
        k = 3 + (i % 3)  # 3..5 nodes
        edge_index = torch.tensor(
            [list(range(k - 1)), list(range(1, k))], dtype=torch.long
        )
        graphs.append(
            Data(
                x=torch.randn(k, num_features),
                edge_index=edge_index,
                y_act=torch.tensor(i % num_classes, dtype=torch.long),
                y_label=torch.tensor(i % 2, dtype=torch.long),
            )
        )
    return graphs


def _run_smoke(task: str, model_name: str, num_features: int, num_classes: int, model_args: dict):
    pl.seed_everything(0, workers=True)
    loader = DataLoader(_synthetic_graphs(8, num_features, num_classes), batch_size=4)
    module = build_lightning_module(
        task=task, model_name=model_name, model_args=model_args, scheduler=None,
    )
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    # A finite training loss must have been recorded.
    train_loss = trainer.callback_metrics.get("train/loss")
    assert train_loss is not None and torch.isfinite(train_loss)


def test_training_smoke_anomaly_aware_gcn_lstm():
    _run_smoke(
        task="anomaly_aware",
        model_name="gcn_lstm",
        num_features=6,
        num_classes=5,
        model_args={
            "num_node_features": 6,
            "gcn_hidden_dim": 8,
            "lstm_hidden_dim": 8,
            "num_activity_classes": 5,
            "lstm_num_layers": 1,
        },
    )


def test_training_smoke_binary_graph_pool_mlp():
    _run_smoke(
        task="binary",
        model_name="graph_pool_mlp",
        num_features=6,
        num_classes=2,
        model_args={"input_dim": 6, "hidden_dim": 8, "dropout": 0.0, "pooling": "mean"},
    )
