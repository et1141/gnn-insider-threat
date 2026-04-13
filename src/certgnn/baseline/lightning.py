"""PyTorch Lightning wrapper for the graph pooling baseline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

from certgnn.baseline.model import GraphPoolingMLP


def _extract_labels(batch: Any) -> torch.Tensor:
    if hasattr(batch, "y_label") and batch.y_label is not None:
        return batch.y_label.view(-1).long()
    if hasattr(batch, "y") and batch.y is not None:
        return batch.y.view(-1).long()
    raise AttributeError("Batch does not contain y_label or y")


def compute_binary_metrics(y_true: torch.Tensor, y_prob: torch.Tensor) -> dict[str, float]:
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    y_prob_np = y_prob.detach().cpu().numpy().astype(float)
    y_pred_np = (y_prob_np >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true_np, y_prob_np)
    except ValueError:
        roc_auc = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


class GraphBaselineLightningModule(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pooling: str = "mean_max",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphPoolingMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pooling=pooling,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._val_outputs: list[dict[str, torch.Tensor]] = []
        self._test_outputs: list[dict[str, torch.Tensor]] = []

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, prefix: str):
        logits = self(batch)
        labels = _extract_labels(batch)
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        batch_size = int(labels.shape[0])
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(prefix != "train"),
            batch_size=batch_size,
        )
        return loss, labels.detach(), probs.detach()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, probs = self._shared_step(batch, "val")
        self._val_outputs.append({"y_true": labels.cpu(), "y_prob": probs.cpu(), "loss": loss.detach().cpu()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, labels, probs = self._shared_step(batch, "test")
        self._test_outputs.append({"y_true": labels.cpu(), "y_prob": probs.cpu(), "loss": loss.detach().cpu()})
        return loss

    def _aggregate_outputs(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if not outputs:
            return {}
        y_true = torch.cat([item["y_true"] for item in outputs], dim=0)
        y_prob = torch.cat([item["y_prob"] for item in outputs], dim=0)
        loss = torch.stack([item["loss"] for item in outputs]).mean().item()
        metrics = compute_binary_metrics(y_true, y_prob)
        metrics["loss"] = float(loss)
        return metrics

    def on_validation_epoch_end(self):
        metrics = self._aggregate_outputs(self._val_outputs)
        self._val_outputs.clear()
        if metrics:
            self.log_dict({f"val/{key}": value for key, value in metrics.items()}, prog_bar=True)

    def on_test_epoch_end(self):
        metrics = self._aggregate_outputs(self._test_outputs)
        self._test_outputs.clear()
        if metrics:
            self.log_dict({f"test/{key}": value for key, value in metrics.items()}, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
