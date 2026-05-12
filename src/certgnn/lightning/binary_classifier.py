"""Binary classification Lightning task — focal-loss baseline (2-class output).

Models that output ``[N, 2]`` logits, trained with batch-balanced focal
loss. ``softmax[:, 1]`` becomes the anomaly score for binary metrics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from certgnn.lightning.base import BaseLightningModule
from certgnn.lightning.metrics import binary_metrics_from_scores
from certgnn.losses import focal_loss_with_dynamic_pos_weight


def _extract_label(batch) -> torch.Tensor:
    if hasattr(batch, "y_label") and batch.y_label is not None:
        return batch.y_label.view(-1).long()
    if hasattr(batch, "y") and batch.y is not None:
        return batch.y.view(-1).long()
    raise AttributeError("Batch has no y_label or y attribute for binary classification.")


class BinaryClassifierLightning(BaseLightningModule):
    """Binary task with focal loss + dynamic positive class weighting."""

    monitor_metric = "val/roc_auc"
    monitor_mode = "max"

    def __init__(
        self,
        model_name: str = "graph_pool_mlp",
        model_args: dict | None = None,
        gamma: float = 2.0,
        pos_weight_clamp: float = 1000.0,
        **base_kwargs,
    ):
        super().__init__(model_name=model_name, model_args=model_args, **base_kwargs)
        self.save_hyperparameters("gamma", "pos_weight_clamp")

    def compute_loss(self, batch, logits: torch.Tensor) -> torch.Tensor:
        labels = _extract_label(batch)
        return focal_loss_with_dynamic_pos_weight(
            logits, labels,
            gamma=self.hparams.gamma,
            pos_weight_clamp=self.hparams.pos_weight_clamp,
        )

    def collect_eval(self, batch, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits.float(), dim=1)
        scores = probs[:, 1]
        return scores, _extract_label(batch)

    def epoch_metrics(self, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return binary_metrics_from_scores(scores.numpy(), labels.numpy())


__all__ = ["BinaryClassifierLightning"]
