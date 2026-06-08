"""Binary classification Lightning task — focal-loss baseline (2-class output).

Models that output ``[N, 2]`` logits, trained with focal loss.
``softmax[:, 1]`` becomes the anomaly score for binary metrics.
Checkpoint/scheduler track ``val/pr_auc`` — the meaningful objective at
~0.2% positive rate (ROC-AUC is misleadingly optimistic here).

The focal variant is selectable via ``focal_variant``:

* ``"fixed_alpha"`` (default) — constant ``alpha`` class weight; stable.
* ``"dynamic_pos_weight"`` — legacy per-batch ``n_neg/n_pos`` weighting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from certgnn.lightning.base import BaseLightningModule
from certgnn.lightning.metrics import binary_metrics_from_scores
from certgnn.losses import (
    focal_loss_fixed_alpha,
    focal_loss_with_dynamic_pos_weight,
)

FOCAL_VARIANTS = ("fixed_alpha", "dynamic_pos_weight")


def _extract_label(batch) -> torch.Tensor:
    if hasattr(batch, "y_label") and batch.y_label is not None:
        return batch.y_label.view(-1).long()
    if hasattr(batch, "y") and batch.y is not None:
        return batch.y.view(-1).long()
    raise AttributeError("Batch has no y_label or y attribute for binary classification.")


class BinaryClassifierLightning(BaseLightningModule):
    """Binary task with selectable focal loss (fixed-alpha or dynamic weight)."""

    monitor_metric = "val/pr_auc"
    monitor_mode = "max"

    def __init__(
        self,
        model_name: str = "graph_pool_mlp",
        model_args: dict | None = None,
        focal_variant: str = "fixed_alpha",
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight_clamp: float = 1000.0,
        **base_kwargs,
    ):
        if focal_variant not in FOCAL_VARIANTS:
            raise ValueError(
                f"Unknown focal_variant {focal_variant!r}. Choose from {FOCAL_VARIANTS}."
            )
        super().__init__(model_name=model_name, model_args=model_args, **base_kwargs)
        self.save_hyperparameters("focal_variant", "gamma", "alpha", "pos_weight_clamp")

    def compute_loss(self, batch, logits: torch.Tensor) -> torch.Tensor:
        labels = _extract_label(batch)
        if self.hparams.focal_variant == "dynamic_pos_weight":
            return focal_loss_with_dynamic_pos_weight(
                logits, labels,
                gamma=self.hparams.gamma,
                pos_weight_clamp=self.hparams.pos_weight_clamp,
            )
        return focal_loss_fixed_alpha(
            logits, labels,
            gamma=self.hparams.gamma,
            alpha=self.hparams.alpha,
        )

    def collect_eval(self, batch, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits.float(), dim=1)
        scores = probs[:, 1]
        return scores, _extract_label(batch)

    def epoch_metrics(self, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return binary_metrics_from_scores(scores.numpy(), labels.numpy())


__all__ = ["BinaryClassifierLightning"]
