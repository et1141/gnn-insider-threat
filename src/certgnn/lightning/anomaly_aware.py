"""Anomaly-aware Lightning task — paper-faithful (192-class output).

Trains the model to predict the masked activity. On normal samples it's
plain CE on the true class. On malicious samples the target is a uniform
distribution over the **non-true** classes — pushing the model away from
predicting the masked activity. At inference, residual ``p(true_class)``
becomes the per-graph anomaly signal: low confidence ⇒ flagged.

Final binary metrics use ``1 − p(true_class)`` as the anomaly score
against the per-graph binary ``y_label``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from certgnn.lightning.base import BaseLightningModule
from certgnn.lightning.metrics import binary_metrics_from_scores
from certgnn.losses import anomaly_aware_loss, standard_cross_entropy


class AnomalyAwareLightning(BaseLightningModule):
    """Activity-prediction task with the paper's anomaly-aware soft-label CE."""

    monitor_metric = "val/loss"
    monitor_mode = "min"

    def __init__(
        self,
        model_name: str = "gcn_lstm",
        model_args: dict | None = None,
        loss_type: str = "anomaly_aware",
        target_fpr: float = 0.05,
        **base_kwargs,
    ):
        super().__init__(model_name=model_name, model_args=model_args, **base_kwargs)
        # save_hyperparameters() in the base call already grabbed our locals
        # via inspection; explicitly re-saving here picks up loss_type/target_fpr.
        self.save_hyperparameters("loss_type", "target_fpr")

    def compute_loss(self, batch, logits: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss_type == "anomaly_aware":
            return anomaly_aware_loss(logits, batch.y_act, batch.y_label)
        if self.hparams.loss_type == "standard":
            return standard_cross_entropy(logits, batch.y_act)
        raise ValueError(f"Unknown loss_type {self.hparams.loss_type!r}")

    def collect_eval(self, batch, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Anomaly score = 1 − p(true_class). Higher = more anomalous.
        probs = F.softmax(logits.float(), dim=1)
        p_true = probs.gather(1, batch.y_act.unsqueeze(1)).squeeze(1)
        scores = 1.0 - p_true
        return scores, batch.y_label.long()

    def epoch_metrics(self, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return binary_metrics_from_scores(
            scores.numpy(), labels.numpy(), target_fpr=self.hparams.target_fpr,
        )


__all__ = ["AnomalyAwareLightning"]
