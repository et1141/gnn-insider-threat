"""PyTorch Lightning module for insider threat detection with custom loss.

Wraps GCNLSTMInsiderThreat with training, validation, and evaluation logic.
Implements the anomaly-aware loss from the paper: suppress true activity class
for malicious examples, push model to flag anomalies via low confidence.
"""

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import pytorch_lightning as pl

from certgnn.models.gcn_lstm import GCNLSTMInsiderThreat


def _binary_metrics_from_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    target_fpr: float = 0.05,
) -> dict[str, float]:
    """Compute binary classification metrics from anomaly scores.

    The model outputs softmax probabilities over activity classes. We use
    `1 - p(true_class)` as the anomaly score: low confidence in the true
    activity = more anomalous. Binary label: 0 = normal, 1 = malicious.

    Returns NaN for ROC/PR-AUC when only one class is present (instead of
    raising) so the caller can still log all the count metrics.

    Non-finite scores (NaN/inf — typically from NaN logits in the model
    forward pass) are filtered out before any metric is computed, and the
    count of dropped samples is exposed as `n_nonfinite` so the caller can
    surface it (e.g., to W&B) and warn. If every score is non-finite,
    every metric except the counts is NaN.

    Args:
        scores: float array of anomaly scores, higher = more anomalous.
        labels: int array of binary ground truth (0/1).
        threshold: classification threshold on the anomaly score.
        target_fpr: target false-positive rate at which to report TPR. Paper
            section V-C uses 0.05 for r5.2 and 0.09 for r6.2; pick the
            largest threshold whose FPR is still ≤ target_fpr (eq. 30-31).
    """
    labels = labels.astype(int)
    scores = scores.astype(float)

    # Filter out non-finite scores (NaN/inf). They typically signal a NaN
    # logit somewhere in the forward pass (e.g., NaN in batch.x, bf16 softmax
    # overflow on extreme logits, GCN normalisation on isolated nodes).
    # roc_auc_score raises on non-finite input, so without this guard the
    # whole epoch_end blows up. Surface the count via `n_nonfinite` so the
    # failure mode is visible in W&B instead of silenced.
    finite_mask = np.isfinite(scores)
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite > 0:
        scores = scores[finite_mask]
        labels = labels[finite_mask]

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    n_total = labels.size

    out: dict[str, float] = {
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "n_nonfinite": float(n_nonfinite),
        "pos_frac": float(n_pos) / max(1, n_total),
        "target_fpr": float(target_fpr),
    }

    if n_total == 0:
        # All samples had non-finite scores; nothing meaningful to compute.
        for k in (
            "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "fpr", "tpr",
            "roc_auc", "pr_auc",
            "tpr_at_fpr_target", "fpr_at_target", "threshold_at_fpr_target",
        ):
            out[k] = math.nan
        return out

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out.update({
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    })

    out["accuracy"] = (tp + tn) / max(1, n_total)
    out["precision"] = tp / max(1, tp + fp)
    out["recall"] = tp / max(1, tp + fn)
    out["fpr"] = fp / max(1, fp + tn)
    out["tpr"] = out["recall"]

    if n_pos > 0 and n_neg > 0:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
        fpr_arr, tpr_arr, thresh_arr = roc_curve(labels, scores)
        # Paper eq. 30-31: t_optimal = max{t | FPR(t) ≤ target}. roc_curve
        # returns thresholds in decreasing order and fpr_arr ascending, so
        # the rightmost index where fpr is still ≤ target is the operating
        # point with maximum TPR meeting the FPR constraint. If no point
        # satisfies the constraint (target_fpr smaller than the smallest
        # achievable FPR), fall back to the strictest threshold (idx 0,
        # FPR=0).
        valid = np.where(fpr_arr <= target_fpr)[0]
        idx = int(valid[-1]) if valid.size > 0 else 0
        out["tpr_at_fpr_target"] = float(tpr_arr[idx])
        out["fpr_at_target"] = float(fpr_arr[idx])
        out["threshold_at_fpr_target"] = float(thresh_arr[idx])
    else:
        out["roc_auc"] = math.nan
        out["pr_auc"] = math.nan
        out["tpr_at_fpr_target"] = math.nan
        out["fpr_at_target"] = math.nan
        out["threshold_at_fpr_target"] = math.nan

    return out


class InsiderThreatLightning(pl.LightningModule):
    """Lightning module for insider threat detection.

    Loss function:
    - For normal samples: standard cross-entropy on true activity
    - For malicious samples: uniform distribution over all BUT the true activity
      (train model to flag true activity as anomalous)

    Evaluation metric: AUC using model's confidence in true activity as anomaly score
    (low confidence = anomalous, high confidence = normal)
    """

    def __init__(
        self,
        num_node_features: int = 54,
        gcn_hidden_dim: int = 16,
        lstm_hidden_dim: int = 32,
        num_activity_classes: int = 192,
        lstm_num_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        # "anomaly_aware" implements paper eq. 24-26 (recommended);
        # "standard" is plain F.cross_entropy on true class — useful as a
        # debug baseline but ignores y_label so malicious samples train the
        # model to predict the true class (opposite of paper's intent).
        loss_type: str = "standard",
        # Paper section V-C uses 0.05 for r5.2 and 0.09 for r6.2; override
        # via train.py --target-fpr when running on r6.2.
        target_fpr: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = GCNLSTMInsiderThreat(
            num_node_features=num_node_features,
            gcn_hidden_dim=gcn_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_activity_classes=num_activity_classes,
            lstm_num_layers=lstm_num_layers,
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_activity_classes = num_activity_classes
        self.loss_type = loss_type
        self.target_fpr = target_fpr

        self.val_preds: list[torch.Tensor] = []
        self.val_labels: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_labels: list[torch.Tensor] = []
        # One-shot warnings so we don't spam the log on every epoch:
        # - single_class: val/test set degenerated to one class (no AUC).
        # - nan_scores:   model produced NaN logits → NaN scores collected.
        self._warned_single_class_val = False
        self._warned_single_class_test = False
        self._warned_nan_scores_val = False
        self._warned_nan_scores_test = False
        self._warned_nonfinite_input_train = False
        self._warned_nonfinite_input_val = False
        self._warned_nonfinite_input_test = False
        self._warned_nonfinite_logits_train = False
        self._warned_nonfinite_logits_val = False
        self._warned_nonfinite_logits_test = False

    def forward(self, batch_data):
        return self.model(batch_data)

    def _sanitize_batch_x(self, batch, stage: str) -> int:
        """Replace NaN/inf in input features with finite values."""
        bad = ~torch.isfinite(batch.x)
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            warned_attr = f"_warned_nonfinite_input_{stage}"
            if not getattr(self, warned_attr):
                logger.warning(
                    f"{stage}: found {n_bad} non-finite values in batch.x; "
                    "replacing with finite values via torch.nan_to_num. "
                    "This points to a preprocessing/scaling issue upstream."
                )
                setattr(self, warned_attr, True)
            batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)
        return n_bad

    def _sanitize_logits(self, logits: torch.Tensor, stage: str) -> tuple[torch.Tensor, int]:
        """Replace NaN/inf logits so loss/metrics stay finite."""
        bad = ~torch.isfinite(logits)
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            warned_attr = f"_warned_nonfinite_logits_{stage}"
            if not getattr(self, warned_attr):
                logger.warning(
                    f"{stage}: found {n_bad} non-finite logits; replacing with "
                    "finite values via torch.nan_to_num to avoid NaN loss/metrics."
                )
                setattr(self, warned_attr, True)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits, n_bad

    def _anomaly_aware_loss(self, logits, y_act, y_label):
        """Soft-label cross-entropy from paper section IV-E (eq. 24-26).

        Builds the soft target distribution Y' for each sample:
        - Normal (y_label=0): Y' = one-hot(true_class) → standard CE on the
          true activity.
        - Malicious (y_label=1): Y' = uniform over all classes EXCEPT the true
          class (mass 1/(M−1) on each non-true class). Pushes the model AWAY
          from predicting the true class on malicious samples; the residual
          confidence in the true class then becomes the anomaly signal used
          at inference (low p(true) → flagged).

        Equivalent vectorised form of paper eq. 24:
            Y' = Y ⊙ (1 − Ω) + (1 / (M − 1)) · (1 − Y) ⊙ Ω
        where Y = one-hot(true_class), Ω_i = y_label_i broadcast across
        the class dimension, M = num activity classes.

        Implementation uses F.log_softmax + soft-label NLL so the loss stays
        stable under bf16/fp16 autocast (no log(p + eps) underflow that the
        previous "softmax + log(probs + 1e-10)" formulation suffered from
        on RTX 3090 with 16-mixed).
        """
        num_classes = logits.size(1)

        max_activity = y_act.max().item()
        if max_activity >= num_classes:
            raise RuntimeError(
                f"Activity index {max_activity} >= num_classes {num_classes}. "
                f"Check metadata['num_classes'] = {num_classes}, max(y_act) = {max_activity}"
            )

        Y = F.one_hot(y_act, num_classes).to(logits.dtype)
        omega = (y_label > 0).to(logits.dtype).unsqueeze(1)
        Y_prime = Y * (1.0 - omega) + (1.0 - Y) / (num_classes - 1) * omega

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(Y_prime * log_probs).sum(dim=1).mean()
        return loss

    def _standard_loss(self, logits, y_act):
        """Standard cross-entropy loss from the paper."""
        return F.cross_entropy(logits, y_act)

    def _compute_loss(self, logits, batch):
        if self.loss_type == "standard":
            return self._standard_loss(logits, batch.y_act)
        if self.loss_type == "anomaly_aware":
            return self._anomaly_aware_loss(logits, batch.y_act, batch.y_label)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def training_step(self, batch, batch_idx):
        n_bad_x = self._sanitize_batch_x(batch, stage="train")
        logits = self.forward(batch)
        logits, n_bad_logits = self._sanitize_logits(logits, stage="train")
        loss = self._compute_loss(logits, batch)

        batch_size = batch.y_act.shape[0]
        # on_step=True: per-batch loss curve in W&B (epochs are 20+ min, otherwise
        # we'd see one point per epoch). on_epoch=True: still aggregate per-epoch
        # for ReduceLROnPlateau / EarlyStopping clients.
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log("train/nonfinite_input_values", float(n_bad_x), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/nonfinite_logits_values", float(n_bad_logits), on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        n_bad_x = self._sanitize_batch_x(batch, stage="val")
        logits = self.forward(batch)
        logits, n_bad_logits = self._sanitize_logits(logits, stage="val")
        loss = self._compute_loss(logits, batch)

        probs = F.softmax(logits.float(), dim=1)
        scores = probs.gather(1, batch.y_act.unsqueeze(1)).squeeze(1)

        self.val_preds.append(scores.detach().cpu())
        self.val_labels.append(batch.y_label.detach().cpu())

        batch_size = batch.y_act.shape[0]
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.log("val/nonfinite_input_values", float(n_bad_x), on_epoch=True, on_step=False, batch_size=batch_size)
        self.log("val/nonfinite_logits_values", float(n_bad_logits), on_epoch=True, on_step=False, batch_size=batch_size)

    def _log_eval_metrics(
        self,
        prefix: str,
        preds: list[torch.Tensor],
        labels: list[torch.Tensor],
        warned_single_class_attr: str,
        warned_nan_scores_attr: str,
    ) -> None:
        if len(preds) == 0:
            return

        preds_np = torch.cat(preds).float().numpy()
        labels_np = torch.cat(labels).int().numpy()
        n_total = labels_np.size

        # Low confidence in true activity = anomalous
        anomaly_scores = 1.0 - preds_np

        metrics = _binary_metrics_from_anomaly_scores(
            anomaly_scores, labels_np, target_fpr=self.target_fpr
        )

        n_nonfinite = int(metrics["n_nonfinite"])
        if n_nonfinite > 0 and not getattr(self, warned_nan_scores_attr):
            logger.warning(
                f"{prefix}: {n_nonfinite}/{n_total} anomaly scores were NaN/inf "
                "and dropped before metric computation. The model produced NaN "
                "logits during forward — likely root causes: (a) NaN/inf in "
                "batch.x (preprocessing/scaling bug), (b) bf16 softmax on "
                "extreme logits (model init or LR too aggressive), (c) GCN "
                "normalisation on a graph with isolated nodes / no edges. "
                "Per-epoch count is logged as `{prefix}/n_nonfinite` in W&B."
            )
            setattr(self, warned_nan_scores_attr, True)

        n_pos = int(metrics["n_pos"])
        n_neg = int(metrics["n_neg"])
        if (n_pos == 0 or n_neg == 0) and not getattr(self, warned_single_class_attr):
            logger.warning(
                f"{prefix} set has only one class (n_pos={n_pos}, n_neg={n_neg}); "
                "ROC/PR-AUC will be NaN. Likely caused by current RandomSplit + skewed "
                "user fractions in configs/config.yaml."
            )
            setattr(self, warned_single_class_attr, True)

        # Legacy names kept so EarlyStopping/ModelCheckpoint monitors keep working
        if not math.isnan(metrics["roc_auc"]):
            self.log(f"{prefix}_auc", metrics["roc_auc"], on_epoch=True, prog_bar=True, batch_size=n_total)

        # Namespaced metrics for the W&B dashboard
        for key, value in metrics.items():
            self.log(f"{prefix}/{key}", value, on_epoch=True, batch_size=n_total)

    def on_validation_epoch_end(self):
        self._log_eval_metrics(
            prefix="val",
            preds=self.val_preds,
            labels=self.val_labels,
            warned_single_class_attr="_warned_single_class_val",
            warned_nan_scores_attr="_warned_nan_scores_val",
        )
        self.val_preds.clear()
        self.val_labels.clear()

        if self.device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    def test_step(self, batch, batch_idx):
        n_bad_x = self._sanitize_batch_x(batch, stage="test")
        logits = self.forward(batch)
        logits, n_bad_logits = self._sanitize_logits(logits, stage="test")
        probs = F.softmax(logits.float(), dim=1)
        scores = probs.gather(1, batch.y_act.unsqueeze(1)).squeeze(1)
        self.test_preds.append(scores.detach().cpu())
        self.test_labels.append(batch.y_label.detach().cpu())
        batch_size = batch.y_act.shape[0]
        self.log("test/nonfinite_input_values", float(n_bad_x), on_epoch=True, on_step=False, batch_size=batch_size)
        self.log("test/nonfinite_logits_values", float(n_bad_logits), on_epoch=True, on_step=False, batch_size=batch_size)

    def on_test_epoch_end(self):
        self._log_eval_metrics(
            prefix="test",
            preds=self.test_preds,
            labels=self.test_labels,
            warned_single_class_attr="_warned_single_class_test",
            warned_nan_scores_attr="_warned_nan_scores_test",
        )
        self.test_preds.clear()
        self.test_labels.clear()

    def on_train_epoch_end(self) -> None:
        if self.device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
