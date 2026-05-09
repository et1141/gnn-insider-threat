"""Metric helpers shared by both Lightning tasks.

Both the anomaly-score task (``1 - p(true_class)``) and the binary task
(``softmax[:, 1]``) end up computing the same binary metrics on a
``(score, label)`` pair, so the implementation lives here once.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def binary_metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    target_fpr: float | None = 0.05,
) -> dict[str, float]:
    """Compute binary classification metrics from continuous scores.

    Higher score = more anomalous (label=1). NaN/inf scores are filtered
    out before any sklearn call (``roc_auc_score`` raises on non-finite
    input); the dropped count is exposed as ``n_nonfinite`` so the caller
    can surface it (e.g. to W&B) instead of silently masking a NaN-logit
    failure mode.

    When the label distribution degenerates to one class, ROC/PR-AUC are
    set to NaN rather than raising — count metrics still flow through.

    Args:
        scores: per-sample continuous score, higher = more anomalous.
        labels: per-sample binary ground truth (0/1).
        threshold: classification threshold on the score for hard preds.
        target_fpr: when set, also report the TPR/threshold at the largest
            point with ``FPR ≤ target_fpr`` (paper section V-C operating
            point: 0.05 for r5.2, 0.09 for r6.2).
    """
    labels = labels.astype(int)
    scores = scores.astype(float)

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
    }
    if target_fpr is not None:
        out["target_fpr"] = float(target_fpr)

    nan_keys = (
        "tp", "tn", "fp", "fn",
        "accuracy", "precision", "recall", "fpr", "tpr",
        "roc_auc", "pr_auc",
    )
    if target_fpr is not None:
        nan_keys = (*nan_keys, "tpr_at_fpr_target", "fpr_at_target", "threshold_at_fpr_target")

    if n_total == 0:
        for k in nan_keys:
            out[k] = math.nan
        return out

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out.update({"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)})
    out["accuracy"] = (tp + tn) / max(1, n_total)
    out["precision"] = tp / max(1, tp + fp)
    out["recall"] = tp / max(1, tp + fn)
    out["fpr"] = fp / max(1, fp + tn)
    out["tpr"] = out["recall"]

    if n_pos > 0 and n_neg > 0:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
        if target_fpr is not None:
            fpr_arr, tpr_arr, thresh_arr = roc_curve(labels, scores)
            valid = np.where(fpr_arr <= target_fpr)[0]
            idx = int(valid[-1]) if valid.size > 0 else 0
            out["tpr_at_fpr_target"] = float(tpr_arr[idx])
            out["fpr_at_target"] = float(fpr_arr[idx])
            out["threshold_at_fpr_target"] = float(thresh_arr[idx])
    else:
        out["roc_auc"] = math.nan
        out["pr_auc"] = math.nan
        if target_fpr is not None:
            out["tpr_at_fpr_target"] = math.nan
            out["fpr_at_target"] = math.nan
            out["threshold_at_fpr_target"] = math.nan

    return out


__all__ = ["binary_metrics_from_scores"]
