"""Tests for paper-faithful loss and operating-point metrics.

References:
- Paper section IV-E (eq. 24-26): soft-label CE with uniform-over-non-true
  target on malicious samples.
- Paper section V-C (eq. 30-31): operating threshold = max{t | FPR(t) ≤ target}.
"""

import math

import numpy as np
import pytest
import torch

from certgnn.lightning.metrics import binary_metrics_from_scores
from certgnn.losses import anomaly_aware_loss


# ---------------------------------------------------------------------------
# anomaly_aware_loss — paper eq. 24-26
# ---------------------------------------------------------------------------

def test_anomaly_aware_normal_sample_equals_cross_entropy():
    """For normal (q=0) the soft target is one-hot(true_class), so the loss
    must equal `F.cross_entropy` on the same logits."""
    torch.manual_seed(0)

    logits = torch.randn(8, 4)
    y_act = torch.randint(0, 4, (8,))
    y_label = torch.zeros(8, dtype=torch.long)

    loss = anomaly_aware_loss(logits, y_act, y_label, num_classes=4)
    expected = torch.nn.functional.cross_entropy(logits, y_act)

    assert torch.allclose(loss, expected, atol=1e-6), (loss.item(), expected.item())


def test_anomaly_aware_malicious_sample_matches_uniform_target():
    """For malicious (q=1) the target is uniform 1/(M-1) on non-true classes,
    so loss = -mean over j!=true of log p(j) / (M-1)."""
    M = 4
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    y_act = torch.tensor([0])
    y_label = torch.tensor([1])

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)[0]
    expected = -(log_probs[1] + log_probs[2] + log_probs[3]) / (M - 1)

    loss = anomaly_aware_loss(logits, y_act, y_label, num_classes=M)
    assert torch.allclose(loss, expected, atol=1e-6), (loss.item(), expected.item())


def test_anomaly_aware_malicious_pushes_away_from_true_class():
    """The whole point of the malicious branch: predicting the true class
    confidently should be punished worse than predicting any other class."""
    logits_predicts_true = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    logits_predicts_other = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    y_act = torch.tensor([0])
    y_label = torch.tensor([1])

    loss_true = anomaly_aware_loss(logits_predicts_true, y_act, y_label, num_classes=4)
    loss_other = anomaly_aware_loss(logits_predicts_other, y_act, y_label, num_classes=4)
    assert loss_true > loss_other


def test_anomaly_aware_mixed_batch_is_average_of_branches():
    """A batch with one normal + one malicious sample should equal the mean of
    the two branch losses computed separately."""
    logits = torch.tensor([[1.0, 2.0, -0.5, 0.3], [0.2, 0.1, 0.9, 0.4]])
    y_act = torch.tensor([1, 2])
    y_label = torch.tensor([0, 1])

    mixed = anomaly_aware_loss(logits, y_act, y_label, num_classes=4)
    normal_only = anomaly_aware_loss(logits[:1], y_act[:1], y_label[:1], num_classes=4)
    malicious_only = anomaly_aware_loss(logits[1:], y_act[1:], y_label[1:], num_classes=4)
    expected = (normal_only + malicious_only) / 2

    assert torch.allclose(mixed, expected, atol=1e-6), (mixed.item(), expected.item())


def test_anomaly_aware_is_finite_under_extreme_logits():
    """The whole reason we rewrote this in F.log_softmax: huge logits used
    to produce NaN under fp16 with the old `softmax + log(probs + 1e-10)`."""
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    y_act = torch.tensor([0])
    for label in (0, 1):
        loss = anomaly_aware_loss(logits, y_act, torch.tensor([label]), num_classes=4)
        assert torch.isfinite(loss), label


# ---------------------------------------------------------------------------
# binary_metrics_from_scores — paper eq. 30-31
# ---------------------------------------------------------------------------

def test_threshold_at_fpr_target_picks_largest_valid_fpr():
    """Among thresholds whose FPR ≤ target, pick the operating point with
    maximum TPR — i.e. the rightmost valid index in roc_curve output."""
    rng = np.random.default_rng(0)
    n_neg = 200
    n_pos = 50

    neg_scores = rng.uniform(0.0, 0.6, size=n_neg)
    pos_scores = rng.uniform(0.4, 1.0, size=n_pos)
    scores = np.concatenate([neg_scores, pos_scores])
    labels = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])

    target = 0.05
    metrics = binary_metrics_from_scores(scores, labels, target_fpr=target)

    assert metrics["fpr_at_target"] <= target + 1e-12
    from sklearn.metrics import roc_curve
    fpr_arr, _, _ = roc_curve(labels, scores)
    valid = np.where(fpr_arr <= target)[0]
    rightmost = int(valid[-1])
    if rightmost + 1 < len(fpr_arr):
        assert fpr_arr[rightmost + 1] > target


def test_threshold_at_fpr_target_falls_back_when_target_too_strict():
    """If target_fpr is below every achievable FPR (smallest is 0 here, so we
    use a negative target), fall back to the strictest threshold (idx 0)."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    metrics = binary_metrics_from_scores(scores, labels, target_fpr=-0.01)

    assert metrics["fpr_at_target"] == pytest.approx(0.0)
    assert metrics["tpr_at_fpr_target"] == pytest.approx(0.0)


def test_target_fpr_reported_in_metrics():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    metrics = binary_metrics_from_scores(scores, labels, target_fpr=0.09)
    assert metrics["target_fpr"] == pytest.approx(0.09)


def test_nonfinite_scores_are_filtered_and_counted():
    """NaN/inf in scores must not crash sklearn — they get dropped before
    any metric is computed and the count is exposed as `n_nonfinite`."""
    scores = np.array([0.1, np.nan, 0.8, np.inf, 0.4, -np.inf])
    labels = np.array([0, 1, 1, 0, 0, 1])

    metrics = binary_metrics_from_scores(scores, labels, target_fpr=0.5)

    assert metrics["n_nonfinite"] == 3.0
    # The 3 finite samples (0, 2, 4) had labels (0, 1, 0): so n_pos=1, n_neg=2.
    assert metrics["n_pos"] == 1
    assert metrics["n_neg"] == 2
    assert math.isfinite(metrics["roc_auc"])


def test_all_nonfinite_returns_nan_metrics_no_crash():
    """If every score is non-finite, return NaN metrics instead of raising."""
    scores = np.array([np.nan, np.nan, np.inf])
    labels = np.array([0, 1, 1])

    metrics = binary_metrics_from_scores(scores, labels)
    assert metrics["n_nonfinite"] == 3.0
    assert metrics["n_pos"] == 0
    assert metrics["n_neg"] == 0
    for k in ("roc_auc", "pr_auc", "tpr_at_fpr_target", "accuracy", "precision"):
        assert math.isnan(metrics[k]), k


def test_single_class_returns_nan_aucs():
    """No raise, just NaN for things that are undefined when only one class
    is present. Keeps the count metrics (TP/TN/FP/FN/etc.) intact."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 0, 0])

    metrics = binary_metrics_from_scores(scores, labels)
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["pr_auc"])
    assert math.isnan(metrics["tpr_at_fpr_target"])
    assert math.isnan(metrics["fpr_at_target"])
    assert math.isnan(metrics["threshold_at_fpr_target"])
    # Counts must still be computed
    assert metrics["n_pos"] == 0
    assert metrics["n_neg"] == 3
