"""Loss functions for the insider-threat detector.

Two task-specific losses live here:

* ``anomaly_aware_loss`` — paper-faithful, multi-class (192 activity codes).
  Implements the paper's section IV-E soft-label CE (eq. 24-26): on
  malicious samples the target distribution is uniform over non-true
  classes, so the model is pushed AWAY from predicting the true activity
  on attacks. Residual ``p(true_class)`` is then the anomaly signal at
  inference.

* ``focal_loss_with_dynamic_pos_weight`` — binary classification with
  Lin et al.'s focal loss; class weights are derived from in-batch
  positive/negative counts so highly imbalanced batches don't drown the
  malicious gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def anomaly_aware_loss(
    logits: torch.Tensor,
    y_act: torch.Tensor,
    y_label: torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Soft-label cross-entropy from the paper (eq. 24-26).

    For each sample i the soft target ``Y'_i`` is::

        Y'_i = one_hot(y_act_i)                          if y_label_i == 0
        Y'_i = (1 - one_hot(y_act_i)) / (M - 1)          if y_label_i == 1

    where M is the number of activity classes. Equivalent vectorised form::

        Y' = Y ⊙ (1 − Ω) + (1 / (M − 1)) · (1 − Y) ⊙ Ω

    Implementation uses ``F.log_softmax`` + soft-label NLL so the loss
    stays stable under bf16/fp16 autocast (no ``log(p + eps)`` underflow
    that a "softmax + log" formulation suffers from on certain GPUs).
    """
    if num_classes is None:
        num_classes = logits.size(1)

    max_activity = int(y_act.max().item()) if y_act.numel() > 0 else 0
    if max_activity >= num_classes:
        raise RuntimeError(
            f"Activity index {max_activity} >= num_classes {num_classes}. "
            "Check metadata['num_classes'] vs max(y_act)."
        )

    Y = F.one_hot(y_act, num_classes).to(logits.dtype)
    omega = (y_label > 0).to(logits.dtype).unsqueeze(1)
    Y_prime = Y * (1.0 - omega) + (1.0 - Y) / (num_classes - 1) * omega

    log_probs = F.log_softmax(logits, dim=1)
    return -(Y_prime * log_probs).sum(dim=1).mean()


def standard_cross_entropy(logits: torch.Tensor, y_act: torch.Tensor) -> torch.Tensor:
    """Plain cross-entropy on activity index (debug baseline)."""
    return F.cross_entropy(logits, y_act)


def focal_loss_with_dynamic_pos_weight(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    pos_weight_clamp: float = 1000.0,
) -> torch.Tensor:
    """Focal loss with batch-derived positive class weighting.

    The positive class weight is recomputed per-batch as ``n_neg / n_pos``
    (clamped at ``pos_weight_clamp`` to keep the gradient bounded on
    near-empty positive batches). The focal modulator
    ``(1 − p_t)^gamma`` then downweights well-classified examples.

    Args:
        logits: ``[N, 2]`` raw scores.
        labels: ``[N]`` integer labels in ``{0, 1}``.
        gamma: Focal-loss focusing parameter.
        pos_weight_clamp: Upper bound for the dynamic positive weight.
    """
    n_neg = (labels == 0).sum().float()
    n_pos = (labels == 1).sum().float()

    if n_pos > 0:
        pos_weight = torch.clamp(n_neg / n_pos, max=pos_weight_clamp)
    else:
        pos_weight = torch.tensor(1.0, device=logits.device, dtype=logits.dtype)

    class_weights = torch.stack([torch.tensor(1.0, device=logits.device), pos_weight])

    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    weighted = class_weights[labels] * focal
    return weighted.mean()


__all__ = [
    "anomaly_aware_loss",
    "focal_loss_with_dynamic_pos_weight",
    "standard_cross_entropy",
]
