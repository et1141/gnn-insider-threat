"""PyTorch Lightning task modules.

Each task is a thin subclass of :class:`BaseLightningModule` that
declares its loss function, evaluation extraction, and metric set.
The base class owns the training loop, callbacks, optimizer, sanitization,
and logging — so adding a new task means dropping a file with three method
overrides.

Example::

    from certgnn.lightning import build_lightning_module

    module = build_lightning_module(
        task="anomaly_aware",
        model_name="gcn_lstm",
        model_args={"num_node_features": 54, ...},
        learning_rate=1e-3,
    )
"""

from __future__ import annotations

from typing import Callable

from certgnn.lightning.anomaly_aware import AnomalyAwareLightning
from certgnn.lightning.base import BaseLightningModule
from certgnn.lightning.binary_classifier import BinaryClassifierLightning

LIGHTNING_REGISTRY: dict[str, Callable[..., BaseLightningModule]] = {
    "anomaly_aware": AnomalyAwareLightning,
    "binary": BinaryClassifierLightning,
}


def build_lightning_module(task: str, **kwargs) -> BaseLightningModule:
    """Instantiate a Lightning task module by registry name."""
    if task not in LIGHTNING_REGISTRY:
        raise KeyError(
            f"Unknown task {task!r}. Available: {sorted(LIGHTNING_REGISTRY)}"
        )
    return LIGHTNING_REGISTRY[task](**kwargs)


__all__ = [
    "AnomalyAwareLightning",
    "BaseLightningModule",
    "BinaryClassifierLightning",
    "LIGHTNING_REGISTRY",
    "build_lightning_module",
]
