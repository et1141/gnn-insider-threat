"""Model architectures and a small registry to wire them up via config.

Adding a new architecture amounts to:

1. Drop a new file under ``certgnn/models/`` that defines an ``nn.Module``.
2. Register it in ``MODEL_REGISTRY`` below.
3. Reference it as ``training.model: <name>`` in ``configs/training.yaml``.

The Lightning module instantiates the architecture via ``build_model``,
so the training loop, callbacks, optimizer, and logging stay shared.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from certgnn.models.gcn_lstm import GCN, ActivityPredictor, GCNLSTMInsiderThreat
from certgnn.models.graph_pool_mlp import GraphPoolingMLP

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "gcn_lstm": GCNLSTMInsiderThreat,
    "graph_pool_mlp": GraphPoolingMLP,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by registry name.

    ``kwargs`` are forwarded to the architecture's constructor — typically
    sourced from ``config.training.model_args``.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "ActivityPredictor",
    "GCN",
    "GCNLSTMInsiderThreat",
    "GraphPoolingMLP",
    "MODEL_REGISTRY",
    "build_model",
]
