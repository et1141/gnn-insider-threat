"""Lightning callbacks + a fabryka that builds the standard set from config."""

from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from certgnn.callbacks.gpu_metrics import GPUMetricsCallback


def build_callbacks(
    monitor: str,
    monitor_mode: str = "min",
    patience: int = 5,
    save_top_k: int = 3,
    enable_gpu_metrics: bool = True,
    checkpoint_filename: str | None = None,
) -> list[pl.Callback]:
    """Standard set of callbacks tuned to whichever monitor the task uses.

    The Lightning module's ``monitor_metric`` / ``monitor_mode`` class
    attributes drive ``monitor`` / ``monitor_mode`` here, so swapping
    tasks doesn't require changing the trainer wiring.
    """
    if checkpoint_filename is None:
        # Replace path-unsafe slashes from monitor name (val/loss → val_loss).
        slug = monitor.replace("/", "_")
        checkpoint_filename = f"{{epoch:02d}}-{{{slug}:.4f}}"

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=save_top_k,
            filename=checkpoint_filename,
            save_last=True,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=patience,
        ),
    ]
    if enable_gpu_metrics:
        callbacks.append(GPUMetricsCallback())
    return callbacks


__all__ = ["GPUMetricsCallback", "build_callbacks"]
