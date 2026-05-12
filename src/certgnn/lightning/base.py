"""Base PyTorch Lightning module shared by every task in this project.

Holds the cross-cutting concerns — model construction from the registry,
optimizer/scheduler, NaN/inf sanitization, eval-batch buffering — so the
task-specific subclasses (``AnomalyAwareLightning``,
``BinaryClassifierLightning``) only declare:

* ``compute_loss(batch, logits)`` — what the loss function is.
* ``collect_eval(batch, logits)`` — how to extract ``(score, label)``
  for downstream metrics.
* ``epoch_metrics(scores, labels)`` — what metrics dict to log.

Adding a new task = drop a subclass that implements those three hooks.
The training loop, callbacks, and hyperparameter logging do not change.
"""

from __future__ import annotations

import gc
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger

from certgnn.models import build_model


class BaseLightningModule(pl.LightningModule):
    """Reusable Lightning skeleton for any (model, loss, metric) task.

    Args:
        model_name: Registry key for the architecture (see
            ``certgnn.models.MODEL_REGISTRY``).
        model_args: kwargs forwarded to the architecture's constructor.
        learning_rate: Initial LR.
        weight_decay: L2 regularisation coefficient.
        optimizer: ``"adam"`` or ``"adamw"``.
        scheduler: ``None`` to disable, or ``"plateau"`` for
            ``ReduceLROnPlateau`` on ``val/loss``.
        scheduler_args: kwargs forwarded to the scheduler.
    """

    # Subclasses set this to the metric Lightning's monitors should track.
    # Used by configure_optimizers when scheduler="plateau".
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"

    def __init__(
        self,
        model_name: str,
        model_args: dict | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = "adam",
        scheduler: str | None = "plateau",
        scheduler_args: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(model_name, **(model_args or {}))
        self._eval_buffer: dict[str, list] = {"val": [], "test": []}

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    def compute_loss(self, batch: Any, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def collect_eval(self, batch: Any, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scores, labels)`` for one batch, both 1-D, on CPU."""
        raise NotImplementedError

    def epoch_metrics(self, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: Any) -> torch.Tensor:
        return self.model(batch)

    # ------------------------------------------------------------------
    # Numerical sanitization
    # ------------------------------------------------------------------
    def _sanitize_input(self, batch: Any) -> int:
        """Replace NaN/inf values in ``batch.x`` with finite ones in place."""
        x = getattr(batch, "x", None)
        if x is None:
            return 0
        bad = ~torch.isfinite(x)
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            if not getattr(self, "_warned_input", False):
                logger.warning(
                    f"Found {n_bad} non-finite values in batch.x; "
                    "replacing via torch.nan_to_num. Points to a preprocessing/scaling issue."
                )
                self._warned_input = True
            batch.x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return n_bad

    def _sanitize_logits(self, logits: torch.Tensor) -> tuple[torch.Tensor, int]:
        bad = ~torch.isfinite(logits)
        n_bad = int(bad.sum().item())
        if n_bad > 0:
            if not getattr(self, "_warned_logits", False):
                logger.warning(
                    f"Found {n_bad} non-finite logits; replacing via torch.nan_to_num."
                )
                self._warned_logits = True
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits, n_bad

    # ------------------------------------------------------------------
    # Training / eval steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch: Any, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        self._sanitize_input(batch)
        logits = self(batch)
        logits, _ = self._sanitize_logits(logits)
        loss = self.compute_loss(batch, logits)
        return loss, logits

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, "train")
        batch_size = self._infer_batch_size(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logits = self._shared_step(batch, "val")
        batch_size = self._infer_batch_size(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        scores, labels = self.collect_eval(batch, logits)
        self._eval_buffer["val"].append((scores.detach().cpu(), labels.detach().cpu()))
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, logits = self._shared_step(batch, "test")
        batch_size = self._infer_batch_size(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        scores, labels = self.collect_eval(batch, logits)
        self._eval_buffer["test"].append((scores.detach().cpu(), labels.detach().cpu()))
        return loss

    def _flush_eval(self, prefix: str) -> None:
        buf = self._eval_buffer[prefix]
        if not buf:
            return
        scores = torch.cat([s for s, _ in buf], dim=0)
        labels = torch.cat([lbl for _, lbl in buf], dim=0)
        buf.clear()
        metrics = self.epoch_metrics(scores, labels)
        if metrics:
            n = int(labels.shape[0])
            self.log_dict(
                {f"{prefix}/{k}": float(v) for k, v in metrics.items()},
                on_epoch=True, prog_bar=(prefix == "val"), batch_size=n,
            )

    def on_validation_epoch_end(self) -> None:
        self._flush_eval("val")
        self._maybe_release_mps()

    def on_test_epoch_end(self) -> None:
        self._flush_eval("test")
        self._maybe_release_mps()

    def on_train_epoch_end(self) -> None:
        self._maybe_release_mps()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        for attr in ("y_act", "y_label", "y"):
            value = getattr(batch, attr, None)
            if value is not None:
                return int(value.shape[0])
        return 1

    def _maybe_release_mps(self) -> None:
        """Apple Silicon MPS pool grows without GC pressure between epochs."""
        if self.device.type == "mps":
            gc.collect()
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[self.hparams.optimizer]
        optimizer = opt_cls(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler is None:
            return optimizer
        if self.hparams.scheduler == "plateau":
            args = {"mode": self.monitor_mode, "factor": 0.5, "patience": 3, "min_lr": 1e-6}
            args.update(self.hparams.scheduler_args or {})
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **args)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor_metric},
            }
        raise ValueError(f"Unknown scheduler {self.hparams.scheduler!r}")


__all__ = ["BaseLightningModule"]
