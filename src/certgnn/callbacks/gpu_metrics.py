"""Log GPU memory & utilization to whatever logger Lightning is using.

MPS  — ``torch.mps``: allocated (PyTorch tensors) + driver (total Metal).
       Utilization % is not available without ``sudo powermetrics``.
CUDA — ``torch.cuda``: allocated, reserved, and utilization % via pynvml
       (ships with ``nvidia-ml-py``, which is a W&B dependency on CUDA
       machines).
CPU  — no-op.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class GPUMetricsCallback(pl.Callback):
    """Per-epoch GPU stats — namespaced under ``gpu/*``."""

    def on_train_epoch_end(self, trainer, pl_module):
        device = pl_module.device

        if device.type == "mps":
            pl_module.log(
                "gpu/allocated_gb",
                torch.mps.current_allocated_memory() / 1024**3,
                on_epoch=True,
            )
            pl_module.log(
                "gpu/driver_gb",
                torch.mps.driver_allocated_memory() / 1024**3,
                on_epoch=True,
            )
        elif device.type == "cuda":
            idx = device.index or 0
            pl_module.log("gpu/allocated_gb", torch.cuda.memory_allocated(idx) / 1024**3, on_epoch=True)
            pl_module.log("gpu/reserved_gb", torch.cuda.memory_reserved(idx) / 1024**3, on_epoch=True)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pl_module.log("gpu/utilization_pct", float(util), on_epoch=True)
                pl_module.log("gpu/vram_used_gb", mem_info.used / 1024**3, on_epoch=True)
            except Exception:
                pass
