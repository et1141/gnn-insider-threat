"""Simple training script

Usage:
    uv run python src/certgnn/train.py

Or with custom arguments:
    uv run python src/certgnn/train.py \
        --processed-dir data/processed/r5.2 \
        --batch-size 64 \
        --max-epochs 20 \
        --val-size 0.1 \
        --test-size 0.2
"""

import argparse
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)


def _proc_memory_metrics() -> dict:
    """Return the training process's memory footprint, suitable for W&B.

    rss/vms come from psutil. On macOS we additionally parse `vmmap -summary`
    for `Physical footprint` — this is what Activity Monitor calls "Memory" and
    includes compressed + swapped pages, which is the number that visibly grows
    during long runs on Apple Silicon.
    """
    metrics: dict = {}
    try:
        import psutil
        p = psutil.Process(os.getpid())
        mem = p.memory_info()
        metrics["proc/rss_gb"] = mem.rss / 1024**3
        metrics["proc/vms_gb"] = mem.vms / 1024**3
        try:
            metrics["proc/uss_gb"] = p.memory_full_info().uss / 1024**3
        except Exception:
            pass
    except Exception:
        pass

    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["vmmap", "-summary", str(os.getpid())],
                capture_output=True, text=True, timeout=15,
            ).stdout
            scale = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
            m = re.search(r"Physical footprint:\s+([\d.]+)([KMGT])", out)
            if m:
                metrics["proc/physical_footprint_gb"] = float(m.group(1)) * scale[m.group(2)]
            m = re.search(r"Physical footprint \(peak\):\s+([\d.]+)([KMGT])", out)
            if m:
                metrics["proc/physical_footprint_peak_gb"] = float(m.group(1)) * scale[m.group(2)]
            m = re.search(r"swapped_out=([\d.]+)([KMGT])", out)
            if m:
                metrics["proc/swapped_out_gb"] = float(m.group(1)) * scale[m.group(2)]
        except Exception:
            pass
    return metrics


def _cgroup_memory_metrics() -> dict:
    """Return cgroup memory usage/limit when available (Linux/SLURM)."""
    metrics: dict = {}
    try:
        usage_path = Path("/sys/fs/cgroup/memory.current")
        limit_path = Path("/sys/fs/cgroup/memory.max")
        if usage_path.exists() and limit_path.exists():
            used = int(usage_path.read_text().strip())
            raw_limit = limit_path.read_text().strip()
            if raw_limit != "max":
                limit = int(raw_limit)
                if limit > 0:
                    metrics["proc/cgroup_used_gb"] = used / 1024**3
                    metrics["proc/cgroup_limit_gb"] = limit / 1024**3
                    metrics["proc/cgroup_used_pct"] = 100.0 * used / limit
    except Exception:
        pass
    return metrics


class GPUMetricsCallback(pl.Callback):
    """Log GPU + process memory to W&B at epoch end; flush MPS pool mid-epoch.

    MPS  — memory metrics only (utilization % requires sudo powermetrics).
           Metal command buffers accumulate each batch; flushed every
           MPS_FLUSH_INTERVAL steps to prevent unbounded growth.
    CUDA — memory + utilization % via pynvml (bundled with W&B on CUDA hosts).
    Process — rss/vms/uss + macOS physical footprint, the number Activity
              Monitor displays. This is the metric to watch for heap leaks.
    """

    MPS_FLUSH_INTERVAL = 200

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.device.type == "mps" and batch_idx % self.MPS_FLUSH_INTERVAL == 0:
            import gc
            gc.collect()
            torch.mps.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        if trainer.logger is None:
            return

        metrics: dict = {}
        if device.type == "mps":
            metrics["gpu/main/allocated_gb"] = torch.mps.current_allocated_memory() / 1024**3
            metrics["gpu/detail/driver_gb"] = torch.mps.driver_allocated_memory() / 1024**3
        elif device.type == "cuda":
            idx = device.index or 0
            metrics["gpu/main/allocated_gb"] = torch.cuda.memory_allocated(idx) / 1024**3
            metrics["gpu/main/reserved_gb"] = torch.cuda.memory_reserved(idx) / 1024**3
            total_mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            metrics["gpu/main/total_gb"] = total_mem
            metrics["gpu/main/allocated_pct"] = 100.0 * metrics["gpu/main/allocated_gb"] / max(total_mem, 1e-12)
            metrics["gpu/main/reserved_pct"] = 100.0 * metrics["gpu/main/reserved_gb"] / max(total_mem, 1e-12)
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                metrics["gpu/main/free_gb"] = free_b / 1024**3
                metrics["gpu/main/used_pct"] = 100.0 * (1.0 - (free_b / max(total_b, 1)))
            except Exception:
                pass
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                metrics["gpu/main/utilization_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu/detail/vram_used_gb"] = mem_info.used / 1024**3
                metrics["gpu/detail/vram_used_pct"] = 100.0 * mem_info.used / max(mem_info.total, 1)
            except Exception:
                pass

        metrics.update(_proc_memory_metrics())
        metrics.update(_cgroup_memory_metrics())
        trainer.logger.log_metrics(metrics, step=trainer.global_step)


class EpochTimingCallback(pl.Callback):
    """Log wall-clock train epoch duration to W&B."""

    def __init__(self) -> None:
        self._epoch_start_time: float | None = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.logger is None or self._epoch_start_time is None:
            return
        epoch_sec = time.perf_counter() - self._epoch_start_time
        trainer.logger.log_metrics({"time/epoch_train_sec": epoch_sec}, step=trainer.global_step)

from certgnn.datamodule import InsiderThreatDataModule
from certgnn.lightning_model import InsiderThreatLightning
from certgnn.split import RandomSplit
from certgnn.utils import get_project_root


def main():
    parser = argparse.ArgumentParser(
        description="Train GCN+LSTM model for insider threat detection"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/r5.2",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Fraction of data for validation"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device (0, 1, ...). None = CPU"
    )
    parser.add_argument(
        "--precision-mode",
        choices=["auto", "32", "bf16"],
        default="auto",
        help="'auto' picks bf16 on supported CUDA, else fp32. Use '32' to force "
             "full precision for NaN debugging, or 'bf16' to force bf16 on CUDA.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="DataLoader worker processes. Use 0 on shared servers where worker "
             "subprocesses get OOM-killed. Use 1 on macOS (workers die per phase, "
             "kernel reclaims heap). Use 2-6 on SLURM/dedicated NVIDIA nodes.",
    )
    parser.add_argument(
        "--max-local-chunks", type=int, default=None,
        help="How many chunk files to keep loaded in RAM at once (per worker). "
             "Auto-detected from available/cgroup memory if not set. Override "
             "explicitly when auto-detection picks wrong value on shared machines.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=None,
        help="DataLoader prefetch_factor (batches each worker prefetches). "
             "Default: 4 on CUDA, 2 elsewhere. Ignored when --num-workers=0.",
    )
    parser.add_argument(
        "--persistent-workers", choices=["auto", "true", "false"], default="auto",
        help="Keep DataLoader workers alive between phases. 'auto' = True on "
             "CUDA (with workers>0), False on MPS (heap doesn't shrink between "
             "epochs on macOS). Force 'true' on dedicated CUDA nodes if you want "
             "to skip phase-respawn cost.",
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=50,
        help="Lightning's log_every_n_steps. With bs=512 a value of 50 is a "
             "log per ~25k samples — frequent enough to see live train_loss "
             "without flooding W&B.",
    )
    parser.add_argument(
        "--target-fpr", type=float, default=0.05,
        help="Target false-positive rate at which val/test report TPR. Paper "
             "section V-C uses 0.05 for r5.2 and 0.09 for r6.2; metric is "
             "logged as val/tpr_at_fpr_target and test/tpr_at_fpr_target.",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Global gradient norm clipping value. Helps avoid NaN divergence "
             "on large unnormalized feature spikes, especially with mixed precision.",
    )
    parser.add_argument(
        "--loss-type", choices=["standard", "anomaly_aware"], default="standard",
        help="'anomaly_aware' implements paper eq. 24-26 (one-hot CE for "
             "normal, uniform-over-non-true for malicious — recommended for "
             "paper-faithful runs). 'standard' is plain F.cross_entropy on "
             "the true class — useful as a debug baseline but ignores "
             "y_label, so malicious samples train the model to predict the "
             "true class (opposite of paper's intent).",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="gnn-insider-threat", help="W&B project name"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging"
    )
    args = parser.parse_args()

    root = get_project_root()
    processed_dir = root / args.processed_dir

    with open(processed_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    num_activity_classes = metadata["num_classes"]
    num_activity_types = metadata["num_activity_types"]

    # Determine accelerator: CUDA GPU > Apple MPS > CPU
    if args.gpu is not None:
        accelerator, devices = "gpu", [args.gpu]
    elif torch.cuda.is_available():
        accelerator, devices = "gpu", "auto"
    elif torch.backends.mps.is_available():
        accelerator, devices = "mps", "auto"
    else:
        accelerator, devices = "cpu", "auto"

    # Auto-precision. fp16-mixed on the 53k-param GCN+LSTM with CE on 192/216
    # classes overflows on RTX 3090 (NaN losses). bf16-mixed has fp32-range
    # exponent so it doesn't overflow; almost as fast as fp16 on Ampere.
    if args.precision_mode == "32":
        precision = "32-true"
    elif args.precision_mode == "bf16":
        precision = "bf16-mixed"
    elif accelerator == "gpu" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    else:
        precision = "32-true"

    if accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    logger.info(
        f"accelerator={accelerator} | precision={precision} | "
        f"batch_size={args.batch_size} | max_epochs={args.max_epochs} | "
        f"num_activity_classes={num_activity_classes}"
    )

    # W&B logger
    wandb_logger = None
    if not args.no_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            config={
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "learning_rate": args.learning_rate,
                "val_size": args.val_size,
                "test_size": args.test_size,
                "seed": args.seed,
                "accelerator": accelerator,
                "precision": precision,
                "num_activity_classes": num_activity_classes,
                "num_activity_types": num_activity_types,
                "processed_dir": str(processed_dir),
                "loss_type": args.loss_type,
                "target_fpr": args.target_fpr,
            },
        )

    split = RandomSplit(val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    if args.persistent_workers == "auto":
        persistent_workers = None
    else:
        persistent_workers = args.persistent_workers == "true"

    datamodule = InsiderThreatDataModule(
        processed_dir=processed_dir,
        split_strategy=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_local_chunks=args.max_local_chunks,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
        accelerator=accelerator,
    )

    model = InsiderThreatLightning(
        learning_rate=args.learning_rate,
        num_activity_classes=num_activity_classes,
        loss_type=args.loss_type,
        target_fpr=args.target_fpr,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        dirpath=root / "checkpoints",
        filename="insider-threat-{epoch:02d}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min",
    )

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        RichProgressBar(),
        EpochTimingCallback(),
    ]
    if accelerator in ("mps", "gpu"):
        callbacks.append(GPUMetricsCallback())

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=args.gradient_clip_val,
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
