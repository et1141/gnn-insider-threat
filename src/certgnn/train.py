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
import pickle
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)

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
        "--num-workers", type=int, default=0, help="DataLoader worker processes (keep 0 on low-RAM machines)"
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
    import torch
    if args.gpu is not None:
        accelerator, devices = "gpu", [args.gpu]
    elif torch.cuda.is_available():
        accelerator, devices = "gpu", "auto"
    elif torch.backends.mps.is_available():
        accelerator, devices = "mps", "auto"
    else:
        accelerator, devices = "cpu", "auto"

    precision = "16-mixed" if accelerator == "gpu" else "32-true"

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
            },
        )

    split = RandomSplit(val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    datamodule = InsiderThreatDataModule(
        processed_dir=processed_dir,
        split_strategy=split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = InsiderThreatLightning(
        learning_rate=args.learning_rate,
        num_activity_classes=num_activity_classes,
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

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=wandb_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
