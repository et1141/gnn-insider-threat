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
        "--batch-size", type=int, default=32, help="Batch size for training"
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
    args = parser.parse_args()

    root = get_project_root()
    processed_dir = root / args.processed_dir

    # Load metadata to get num_classes and activity types
    metadata_path = processed_dir / "metadata.pkl"
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    num_activity_classes = metadata["num_classes"]
    num_activity_types = metadata["num_activity_types"]
    activity_types = metadata["activity_types"]

    # Also load label encoder to verify encoding
    encoder_path = processed_dir / "label_encoder.pkl"
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    encoder_classes = list(label_encoder.classes_)

    print("\n" + "=" * 70)
    print("Insider Threat Detection: GCN + LSTM")
    print("=" * 70)
    print(f"Processed data: {processed_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Split: val={args.val_size}, test={args.test_size}")
    print("\n[Activity Types Encoding]")
    print(f"  Found {num_activity_types} activity types:")
    for i, atype in enumerate(activity_types):
        print(f"    {i}: {atype}")
    print(f"  Encoder sees: {encoder_classes}")
    if activity_types != encoder_classes:
        print(f"  ⚠️  WARNING: Activity types differ from encoder classes!")
    print(f"  Total classes: {num_activity_types} types × 24 hours = {num_activity_classes}")
    print("\n[Paper Comparison]")
    print(f"  Paper uses: 8 activity types × 24 hours = 192 classes")
    print(f"  This run: {num_activity_types} activity types × 24 hours = {num_activity_classes} classes")
    if num_activity_classes != 192:
        print(f"  ⚠️  MISMATCH: Expected 192 classes from paper, got {num_activity_classes}")
        print(f"  Model will be initialized with {num_activity_classes} output classes")
    print("=" * 70 + "\n")

    # Create split strategy
    split = RandomSplit(val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    # Create data module
    datamodule = InsiderThreatDataModule(
        processed_dir=processed_dir,
        split_strategy=split,
        batch_size=args.batch_size,
        num_workers=0,  # Required for StreamingChunkDataset
    )

    # Create model with correct num_classes from metadata
    model = InsiderThreatLightning(
        learning_rate=args.learning_rate,
        num_activity_classes=num_activity_classes,
    )

    # Callbacks
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

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        accelerator="gpu" if args.gpu is not None else "cpu",
        devices=[args.gpu] if args.gpu is not None else "auto",
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("Checkpoints saved to: checkpoints/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
