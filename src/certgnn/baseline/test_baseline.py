"""Fast test script (dry-run) for the CERT baseline."""

from __future__ import annotations
import re

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from certgnn.baseline.data import list_split_chunks, load_processed_metadata
from certgnn.streaming_dataset import SequentialChunkDataset
from certgnn.baseline.lightning import GraphBaselineLightningModule
from certgnn.utils import get_project_root, load_config


def _sort_chunks_numerically(chunk_list: list[str]) -> list[str]:
    def extract_number(filename: str) -> int:
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    return sorted(chunk_list, key=extract_number)


def main() -> None:
    config = load_config()
    baseline_cfg = config.get("baseline", {})
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Limit chunks for dry-run
    train_chunks = _sort_chunks_numerically(list_split_chunks(processed_dir, "train"))[:3]
    val_chunks = _sort_chunks_numerically(list_split_chunks(processed_dir, "val"))[:1]

    print(f"Starting test run with train chunks: {train_chunks}")
    print(f"Starting test run with val chunks: {val_chunks}")

    batch_size = int(baseline_cfg.get("batch_size", 64))

    train_loader = DataLoader(
        SequentialChunkDataset(processed_dir, train_chunks, is_training=True), 
        batch_size=batch_size, 
        num_workers=0
    )
    val_loader = DataLoader(
        SequentialChunkDataset(processed_dir, val_chunks, is_training=False), 
        batch_size=batch_size, 
        num_workers=0
    )

    metadata = load_processed_metadata(processed_dir)
    model = GraphBaselineLightningModule(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=int(baseline_cfg.get("hidden_dim", 128)),
    )

    # Setup W&B logger in offline mode
    wandb_logger = WandbLogger(
        project=str(baseline_cfg.get("wandb_project", "gnn-insider-threat-baseline")),
        mode="offline",
        name="test-baseline-graphpoolingMLP",
        log_model=False
    )

    # Fast dev run limited to 1 epoch, no checkpointing
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Test run completed successfully.")


if __name__ == "__main__":
    main()