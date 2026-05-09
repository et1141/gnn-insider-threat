"""Training entrypoint for the CERT baseline."""

from __future__ import annotations

import re
import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from certgnn.baseline.data import list_split_chunks, load_processed_metadata
from certgnn.streaming_dataset import SequentialChunkDataset
from certgnn.baseline.lightning import GraphBaselineLightningModule
from certgnn.utils import get_project_root, load_config


def _load_wandb_logger(project: str, mode: str, name: str):
    try:
        from pytorch_lightning.loggers import WandbLogger
        return WandbLogger(project=project, mode=mode, name=name, log_model=False)
    except Exception:
        print("W&B logger unavailable; continuing without experiment logging.")
        return None


def _save_metrics(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=float))


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

    seed = int(baseline_cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    metadata = load_processed_metadata(processed_dir)
    batch_size = int(baseline_cfg.get("batch_size", 64))
    
    num_workers = 0 

    train_chunks = _sort_chunks_numerically(list_split_chunks(processed_dir, "train"))
    val_chunks = _sort_chunks_numerically(list_split_chunks(processed_dir, "val"))
    test_chunks = _sort_chunks_numerically(list_split_chunks(processed_dir, "test"))

    train_dataset = SequentialChunkDataset(processed_dir, train_chunks, is_training=True)
    val_dataset = SequentialChunkDataset(processed_dir, val_chunks, is_training=False)
    test_dataset = SequentialChunkDataset(processed_dir, test_chunks, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model = GraphBaselineLightningModule(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=int(baseline_cfg.get("hidden_dim", 128)),
        dropout=float(baseline_cfg.get("dropout", 0.2)),
        pooling=str(baseline_cfg.get("pooling", "mean_max")),
        learning_rate=float(baseline_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(baseline_cfg.get("weight_decay", 1e-4)),
        gamma=float(baseline_cfg.get("gamma", 2.0))
    )

    wandb_logger = _load_wandb_logger(
        project=str(baseline_cfg.get("wandb_project", "gnn-insider-threat-baseline")),
        mode=str(baseline_cfg.get("wandb_mode", "offline")),
        name=str(baseline_cfg.get("run_name", "graph-pooling-mlp")),
    )

    checkpoint = ModelCheckpoint(
        monitor="val/roc_auc",
        mode="max",
        save_top_k=1,
        filename="baseline-{epoch:02d}-{val_roc_auc:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="val/roc_auc",
        mode="max",
        patience=int(baseline_cfg.get("patience", 5)),
    )

    trainer = pl.Trainer(
        max_epochs=int(baseline_cfg.get("max_epochs", 10)),
        accelerator=str(baseline_cfg.get("accelerator", "auto")),
        devices=baseline_cfg.get("devices", "auto"),
        logger=wandb_logger,
        callbacks=[checkpoint, early_stopping],
        log_every_n_steps=int(baseline_cfg.get("log_every_n_steps", 50)),
        deterministic=True,
        default_root_dir=str(root / "artifacts" / "baseline"),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    metrics = {}
    if test_results:
        metrics = test_results[0]
    elif trainer.callback_metrics:
        metrics = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items() if hasattr(value, "detach")}

    _save_metrics(root / "reports" / "baseline" / "metrics.json", metrics)


if __name__ == "__main__":
    main()