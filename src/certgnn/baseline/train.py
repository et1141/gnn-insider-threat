"""Training entrypoint for the CERT baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from certgnn.baseline.data import (
    build_dataloader,
    list_split_chunks,
    load_processed_metadata,
)
from certgnn.chunk_store import DvcChunkStore
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


def main() -> None:
    config = load_config()
    baseline_cfg = config.get("baseline", {})
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    seed = int(baseline_cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    def _prefetch_chunks(processed_dir: Path, chunk_names: list[str]) -> None:
        if not chunk_names:
            return
        store = DvcChunkStore(processed_dir)
        print(f"Prefetching {len(chunk_names)} chunks...")
        for name in chunk_names:
            store.pull_chunk(name)

    metadata = load_processed_metadata(processed_dir)
    batch_size = int(baseline_cfg.get("batch_size", 64))
    max_local_chunks = int(baseline_cfg.get("max_local_chunks", 2))
    num_workers = int(baseline_cfg.get("num_workers", 0))

    prefetch_chunks = bool(baseline_cfg.get("prefetch_chunks", False))
    delete_after_eviction = bool(baseline_cfg.get("delete_after_eviction", True))

    if prefetch_chunks:
        train_chunks = list_split_chunks(processed_dir, "train")
        val_chunks = list_split_chunks(processed_dir, "val")
        test_chunks = list_split_chunks(processed_dir, "test")
        _prefetch_chunks(processed_dir, train_chunks + val_chunks + test_chunks)

    train_loader = build_dataloader(
        processed_dir=processed_dir,
        split="train",
        batch_size=batch_size,
        max_local_chunks=max_local_chunks,
        shuffle=True,
        num_workers=num_workers,
        delete_after_eviction=False,
    )
    val_loader = build_dataloader(
        processed_dir=processed_dir,
        split="val",
        batch_size=batch_size,
        max_local_chunks=max_local_chunks,
        shuffle=False,
        num_workers=num_workers,
        delete_after_eviction=delete_after_eviction,
    )
    test_loader = build_dataloader(
        processed_dir=processed_dir,
        split="test",
        batch_size=batch_size,
        max_local_chunks=max_local_chunks,
        shuffle=False,
        num_workers=num_workers,
        delete_after_eviction=delete_after_eviction,
    )

    model = GraphBaselineLightningModule(
        input_dim=int(metadata["feature_dim"]),
        hidden_dim=int(baseline_cfg.get("hidden_dim", 128)),
        dropout=float(baseline_cfg.get("dropout", 0.2)),
        pooling=str(baseline_cfg.get("pooling", "mean_max")),
        learning_rate=float(baseline_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(baseline_cfg.get("weight_decay", 1e-4)),
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
