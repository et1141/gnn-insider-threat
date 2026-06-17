"""Unified training entry point.

Reads ``configs/config.yaml`` and dispatches to:

* the right DataModule mode (``random_split`` for paper-faithful chunks,
  ``presplit`` for user-level-split chunks),
* the right Lightning task (``anomaly_aware`` or ``binary``),
* the right model architecture from ``MODEL_REGISTRY``,
* the standard callback set from ``build_callbacks``,
* a WandB logger when configured.

A handful of CLI flags override the most common knobs without touching
the YAML; everything else flows through config.

Examples::

    # Use whatever variant/task/model is in config.yaml
    uv run train

    # Smoke-test a single batch on the binary task (focal loss)
    uv run train --task binary --model gcn_transformer --fast-dev-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from certgnn.callbacks import build_callbacks
from certgnn.data import InsiderThreatDataModule
from certgnn.lightning import build_lightning_module
from certgnn.split import RandomSplit
from certgnn.utils import get_project_root, load_config


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _get(d: dict, *keys: str, default: Any = None) -> Any:
    """Traverse a nested dict, returning ``default`` on any missing key."""
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _datamodule_mode_for_variant(variant: str) -> str:
    if variant == "paper_faithful":
        return "random_split"
    if variant == "user_level_split":
        return "presplit"
    raise ValueError(f"Unknown preprocessing.variant {variant!r}")


def _build_datamodule(config: dict, processed_dir: Path) -> InsiderThreatDataModule:
    variant = _get(config, "preprocessing", "variant", default="paper_faithful")
    mode = _datamodule_mode_for_variant(variant)
    data_cfg = _get(config, "training", "data", default={}) or {}

    split_strategy = None
    if mode == "random_split":
        split_strategy = RandomSplit(
            val_size=float(data_cfg.get("val_size", 0.1)),
            test_size=float(data_cfg.get("test_size", 0.2)),
            seed=int(data_cfg.get("sampler_seed", 42)),
        )

    return InsiderThreatDataModule(
        processed_dir=processed_dir,
        mode=mode,
        split_strategy=split_strategy,
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        max_local_chunks=data_cfg.get("max_local_chunks"),
        pin_memory=data_cfg.get("pin_memory"),
        persistent_workers=data_cfg.get("persistent_workers"),
        prefetch_factor=data_cfg.get("prefetch_factor"),
        accelerator=data_cfg.get("accelerator"),
        sampler_seed=int(data_cfg.get("sampler_seed", 42)),
    )


def _build_module(config: dict, feature_dim: int, num_classes: int):
    train_cfg = config.get("training", {})
    task = train_cfg.get("task", "binary")
    model_name = train_cfg.get("model", "graph_pool_mlp")

    # model_args is keyed per model; loss_args per task. train.py picks the
    # matching subsection so swapping model/task via CLI doesn't pull in
    # kwargs meant for the old choice. Falls back to flat dict for backward
    # compat with older config files.
    def _per_key(cfg_block: dict, key: str) -> dict:
        block = cfg_block or {}
        if isinstance(block, dict) and key in block and isinstance(block[key], dict):
            return dict(block[key])
        return dict(block)

    model_args = _per_key(train_cfg.get("model_args"), model_name)
    loss_args = _per_key(train_cfg.get("loss_args"), task)

    # Sensible defaults per architecture if the user didn't pin them.
    if model_name == "graph_pool_mlp":
        model_args.setdefault("input_dim", feature_dim)
    elif model_name in ("gcn_lstm", "gcn_transformer"):
        model_args.setdefault("num_node_features", feature_dim)
        # Binary task uses focal loss on y_label (2 logits); anomaly_aware needs
        # full activity vocabulary from metadata (~192 classes).
        out_dim = 2 if task == "binary" else num_classes
        model_args.setdefault("num_activity_classes", out_dim)

    opt_cfg = train_cfg.get("optimizer", {}) or {}
    sched_cfg = train_cfg.get("scheduler") or {}

    return build_lightning_module(
        task=task,
        model_name=model_name,
        model_args=model_args,
        learning_rate=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
        optimizer=str(opt_cfg.get("name", "adam")),
        scheduler=sched_cfg.get("type") if isinstance(sched_cfg, dict) else sched_cfg,
        scheduler_args={k: v for k, v in sched_cfg.items() if k != "type"} if isinstance(sched_cfg, dict) else None,
        **loss_args,
    )


def _build_logger(config: dict) -> Any:
    wandb_cfg = config.get("training", {}).get("wandb") or {}
    if not wandb_cfg or wandb_cfg.get("disabled"):
        return None
    try:
        from pytorch_lightning.loggers import WandbLogger
        logger_kwargs: dict[str, Any] = {
            "project": wandb_cfg.get("project", "gnn-insider-threat"),
            "mode": wandb_cfg.get("mode", "offline"),
            "name": wandb_cfg.get("run_name"),
            "log_model": False,
        }
        if wandb_cfg.get("entity"):
            logger_kwargs["entity"] = wandb_cfg["entity"]
        return WandbLogger(**logger_kwargs)
    except Exception as exc:
        print(f"W&B logger unavailable ({exc}); continuing without experiment logging.")
        return None


def _load_metadata(processed_dir: Path) -> dict:
    import pickle
    metadata_path = processed_dir / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run preprocessing before training."
        )
    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified GNN insider-threat trainer")
    parser.add_argument("--task", choices=["anomaly_aware", "binary"],
                        help="Override training.task in config.yaml")
    parser.add_argument("--model", help="Override training.model in config.yaml")
    parser.add_argument("--max-epochs", type=int, help="Override training.trainer.max_epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.data.batch_size")
    parser.add_argument("--lr", type=float, help="Override training.optimizer.lr")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run a single train/val/test batch end-to-end (Lightning's fast_dev_run)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logger even if config.yaml has it.")
    args = parser.parse_args()

    config = load_config()

    if args.task is not None:
        config.setdefault("training", {})["task"] = args.task
    if args.model is not None:
        config.setdefault("training", {})["model"] = args.model
    if args.lr is not None:
        config.setdefault("training", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.batch_size is not None:
        config.setdefault("training", {}).setdefault("data", {})["batch_size"] = args.batch_size
    if args.max_epochs is not None:
        config.setdefault("training", {}).setdefault("trainer", {})["max_epochs"] = args.max_epochs
    if args.no_wandb:
        config.setdefault("training", {}).setdefault("wandb", {})["disabled"] = True

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    seed = int(_get(config, "training", "seed", default=42))
    pl.seed_everything(seed, workers=True)

    # Enable TF32 matmuls on Ampere+ GPUs — free throughput, no config knob.
    torch.set_float32_matmul_precision("high")

    metadata = _load_metadata(processed_dir)
    feature_dim = int(metadata["feature_dim"])
    num_classes = int(metadata["num_classes"])

    datamodule = _build_datamodule(config, processed_dir)
    module = _build_module(config, feature_dim, num_classes)

    trainer_cfg = config.get("training", {}).get("trainer", {}) or {}
    callback_cfg = config.get("training", {}).get("callbacks", {}) or {}
    callbacks = build_callbacks(
        monitor=module.monitor_metric,
        monitor_mode=module.monitor_mode,
        patience=int(callback_cfg.get("patience", trainer_cfg.get("patience", 5))),
        save_top_k=int(callback_cfg.get("save_top_k", 3)),
        enable_gpu_metrics=bool(callback_cfg.get("enable_gpu_metrics", True)),
    )

    logger = _build_logger(config)

    trainer = pl.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 10)),
        accelerator=str(trainer_cfg.get("accelerator", "auto")),
        devices=trainer_cfg.get("devices", "auto"),
        precision=trainer_cfg.get("precision", 32),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        deterministic=bool(trainer_cfg.get("deterministic", True)),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=str(root / "artifacts"),
    )

    trainer.fit(module, datamodule=datamodule)
    test_results = trainer.test(module, datamodule=datamodule, ckpt_path="best") if not args.fast_dev_run else []

    metrics: dict[str, float] = {}
    if test_results:
        metrics = {k: float(v) for k, v in test_results[0].items()}
    elif trainer.callback_metrics:
        metrics = {
            k: float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
            for k, v in trainer.callback_metrics.items()
        }

    out = root / "reports" / "training" / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
