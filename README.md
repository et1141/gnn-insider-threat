# Insider Threat Detection with GCN and Bi-LSTM

A PyTorch project for detecting insider threats in the CMU CERT r5.2 dataset.
Two preprocessing variants and two model architectures coexist behind a
unified config + Lightning structure.

## Setup

```bash
uv sync --extra dev
```

## Layout

```
src/certgnn/
├── preprocessing/         # DuckDB feature pipeline + two variants
│   ├── paper_faithful.py     #  fractional users, single chunk stream
│   └── user_level_split.py   #  fixed-count + scenario-stratified split
├── data/                  # chunk store, datasets, sampler, DataModule
├── models/                # gcn_lstm.py + graph_pool_mlp.py + registry
├── losses.py              # anomaly_aware + focal + standard CE
├── lightning/             # BaseLightningModule + two task subclasses
├── callbacks/             # GPUMetricsCallback + build_callbacks fabric
├── train.py               # config-driven entry point
└── ...
```

## Run

```bash
uv run preprocess              # variant picked from config.yaml
uv run train                   # task + model picked from config.yaml
uv run pytest                  # 31 tests
uv run ruff check .
```

CLI overrides for the most common knobs:

```bash
uv run train --task binary --model graph_pool_mlp --max-epochs 5 --fast-dev-run
uv run preprocess-paper-faithful --stream    # bypass dispatcher, pin variant
uv run preprocess-user-split --stream
```

## Config knobs

`configs/config.yaml`:

* `preprocessing.variant` — `paper_faithful` | `user_level_split`
* `training.task` — `anomaly_aware` (192-class) | `binary` (2-class)
* `training.model` — `gcn_lstm` | `graph_pool_mlp`
* `training.loss_args.binary.focal_variant` — `fixed_alpha` (default, constant
  `alpha` class weight; stable at ~0.2% positives) | `dynamic_pos_weight`
  (legacy per-batch `n_neg/n_pos` weighting). Tune `alpha` (fixed) or
  `pos_weight_clamp` (dynamic) accordingly.
* `training.{model_args, loss_args, optimizer, scheduler, data, trainer, wandb}`

Adding a new architecture: drop `src/certgnn/models/<name>.py`, register
in `MODEL_REGISTRY`, set `training.model: <name>` in config.yaml.

Adding a new task: drop a subclass of `BaseLightningModule` overriding
`compute_loss`, `collect_eval`, `epoch_metrics`; register in
`LIGHTNING_REGISTRY`.

## Dataset

```bash
uv run download-data
cd data/raw/cmu_cert_r5.2
tar -xjf r5.2.tar.bz2
tar -xjf answers.tar.bz2
```
