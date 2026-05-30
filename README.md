# Insider Threat Detection with GCN + Bi-LSTM

[![CI](https://github.com/OWNER/gnn-insider-threat/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

Deep-learning project that detects malicious insider activity in the
**CMU CERT r5.2** dataset by modelling each user's activity sub-sessions as
graphs and learning to predict the next masked activity. It reimplements and
hardens the GCN + Bi-LSTM *anomaly-aware* approach of
[arXiv:2512.18483](https://arxiv.org/abs/2512.18483v1), plus a lighter
graph-pooling baseline.

> Course context: this repository targets the engineering / MLOps rubric —
> reproducible pipeline, config-driven training, tests, CI, and experiment
> tracking. See [docs/project_overview.md](docs/project_overview.md) for the
> brief and [docs/REFACTOR_REPORT.md](docs/REFACTOR_REPORT.md) for the cleanup
> history.

---

## 1. Problem

Insider threat detection is a needle-in-a-haystack anomaly problem: a handful of
malicious actions are hidden among millions of benign log events (logon, device,
file, email, http). We frame it as **self-supervised activity prediction**: the
model learns each user's normal behaviour by predicting a masked activity from
its session graph. On malicious samples the target is inverted (pushed *away*
from the true activity), so at inference a low residual probability of the true
activity becomes the anomaly score.

## 2. Dataset

- **Source:** CMU CERT Insider Threat dataset, release **r5.2**
  ([Figshare article 12841247](https://figshare.com/articles/dataset/Insider_Threat_Test_Dataset/12841247)).
- **Scale:** tens of millions of events; 99 malicious users across several
  attack scenarios.
- **Not in git.** Raw archives and the multi-GB processed graph chunks are
  versioned with **DVC** and streamed from a Google Drive remote on demand
  (see §6). Only `.dvc` pointer files and a small chunk manifest are committed.

Download the raw data:

```bash
uv run download-data
cd data/raw/cmu_cert_r5.2
tar -xjf r5.2.tar.bz2
tar -xjf answers.tar.bz2
```

## 3. Project structure

```text
.
├── configs/                 # split YAML config (merged at load time)
│   ├── paths.yaml           #   filesystem layout + dataset download
│   ├── preprocessing.yaml   #   variant selection + preprocessing knobs
│   ├── training.yaml        #   task/model/optimizer/trainer/wandb
│   └── chunks_manifest.json # generated: chunk -> #graphs (data manifest)
├── src/certgnn/
│   ├── preprocessing/       # DuckDB feature pipeline + two split variants
│   │   ├── common.py        #   shared raw→features→graph pipeline
│   │   ├── paper_faithful.py#   fractional users, single chunk stream
│   │   ├── user_level_split.py # fixed-count, scenario-stratified split
│   │   └── features_duckdb.py  # SQL/UDF feature extraction
│   ├── data/                # DVC chunk store, streaming datasets, sampler, DataModule
│   ├── models/              # gcn_lstm + graph_pool_mlp + name registry
│   ├── lightning/           # BaseLightningModule + 2 task subclasses + metrics
│   ├── losses.py            # anomaly-aware soft-label CE + focal loss
│   ├── callbacks/           # checkpoint/early-stop/GPU-metrics factory
│   ├── split.py             # pluggable train/val/test split strategies
│   ├── train.py             # unified config-driven entry point
│   ├── download_data.py     # Figshare download script
│   └── feature_extraction.py# reference Python features (test oracle)
├── tests/                   # pytest: data, model, loss, training smoke
├── scripts/                 # train.sh (macOS) + slurm_train.sh (cluster)
├── notebooks/               # EDA only (not part of the pipeline)
├── dvc.yaml / dvc.lock      # reproducible download→preprocess→train DAG
└── pyproject.toml           # uv project + ruff/pytest config
```

## 4. Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev          # create .venv and install runtime + dev deps
```

## 5. Quickstart

```bash
uv run preprocess            # build graphs for the variant in configs/preprocessing.yaml
uv run train                 # train the task+model in configs/training.yaml
uv run pytest                # run the test suite
uv run ruff check .          # lint
```

Common CLI overrides (everything else flows through config):

```bash
uv run train --task binary --model graph_pool_mlp --max-epochs 10
uv run train --fast-dev-run --no-wandb        # 1 batch end-to-end, no logging
uv run preprocess-paper-faithful --stream     # pin a variant, push chunks to DVC
uv run preprocess-user-split --stream
```

## 6. Pipeline & reproducibility

The flow is a DVC DAG (`dvc dag` to visualise):

```text
download ──> preprocess_paper_faithful ─┐
         └─> preprocess_user_split ─────┴─> train
```

- **`dvc repro`** reruns only the stages whose code/params/data changed.
- **Data versioning:** processed graph chunks are pushed to / pulled from the
  DVC remote (Google Drive) by `DvcChunkStore`, so training streams chunks
  without ever materialising the full dataset locally.
- **Determinism:** fixed seeds (`pl.seed_everything`), fixed split strategies
  (`RandomSplit` / user-level split with a seed), `deterministic=True` trainer,
  and the best checkpoint is saved by `ModelCheckpoint`.
- **Config:** all hyperparameters live in `configs/*.yaml`; `load_config()`
  deep-merges them into one dict.

### Two preprocessing variants

| Variant | User selection | Split | Leakage |
|---|---|---|---|
| `paper_faithful` | fractional | per-graph `RandomSplit` at train time | sub-session variants can leak across folds (see §9) |
| `user_level_split` | fixed-count, scenario-balanced | materialised `train_/val_/test_` chunks, split **by user** | no cross-fold sub-session leak |

## 7. Models & tasks

| Model (`training.model`) | Task (`training.task`) | Output | Loss | Monitor |
|---|---|---|---|---|
| `gcn_lstm` | `anomaly_aware` | activity logits (`num_classes`) | anomaly-aware soft-label CE | `val/loss` ↓ |
| `graph_pool_mlp` | `binary` | 2-class logits | focal loss + dynamic pos-weight | `val/roc_auc` ↑ |

Adding an architecture: drop `src/certgnn/models/<name>.py`, register it in
`MODEL_REGISTRY`, set `training.model: <name>`. Adding a task: subclass
`BaseLightningModule` (override `compute_loss` / `collect_eval` /
`epoch_metrics`) and register it in `LIGHTNING_REGISTRY`.

## 8. Experiment tracking (W&B)

Runs log hyperparameters, training curves, and the operating-point metrics
(`roc_auc`, `pr_auc`, `tpr_at_fpr_target`, …) to Weights & Biases.

- Project: **`gnn-insider-threat`**
- Dashboard: `https://wandb.ai/<your-entity>/gnn-insider-threat` *(set your entity)*
- Default mode is `offline` (configs/training.yaml → `training.wandb.mode`).
  Sync offline runs with `wandb sync outputs/wandb/offline-run-*`, or set
  `mode: online`.

## 9. Limitations

- **Leakage in `paper_faithful`.** A per-graph random split scatters the ~50
  masked variants of one sub-session across train/val/test, inflating test AUC.
  The `user_level_split` variant fixes this by splitting at the user level;
  prefer it for honest evaluation.
- **Threshold reporting.** Confusion-matrix metrics use a fixed `0.5` threshold;
  for deployment use `threshold_at_fpr_target` instead
  (see [docs/notes/target-fpr.md](docs/notes/target-fpr.md)).

## 10. References

- Paper: [arXiv:2512.18483](https://arxiv.org/abs/2512.18483v1) — GCN + Bi-LSTM
  anomaly-aware insider threat detection.
- Dataset: CMU CERT Insider Threat r5.2.
- Author reference code: `context/source_code/` (kept locally, gitignored).
