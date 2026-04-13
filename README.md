# Insider Threat Detection with GCN and Bi-LSTM

A PyTorch project for detecting insider threats using Graph Convolutional Networks and Bidirectional LSTM.

## Setup

```bash
uv sync --extra dev
```

## Run Tests

```bash
uv run pytest
```

## Baseline training

The baseline uses split-aware processed graph chunks and a lightweight graph pooling + MLP model.

```bash
uv run preprocess
uv run train-baseline
```

The training command logs to Weights & Biases in offline mode by default and writes summary metrics to `reports/baseline/metrics.json`.

## Lint

```bash
uv run ruff check .
```

## Download dataset and unpack (only r5.2 for now)

```bash
uv run python src/download_data.py
```

```bash
cd data/raw/cmu_cert_r5.2
tar -xjf r5.2.tar.bz2
tar -xjf answers.tar.bz2
```
