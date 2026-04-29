# Incident: NaN Training on RTX 3090

## Date
2026-04-29

## Summary

Training run stopped early because losses became non-finite in epoch 0.
Validation and test metrics were mostly `NaN` due to non-finite model outputs.

## Evidence

- W&B run: [hj847l2g](https://wandb.ai/adrian-walczak119-iiuwr/gnn-insider-threat/runs/hj847l2g?nw=nwuseradrianwalczak119)
- SLURM log: `logs/train-10524.txt`
- Config signals:
  - `precision=bf16-mixed`
  - `batch_size=1024`
  - `num_activity_classes=216`

## What Was Observed

- `train_loss_step: nan`
- `val_loss: nan`
- `test/n_nonfinite: 146199`
- warnings from `certgnn.lightning_model` about non-finite anomaly scores

## Root Cause

Primary root cause:

- Input features contain very large count-like values (up to millions) and were fed to GCN without normalization/compression.
- Combined with mixed precision, this leads to unstable activations/gradients and early NaN logits.

Secondary findings:

- `val/test has only one class` warnings are a downstream effect when non-finite predictions are dropped, leaving no valid samples.
- Job uses one visible GPU (`CUDA_VISIBLE_DEVICES: [0]`), so low aggregate GPU utilization was expected.

## Fix Applied

- Added feature stabilization before GCN:
  - `nan_to_num`, clamp, `log1p`, `layer_norm`
- Added gradient clipping in trainer.
- Added `--precision-mode` CLI override for controlled debug runs.
- Added tests in `tests/test_model_stability.py`.

## Recommended Runbook

1. First stability run:
   - `--precision-mode 32`
   - reduced batch size (e.g., `512`)
   - `--gradient-clip-val 1.0`
2. If stable, switch to `--precision-mode auto`.
3. Track:
   - `train_loss`
   - `val_loss`
   - `val/n_nonfinite` and `test/n_nonfinite`

## Open Items

- Add gradient-norm logging to W&B to confirm clipping effectiveness.
- Consider split strategy upgrade to reduce label-degenerate evaluation folds.
- TODO:: Confirm next SLURM run has finite train/val loss after full epoch.
- TODO:: Attach W&B curves (train_loss, val_loss, val/n_nonfinite).
