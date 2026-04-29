# ADR 0001: Training Stability and NaN Mitigation on CUDA

## Status
Accepted

## Date
2026-04-29

## Context

Training on RTX 3090 produced `NaN` losses and invalid validation/test metrics.
The issue persisted across several commits intended to fix precision and dataloader behavior.

Evidence:

- W&B run: [hj847l2g](https://wandb.ai/adrian-walczak119-iiuwr/gnn-insider-threat/runs/hj847l2g?nw=nwuseradrianwalczak119)
- Log file: `logs/train-10524.txt`
- Symptoms:
  - `train_loss_step: nan`, `val_loss: nan` in epoch 0
  - `test/n_nonfinite = 146199`
  - warnings about non-finite anomaly scores from model outputs

## Decision

Adopt explicit numerical stabilization in the model input path and optimizer setup:

1. Stabilize graph node features before GCN:
   - replace non-finite values (`nan_to_num`)
   - clamp extreme values
   - apply `log1p` dynamic-range compression
   - apply per-node `layer_norm` across feature dimension
2. Enable gradient clipping in training (`gradient_clip_algorithm="norm"`).
3. Add CLI precision override for controlled debugging:
   - `--precision-mode auto|32|bf16`

Implemented in:

- `src/certgnn/model.py`
- `src/certgnn/train.py`
- `tests/test_model_stability.py`

## Consequences

Positive:

- Much lower risk of early NaN divergence with large count-like features.
- Faster diagnosis path (`--precision-mode 32` for deterministic stability checks).
- Safer mixed-precision operation on CUDA.

Trade-offs:

- Input transformation changes feature distribution relative to raw counts.
- Potential small behavior shift vs. prior runs; old metrics are not directly comparable.

Risks:

- If instability remains, root cause may include data construction or split policy.
- Model can still underutilize GPU due to small parameter count and I/O limits.

## Validation

- Code-level validation completed (stability path + dedicated tests added).
- TODO:: Run a full SLURM training job and confirm no NaN in `train_loss`/`val_loss`.
- TODO:: Attach W&B plots for `train_loss`, `val_loss`, `val/n_nonfinite`, `gpu/utilization_pct`.
