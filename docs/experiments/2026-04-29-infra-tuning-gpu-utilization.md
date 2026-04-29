# 2026-04-29 22:45 - Infra tuning vs quality trade-off

- Context: Improve GPU utilization and epoch throughput on `student-nvidia` without reintroducing NaN instability.
- Evidence:
  - Run A (baseline): `f7i4qxzx`
  - Run B (larger batch/workers, 32 GB): `gttgodlk`
  - Run C (larger batch/workers, higher CPU/RAM/prefetch): `jhvqr5jt`
  - W&B links:
    - https://wandb.ai/adrian-walczak119-iiuwr/gnn-insider-threat/runs/f7i4qxzx
    - https://wandb.ai/adrian-walczak119-iiuwr/gnn-insider-threat/runs/gttgodlk
    - https://wandb.ai/adrian-walczak119-iiuwr/gnn-insider-threat/runs/jhvqr5jt

## Command/config diff

- Run A: `bs=1024`, `workers=4`, `prefetch=4`, `precision=32`, `cpus=8`, `mem=32G`, `max_epochs=5`.
- Run B: `bs=2048`, `workers=6`, `prefetch=4`, `precision=auto(bf16)`, `cpus=8`, `mem=32G`, `max_epochs=5`.
- Run C: `bs=2048`, `workers=8`, `prefetch=6`, `precision=auto(bf16)`, `cpus=16`, `mem=60G`, `max_epochs=5`.

## Hypothesis

- Increasing batch size, workers, and host resources should reduce GPU idle time and shorten epoch wall-clock time.

## Outcome

- Throughput improved:
  - epoch time: `6:05` (A) -> `4:40` (B) -> `3:49` (C).
- Stability stayed good:
  - `test/n_nonfinite = 0` in all three runs.
- Quality dropped in larger-batch runs:
  - `test/roc_auc`: `0.7808` -> `0.7530` -> `0.7241`
  - `test/pr_auc`: `0.0183` -> `0.0146` -> `0.00594`
  - `test/tpr_at_fpr_target`: `0.6099` -> `0.4780` -> `0.3407`

## Interpretation

- Infra tuning worked for speed, but the comparison is not update-equivalent.
- Doubling batch size halved steps per epoch (`500 -> 250`) while keeping `max_epochs=5`, so larger-batch runs used fewer optimizer updates.
- Most of the quality regression is likely undertraining from reduced update count, not only infra side effects.

## Next step

- Keep infra profile from Run C (`bs=2048`, `workers=8`, `prefetch=6`, `cpus=16`, `mem=60G`, `bf16`).
- Increase training budget to restore update parity:
  - run C2 with more epochs.
- Compare A vs C2 on:
  - wall-clock to target quality,
  - `test/roc_auc`, `test/pr_auc`, `test/tpr_at_fpr_target`,
  - `*_n_nonfinite`.
