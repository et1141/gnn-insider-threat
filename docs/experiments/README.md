# Experiment Log

Use one short file per experiment iteration when needed, or append to a running log.

## Required Fields

- Date/time
- Command/config diff
- Hypothesis
- Outcome
- Next step
- `TODO::` entries for manual checks/artifacts

## Example Entry

```md
## 2026-04-30 10:15
- Command: python src/certgnn/train.py --precision-mode 32 --batch-size 512
- Hypothesis: fp32 + smaller batch removes NaN in epoch 0.
- Outcome: preliminary stable first 3 epochs, val_loss finite.
- Next step: test precision-mode auto with same batch size.
- TODO:: Confirm full run does not regress after epoch 3.
- TODO:: Add train/val loss plot from W&B.
```
