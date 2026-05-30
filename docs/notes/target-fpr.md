# Target FPR in This Project

## What It Is

`target_fpr` is the false-positive-rate budget used to choose an operating threshold for anomaly scores.

- FPR (false positive rate) = `FP / (FP + TN)`.
- In this project, anomaly score is `1 - p(true_activity_class)`.
- Higher score => more anomalous.

The threshold is not fixed at `0.5`. It is selected from ROC points to satisfy the FPR budget.

## How the Paper Defines It

The implementation follows the paper rule cited in code comments:

- Paper section V-C, eq. 30-31.
- Operating threshold is chosen as:
  - `t_opt = max { t | FPR(t) <= target_fpr }`

Configured values (as documented in code):

- r5.2 -> `target_fpr = 0.05`
- r6.2 -> `target_fpr = 0.09`

References in repository:

- `src/certgnn/lightning/metrics.py` (`binary_metrics_from_scores`)
- `src/certgnn/lightning/anomaly_aware.py`
- `tests/test_loss_and_metrics.py`

## How We Implement It

Current logic (`binary_metrics_from_scores` in `src/certgnn/lightning/metrics.py`):

1. Compute ROC arrays with `roc_curve(labels, scores)`.
2. Find indices where `fpr_arr <= target_fpr`.
3. Pick the rightmost valid index (largest threshold that still meets budget).
4. Log:
   - `*_tpr_at_fpr_target`
   - `*_fpr_at_target`
   - `*_threshold_at_fpr_target`

If no ROC point satisfies the constraint, code falls back to the strictest point (`idx=0`), consistent with current tests.

## Why `AUC` Can Improve While `TN=0`

Not a contradiction.

- `roc_auc` evaluates ranking quality across all thresholds.
- `tn/fp/fn/tp` in current logs are computed at a fixed threshold (`0.5` in metric helper).

So ranking may improve (`AUC` up) while fixed-threshold confusion values stay poor (`TN=0`, `FPR=1`).

## How We Should Use It

For this project, operational decisions should use `threshold_at_fpr_target`, not fixed `0.5`.

Practical policy:

- Tune threshold on validation split only.
- Keep test split for final reporting.
- Report at least:
  - `val/test roc_auc`, `pr_auc`
  - `fpr_at_target`, `tpr_at_fpr_target`
  - `threshold_at_fpr_target`

## Current Project State

In recent stable run metrics:

- `target_fpr` is `0.05`.
- `threshold_at_fpr_target` is around `0.99416` (val/test close to each other).
- This indicates fixed `0.5` is too low for deployment in this setup.

## TODO

- TODO:: Switch confusion-matrix metrics (`tn/fp/fn/tp/accuracy/precision/recall`) to be reported at `threshold_at_fpr_target` in addition to fixed `0.5`.
- TODO:: Add a small W&B panel that highlights `threshold_at_fpr_target` per epoch and the corresponding `tpr_at_fpr_target`.
