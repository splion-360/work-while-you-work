# BGE-M3 Binary Row-Distribution Evaluation

Status: concluded on 2026-09-07.

## Decision

Collapse the source labels into two targets:

| Source label | Binary label |
| --- | --- |
| No Fit | No Fit |
| Potential Fit | Good Fit |
| Good Fit | Good Fit |

The production score is the calibrated probability of `Good Fit`, multiplied by 100. The classifier uses the existing frozen BGE-M3 token vectors and row-distribution K-NRM features. Only the standardized linear `77 → 2` layer is trained.

## Train

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/train_classifier.py \
  --feature-report artifacts/data__bge_m3_row_distribution_features.json \
  --report artifacts/train__bge_m3_row_distribution_binary.json \
  --device cuda \
  --fold 0 --fold 1 --fold 2 \
  --label-mode binary-potential-positive \
  --max-epochs 50 \
  --patience 50 \
  --experiment resume-jd-bge-m3-row-distribution-binary
```

All folds ran for 50 epochs. Parent MLflow run: `8dcf84d370b04a469feac0411306dd27`.

## Outer-Test Results

These rows remain graph-disjoint from each fold's training partition.

| Fold | Rows | Accuracy | Macro F1 | Good Fit precision | Good Fit recall | Good Fit F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10,644 | 0.7156 | 0.6766 | 0.4775 | 0.6897 | 0.5643 |
| 1 | 10,405 | 0.7139 | 0.6815 | 0.4670 | 0.7648 | 0.5799 |
| 2 | 10,284 | 0.6802 | 0.6535 | 0.4500 | 0.7317 | 0.5573 |
| Pooled | 31,333 | 0.7034 | 0.6707 | 0.4643 | 0.7280 | 0.5670 |

Pooled uncalibrated probability metrics: ROC-AUC `0.7868`, average precision `0.5860`, log loss `0.5642`, Brier score `0.1922`, and 10-bin ECE `0.1800`.

## Calibrate

Platt scaling is fitted separately on each fold's graph-disjoint inner-validation rows. Outer-test labels are not used to fit calibration.

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/calibrate_binary_classifier.py \
  --training-report artifacts/train__bge_m3_row_distribution_binary.json \
  --feature-report artifacts/data__bge_m3_row_distribution_features.json \
  --report artifacts/eval__bge_m3_row_distribution_binary_calibration.json
```

| Pooled metric | Before | After |
| --- | ---: | ---: |
| Log loss | 0.5642 | 0.4764 |
| Brier score | 0.1922 | 0.1552 |
| 10-bin ECE | 0.1800 | 0.0320 |
| Accuracy at 0.5 | 0.7034 | 0.7784 |
| Macro F1 at 0.5 | 0.6707 | 0.6858 |
| Good Fit precision at 0.5 | 0.4643 | 0.6185 |
| Good Fit recall at 0.5 | 0.7280 | 0.4414 |

Calibration materially improves probability reliability. The `0.5` classification threshold becomes more conservative and lowers Good Fit recall; score calibration and the operational decision threshold are separate choices.

## Threshold Selection

For each fold, all distinct calibrated probabilities from the inner-validation partition are evaluated. The threshold with maximum validation macro F1 is frozen, then applied to the untouched outer-test partition. Outer-test labels do not influence threshold selection.

| Fold | Selected threshold | Validation macro F1 |
| ---: | ---: | ---: |
| 0 | 0.3790 | 0.6724 |
| 1 | 0.3942 | 0.7327 |
| 2 | 0.3944 | 0.7188 |

Outer-test performance using each fold's validation-selected threshold:

| Fold | Accuracy | Macro F1 | Good Fit precision | Good Fit recall | Good Fit F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7393 | 0.6889 | 0.5095 | 0.6305 | 0.5636 |
| 1 | 0.7718 | 0.7107 | 0.5533 | 0.6044 | 0.5777 |
| 2 | 0.7483 | 0.6859 | 0.5420 | 0.5497 | 0.5458 |
| Pooled | 0.7531 | 0.6952 | 0.5333 | 0.5948 | 0.5624 |

| Fold | No Fit precision | No Fit recall | No Fit F1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.8527 | 0.7789 | 0.8141 |
| 1 | 0.8577 | 0.8301 | 0.8437 |
| 2 | 0.8282 | 0.8237 | 0.8260 |
| Pooled | 0.8461 | 0.8107 | 0.8280 |

Compared with calibrated `0.5` classification, pooled Good Fit recall rises from `0.4414` to `0.5948`, Good Fit precision falls from `0.6185` to `0.5333`, and macro F1 rises from `0.6858` to `0.6952`.

## Package

Fold 1 is selected because it has the highest inner-validation macro F1 (`0.6995`). Selection does not use outer-test results.

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/package_binary_scorer.py \
  --training-report artifacts/train__bge_m3_row_distribution_binary.json \
  --calibration-report artifacts/eval__bge_m3_row_distribution_binary_calibration.json \
  --feature-report artifacts/data__bge_m3_row_distribution_features.json \
  --token-metadata artifacts/model__bge_m3_multivector_generation.json \
  --output exports/binary-scorer
```

The generated bundle contains `classifier.pt` and `manifest.json`. The manifest pins the checkpoint hash, BGE-M3 fingerprint, 77-feature contract, fold-1 Platt coefficients, and fold-1 validation-selected threshold. The BGE-M3 weights remain external to the bundle.

## Production Score

The linear classifier produces a No Fit logit and a Good Fit logit. Define the margin as:

```text
m = Good Fit logit − No Fit logit
```

Each fold calibrates that margin with `P(Good Fit) = sigmoid(am + b)`:

| Fold | Slope `a` | Intercept `b` |
| ---: | ---: | ---: |
| 0 | 0.7692 | −0.6399 |
| 1 | 1.1530 | −1.0328 |
| 2 | 1.0803 | −1.1503 |

The packaged scorer uses fold 1, so its displayed score is:

```text
Score = 100 × sigmoid(1.1530m − 1.0328)
```

The score remains a calibrated probability percentage. The packaged fold-1 threshold classifies scores of at least `39.42` as Good Fit.

## No Fit Performance

Calibrated outer-test performance at the `0.5` baseline threshold:

| Fold | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.8112 | 0.9100 | 0.8577 |
| 1 | 0.8321 | 0.9046 | 0.8668 |
| 2 | 0.8046 | 0.8877 | 0.8441 |
| Pooled | 0.8160 | 0.9010 | 0.8564 |

Fold 1 is the packaged production candidate. The pooled row summarizes predictions from all three independently trained and calibrated folds; it is evaluation evidence, not a fourth model.

## Good Fit Performance

Calibrated outer-test performance at the `0.5` baseline threshold:

| Fold | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.6288 | 0.4184 | 0.5024 |
| 1 | 0.6346 | 0.4756 | 0.5437 |
| 2 | 0.5935 | 0.4320 | 0.5000 |
| Pooled | 0.6185 | 0.4414 | 0.5152 |

At the calibrated `0.5` threshold, the model favors Good Fit precision over recall. Fold 1 identifies 47.56% of actual Good Fit pairs, and 63.46% of its Good Fit predictions are correct under the merged label definition.

## Validation

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/ruff check src scripts tests
.venv/bin/pytest -q
```

Result: `57 passed`. A smoke test loaded the packaged scorer and processed a real cached resume-JD pair through row-distribution pooling, standardization, classification, and calibration.

## Boundary

This work produces a deployable scoring bundle and Python inference boundary in the ML worktree. Replacing the current Ollama scorer in the job-tracker service remains a separate integration change in the job-tracker worktree.
