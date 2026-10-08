# BGE-M3 Row-Distribution Evaluation

Status: concluded on 2026-09-07.

## Objective

Test whether retaining the distribution of per-JD-token matching evidence improves three-class resume-JD classification over global K-NRM pooling.

## Architecture

For normalized BGE-M3 token matrices `R ∈ ℝⁿˣ¹⁰²⁴` and `J ∈ ℝᵐˣ¹⁰²⁴`:

```text
S = JRᵀ ∈ ℝᵐˣⁿ
```

Each JD token receives a length-normalized response from each of the 11 fixed K-NRM kernels:

```text
Hᵢₖ = 0.01 log(max((1/n) Σⱼ exp(−(Sᵢⱼ − μₖ)² / (2σₖ²)), 10⁻¹⁰))
```

For every kernel, the model retains the mean, population standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum over JD tokens. This produces `11 × 7 = 77` fixed features. A standardized linear `77 → 3` classifier is the only learned component. BGE-M3 and the kernels remain frozen.

## Reproduce

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/materialize_knrm_features.py \
  --token-metadata artifacts/model__bge_m3_multivector_generation.json \
  --device cuda \
  --all-folds \
  --pooling row-distribution \
  --report artifacts/data__bge_m3_row_distribution_features.json
```

The all-folds artifact computes every pair once. Training and evaluation continue to select rows using each fold's existing role column.

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/train_classifier.py \
  --feature-report artifacts/data__bge_m3_row_distribution_features.json \
  --report artifacts/train__bge_m3_row_distribution_classifier.json \
  --device cuda \
  --fold 0 --fold 1 --fold 2 \
  --max-epochs 50 \
  --patience 50 \
  --experiment resume-jd-bge-m3-row-distribution
```

```bash
cd "$(git rev-parse --show-toplevel)/ml"
for fold in 0 1 2; do
  .venv/bin/python scripts/evaluate_classifier.py \
    --training-report artifacts/train__bge_m3_row_distribution_classifier.json \
    --feature-report artifacts/data__bge_m3_row_distribution_features.json \
    --fold "$fold" \
    --report "artifacts/eval__bge_m3_row_distribution_fold${fold}.json"
done
```

```bash
cd "$(git rev-parse --show-toplevel)/ml"
.venv/bin/python scripts/diagnose_knrm_features.py \
  --feature-reports artifacts/data__bge_m3_row_distribution_features.json \
  --folds 0 1 2 \
  --output-root data/analysis/row_distribution_diagnosis \
  --report artifacts/eval__bge_m3_row_distribution_diagnosis.json
```

## Run Identity

| Property | Value |
| --- | --- |
| Source commit used by MLflow | `9e17d1e` |
| Feature fingerprint | `e90ccb0920f89bd28445c973ad1400d318c8641f760313a68589a65dd90dff23` |
| Materialized pairs | 93,733 |
| Feature dimension | 77 |
| Feature runtime | 750.498 s |
| Parent MLflow run | `4ef0f4261a28429e8e9a66aa080e0b35` |
| Fold-0 run | `e21c99431bdd4aef9fb9b64c5958058b` |
| Fold-1 run | `9be7f708aab648dbb9dc8901f751e5d9` |
| Fold-2 run | `75913c05da6d49a181770f96902dcf5c` |

All three folds executed 50 epochs. Best validation-loss checkpoints occurred at epochs 4, 7, and 8.

## Outer-Test Results

| Fold | Rows | Global accuracy | Distribution accuracy | Delta | Global macro F1 | Distribution macro F1 | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 10,644 | 0.6137 | 0.5922 | −0.0215 | 0.4200 | 0.4481 | +0.0281 |
| 1 | 10,405 | 0.5942 | 0.5949 | +0.0007 | 0.4224 | 0.4515 | +0.0291 |
| 2 | 10,284 | 0.5888 | 0.5397 | −0.0491 | 0.4119 | 0.4262 | +0.0143 |
| Mean | 10,444 | 0.5989 | 0.5756 | −0.0233 | 0.4181 | 0.4419 | +0.0238 |
| Sample SD | — | 0.0131 | 0.0311 | — | 0.0055 | 0.0138 | — |

The test-row-weighted distributional results are accuracy `0.5758` and macro F1 `0.4420`.

## Mean Class F1

| Class | Global K-NRM | Row distribution | Delta |
| --- | ---: | ---: | ---: |
| No Fit | 0.7718 | 0.7404 | −0.0313 |
| Potential Fit | 0.2114 | 0.2955 | +0.0840 |
| Good Fit | 0.2710 | 0.2899 | +0.0188 |

The row-distribution representation improves Potential Fit F1 in every fold. Good Fit improves in folds 0 and 1 but falls in fold 2. No Fit F1 falls in every-fold mean, producing the lower overall accuracy.

## Representation Diagnosis

| Measurement | Global K-NRM | Row distribution |
| --- | ---: | ---: |
| Effective feature dimension | 1.243 | 4.385 |
| Effective dimension after length residualization | 3.201 | 4.466 |
| Fold-0 outer-train-to-test 10-neighbor purity | 0.6430 | 0.6536 |
| Fold-1 outer-train-to-test 10-neighbor purity | 0.6531 | 0.6638 |
| Fold-2 outer-train-to-test 10-neighbor purity | 0.6434 | 0.6413 |
| Maximum absolute JD-length Spearman ρ | 1.000 | 0.900 |

The representation is less collapsed and less dominated by JD length, but length association remains. Sixteen channels are constant across the combined outer-test folds, primarily statistics from saturated extreme kernels. The t-SNE projections remain visibly label-mixed in every fold.

## Decision

Row-distribution pooling is the better measured representation when balanced three-class performance is the objective. It raises mean macro F1 by 0.0238 and materially recovers Potential Fit without adding a nonlinear classifier.

It is not an overall accuracy improvement. The 0.0233 mean accuracy reduction and larger fold variance show that the additional distributional information trades some No Fit discrimination for minority-class recall. The experiment confirms that preserving per-token coverage is useful, but fixed global summary statistics still do not cleanly separate the three labels.
