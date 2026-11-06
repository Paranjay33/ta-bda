# Evaluation protocol

## Sequence (enforced by the code)

1. Validate data and build/verify the frozen split (`hinglish.prepare_data`).
2. Train every model on the **train** split only.
3. Select using **validation** only: E3/E4 keep the epoch with the best validation Macro-F1
   (`load_best_model_at_end`, early-stopping patience 2). E1/E2 have a fixed configuration —
   validation is used only for reporting and for fitting score calibration.
4. Fit calibration on **validation** only (E3/E4 temperature on logits; E2 temperature on OvR
   margins; E1 none).
5. Freeze: model manifests record version, hashes and status `locked_test_evaluation: not_run`.
6. Locked test evaluation, once: `bash scripts/evaluate_all_locked.sh`
   (Spark scores TEST with the saved `PipelineModel`s; `hinglish.evaluate_locked` scores E3/E4
   and computes metrics for all four). Every run is appended to
   `artifacts/metrics/locked_test_log.jsonl`, so a repeated test evaluation is visible.
7. `python -m hinglish.report` builds the comparison table, the E3/E4 significance test and the
   error analysis from saved artifacts.

No hyperparameter, epoch, threshold, calibration or Auto-policy decision reads the test split.
The API's Auto policy is configuration (`PREFERRED_MODEL`), never "highest test score".

## Metrics

Computed by `hinglish.evaluate.compute_classification_metrics` (scikit-learn) for every model;
Spark also computes them natively (`*_metrics_spark.json`) as a cross-check.

- Accuracy; macro precision/recall/F1 (unweighted mean over the 3 classes — **primary metric:
  Macro-F1**); weighted F1 (by gold support); per-class precision/recall/F1/support; confusion
  matrix (rows = gold, columns = predicted, canonical order negative/neutral/positive).
- Probabilistic (where scores are probabilities): multiclass log-loss, multiclass Brier score
  (sum of squared errors over classes, averaged over samples), ECE with 15 equal-width
  confidence bins. For E2 these are reported for the calibrated map only — raw SVM margins are
  not probabilities. ROC-AUC is not reported.
- E3 vs E4: paired bootstrap (2,000 resamples of the test set, seed 42) 95% CI on the Macro-F1
  difference, and an exact McNemar test on per-sample correctness.

## Matched E3/E4 protocol

Same entry point (`python -m hinglish.train`), split, seed (42), max length (128), batch sizes
(16/32), learning rate (2e-5), epochs (5, early stopping 2), weight decay (0.01), warmup ratio
(0.06), fp16, optimizer (8-bit AdamW), selection rule (validation Macro-F1) and calibration.
`train.py` refuses to train E3/E4 from any checkpoint other than
`FacebookAI/xlm-roberta-base` / `l3cube-pune/hing-roberta`.

### Deviations from the midterm protocol (identical for both arms)

| Change | Reason |
|---|---|
| Verified, de-duplicated split instead of the supplied leaky split | integrity (see `docs/DATASET.md`) |
| `warmup_ratio=0.06` applied | it was in config but commented out in code |
| 8-bit AdamW (bitsandbytes) instead of fp32 AdamW | both checkpoints have ~278M parameters (250k-token vocabulary); fp32 weights+grads+Adam moments (~4.4 GB) exceed the 4 GB GPU |
| PyTorch `.bin` checkpoints instead of safetensors | safetensors 0.8.0 save crashed natively in this environment |

Because of these changes the new numbers are **not** comparable to the midterm table; they
replace it.

## What the test numbers mean

One dataset (SentiMix tweets, preprocessed upstream), one split, one seed. They estimate
in-distribution performance on held-out SentiMix tweets. They do not establish performance on
other domains, raw (non-preprocessed) text, or new time periods.

## v3 protocol (Kaggle, optional second run)

`configs/kaggle_v3.yaml` + `python -m hinglish.kaggle_v3` (see `kaggle/README.md`). Identical for E3 and E4:
full fp32 AdamW, max 30 epochs, early stopping patience 3, learning-rate search {1e-5, 2e-5, 3e-5}
on seed 42 using **validation** Macro-F1, the chosen rate re-run with seeds 42/43/44, served model =
best validation seed; validation mean ± sd across seeds is reported. Everything else (split, max
length, batch sizes, warmup, weight decay, calibration) is unchanged. Kaggle never reads the test split.

After `python -m hinglish.import_kaggle_v3 <zip>` (verifies split identity, weight hashes, label
order; archives v2 results under `artifacts/metrics/runs/`), `bash scripts/evaluate_v3_locked.sh`
evaluates the two v3 models on the locked test split **once** (it refuses to repeat). Because the
test split was already used for v2, v3 is reported as a **second run**, next to v2, with both
entries in `locked_test_log.jsonl`; no v3 setting was chosen using test data.
