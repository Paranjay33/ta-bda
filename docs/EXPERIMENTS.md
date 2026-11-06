# Experiments (E1-E4)

All four are 3-class classifiers over `negative=0, neutral=1, positive=2`
(`configs/label_mapping.json`), trained on the frozen split's `normalized_text`
(`docs/DATASET.md`) and evaluated under `docs/EVALUATION_PROTOCOL.md`.
Results: `artifacts/metrics/comparison_report.md` (machine-generated) and `README.md`.

## E1 — Word TF-IDF → Multinomial Logistic Regression (Scala + Spark MLlib)

- Pipeline (one `Pipeline`, fitted on train only, saved as a `PipelineModel`):
  `RegexTokenizer(\s+, no lowercasing)` → `CountVectorizer(vocab 20,000)` → `IDF` →
  `LogisticRegression(family=multinomial, maxIter=100, regParam=0)`.
- Code: `spark/src/main/scala/hinglish/experiments/E1LogisticRegression.scala`,
  `features/FeaturePipeline.scala`, shared driver `experiments/ClassicalExperiment.scala`.
- Scores: Spark `probability` column, served as `model_probability` (not separately calibrated;
  with `regParam=0` they are extremely over-confident — see the model card).
- Run: `bash scripts/run_spark.sh e1` (train) · `... e1 --final-test-eval` (locked test).

## E2 — Word + character 3-gram TF-IDF → One-vs-Rest LinearSVC (Scala + Spark MLlib)

- Word stages as E1; character stages: custom `CharacterSplitter` (one token per Unicode code
  point) → Spark `NGram(n=3)` → `CountVectorizer(20,000)` → `IDF` — genuine character n-grams;
  `VectorAssembler` concatenates both; `OneVsRest(LinearSVC(maxIter=100))` because `LinearSVC`
  is binary-only.
- Scores: `rawPrediction` = one SVM margin per class (not probabilities). A temperature
  `softmax(m/T)` is fitted on validation margins (`hinglish.calibrate_classical`); it is
  argmax-preserving, so the served class is always the native OvR decision. A multinomial Platt
  map was evaluated and rejected because it changed the predicted class on ~18% of validation
  rows (it would be a different, validation-trained classifier). Both are recorded in the manifest.
- Fix made during this work: `CharacterSplitter` lacked a `DefaultParamsReadable` companion,
  so a saved E2 model could never be loaded.

## E3 — `FacebookAI/xlm-roberta-base` (PyTorch + Hugging Face)
## E4 — `l3cube-pune/hing-roberta` (PyTorch + Hugging Face)

- One entry point for both: `python -m hinglish.train --experiment-id E3|E4 --model-name-or-path …`
  (`bash scripts/train_transformers.sh`). Only the checkpoint differs; the script refuses any
  other checkpoint under these experiment ids.
- Matched settings and the documented deviations from the midterm run: see
  `docs/EVALUATION_PROTOCOL.md`.
- Outputs per run (`artifacts/models/<exp>/v2-clean-split/`): `best/` (weights, tokenizer,
  `calibration.json`), `checkpoints/` (last kept checkpoint, for resume), `manifest.json`.
- Note: `l3cube-pune/hing-roberta` uses the XLM-RoBERTa architecture/tokenizer
  (`XLMRobertaForSequenceClassification`), the same family as E3 — which makes the comparison
  a clean test of pretraining data.

## Legacy midterm run

`artifacts/models/e3/{best,checkpoint-3644,checkpoint-4555}` (midterm E3, trained/selected on
the leaky split) is preserved untouched and is **not** registered for serving. No E4 midterm
checkpoint exists.

## Manifests

Each model manifest (`schema_version 2`) records: experiment id, display name, version, source
checkpoint and Hub commit hash (E3/E4), seed, dataset identity (split-manifest SHA-256, test
sample-id hash, raw-file hashes, preprocessing fingerprint, label-map hash), hyperparameters,
per-epoch validation log and selected epoch (E3/E4), validation metrics (+ probabilistic),
calibration, score semantics, artifact SHA-256 and size, runtime, environment, and status
(`training`, `selection`, `locked_test_evaluation`). The locked evaluation updates the status
and adds `test_metrics_summary`.
