# HinglishSense

A working Hinglish (Hindi-English code-mixed, Roman script) sentiment analysis platform backed by
the project's four TA/BDA experiments. Type a sentence, pick a model, get a real prediction from a
trained artifact — with honest score semantics, model version and inference time — compare E3 vs
E4, analyse CSVs in batch, and inspect locked-test research results.

> Research prototype on SemEval-2020 Task 9 (SentiMix) tweets. Not a judgement of anyone's feelings.

## Research question

> Does Hindi-English domain-adapted pretraining (E4, HingRoBERTa) give a measurable improvement over
> generic multilingual pretraining (E3, XLM-R) for three-class Hinglish sentiment, under a matched
> protocol?

| Exp | Area | Approach | Stack |
|---|---|---|---|
| E1 | BDA | Word TF-IDF + multinomial Logistic Regression | Scala + Spark MLlib 3.5.1 |
| E2 | BDA | Word + genuine char-3-gram TF-IDF + One-vs-Rest LinearSVC | Scala + Spark MLlib 3.5.1 |
| E3 | TA | Fine-tuned `FacebookAI/xlm-roberta-base` | PyTorch + Hugging Face |
| E4 | TA | Fine-tuned `l3cube-pune/hing-roberta` | PyTorch + Hugging Face |

## Results (locked test split)

<!-- RESULTS:START -->
Evaluated **once** on the frozen, de-duplicated test split (1,457 tweets, seed 42); copied from the
machine-generated `artifacts/metrics/comparison_report.md`.

| Model | Test Accuracy | **Test Macro-F1** | Weighted-F1 | F1 neg / neu / pos | Val Macro-F1 | Test ECE |
|---|---:|---:|---:|---|---:|---:|
| E1 TF-IDF + LR (Spark) | 49.83% | **50.13%** | 49.89% | 51.4 / 45.7 / 53.3 | 49.81% | 0.499 (uncalibrated) |
| E2 Word+Char + OvR-SVM (Spark) | 54.98% | **55.33%** | 54.98% | 57.9 / 49.4 / 58.7 | 54.12% | 0.045 |
| E3 XLM-R | 64.72% | **65.12%** | 64.66% | 67.5 / 56.6 / 71.3 | 63.40% | 0.039 |
| E4 HingRoBERTa | 67.81% | **68.10%** | 67.58% | 71.2 / 58.9 / 74.2 | 68.35% | 0.035 |

**Answer to the research question:** under the matched protocol, domain-adapted pretraining (E4)
beats generic multilingual pretraining (E3) by **+2.98 pp Macro-F1** (95% paired-bootstrap CI
+0.89 to +5.24 pp; McNemar exact p = 0.005). The gain is real but much smaller than the midterm's
+11 pp, which came from a leaky split. E4 also reached its best validation score after 1 epoch,
whereas E3 was still improving at the 5-epoch cap.

Serving (warm, median of 20 requests; `artifacts/metrics/serving_latency.json`): E3 18 ms, E4 20 ms
(RTX 3050 Laptop GPU); E1 436 ms, E2 598 ms via the persistent Spark service (local[2]).
<!-- RESULTS:END -->

**Important:** the midterm figures (E3 74.10%, E4 85.16% Macro-F1) are not reproduced here and must
not be compared with the table above. The audit showed the E3 figure was a *validation* score on a
supplied split where 1,446 of 3,000 validation rows were copies of training rows, and no E4 midterm
artifact exists. See `docs/INITIAL_AUDIT.md`.

## Architecture

```
Browser ──► FastAPI (backend/app) ── serves frontend/ (supplied HTML/CSS/JS) + /api/v1
              │  TextPipeline (shared normalization + dataset-style transform)
              │  ModelRegistry (from manifests) · Auto policy · SQLite/PostgreSQL (Alembic)
              ├─► E3 / E4  fine-tuned checkpoints, PyTorch, loaded once, temperature-calibrated
              └─► Spark inference service (Scala, one JVM, local[2]) ─► E1 / E2 PipelineModels
Offline: SentiMix TSV → prepare_data (dedup + frozen split) → Spark E1/E2 · HF E3/E4
         → evaluate_locked (once) → report (comparison, significance, error analysis)
```
Details: `docs/ARCHITECTURE.md`.

## Quick start (Windows/Git Bash or Linux, from this folder)

Requirements: Python 3.12 venv at `.venv` (existing one is reused), JDK 17 (`JAVA_HOME`, or the
portable JDK in `../../tools/`), sbt 1.9, the data under `data/raw/sentimix_supplied/`, and trained
artifacts under `artifacts/models/` (or train them, below). ≥ 5 GB free on the system drive.

```bash
bash scripts/preflight.sh                 # what is installed / present / missing
bash scripts/run_spark.sh compile         # once: compile Scala + export classpath
bash scripts/start_local.sh               # Spark service :8090 + API/UI on http://127.0.0.1:8000
```
UI: <http://127.0.0.1:8000> · API docs: <http://127.0.0.1:8000/docs>

## Train, evaluate, report

```bash
PYTHONPATH=python/src .venv/Scripts/python -m hinglish.prepare_data   # frozen split (verifies if it exists)
bash scripts/train_all.sh                 # E1, E2 (+ calibration), E3, E4
bash scripts/evaluate_all_locked.sh       # locked test once + comparison + significance + error analysis
PYTHONPATH=python/src .venv/Scripts/python -m hinglish.model_cards    # regenerate docs/MODEL_CARD_E*.md
bash scripts/run_scalability_benchmark.sh # BDA 1x/5x/10x (replicated workload, local mode)
```
(On Linux use `.venv/bin/python`.) Guides: `docs/TRAINING_GUIDE.md`, `docs/EVALUATION_PROTOCOL.md`.

**Train E3/E4 on Kaggle (v3: fp32 AdamW, up to 30 epochs, LR search, 3 seeds):** follow `kaggle/README.md`, then
`python -m hinglish.import_kaggle_v3 <zip>` and `bash scripts/evaluate_v3_locked.sh`.

## Tests

```bash
bash scripts/test_all.sh                       # structure, Python, API, Scala (sbt test)
RUN_UI=1 RUN_LIVE=1 bash scripts/test_all.sh   # + Playwright browser tests and real-model tests
```
See `docs/TESTING.md`; latest executed results in `docs/IMPLEMENTATION_PROGRESS.md`.

## Docker

```bash
cp .env.example .env && docker compose up --build    # http://localhost:8000
```
Prepared but **not executed on the build machine (Docker not installed)** — see `docs/DEPLOYMENT.md`.

## Data

SentiMix as supplied (`data/raw/sentimix_supplied/`), already lowercased/stop-word-filtered/stemmed
upstream. Pooled, de-duplicated, conflicting labels removed, stratified 80/10/10 (seed 42):
train 11,654 · validation 1,457 · test 1,457. Not redistributed. `docs/DATASET.md`,
`docs/DATASET_CARD.md`, `docs/PREPROCESSING.md`.

## Documentation

`docs/`: INITIAL_AUDIT · ARCHITECTURE · DATASET · DATASET_CARD · PREPROCESSING · EXPERIMENTS ·
TRAINING_GUIDE · EVALUATION_PROTOCOL · MODEL_CARD_E1..E4 · API_REFERENCE · FRONTEND_GUIDE · DATABASE ·
DEPLOYMENT · OPERATIONS_RUNBOOK · TESTING · ERROR_ANALYSIS · SCALABILITY_REPORT · LIMITATIONS ·
REFERENCES · IMPLEMENTATION_PROGRESS.

## Known limitations (short)

English negators/contrast words were removed from the data upstream; one split and one seed;
Spark in local mode; Docker/cloud not executed; E1 probabilities uncalibrated. Full list:
`docs/LIMITATIONS.md`.
