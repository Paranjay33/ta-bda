# Training guide

All commands run from the project root (`hinglish-sentiment/hinglish-sentiment/`) in Git Bash
or any POSIX shell. Scripts use the existing `.venv` and never recreate it.

## 0. Preflight

```bash
bash scripts/preflight.sh        # toolchain, CUDA, data, cached checkpoints, artifacts, disk
```

Requirements: Python 3.12 venv with `python/requirements.txt`; JDK 17 (the scripts use
`JAVA_HOME`, else the portable `../../tools/jdk-17*`); sbt 1.9.x; an NVIDIA GPU for E3/E4
(4 GB works with the 8-bit optimizer, slowly if another program uses the GPU);
**at least 5 GB free on the system drive** — Windows needs page-file room for pinned GPU
memory, and a full C: drive caused native crashes during this project (temp files are already
redirected to `./.tmp`).

## 1. Data split (once)

```bash
PYTHONPATH=python/src .venv/Scripts/python -m hinglish.prepare_data      # creates / verifies
```

## 2. Spark experiments (E1, E2)

```bash
bash scripts/run_spark.sh compile        # sbt compile + export runtime classpath
bash scripts/run_spark.sh e1             # train on train, report validation, save PipelineModel
bash scripts/run_spark.sh e2
PYTHONPATH=python/src .venv/Scripts/python -m hinglish.calibrate_classical   # E2 temperature, E1 record
```

Smoke runs: append `--smoke-test` (64 rows, writes to `*-smoke` run dirs).

## 3. Transformer experiments (E3, E4)

```bash
bash scripts/train_transformers.sh E3 E4     # sequential; logs in artifacts/logs/train_e?.log
```

Equivalent single run: `bash scripts/run_e4.sh` (add `--smoke-test` for a 1-epoch sanity
pass, `--resume` to continue from `checkpoints/`). Base checkpoints are read from the local
Hugging Face cache (`HF_HUB_OFFLINE=1`); on a fresh machine unset it once to download.

Measured on the project laptop (RTX 3050 Laptop 4 GB, shared with another GPU-heavy program
during the run): see `training_runtime_seconds` in each manifest.

## 4. Everything in one go

```bash
bash scripts/train_all.sh       # split → compile → E1 → E2 → calibration → E3 → E4
```

The orchestrator logs `START/END exit=` per step to `artifacts/logs/train_status.log`,
continues past a failed step and exits non-zero if any step failed.

## 5. Locked test evaluation + reports (after everything is frozen)

```bash
bash scripts/evaluate_all_locked.sh   # Spark --final-test-eval, evaluate_locked, report
```

## 6. Scalability benchmark (BDA)

```bash
bash scripts/run_scalability_benchmark.sh
```

## Do not

- re-run `prepare_data --force-new-split` after any test evaluation;
- tune anything after looking at `*_test_metrics.json`;
- train E3/E4 at the same time on a 4 GB GPU.

## 7. Optional: v3 training on Kaggle (bigger GPU, more epochs, LR search)

Step-by-step: `kaggle/README.md`. In short: `python scripts/make_kaggle_bundle.py` → upload
`dist/hinglishsense_kaggle_bundle.zip` as a **private** Kaggle dataset → import
`kaggle/hinglishsense_v3_train.ipynb`, GPU + Internet on → Save & Run All → download
`hinglishsense_v3_models.zip` → `python -m hinglish.import_kaggle_v3 <zip>` →
`bash scripts/evaluate_v3_locked.sh` → restart the app.
