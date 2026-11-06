# Initial workspace audit (Phase 0)

Audit date: 2026-11-06. Machine: Windows 11 laptop (AMD Ryzen 7 7435HS, 16 logical CPUs,
15.8 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM, driver 595.79).
Only the task workspace `D:\SEMESTER V\TA-BDA` was inspected.

A source backup (everything except the virtual environment and the 7.3 GB of model
weights, which were left untouched in place) was written to
`D:\SEMESTER V\TA-BDA\_original_backup_2026-11-06\` before any change. The two original
ZIP archives and the PPTX were not modified.

## 1. Assets located

| Asset | Location | Notes |
|---|---|---|
| Research repo | `hinglish-sentiment/hinglish-sentiment/` | Python (E3/E4) + Scala/Spark (E1/E2), configs, docs, tests |
| Frontend | `hinglishsense.zip` (inside the repo root) | Not extracted before; extracted to `frontend/` (index.html, css/style.css, js/app.js, empty assets/) |
| Midterm deck | `Hinglish_Sentiment_Analysis_Midterm (2).pptx` | Slide 20 = UI concept; slide 19 = E3/E4 results |
| Older copy of the repo | `hinglish_project.zip` | Contains the same code + `data/raw/*.txt`; left untouched |
| Existing venv | `.venv/` (Python 3.12.8) | Re-used; packages only **added** (see §6) |

The spec mentions `references/hinglishsense/`; no such folder existed — the frontend was
only present as `hinglishsense.zip`.

## 2. Data files found

| File | Rows | Schema (verified) | Notes |
|---|---:|---|---|
| `FinalTrainingOnly.tsv` | 14,594 | `uid \t text \t label`, no header, labels 0/1/2 | negative 4,252 · neutral 5,473 · positive 4,869 |
| `ValidationOnly.tsv` | 3,000 | same | negative 890 · neutral 1,128 · positive 982 |
| `FinalTest.tsv` | 3,000 | `uid \t text`, **no labels** | official test text only — cannot be used for evaluation |
| `train.txt`, `validation.txt` | 14,569 / 2,999 | `text \t label` with header | produced by `add_headers.py`; it silently dropped 25 training rows (texts with < 2 tokens) |
| `test.txt` | **2** | `text \t label` | unknown provenance; the old "locked test" was computed on these 2 rows |

Raw-file SHA-256 values are recorded in `data/processed/split_manifest.json`.

### Findings that changed the plan

1. **The text is already preprocessed upstream.** Across ~230k training tokens, none of the
   198 NLTK English stop-words occurs (so `not`, `no`, `but` are absent), 98.6% of token mass
   is a Porter-stemmer fixed point (vs ~42% for ordinary English) and the text is lowercase
   (`movie`→`movi`, `happy`→`happi`). 44% of rows end in a literal ` nan` artefact.
   Consequence: live user text must be mapped into the same space (see `docs/PREPROCESSING.md`).
2. **The supplied train/validation files are not disjoint.** 1,446 of the 3,000 validation
   rows are exact copies (same uid, text and label) of training rows; 1,458 validation rows
   share their text with a training row. uids are not unique across files (2,555 overlap,
   only 1,446 with identical text). 114 texts carry conflicting labels.
3. **Label semantics** were spot-checked by reading random samples: 0 = negative,
   1 = neutral (noisy, as usual for SentiMix), 2 = positive.

## 3. Existing results and checkpoints

| Item | Finding |
|---|---|
| `artifacts/models/e3/` (7.3 GB) | Legacy midterm E3 run: `best/`, `checkpoint-3644/`, `checkpoint-4555/` (safetensors + optimizer states). Left untouched. |
| E3 midterm numbers (74.02% acc / 74.10% Macro-F1 / loss 0.6438) | `checkpoint-4555/trainer_state.json` shows these are **epoch-5 VALIDATION** metrics on `validation.txt`, i.e. on the leaky split above. |
| `artifacts/metrics/e3_locked_test_metrics.json` | Computed on the 2-row `test.txt` (accuracy 0.0) — meaningless. |
| `artifacts/models/e4/` | **Empty** (only an empty `checkpoint-4/` folder). No E4 checkpoint, manifest or log exists, so the midterm E4 numbers (84.99% / 85.16%) cannot be verified. |
| E1/E2 artifacts | None. |
| `configs/label_mapping.json` | Had been edited to `{"0":0,"1":1,"2":2}`, breaking the canonical names the code and tests expect. |
| `configs/hea` | Empty stray file (0 bytes); removed (copy kept in the backup). |

## 4. Code audit

### Python (`python/src/hinglish`)
- Config/label loading, normalization, CoNLL/CSV schema detection, metrics, manifest: sound.
- `train.py`: `warmup_ratio` from config was **commented out**; output dir `artifacts/models/e3`
  with `save_total_limit=2` would have rotated away the legacy checkpoints on a re-run;
  locked-test step re-used the same script (fine) but the test file was the 2-row file.
- `scalability_compare.py`: scored accuracy on a train/test split of *replicated* rows (copies
  of training rows in its test set) and used the removed `multi_class=` sklearn argument.

### Scala / Spark (`spark/`)
- Had **never been compiled**. On compilation/execution the following real defects surfaced:
  - `DataIngestion` parsed only CoNLL blocks; the supplied TSV would have produced 0 rows.
  - No normalization in the Spark branch; no validation split; models were never persisted.
  - `EvaluationUtils`: per-class `support` counted *predicted* rows instead of gold rows
    (wrong weighted-F1).
  - `CharacterSplitter` had no `DefaultParamsReadable` companion: a saved E2 `PipelineModel`
    could be written but **never loaded** (found by the new save/reload check).
  - All original ScalaTest specs failed to compile (`import spark.implicits._` on a `var`).

### Frontend (`hinglishsense.zip`)
- Vanilla HTML/CSS/JS, cream/purple design, accessible custom listbox, example chips,
  500-char counter, result panel with probability bars.
- `analyzeSentiment()` was an empty stub; the result panel only showed "Connect the model API".
- Models section and insight hard-coded 74.10% / 85.16% / +11.06 pp.

## 5. Baseline tests before any change

| Suite | Result |
|---|---|
| `python -m pytest python/tests` | **3 failed, 14 passed** (all 3 caused by the edited label map) |
| `tests/test_repo_structure.py` | would fail: `README.md` missing |
| `sbt test` | could not run (no Java on PATH); after setup: compile errors (see §4) |

## 6. Toolchain and network

| Tool | Status |
|---|---|
| Python 3.12.8 venv | torch 2.5.1+cu121, transformers 4.46.3, datasets 5.0.1, accelerate 1.15.0, sklearn 1.9.1, pandas 3.0.5 |
| Added to the venv | nltk, fastapi, uvicorn, sqlalchemy, alembic, python-multipart, bitsandbytes, playwright (nothing removed or downgraded) |
| Java | not on PATH; a JDK 24 existed in `~/.jdks` (unsupported by Spark 3.5.1). Portable Temurin **JDK 17.0.21** placed in `D:\SEMESTER V\TA-BDA\tools\` |
| sbt | installed (1.9.9 launcher); caches redirected into the project (`.cache/`) |
| Spark distribution / winutils | none; not needed (see `docs/ARCHITECTURE.md`, Windows FS shim) |
| Docker | **not installed** — Compose files are written but cannot be executed here |
| Node | 22.20.0 (not needed) |
| Network | huggingface.co, Maven Central, PyPI, Adoptium reachable. Both base checkpoints already in the local HF cache. |
| Disk | D: 44 GB free. C: was 96% full (13 GB) and later dropped to 354 MB, which caused native crashes during training (see `docs/OPERATIONS_RUNBOOK.md`). |
| GPU contention | A game (VALORANT) was running during the session and shared the 4 GB GPU, slowing transformer training several-fold. |

## 7. What could run and what was blocked at audit time

- Runnable: data validation and splitting; E1/E2 (after JDK 17); E3/E4 fine-tuning (GPU present,
  checkpoints cached); API/frontend; Python + Scala tests; browser tests via installed Edge.
- Blocked: Docker build/compose (no Docker); hosted deployment (no credentials); official-test
  evaluation (no gold labels for `FinalTest.tsv`).
