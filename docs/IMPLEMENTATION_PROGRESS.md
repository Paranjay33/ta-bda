# Implementation progress (final state, 2026-11-07)

Legend: **done (tested)** = implemented and verified by an executed test/run; done (untested);
blocked (external).

| Area | Status | Evidence |
|---|---|---|
| Phase 0 audit + source backup | **done** | `docs/INITIAL_AUDIT.md`; `../../_original_backup_2026-11-06/` |
| Label map restored + raw-label aliases | **done (tested)** | `python/tests` |
| Headerless-TSV schema parser | **done (tested)** | `test_data_contract.py` |
| Frozen de-duplicated split + manifest + freeze guard | **done (tested)** | `data/processed/split_manifest.json`; rebuild verifies identical |
| Dataset-style serving transform | **done (tested)** | unit tests + live `model_input` checks |
| Spark build (JDK 17, sbt), Windows FS shim (no winutils) | **done (tested)** | `sbt compile`, `sbt test` 13/13 |
| Spark fixes: processed ingestion, support count, CharacterSplitter reader, test compile errors | **done (tested)** | `ServingAndContractSpec` |
| E1 / E2 trained, persisted, reloaded-and-verified | **done** | val Macro-F1 0.4981 / 0.5412 |
| E2 calibration (temperature on margins, argmax-preserving) | **done** | manifest; Platt rejected (18% argmax change) |
| E3 trained (XLM-R) | **done** | 5 epochs, selected ep 5, val 0.6340, 4,548 s (GPU shared with a game for most of it) |
| E4 trained (HingRoBERTa) | **done** | early-stopped after ep 3, selected ep 1, val 0.6835, 745 s |
| Locked test evaluation (once each) | **done** | `locked_test_log.jsonl` has exactly 4 entries |
| Comparison, E3/E4 significance, error analysis, model cards | **done** | `artifacts/metrics/comparison_report.md`, `docs/MODEL_CARD_E*.md` |
| Spark ↔ Python metric cross-check | **done** | validation and test: max abs diff ≤ 1.2e-16, identical confusion matrices |
| Scalability benchmark (fixed: warm-up, repeats, cache-eviction bug) | **done** | `docs/SCALABILITY_REPORT.md` |
| Persistent Spark inference service | **done (tested)** | live tests; cold start 18 s, warm E1 436 ms / E2 598 ms |
| FastAPI backend (registry, Auto, single/compare/batch, history, analytics, research, security) | **done (tested)** | 22 API tests |
| Database (SQLAlchemy + Alembic, privacy default) | **done (tested)** | API tests + e2e DB check |
| Frontend (supplied UI, wired to real inference) | **done (tested)** | Playwright/Edge 12 passed (+1 skip), screenshots in `artifacts/logs/ui_*.png` |
| End-to-end acceptance (§15.5) incl. restart persistence | **done (tested)** | `scripts/e2e_acceptance.py`: 12/12 before, 4/4 after restart |
| Docker / Compose | done (untested) — **blocked: Docker not installed** | `compose.yaml`, `deploy/` |
| Cloud deployment | **blocked: no account/credentials** (user chose to do this next) | `docs/DEPLOYMENT.md` |

## Test runs executed (final)

`RUN_UI=1 RUN_LIVE=1 bash scripts/test_all.sh` → exit 0 (`artifacts/logs/test_all.log`):

| Suite | Result |
|---|---|
| Repository structure | OK |
| Python research package (`python/tests`) | 39 passed |
| API integration (`backend/tests/test_api.py`) | 22 passed |
| Scala / Spark (`sbt test`) | 13 passed |
| Browser UI (`backend/tests/test_ui.py`, Edge) | 12 passed, 1 skipped (unavailable-model UI state cannot occur while all models are up; it passed in an earlier run when E3/E4 were untrained) |
| Live models (`backend/tests/test_live_models.py`) | 6 passed |

`python scripts/e2e_acceptance.py --phase before` → 12/12; `--phase after-restart` → 4/4.

## Defects found and fixed during the work (beyond the audit list)

- Batch CSV silently dropped blank lines (pandas `skip_blank_lines`); now reported as skipped rows
  (regression test added).
- Python scalability baseline scored "accuracy" on replicated copies of training rows and used a
  removed sklearn argument.
- Spark scalability benchmark evicted its cached base data after the 1× run and had no warm-up.
- Native crashes during training traced to a full system drive (page file); temp files redirected.
