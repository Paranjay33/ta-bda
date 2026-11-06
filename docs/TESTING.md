# Testing

```bash
bash scripts/test_all.sh                 # structure + Python + API + Scala
RUN_UI=1 RUN_LIVE=1 bash scripts/test_all.sh   # + browser and real-model tests (server must be running)
```

| Suite | Location | What it covers | Data |
|---|---|---|---|
| Repo structure | `tests/test_repo_structure.py` | required dirs/files, contiguous label ids | — |
| Research package | `python/tests/` | config + label aliases, CoNLL/CSV/headerless-TSV schema detection, validation, split building (determinism, dedup, conflict removal, no overlap), frozen-split hashes, **Python↔Scala normalization contract on the real split**, dataset-style transform, calibration, metrics, normalization rules | synthetic fixtures + the frozen split (integration tests skip if absent) |
| API integration | `backend/tests/test_api.py` | health/ready, models, predict contract & canonical score order, score semantics, 400/413/422/429/401/503, Auto fallback, compare (full + partial), batch CSV (order, skipped rows, download, limits), history privacy, analytics, research endpoints, production-auth guard, same-origin frontend | **test-double adapters** defined only in the test file |
| Scala / Spark | `spark/src/test/scala/` (`sbt test`) | CoNLL parsing, label filtering, genuine char n-grams, canonical labels, processed-contract reader, label/label_id guard, E1+E2 fit → save → reload → single-text scoring via the serving code, E2 stage wiring, gold-label support regression | tiny synthetic fixtures |
| Browser UI | `backend/tests/test_ui.py` (Playwright + Edge) | design preserved, no console errors, counter/clear/chips, validation, model status from API, listbox keyboard, live prediction per available model (score label, bars sum to 100%), compare, unavailable → 503, batch upload + download, mobile overflow & menu | running stack |
| Live models | `backend/tests/test_live_models.py` | real predictions match the registered manifest version, score order/sum, bars agree with class, model input transform, DB row matches response, compare consistency, Auto | running stack with real artifacts |

Test doubles never ship: production adapters are `TransformerAdapter` and `SparkAdapter` only.

## Results

The latest executed counts are recorded in `docs/IMPLEMENTATION_PROGRESS.md` (with the exact
commands); they are not repeated here so that this file cannot go stale.
