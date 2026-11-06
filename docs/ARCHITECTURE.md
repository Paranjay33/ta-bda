# Architecture

```
                         Browser
                            │  same origin (http://host:8000)
                            ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ FastAPI app  (backend/app)                                     │
   │  • serves frontend/ (HTML/CSS/vanilla JS) at /                 │
   │  • /api/v1/*  validation · request IDs · JSON logs · limits    │
   │  • TextPipeline = shared normalization + dataset-style         │
   │  • ModelRegistry (from manifests) + Auto policy                │
   │  • InferenceService (single / compare / batch jobs)            │
   │  • SQLAlchemy + Alembic → SQLite (local) / PostgreSQL          │
   └────────┬──────────────────────────────┬───────────────────────┘
            │ in-process (PyTorch)         │ HTTP, localhost / internal network
            ▼                              ▼
   E3 XLM-R, E4 HingRoBERTa        Spark inference service (Scala, one JVM,
   fine-tuned checkpoints,          one SparkSession local[2])
   loaded once, kept resident,      E1 / E2 PipelineModels loaded once
   temperature calibration          (+ E2 temperature on the API side)
            ▲                              ▲
            │ artifacts/models/<exp>/v2-clean-split/ (mounted, read-only)
            │
   ── offline research pipelines (never triggered by web requests) ──
   SentiMix TSV → prepare_data (dedup, frozen split, manifest)
        ├── Scala/Spark: E1, E2 (fit on train, save PipelineModel, val metrics)
        └── Python/HF:  E3, E4 (shared train.py, val-Macro-F1 selection, calibration)
   → evaluate_locked (test, once) → report (comparison, significance, error analysis)
```

## Components

| Component | Code | Responsibility |
|---|---|---|
| Research package | `python/src/hinglish` | config + labels, schema detection, split building, normalization, dataset-style transform, training, calibration, locked evaluation, reports |
| Spark branch | `spark/src/main/scala/hinglish` | ingestion of the processed contract, E1/E2 pipelines, metrics, persistence, scalability benchmark, **persistent inference server** |
| API | `backend/app` | `settings`, `textpipe`, `registry` (adapters), `service`, `db`, `main` (routes) |
| Frontend | `frontend/` | supplied HinglishSense UI, wired to the API (`js/api.js`, `js/app.js`, `js/dashboard.js`) |
| Migrations | `backend/migrations` | Alembic; applied automatically at API start-up |
| Deployment | `compose.yaml`, `deploy/` | API image, Spark image, optional nginx proxy |

## Key decisions

- **One preprocessing implementation.** Spark consumes the processed TSV produced by Python, and
  at serving time receives already-transformed text from the API, so the branches cannot drift.
- **Persistent Spark serving.** `hinglish.serving.SparkInferenceServer` (JDK `HttpServer` +
  Jackson from the Spark classpath; no new dependencies) loads both `PipelineModel`s at start-up,
  warms them, and exposes `/health` and `/predict`. No JVM is started per request. Cross-language
  loading in PySpark was not used, so there is no unverified serialization compatibility.
- **Registry from manifests.** A model is offered only if its manifest says training completed,
  its files exist, its runtime is reachable, and (Spark) the service reports the same version.
  Test metrics are shown only if they were produced by the same model version.
- **Auto is configuration, not test-score-driven.** `PREFERRED_MODEL=e4`; explicit fallback chain
  `e3,e2,e1`; every response carries `model_requested`, `model_used` and a fallback warning.
- **Score semantics are explicit.** `score_type` is `calibrated_probability` (E2/E3/E4) or
  `model_probability` (E1); `confidence` is null unless calibrated.
- **Memory policy.** E3 and E4 (≈1.1 GB fp32 each) are loaded once and stay resident; if the GPU
  cannot hold a model it is placed on CPU and the device is reported in `/api/v1/models`.
- **Windows without winutils.** `hinglish.util.WindowsLocalFileSystem` (installed only on Windows)
  makes Hadoop permission calls no-ops and lists/stats files with plain Java I/O, so Spark can save
  and load models without downloading a third-party native binary. Linux/Docker uses Hadoop's
  default filesystem.
- **Local Spark is local.** `local[*]`/`local[2]` on one machine; `SPARK_MASTER` can point at a real
  cluster, but no cluster run was performed or is claimed.
