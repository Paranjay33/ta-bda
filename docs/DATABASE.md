# Database

SQLAlchemy 2.x models in `backend/app/db.py`; schema owned by Alembic
(`backend/migrations/versions/0001_initial_schema.py`), applied automatically when the API starts
(`Database.migrate()`), or manually:

```bash
cd backend && DATABASE_URL=sqlite:///../data/hinglishsense.db ../.venv/Scripts/alembic upgrade head
```

- Default: SQLite at `data/hinglishsense.db` (WAL mode, foreign keys on) — single-user/local.
- Hosted / several replicas: PostgreSQL via
  `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/hinglishsense` (driver pinned in
  `backend/requirements.txt`). Background batch jobs run in-process threads, so with several
  API replicas use sticky sessions or move jobs to a queue (see `docs/DEPLOYMENT.md`).

## Tables

| Table | Purpose | Notable columns / indexes |
|---|---|---|
| `model_registry` | registered model versions (synced from manifests at start-up) | `experiment_id`, `version` (unique together), `artifact_ref` (path reference, never binaries), `status` trained/evaluated, `validation_macro_f1`, `locked_test_macro_f1` |
| `experiments` | experiment ↔ split/preprocessing identity | `split_manifest_sha256`, `preprocessing_sha256`, `training_status`, `test_status`, `metrics_path` |
| `analysis_history` | one row per prediction (single, compare, batch) | `request_id`, `source`, `batch_job_id` (FK), `model_requested`, `model_used` (idx), `model_version`, `prediction` (idx), `scores` JSON, `score_type`, `inference_ms`, `input_chars`, `normalized_text` (**NULL unless `ALLOW_HISTORY_STORAGE=true`**), `created_at` (idx) |
| `batch_jobs` | batch status/progress | `status` (idx), counts, sanitized `input_filename`, `output_path`, `summary` JSON, `error` |

A `feedback` table was deliberately not added: there is no consented feedback flow, and
feedback must never silently become training labels.

## Privacy and retention

- Raw text is never stored. With `ALLOW_HISTORY_STORAGE=true` only the normalized text is kept
  and the UI says so in the History section.
- Clearing: `DELETE /api/v1/analyses` (UI "Clear history"). For a time-based retention policy run
  e.g. `DELETE FROM analysis_history WHERE created_at < now() - interval '30 days'` from a cron job.
- Batch outputs (CSV containing the uploaded text) are written to `BATCH_OUTPUT_DIR`
  (`data/batch_outputs/` locally, a Docker volume in Compose); delete them on the same schedule.

## Failure behaviour

If the database cannot be initialised, the API still serves predictions (history writes are
skipped and logged), `/health/ready` reports `database: false`, and history/analytics/batch
routes return 503 with an explanatory message.
