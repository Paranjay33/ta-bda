# API reference (v1)

Base URL: same origin as the UI, e.g. `http://127.0.0.1:8000`. Interactive OpenAPI docs: `/docs`.
All errors use one shape and never include stack traces, paths or environment values:

```json
{"error": {"code": "bad_request", "message": "Text is empty.", "request_id": "…"}}
```

| Status | When |
|---|---|
| 400 | invalid input (empty / over-long text, bad CSV, unsupported column) |
| 401 | `AUTH_REQUIRED=true` and missing/wrong `X-API-Key` |
| 404 | unknown job / analysis / experiment |
| 413 | body larger than `MAX_UPLOAD_MB` |
| 422 | schema validation (e.g. `model` not in `auto,e1,e2,e3,e4`) |
| 429 | rate limit (`RATE_LIMIT_PER_MINUTE` per client IP; batch costs 5) |
| 503 | requested model unavailable (reason in message) / database down for DB routes |

Every response carries `X-Request-ID` (you may send your own alphanumeric one).

## Health

- `GET /health/live` → `{"status":"alive","uptime_seconds":…}` (process up)
- `GET /health/ready` → 200 when DB is migrated, warm-up finished and ≥1 model is servable;
  503 otherwise. Body lists every model's `available`/`reason` and `unavailable_models`.

## Models and status

- `GET /api/v1/status` — preprocessing version, Auto resolution, limits, privacy/auth flags.
- `GET /api/v1/models` — per model: `model_id`, `name`, `method`, `family`, `area` (TA/BDA),
  `available`, `unavailable_reason` (`not_trained`, `spark_service_unreachable`,
  `spark_model_not_loaded`, `spark_service_version_mismatch`, `calibration_missing`,
  `load_failed:<Type>`), `version`, `source_checkpoint(_revision)`, `status.runtime`
  (device / Spark master / cold-start seconds), `score_type`, `score_semantics`,
  `validation_macro_f1`, `test_metrics` (only when produced by this version), sizes, runtime;
  plus `auto` (`preferred`, `resolves_to`, `warnings`, `fallback_chain`, `policy`).
- `POST /api/v1/models/reload` — re-read manifests (development, or with API key).

## Inference

### `POST /api/v1/predict`
Request: `{"text": "movie toh acchi thi but ending bakwas thi", "model": "e4"}`
(`model` ∈ `auto|e1|e2|e3|e4`, default `auto`; optional `session_id`).

Response fields: `request_id`, `text`, `normalized_text` (shared normalization), `model_input`
(what every model actually read), `model_requested`, `model_used`, `model_name`, `model_version`,
`prediction`, `label_id`, `scores` (keys always `negative, neutral, positive`), `score_type`,
`confidence` (calibrated top probability or `null`), `top_class_score`, `inference_ms` (warm
model forward pass / Spark call, excluding HTTP), `preprocessing_version`, `warnings`.

### `POST /api/v1/predict/compare`
Request: `{"text": "…", "models": ["e3","e4"]}`. Response: `results.e3` / `results.e4` each either
a full prediction with `"status":"ok"` or `{"status":"unavailable","reason":…}` (never substituted),
`agreement` (`true/false`, or `null` if a model could not run), `model_input`, `note`.

### `POST /api/v1/predict/batch` (multipart)
Fields: `file` (`.csv`, ≤ `MAX_UPLOAD_MB`), `model`, optional `text_column` (auto-detects
`text, sentence, tweet, content, review, comment, message`). ≤ `MAX_BATCH_ROWS` rows.
≤ `SYNC_BATCH_MAX_ROWS` rows → processed synchronously (status `completed` in the response);
larger → background job (`queued`/`running`), poll the job. Empty or over-long rows are kept in
order and marked `skipped` with a reason.

`POST /api/v1/predict/batch/json` — same with `{"texts": [...], "model": "…"}`.

- `GET /api/v1/batch/jobs/{id}` — status, counts, distribution, first 200 rows, `download_url`.
- `GET /api/v1/batch/jobs/{id}/download` — CSV:
  `row,text,status,error,prediction,label_id,score_negative,score_neutral,score_positive,score_type,model_used,model_version`.

## History and analytics (database-backed)

- `GET /api/v1/analyses?page=&page_size=&model=&prediction=&source=` — paginated history.
  `normalized_text` is `null` unless `ALLOW_HISTORY_STORAGE=true`.
- `GET /api/v1/analyses/{id}` · `DELETE /api/v1/analyses` (clear).
- `GET /api/v1/analytics/summary?days=14` — totals, by prediction/model/source, per-model
  distribution, requests per day, latency (single requests) — derived only from stored rows.

## Research artifacts (read-only)

- `GET /api/v1/experiments` · `GET /api/v1/experiments/{e1..e4}` (manifest without filesystem paths,
  validation/test metrics, probabilistic metrics).
- `GET /api/v1/metrics/comparison` — machine-generated E1–E4 rows (`not run` → `null`), E3/E4
  significance, error-analysis summary, confusion matrices, split counts, and the historical
  midterm values explicitly labelled as unverified/non-reproducible.

## Example

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"movie toh acchi thi but ending bakwas thi","model":"e4"}'
```
