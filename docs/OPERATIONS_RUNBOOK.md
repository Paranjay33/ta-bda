# Operations runbook

## Start / stop

| Action | Command |
|---|---|
| Start locally | `bash scripts/start_local.sh` (Ctrl+C stops API and Spark service) |
| Start Spark service only | `bash scripts/run_spark.sh serve` (log: `artifacts/logs/spark_serving.log`) |
| Start API only | `.venv/Scripts/python -m uvicorn app.main:create_app --factory --app-dir backend --port 8000` |
| Compose | `docker compose up -d --build` / `docker compose down` |

## Health checks

- `GET /health/live` — process up. `GET /health/ready` — DB ok, warm-up finished, ≥1 model servable;
  the body lists unavailable models with reasons.
- Spark service: `curl http://127.0.0.1:8090/health` — per-model `loaded`, version, cold-start time.

## Observability

- API logs: one JSON line per request (`request_id`, path, method, status, `duration_ms`); model
  load times and failures; never the request text.
- `X-Request-ID` on every response; quote it when reporting a problem.
- Inference latency per model: `analysis_history.inference_ms`, summarised in
  `GET /api/v1/analytics/summary` and the History & analytics section.
- Batch: `batch_jobs.status/processed_count/failed_count/error`.
- Drift readiness (not drift detection): the analytics endpoint exposes prediction-class mix,
  model usage, latency and volume over time; input length is stored (`input_chars`). With a few
  hundred records these are descriptive only — no statistical drift claim is made.

## Common problems

| Symptom | Cause / fix |
|---|---|
| `e1/e2 unavailable: spark_service_unreachable` | with `start_local.sh` the Spark service is supervised and restarts itself (~30-40 s, verified by killing the JVM); otherwise start `bash scripts/run_spark.sh serve` and check the log |
| Spark service died after the laptop slept (`Futures timed out after [10000 milliseconds]`, `Cannot find endpoint: CoarseGrainedScheduler`) | observed 2026-11-07. Fixed: serving JVM uses long heartbeat/network timeouts, a watchdog exits if the SparkContext stops, and the supervisor restarts it |
| `spark_service_version_mismatch` | Spark service is serving an older model run; restart it after retraining |
| `e3/e4 unavailable: not_trained` | run `bash scripts/train_transformers.sh` |
| Model shows `device: cpu` on a GPU machine | GPU memory full (another program); free it and restart the API |
| Training or saving crashes with `Windows fatal exception: access violation` | **system drive full** — Windows cannot back pinned/paged memory. Free ≥ 5 GB on C: (this happened during the project: C: fell to 354 MB). Temp files are redirected to `./.tmp` by the scripts |
| Training very slow (> 4 s/step) | another program (e.g. a game) is using the 4 GB GPU and memory spills to system RAM; close it |
| `Did not find winutils.exe` warning | harmless; the project's Windows filesystem shim avoids winutils |
| `prepare_data` refuses to run | the frozen split differs from the rebuilt one — investigate; do not `--force-new-split` after test evaluation |
| 429 responses | raise `RATE_LIMIT_PER_MINUTE` or add per-key limits at the proxy |

## Model updates and rollback

1. Train with a new `run_tag` (config `transformer.run_tag` / `spark.run_tag`); the old run stays.
2. Evaluate (locked test only if the protocol allows a new test evaluation; it is logged in
   `artifacts/metrics/locked_test_log.jsonl`).
3. Restart the Spark service and call `POST /api/v1/models/reload` (or restart the API).
4. Roll back by restoring the previous `run_tag` and repeating step 3.

## Backups

`data/processed/` (frozen split + manifest), `artifacts/models/*/v2-clean-split/`,
`artifacts/metrics/`, and the database file/volume.
