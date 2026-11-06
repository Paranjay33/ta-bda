# Deployment

**Status:** the local, non-Docker deployment (`scripts/start_local.sh`) was run and verified on the
project laptop. Docker is **not installed** on that machine, so the Dockerfiles and
`compose.yaml` below were written but **have not been built or run**. No cloud deployment was
performed (no account or credentials were provided); nothing here claims a public URL.

## Option A — local, no Docker (verified)

```bash
cp .env.example .env              # optional
bash scripts/start_local.sh       # Spark service on 127.0.0.1:8090, then API+UI on :8000
```

Open `http://127.0.0.1:8000` (UI) and `http://127.0.0.1:8000/docs` (OpenAPI).

## Option B — Docker Compose (prepared, not executed here)

```bash
cp .env.example .env
docker compose up --build         # http://localhost:8000
docker compose --profile proxy up --build   # + nginx on :80
```

Services: `spark` (Temurin 17 JRE, resolves Spark 3.5.1 deps at build time, runs
`SparkInferenceServer`, internal port 8090 only), `api` (python:3.12 slim, CPU torch by default,
non-root user, health check on `/health/live`), optional `proxy` (nginx). Both read
`./artifacts` and `./data/processed` **read-only** from the host; weights are never baked into
images. API state (SQLite DB, batch outputs) lives in the `hinglishsense-state` volume.

GPU: build with `TORCH_INDEX=https://download.pytorch.org/whl/cu121` and uncomment the
`deploy.resources` block (requires the NVIDIA container toolkit).

## Required artifact layout (mounted)

```
artifacts/models/e1/v2-clean-split/{manifest.json, pipeline_model/}
artifacts/models/e2/v2-clean-split/{manifest.json, calibration.json, pipeline_model/}
artifacts/models/e3/v2-clean-split/{manifest.json, best/{config.json, pytorch_model.bin, tokenizer*, calibration.json}}
artifacts/models/e4/v2-clean-split/{manifest.json, best/…}
artifacts/metrics/*.json            (optional: shown in the research dashboard)
data/processed/split_manifest.json  (split identity for the dashboard)
```

Ship them with `rsync`/object storage (e.g. an S3/GCS bucket synced to the host or an init
container). Do not publish SentiMix data or the weights publicly unless their licences allow it.

## Cloud path (generic VM, recommended for this workload)

| Need | Recommendation |
|---|---|
| Compute | CPU-only: 4 vCPU / 16 GB RAM (E3+E4 fp32 ≈ 2.2 GB RAM, Spark JVM 2 GB); CPU transformer latency was not measured in this project (GPU warm latency was ~20 ms). GPU: any NVIDIA GPU with ≥ 4 GB. |
| Storage | ~5 GB for artifacts + DB volume |
| Database | SQLite for one replica; managed PostgreSQL for more |
| TLS / domain | terminate TLS at a load balancer or in `deploy/nginx.conf` (add a 443 server) |
| Security | `APP_ENV=production` (the API refuses to start without `AUTH_REQUIRED=true` + `API_KEY`), `CORS_ORIGINS` set to your domain, keep port 8090 private |
| Warm-up | `LOAD_MODELS_ON_STARTUP=true`; route traffic only when `/health/ready` is 200 |
| Logs | JSON lines on stdout (no request text) → ship to your log platform |
| Rollback | train/evaluate a new `run_tag`, point config at it, `POST /api/v1/models/reload`; keep the previous run directory to roll back |

Steps: provision VM → install Docker + Compose → copy repo (without `data/raw`) and artifacts →
create `.env` with production values (no secrets in git) → `docker compose --profile proxy up -d
--build` → check `/health/ready` → point DNS at the VM.

## Model modes

- CPU-only: works for all four models (set `TRANSFORMER_DEVICE=cpu`).
- GPU: E3/E4 placed on CUDA; if a model does not fit it falls back to CPU and the device is shown
  in `/api/v1/models`.
- Memory-limited: set `LOAD_MODELS_ON_STARTUP=false` to load E3/E4 lazily on first use, or run
  with `PREFERRED_MODEL=e2` and no transformer artifacts mounted — unavailable models are
  reported as such, never faked.
