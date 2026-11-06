"""HinglishSense API (FastAPI).

    uvicorn app.main:create_app --factory --app-dir backend --port 8000

Serves /api/v1/* and (same-origin) the supplied HTML/CSS/JS frontend at /.
OpenAPI docs: /docs.
"""
from __future__ import annotations

import collections
import json
import logging
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import Database
from .registry import MODEL_IDS, ModelRegistry, ModelUnavailable
from .service import InferenceService, InputError
from .settings import PROJECT_ROOT, Settings

ModelChoice = Literal["auto", "e1", "e2", "e3", "e4"]


# ----------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "path", "method", "status", "duration_ms", "model"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


def _setup_logging(level: str) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our request log (no query/text logging)


log = logging.getLogger("hinglishsense.api")


# ----------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    text: str = Field(..., description="Hinglish text (Roman script)")
    model: ModelChoice = "auto"
    session_id: str | None = Field(None, max_length=64)


class CompareRequest(BaseModel):
    text: str
    models: list[Literal["e1", "e2", "e3", "e4"]] = ["e3", "e4"]
    session_id: str | None = Field(None, max_length=64)


class BatchJsonRequest(BaseModel):
    texts: list[str]
    model: ModelChoice = "auto"


# ----------------------------------------------------------------------------- app factory
class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def check(self, key: str, cost: int = 1) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) + cost > self.per_minute:
                return False
            q.extend([now] * cost)
            return True


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    _setup_logging(settings.log_level)
    state: dict = {"ready": False, "warmup_done": False, "db_ok": False, "started": time.time()}

    db = Database(settings.database_url)
    registry = registry or ModelRegistry(settings)
    service = InferenceService(settings, registry, db)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            db.migrate()
            state["db_ok"] = True
            service.sync_registry_tables()
        except Exception:
            log.exception("database initialisation failed; history/analytics/batch are disabled")
        if settings.load_models_on_startup:
            def _warm():
                registry.warm_up()
                state["warmup_done"] = True
            threading.Thread(target=_warm, daemon=True).start()
        else:
            state["warmup_done"] = True
        log.info("HinglishSense API started", extra={"path": str(PROJECT_ROOT)})
        yield
        log.info("HinglishSense API stopping")

    app = FastAPI(title="HinglishSense API", version="1.0.0", lifespan=lifespan,
                  description="Hinglish (Hindi-English code-mixed) sentiment analysis backed by the "
                              "E1-E4 TA/BDA experiments. Research prototype.")
    app.state.settings, app.state.registry, app.state.service, app.state.db = settings, registry, service, db

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"], expose_headers=["X-Request-ID"])

    max_body = settings.max_upload_mb * 1024 * 1024

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        if not rid.isalnum() or len(rid) > 64:
            rid = uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > max_body:
            resp = _error(413, "payload_too_large", f"Request body exceeds {settings.max_upload_mb} MB.", rid)
        else:
            try:
                resp = await call_next(request)
            except Exception:
                log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
                resp = _error(500, "internal_error", "An unexpected error occurred.", rid)
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith(("/api/", "/health")):
            log.info("request", extra={"request_id": rid, "path": request.url.path, "method": request.method,
                                       "status": resp.status_code,
                                       "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        msgs = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg')}" for e in exc.errors())
        return _error(422, "validation_error", msgs or "Invalid request.", request.state.request_id)

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException):
        code = {400: "bad_request", 401: "unauthorized", 404: "not_found", 413: "payload_too_large",
                429: "rate_limited", 503: "unavailable"}.get(exc.status_code, "error")
        return _error(exc.status_code, code, str(exc.detail), getattr(request.state, "request_id", None))

    # ------------------------------------------------------------------ dependencies
    def require_key(request: Request):
        if settings.auth_required and request.headers.get("X-API-Key") != settings.api_key:
            raise HTTPException(401, "Missing or invalid API key.")

    def rate(cost: int = 1):
        def dep(request: Request):
            ip = request.client.host if request.client else "unknown"
            if not limiter.check(ip, cost):
                raise HTTPException(429, "Rate limit exceeded; try again in a minute.")
        return dep

    def db_required():
        if not state["db_ok"]:
            raise HTTPException(503, "Database unavailable: history, analytics and batch jobs are disabled.")

    # ------------------------------------------------------------------ health
    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive", "uptime_seconds": round(time.time() - state["started"], 1)}

    @app.get("/health/ready", tags=["health"])
    def ready():
        models = {m: dict(zip(("available", "reason"), registry.availability(m))) for m in MODEL_IDS}
        any_ok = any(v["available"] for v in models.values())
        ok = state["db_ok"] and state["warmup_done"] and any_ok
        body = {"status": "ready" if ok else "not_ready", "database": state["db_ok"],
                "warmup_done": state["warmup_done"], "models": models,
                "unavailable_models": [m for m, v in models.items() if not v["available"]]}
        return JSONResponse(body, status_code=200 if ok else 503)

    # ------------------------------------------------------------------ models / status
    @app.get("/api/v1/status", tags=["models"])
    def status():
        try:
            auto_model, auto_warn = registry.resolve_auto()
        except ModelUnavailable as e:
            auto_model, auto_warn = None, [e.reason]
        return {"service": "hinglishsense", "version": app.version, "environment": settings.app_env,
                "preprocessing_version": service.pipe.version, "auto": {"preferred": settings.preferred_model,
                                                                        "resolves_to": auto_model, "warnings": auto_warn},
                "limits": {"max_text_chars": settings.max_text_chars, "max_batch_rows": settings.max_batch_rows,
                           "max_upload_mb": settings.max_upload_mb},
                "history_text_storage": settings.allow_history_storage, "auth_required": settings.auth_required,
                "database": state["db_ok"]}

    @app.get("/api/v1/models", tags=["models"])
    def models():
        try:
            auto_model, auto_warn = registry.resolve_auto()
        except ModelUnavailable as e:
            auto_model, auto_warn = None, [e.reason]
        return {"models": [registry.describe(m) for m in MODEL_IDS],
                "auto": {"preferred": settings.preferred_model, "resolves_to": auto_model, "warnings": auto_warn,
                         "fallback_chain": settings.auto_fallback,
                         "policy": "Preferred model is used only when its artifact is loaded and its validation-based "
                                   "selection is recorded; fallbacks are explicit and reported in every response."}}

    @app.post("/api/v1/models/reload", tags=["models"], dependencies=[Depends(require_key)])
    def reload_models(request: Request):
        if not settings.auth_required and settings.app_env != "development":
            raise HTTPException(401, "Reload requires authentication outside development.")
        registry.refresh()
        if state["db_ok"]:
            service.sync_registry_tables()
        return {"reloaded": True, "models": [registry.describe(m) for m in MODEL_IDS]}

    # ------------------------------------------------------------------ inference
    @app.post("/api/v1/predict", tags=["inference"], dependencies=[Depends(require_key), Depends(rate(1))])
    def predict(req: PredictRequest, request: Request):
        try:
            return service.predict(req.text, req.model, request.state.request_id, session_id=req.session_id).to_dict()
        except InputError as e:
            raise HTTPException(400, str(e))
        except ModelUnavailable as e:
            raise HTTPException(503, f"Model '{e.model_id}' is unavailable ({e.reason}).")

    @app.post("/api/v1/predict/compare", tags=["inference"], dependencies=[Depends(require_key), Depends(rate(2))])
    def compare(req: CompareRequest, request: Request):
        try:
            return service.compare(req.text, list(dict.fromkeys(req.models)), request.state.request_id, req.session_id)
        except InputError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/v1/predict/batch", tags=["inference"],
              dependencies=[Depends(require_key), Depends(rate(5)), Depends(db_required)])
    async def batch_upload(request: Request, file: UploadFile = File(...), model: ModelChoice = Form("auto"),
                           text_column: str | None = Form(None)):
        name = (file.filename or "upload.csv").replace("\\", "/").split("/")[-1][:200]
        if not name.lower().endswith((".csv", ".txt")):
            raise HTTPException(400, "Upload a .csv file.")
        raw = await file.read(max_body + 1)
        if len(raw) > max_body:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
        try:
            df, col = service.parse_csv(raw, text_column or None)
            return service.create_batch(df[col].tolist(), model, name, col, request.state.request_id)
        except InputError as e:
            raise HTTPException(400, str(e))
        except ModelUnavailable as e:
            raise HTTPException(503, f"Model '{e.model_id}' is unavailable ({e.reason}).")

    @app.post("/api/v1/predict/batch/json", tags=["inference"],
              dependencies=[Depends(require_key), Depends(rate(5)), Depends(db_required)])
    def batch_json(req: BatchJsonRequest, request: Request):
        if not req.texts:
            raise HTTPException(400, "texts is empty.")
        if len(req.texts) > settings.max_batch_rows:
            raise HTTPException(400, f"At most {settings.max_batch_rows} texts per batch.")
        try:
            return service.create_batch(req.texts, req.model, None, None, request.state.request_id)
        except InputError as e:
            raise HTTPException(400, str(e))
        except ModelUnavailable as e:
            raise HTTPException(503, f"Model '{e.model_id}' is unavailable ({e.reason}).")

    @app.get("/api/v1/batch/jobs/{job_id}", tags=["inference"], dependencies=[Depends(require_key), Depends(db_required)])
    def batch_job(job_id: str):
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Batch job not found.")
        return job

    @app.get("/api/v1/batch/jobs/{job_id}/download", tags=["inference"],
             dependencies=[Depends(require_key), Depends(db_required)])
    def batch_download(job_id: str):
        path = service.job_output_path(job_id)
        if not path:
            raise HTTPException(404, "No completed output for this job.")
        return FileResponse(path, media_type="text/csv", filename=f"hinglishsense_batch_{job_id[:8]}.csv")

    # ------------------------------------------------------------------ history / analytics
    @app.get("/api/v1/analyses", tags=["history"], dependencies=[Depends(require_key), Depends(db_required)])
    def analyses(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                 model: Literal["e1", "e2", "e3", "e4"] | None = None,
                 prediction: Literal["negative", "neutral", "positive"] | None = None,
                 source: Literal["single", "compare", "batch"] | None = None):
        return service.list_analyses(page, page_size, model, prediction, source)

    @app.get("/api/v1/analyses/{analysis_id}", tags=["history"], dependencies=[Depends(require_key), Depends(db_required)])
    def analysis(analysis_id: str):
        r = service.get_analysis(analysis_id)
        if r is None:
            raise HTTPException(404, "Analysis not found.")
        return r

    @app.delete("/api/v1/analyses", tags=["history"], dependencies=[Depends(require_key), Depends(db_required)])
    def clear_history():
        return {"deleted": service.delete_history()}

    @app.get("/api/v1/analytics/summary", tags=["history"], dependencies=[Depends(require_key), Depends(db_required)])
    def analytics(days: int = Query(14, ge=1, le=365)):
        return service.analytics_summary(days)

    # ------------------------------------------------------------------ research artifacts
    from hinglish.report import collect_rows  # noqa: E402  (shared with the offline report)

    def _read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except Exception:
            return None

    @app.get("/api/v1/experiments", tags=["research"])
    def experiments():
        out = []
        for m in MODEL_IDS:
            e = registry.entries[m]
            mf = e.manifest or {}
            out.append({"experiment_id": m.upper(), "model_id": m, "name": e.info["name"], "area": e.info["area"],
                        "method": e.info["method"], "version": e.version,
                        "training_status": (mf.get("status") or {}).get("training", "not run"),
                        "test_status": "completed" if e.test_metrics else "not run"})
        return {"experiments": out}

    @app.get("/api/v1/experiments/{experiment_id}", tags=["research"])
    def experiment(experiment_id: str):
        m = experiment_id.lower()
        if m not in MODEL_IDS:
            raise HTTPException(404, "Unknown experiment.")
        e = registry.entries[m]
        mf = dict(e.manifest or {})
        mf.pop("environment", None) if settings.app_env == "production" else None
        for k in ("artifact_dir",):
            mf.pop(k, None)  # no filesystem paths to the browser
        return {"experiment_id": m.upper(), "describe": registry.describe(m), "manifest": mf,
                "validation_metrics": (e.validation_metrics or {}).get("metrics"),
                "test_metrics": (e.test_metrics or {}).get("metrics"),
                "test_probabilistic": (e.test_metrics or {}).get("probabilistic")}

    @app.get("/api/v1/metrics/comparison", tags=["research"])
    def comparison():
        cfg = settings.cfg
        mdir = cfg.path("paths", "metrics_dir")
        rows = collect_rows(cfg)
        for r in rows:
            r.pop("test_sample_ids_sha256", None)
        split = _read(cfg.root / cfg.get("dataset", "split_manifest_file")) or {}
        return {
            "rows": rows,
            "e3_e4_significance": _read(mdir / "e3_e4_significance.json"),
            "error_analysis": _read(cfg.path("paths", "error_analysis_dir") / "error_summary.json"),
            "confusion_matrices": {m.upper(): ((registry.entries[m].test_metrics or {}).get("metrics") or {})
                                   .get("confusion_matrix") for m in MODEL_IDS},
            "labels_order": cfg.labels.ordered_names,
            "split": {"policy": split.get("split_policy"), "counts": split.get("counts"),
                      "rows": {k: v.get("rows") for k, v in (split.get("splits") or {}).items()}},
            "historical_midterm": {
                "status": "reported midterm results - NOT reproducible and NOT locked-test results",
                "values": {"E3": {"accuracy": 0.7402, "macro_f1": 0.7410, "eval_loss": 0.6438},
                           "E4": {"accuracy": 0.8499, "macro_f1": 0.8516, "weighted_f1": 0.8490, "eval_loss": 0.8811}},
                "audit_finding": "The E3 figures are VALIDATION metrics (trainer_state.json), and 1,446 of the 3,000 "
                                 "supplied validation rows are exact copies of training rows, so they were measured "
                                 "on a leaky split. No E4 midterm checkpoint exists to verify the E4 figures.",
            },
        }

    # ------------------------------------------------------------------ frontend (same origin)
    if settings.frontend_dir is not None:
        app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")

    return app


def _error(status: int, code: str, message: str, request_id: str | None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id}}, status_code=status)


# Run with the factory so importing this module never loads models:
#   uvicorn app.main:create_app --factory --app-dir backend --host 0.0.0.0 --port 8000
