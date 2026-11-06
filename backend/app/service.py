"""Inference orchestration shared by single, compare and batch routes."""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, select

from .db import AnalysisHistory, BatchJob, Database, ExperimentRow, ModelRegistryRow, utcnow
from .registry import STATIC_INFO, ModelRegistry, ModelUnavailable
from .settings import Settings
from .textpipe import TextPipeline

log = logging.getLogger("hinglishsense.service")

SCORE_NOTES = {
    "model_probability": "Model probability (not separately calibrated). Treat as a relative score.",
    "calibrated_probability": "Calibrated probability (calibration fitted on the validation split only).",
}
TEXT_COLUMN_CANDIDATES = ["text", "sentence", "tweet", "content", "review", "comment", "message"]


@dataclass
class Prediction:
    request_id: str
    text: str
    normalized_text: str
    model_input: str
    model_requested: str
    model_used: str
    model_name: str
    model_version: str | None
    prediction: str
    label_id: int
    scores: dict[str, float]
    score_type: str
    confidence: float | None
    top_class_score: float
    inference_ms: float
    preprocessing_version: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class InputError(ValueError):
    pass


class InferenceService:
    def __init__(self, settings: Settings, registry: ModelRegistry, db: Database):
        self.settings = settings
        self.registry = registry
        self.db = db
        self.pipe = TextPipeline(settings.cfg)
        self.labels = settings.cfg.labels.ordered_names  # canonical order: negative, neutral, positive
        self._jobs_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def validate_text(self, text: str) -> str:
        if text is None or not str(text).strip():
            raise InputError("Text is empty.")
        if len(text) > self.settings.max_text_chars:
            raise InputError(f"Text exceeds {self.settings.max_text_chars} characters.")
        return str(text)

    def resolve(self, model: str) -> tuple[str, list[str]]:
        model = (model or "auto").lower()
        if model == "auto":
            return self.registry.resolve_auto()
        if model not in STATIC_INFO:
            raise InputError(f"Unsupported model '{model}'. Use one of: auto, e1, e2, e3, e4.")
        return model, []

    def _build(self, request_id, prepared, requested, used, raw, ms, warnings) -> Prediction:
        entry = self.registry.entries[used]
        score_type = STATIC_INFO[used]["score_type"]
        scores = {name: float(raw.scores[i]) for i, name in enumerate(self.labels)}
        top = max(scores.values())
        warns = list(warnings)
        if used == "e1":
            warns.append("E1 probabilities are not calibrated and are known to be over-confident "
                         "(validation ECE recorded in its manifest); read them as relative scores.")
        if used == "e2":
            warns.append("E2 class is the native One-vs-Rest SVM decision; bars are SVM margins mapped to "
                         "probabilities by temperature scaling fitted on the validation split.")
        if prepared.model_input.strip() == "":
            warns.append("After preprocessing nothing informative remained (only stop-words/punctuation).")
        return Prediction(
            request_id=request_id, text=prepared.raw, normalized_text=prepared.normalized,
            model_input=prepared.model_input, model_requested=requested, model_used=used,
            model_name=STATIC_INFO[used]["name"], model_version=entry.version,
            prediction=self.labels[raw.label_id], label_id=raw.label_id, scores=scores,
            score_type=score_type,
            confidence=top if score_type == "calibrated_probability" else None,
            top_class_score=top, inference_ms=round(ms, 2),
            preprocessing_version=self.pipe.version, warnings=warns,
        )

    # ------------------------------------------------------------------ single / compare
    def predict(self, text: str, model: str, request_id: str | None = None, source: str = "single",
                session_id: str | None = None) -> Prediction:
        request_id = request_id or uuid.uuid4().hex
        text = self.validate_text(text)
        requested = (model or "auto").lower()
        used, warnings = self.resolve(requested)
        adapter = self.registry.get_adapter(used)
        prepared = self.pipe.prepare(text)
        t0 = time.perf_counter()
        raw = adapter.predict([prepared.model_input])[0]
        ms = (time.perf_counter() - t0) * 1000
        pred = self._build(request_id, prepared, requested, used, raw, ms, warnings)
        self._persist([pred], source=source, session_id=session_id)
        return pred

    def compare(self, text: str, models: list[str], request_id: str | None = None,
                session_id: str | None = None) -> dict:
        request_id = request_id or uuid.uuid4().hex
        text = self.validate_text(text)
        prepared = self.pipe.prepare(text)
        results = {}
        for mid in models:
            try:
                adapter = self.registry.get_adapter(mid)
                t0 = time.perf_counter()
                raw = adapter.predict([prepared.model_input])[0]
                ms = (time.perf_counter() - t0) * 1000
                p = self._build(request_id, prepared, mid, mid, raw, ms, [])
                self._persist([p], source="compare", session_id=session_id)
                results[mid] = {"status": "ok", **p.to_dict()}
            except ModelUnavailable as e:
                results[mid] = {"status": "unavailable", "model_used": None, "model_requested": mid,
                                "model_name": STATIC_INFO[mid]["name"], "reason": e.reason}
        ok = [r for r in results.values() if r["status"] == "ok"]
        agreement = (len({r["prediction"] for r in ok}) == 1) if len(ok) == len(models) else None
        return {"request_id": request_id, "text": text, "normalized_text": prepared.normalized,
                "model_input": prepared.model_input, "results": results, "agreement": agreement,
                "note": "A single sentence cannot show which model is better; see the research dashboard "
                        "for locked-test metrics."}

    # ------------------------------------------------------------------ persistence
    def _persist(self, preds: list[Prediction], source: str, session_id: str | None = None,
                 batch_job_id: str | None = None) -> None:
        store_text = self.settings.allow_history_storage
        try:
            with self.db.Session.begin() as s:
                for p in preds:
                    s.add(AnalysisHistory(
                        request_id=p.request_id, source=source, batch_job_id=batch_job_id,
                        model_requested=p.model_requested, model_used=p.model_used,
                        model_version=p.model_version, prediction=p.prediction, label_id=p.label_id,
                        scores=p.scores, score_type=p.score_type, inference_ms=p.inference_ms,
                        input_chars=len(p.text), normalized_text=p.normalized_text if store_text else None,
                        session_id=session_id))
        except Exception:
            # Inference must not fail because history could not be written; surfaced via /health/ready.
            log.exception("could not persist analysis history")

    def sync_registry_tables(self) -> None:
        with self.db.Session.begin() as s:
            for mid, e in self.registry.entries.items():
                if not e.manifest:
                    continue
                exp = STATIC_INFO[mid]["experiment_id"]
                val = (e.validation_metrics or {}).get("metrics", {}).get("macro_f1")
                test = (e.test_metrics or {}).get("metrics", {}).get("macro_f1")
                row = s.scalar(select(ModelRegistryRow).where(ModelRegistryRow.experiment_id == exp,
                                                              ModelRegistryRow.version == e.version))
                if row is None:
                    row = ModelRegistryRow(experiment_id=exp, version=e.version, model_name=STATIC_INFO[mid]["name"],
                                           artifact_ref=e.manifest.get("artifact_dir", ""), status="registered",
                                           manifest_path=str(e.run_dir / "manifest.json"))
                    s.add(row)
                row.source_checkpoint = e.manifest.get("source_checkpoint")
                row.validation_macro_f1 = val
                row.locked_test_macro_f1 = test
                row.status = "evaluated" if test is not None else "trained"
                ds = e.manifest.get("dataset") or {}
                er = s.scalar(select(ExperimentRow).where(ExperimentRow.experiment_id == exp,
                                                          ExperimentRow.model_version == e.version))
                if er is None:
                    er = ExperimentRow(experiment_id=exp, model_version=e.version,
                                       manifest_path=str(e.run_dir / "manifest.json"), training_status="", test_status="")
                    s.add(er)
                er.split_manifest_sha256 = ds.get("split_manifest_sha256")
                er.preprocessing_sha256 = (ds.get("preprocessing") or {}).get("sha256")
                er.training_status = (e.manifest.get("status") or {}).get("training", "unknown")
                er.test_status = "completed" if e.test_metrics else "not_run"
                er.metrics_path = str(self.settings.cfg.path("paths", "metrics_dir") / f"{mid}_test_metrics.json") \
                    if e.test_metrics else None

    # ------------------------------------------------------------------ batch
    def parse_csv(self, raw: bytes, text_column: str | None) -> tuple[pd.DataFrame, str]:
        for enc in ("utf-8-sig", "utf-8", "cp1252"):
            try:
                # skip_blank_lines=False: blank lines become empty rows that are reported as
                # "skipped" (never silently dropped) and row numbers keep matching the file.
                df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding=enc,
                                 skip_blank_lines=False)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise InputError(f"Could not parse CSV: {type(e).__name__}") from e
        else:
            raise InputError("Could not decode CSV (tried UTF-8 and Windows-1252).")
        if df.empty:
            raise InputError("CSV has no data rows.")
        if text_column:
            if text_column not in df.columns:
                raise InputError(f"Column '{text_column}' not found. Columns: {', '.join(map(str, df.columns))}")
            col = text_column
        else:
            lower = {str(c).strip().lower(): c for c in df.columns}
            col = next((lower[c] for c in TEXT_COLUMN_CANDIDATES if c in lower), None)
            if col is None:
                raise InputError("No text column found; pass text_column. Columns: " + ", ".join(map(str, df.columns)))
        if len(df) > self.settings.max_batch_rows:
            raise InputError(f"CSV has {len(df)} rows; the limit is {self.settings.max_batch_rows}.")
        return df, col

    def create_batch(self, texts: list[str], model: str, filename: str | None, text_column: str | None,
                     request_id: str) -> dict:
        used, warnings = self.resolve(model)
        self.registry.get_adapter(used)  # fail fast with 503 if unavailable
        with self.db.Session.begin() as s:
            job = BatchJob(status="queued", model_requested=(model or "auto").lower(), model_used=used,
                           model_version=self.registry.entries[used].version, input_filename=filename,
                           text_column=text_column, row_count=len(texts), processed_count=0, failed_count=0)
            s.add(job)
            s.flush()
            job_id = job.id
        if len(texts) <= self.settings.sync_batch_max_rows:
            self._run_batch(job_id, texts, used, (model or "auto").lower(), warnings, request_id)
        else:
            threading.Thread(target=self._run_batch, args=(job_id, texts, used, (model or "auto").lower(),
                                                           warnings, request_id), daemon=True).start()
        return self.get_job(job_id)

    def _run_batch(self, job_id, texts, used, requested, warnings, request_id) -> None:
        self._update_job(job_id, status="running")
        out_rows, preds = [], []
        try:
            adapter = self.registry.get_adapter(used)
            prepared, valid_idx = [], []
            for i, t in enumerate(texts):
                t = "" if t is None else str(t)
                if not t.strip():
                    out_rows.append({"row": i + 1, "text": t, "status": "skipped", "error": "empty text"})
                elif len(t) > self.settings.max_text_chars:
                    out_rows.append({"row": i + 1, "text": t, "status": "skipped",
                                     "error": f"longer than {self.settings.max_text_chars} characters"})
                else:
                    prepared.append(self.pipe.prepare(t))
                    valid_idx.append(i)
                    out_rows.append(None)
            chunk = 64
            for start in range(0, len(prepared), chunk):
                part = prepared[start:start + chunk]
                t0 = time.perf_counter()
                raws = adapter.predict([p.model_input for p in part])
                per_row_ms = (time.perf_counter() - t0) * 1000 / max(len(part), 1)
                for p, raw, i in zip(part, raws, valid_idx[start:start + chunk]):
                    pred = self._build(request_id, p, requested, used, raw, per_row_ms, warnings)
                    preds.append(pred)
                    out_rows[i] = {"row": i + 1, "text": p.raw, "status": "ok", "error": "",
                                   "prediction": pred.prediction, "label_id": pred.label_id,
                                   **{f"score_{k}": round(v, 6) for k, v in pred.scores.items()},
                                   "score_type": pred.score_type, "model_used": used,
                                   "model_version": pred.model_version}
                self._update_job(job_id, processed_count=start + len(part))
            self._persist(preds, source="batch", batch_job_id=job_id)
            out_dir = self.settings.batch_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{job_id}.csv"
            cols = ["row", "text", "status", "error", "prediction", "label_id"] + \
                   [f"score_{k}" for k in self.labels] + ["score_type", "model_used", "model_version"]
            pd.DataFrame(out_rows).reindex(columns=cols).to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
            dist = {k: sum(1 for p in preds if p.prediction == k) for k in self.labels}
            self._update_job(job_id, status="completed", processed_count=len(preds),
                             failed_count=sum(1 for r in out_rows if r and r["status"] != "ok"),
                             output_path=str(out_path), completed_at=utcnow(),
                             summary={"distribution": dist, "warnings": warnings,
                                      "score_type": STATIC_INFO[used]["score_type"],
                                      "mean_inference_ms_per_row": round(statistics.fmean(
                                          [p.inference_ms for p in preds]), 2) if preds else None})
        except ModelUnavailable as e:
            self._update_job(job_id, status="failed", error=f"model unavailable: {e.reason}", completed_at=utcnow())
        except Exception as e:
            log.exception("batch %s failed", job_id)
            self._update_job(job_id, status="failed", error=f"internal error ({type(e).__name__})", completed_at=utcnow())

    def _update_job(self, job_id: str, **fields) -> None:
        with self.db.Session.begin() as s:
            job = s.get(BatchJob, job_id)
            for k, v in fields.items():
                setattr(job, k, v)

    def get_job(self, job_id: str, preview_rows: int = 200) -> dict | None:
        with self.db.Session() as s:
            job = s.get(BatchJob, job_id)
            if job is None:
                return None
            d = {c: getattr(job, c) for c in ("id", "status", "model_requested", "model_used", "model_version",
                                              "input_filename", "text_column", "row_count", "processed_count",
                                              "failed_count", "summary", "error", "created_at", "completed_at")}
        d["download_url"] = f"/api/v1/batch/jobs/{job_id}/download" if d["status"] == "completed" else None
        if d["status"] == "completed":
            p = Path(self.job_output_path(job_id) or "")
            if p.exists():
                d["rows"] = pd.read_csv(p, dtype=str, keep_default_na=False).head(preview_rows).to_dict("records")
        return d

    def job_output_path(self, job_id: str) -> str | None:
        with self.db.Session() as s:
            job = s.get(BatchJob, job_id)
            return job.output_path if job and job.status == "completed" else None

    # ------------------------------------------------------------------ history / analytics
    def list_analyses(self, page: int, page_size: int, model: str | None, prediction: str | None,
                      source: str | None) -> dict:
        with self.db.Session() as s:
            q = select(AnalysisHistory)
            if model:
                q = q.where(AnalysisHistory.model_used == model)
            if prediction:
                q = q.where(AnalysisHistory.prediction == prediction)
            if source:
                q = q.where(AnalysisHistory.source == source)
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = s.scalars(q.order_by(AnalysisHistory.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size)).all()
            items = [self._row(r) for r in rows]
        return {"page": page, "page_size": page_size, "total": total, "items": items,
                "text_storage_enabled": self.settings.allow_history_storage}

    def get_analysis(self, analysis_id: str) -> dict | None:
        with self.db.Session() as s:
            r = s.get(AnalysisHistory, analysis_id)
            return self._row(r) if r else None

    def delete_history(self) -> int:
        with self.db.Session.begin() as s:
            return s.execute(delete(AnalysisHistory)).rowcount or 0

    @staticmethod
    def _row(r: AnalysisHistory) -> dict:
        return {"id": r.id, "request_id": r.request_id, "source": r.source, "batch_job_id": r.batch_job_id,
                "model_requested": r.model_requested, "model_used": r.model_used, "model_version": r.model_version,
                "prediction": r.prediction, "label_id": r.label_id, "scores": r.scores, "score_type": r.score_type,
                "inference_ms": r.inference_ms, "input_chars": r.input_chars, "normalized_text": r.normalized_text,
                "created_at": r.created_at.isoformat() + "Z"}

    def analytics_summary(self, days: int = 14) -> dict:
        with self.db.Session() as s:
            total = s.scalar(select(func.count(AnalysisHistory.id))) or 0
            by_pred = dict(s.execute(select(AnalysisHistory.prediction, func.count())
                                     .group_by(AnalysisHistory.prediction)).all())
            by_model = dict(s.execute(select(AnalysisHistory.model_used, func.count())
                                      .group_by(AnalysisHistory.model_used)).all())
            by_source = dict(s.execute(select(AnalysisHistory.source, func.count())
                                       .group_by(AnalysisHistory.source)).all())
            pm = s.execute(select(AnalysisHistory.model_used, AnalysisHistory.prediction, func.count())
                           .group_by(AnalysisHistory.model_used, AnalysisHistory.prediction)).all()
            since = utcnow() - dt.timedelta(days=days)
            day_rows = s.execute(select(AnalysisHistory.created_at).where(AnalysisHistory.created_at >= since)).all()
            lat = s.execute(select(AnalysisHistory.model_used, AnalysisHistory.inference_ms)
                            .where(AnalysisHistory.source != "batch")).all()
        per_day: dict[str, int] = {}
        for (ts,) in day_rows:
            k = ts.date().isoformat()
            per_day[k] = per_day.get(k, 0) + 1
        latency: dict[str, dict] = {}
        for mid in sorted({m for m, _ in lat}):
            vals = [v for m, v in lat if m == mid]
            latency[mid] = {"n": len(vals), "mean_ms": round(statistics.fmean(vals), 2),
                            "median_ms": round(statistics.median(vals), 2)}
        dist_by_model: dict[str, dict] = {}
        for m, p, c in pm:
            dist_by_model.setdefault(m, {k: 0 for k in self.labels})[p] = c
        return {"total": total, "by_prediction": {k: by_pred.get(k, 0) for k in self.labels},
                "by_model": by_model, "by_source": by_source, "distribution_by_model": dist_by_model,
                "per_day": dict(sorted(per_day.items())), "latency_single_requests": latency,
                "note": "Derived only from requests stored in this deployment's database."}
