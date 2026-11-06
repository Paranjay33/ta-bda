"""Model registry + inference adapters.

The registry is built ONLY from on-disk manifests written by the training
pipelines. A model is "available" only if its manifest and artifacts exist
AND its runtime (in-process PyTorch or the persistent Spark service) can
actually serve it. Nothing is hard-coded about availability or metrics.

Score semantics per model (never overstated):
  E1  model_probability        Spark multinomial-LR probabilities, not separately calibrated
  E2  calibrated_probability   temperature-scaled softmax over OvR LinearSVC margins (validation-fitted);
                               argmax-preserving, so the class is the native OvR decision
  E3  calibrated_probability   temperature-scaled softmax (validation-fitted)
  E4  calibrated_probability   temperature-scaled softmax (validation-fitted)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hinglish.calibration import load_calibration, softmax

from .settings import Settings

log = logging.getLogger("hinglishsense.registry")

MODEL_IDS = ["e1", "e2", "e3", "e4"]
STATIC_INFO = {
    "e1": {"experiment_id": "E1", "family": "classical", "area": "BDA",
           "name": "E1 — TF-IDF + Logistic Regression",
           "method": "Word TF-IDF (RegexTokenizer -> CountVectorizer -> IDF) + multinomial Logistic Regression, Scala + Spark MLlib",
           "score_type": "model_probability"},
    "e2": {"experiment_id": "E2", "family": "classical", "area": "BDA",
           "name": "E2 — Word + Character Features + SVM",
           "method": "Word TF-IDF + genuine character 3-gram TF-IDF -> One-vs-Rest LinearSVC, Scala + Spark MLlib",
           "score_type": "calibrated_probability"},
    "e3": {"experiment_id": "E3", "family": "transformer", "area": "TA",
           "name": "E3 — XLM-R",
           "method": "FacebookAI/xlm-roberta-base fine-tuned (PyTorch + Hugging Face), generic multilingual pretraining",
           "score_type": "calibrated_probability"},
    "e4": {"experiment_id": "E4", "family": "transformer", "area": "TA",
           "name": "E4 — HingRoBERTa",
           "method": "l3cube-pune/hing-roberta fine-tuned (PyTorch + Hugging Face), Hindi-English domain-adapted pretraining",
           "score_type": "calibrated_probability"},
}


class ModelUnavailable(Exception):
    def __init__(self, model_id: str, reason: str):
        super().__init__(f"{model_id} unavailable: {reason}")
        self.model_id, self.reason = model_id, reason


@dataclass
class RawPrediction:
    label_id: int
    scores: list[float]


@dataclass
class ModelEntry:
    model_id: str
    run_dir: Path
    manifest: dict | None
    test_metrics: dict | None
    validation_metrics: dict | None
    adapter: Any = None
    load_error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def info(self) -> dict:
        return STATIC_INFO[self.model_id]

    @property
    def version(self) -> str | None:
        return (self.manifest or {}).get("version")

    @property
    def trained(self) -> bool:
        return bool(self.manifest) and (self.manifest.get("status") or {}).get("training") == "completed"

    @property
    def selection_verified(self) -> bool:
        sel = ((self.manifest or {}).get("status") or {}).get("selection", "")
        return self.trained and isinstance(sel, str) and sel != ""


# --------------------------------------------------------------------------- adapters
class TransformerAdapter:
    """Fine-tuned HF checkpoint loaded ONCE and kept resident (lazy, thread-safe)."""

    def __init__(self, entry: ModelEntry, settings: Settings):
        self.entry = entry
        self.settings = settings
        self.best_dir = entry.run_dir / "best"
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.calibration = None
        self.device = None
        self.load_seconds = None
        self.max_length = int(settings.cfg.get("transformer", "max_seq_length", default=128))

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            t0 = time.perf_counter()
            want = self.settings.transformer_device
            device = "cuda" if (want in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
            tok = AutoTokenizer.from_pretrained(str(self.best_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(self.best_dir))
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            expected = self.settings.cfg.labels.id_to_label
            if id2label != expected:
                raise RuntimeError(f"checkpoint label order {id2label} != canonical {expected}")
            model.eval()
            try:
                model.to(device)
            except RuntimeError as e:  # e.g. CUDA OOM on a shared 4 GB GPU
                log.warning("model %s could not be placed on %s (%s); using CPU", self.entry.model_id, device, type(e).__name__)
                device = "cpu"
                model.to("cpu")
                torch.cuda.empty_cache()
            self.calibration = load_calibration(self.best_dir / "calibration.json")
            self.tokenizer, self.model, self.device = tok, model, device
            self.load_seconds = time.perf_counter() - t0
            log.info("loaded %s on %s in %.1fs", self.entry.model_id, device, self.load_seconds)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[RawPrediction]:
        import torch

        if not self.loaded:
            self.load()
        out: list[RawPrediction] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], truncation=True, padding=True,
                                     max_length=self.max_length, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float().cpu().numpy().astype(np.float64)
                probs = self.calibration.apply(logits)
                for row_logits, row_p in zip(logits, probs):
                    out.append(RawPrediction(label_id=int(row_logits.argmax()), scores=[float(x) for x in row_p]))
        return out

    def status(self) -> dict:
        return {"runtime": "pytorch", "loaded": self.loaded, "device": self.device,
                "load_seconds": self.load_seconds}


class SparkAdapter:
    """Calls the persistent Scala/Spark inference service (never starts a JVM per request)."""

    def __init__(self, entry: ModelEntry, settings: Settings):
        self.entry = entry
        self.url = settings.spark_service_url.rstrip("/")
        self.calibration = None
        if entry.model_id == "e2":
            cal = entry.run_dir / "calibration.json"
            self.calibration = load_calibration(cal) if cal.exists() else None
        self._health: dict | None = None
        self._health_at = 0.0

    def health(self, max_age: float = 5.0) -> dict | None:
        import httpx

        if self._health is not None and time.time() - self._health_at < max_age:
            return self._health
        try:
            r = httpx.get(f"{self.url}/health", timeout=2.0)
            self._health = r.json() if r.status_code == 200 else None
        except Exception:
            self._health = None
        self._health_at = time.time()
        return self._health

    @property
    def loaded(self) -> bool:
        h = self.health()
        return bool(h and h.get("models", {}).get(self.entry.model_id, {}).get("loaded"))

    def load(self) -> None:
        if not self.loaded:
            raise ModelUnavailable(self.entry.model_id, "spark_service_unavailable")

    def predict(self, texts: list[str], batch_size: int = 500) -> list[RawPrediction]:
        import httpx

        if self.entry.model_id == "e2" and self.calibration is None:
            raise ModelUnavailable("e2", "calibration_missing")
        out: list[RawPrediction] = []
        for i in range(0, len(texts), batch_size):
            try:
                r = httpx.post(f"{self.url}/predict", json={"model": self.entry.model_id, "texts": texts[i:i + batch_size]},
                               timeout=60.0)
            except Exception as e:
                self._health = None
                raise ModelUnavailable(self.entry.model_id, "spark_service_unreachable") from e
            if r.status_code == 503:
                raise ModelUnavailable(self.entry.model_id, "spark_model_not_loaded")
            r.raise_for_status()
            body = r.json()
            if body.get("version") != self.entry.version:
                raise ModelUnavailable(self.entry.model_id, "spark_service_version_mismatch")
            for res in body["results"]:
                raw = np.asarray(res["scores"], dtype=np.float64)
                scores = self.calibration.apply(raw)[0] if self.calibration is not None else raw
                out.append(RawPrediction(label_id=int(res["label_id"]), scores=[float(x) for x in scores]))
        return out

    def status(self) -> dict:
        h = self.health()
        m = (h or {}).get("models", {}).get(self.entry.model_id, {})
        return {"runtime": "spark", "loaded": bool(m.get("loaded")), "service_reachable": h is not None,
                "spark_master": (h or {}).get("spark_master"), "cold_start_seconds": m.get("load_seconds"),
                "service_version": m.get("version")}


# --------------------------------------------------------------------------- registry
class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.cfg
        self.entries: dict[str, ModelEntry] = {}
        self.refresh()

    def _run_dir(self, mid: str) -> Path:
        key = "spark" if mid in ("e1", "e2") else "transformer"
        return self.cfg.path("paths", "models_dir") / mid / self.cfg.get(key, "run_tag", default="v2-clean-split")

    def refresh(self) -> None:
        mdir = self.cfg.path("paths", "metrics_dir")
        for mid in MODEL_IDS:
            run_dir = self._run_dir(mid)
            manifest = _read_json(run_dir / "manifest.json")
            test = _read_json(mdir / f"{mid}_test_metrics.json")
            val = _read_json(mdir / f"{mid}_validation_metrics.json")
            # Metrics only count if they were produced by THIS registered artifact version.
            if test and manifest and test.get("model_version") != manifest.get("version"):
                test = None
            if val and manifest and val.get("model_version") != manifest.get("version"):
                val = None
            entry = ModelEntry(mid, run_dir, manifest, test, val)
            if entry.trained:
                entry.adapter = (TransformerAdapter if mid in ("e3", "e4") else SparkAdapter)(entry, self.settings)
            old = self.entries.get(mid)
            if old and old.adapter and entry.adapter and old.version == entry.version:
                entry.adapter = old.adapter  # keep loaded weights
            self.entries[mid] = entry

    def warm_up(self) -> None:
        for mid in ("e4", "e3"):
            e = self.entries[mid]
            if e.adapter is not None:
                try:
                    e.adapter.load()
                    e.adapter.predict(["warm up"])
                except Exception as ex:  # recorded, surfaced via /health/ready
                    e.load_error = type(ex).__name__
                    log.error("warm-up of %s failed: %s", mid, ex)

    def availability(self, mid: str) -> tuple[bool, str | None]:
        e = self.entries[mid]
        if not e.manifest:
            return False, "not_trained"
        if not e.trained:
            return False, "training_incomplete"
        if e.load_error:
            return False, f"load_failed:{e.load_error}"
        if isinstance(e.adapter, SparkAdapter):
            h = e.adapter.health()
            if h is None:
                return False, "spark_service_unreachable"
            m = h.get("models", {}).get(mid, {})
            if not m.get("loaded"):
                return False, "spark_model_not_loaded"
            if m.get("version") != e.version:
                return False, "spark_service_version_mismatch"
            if mid == "e2" and e.adapter.calibration is None:
                return False, "calibration_missing"
        if isinstance(e.adapter, TransformerAdapter) and not (e.run_dir / "best" / "calibration.json").exists():
            return False, "artifact_missing"
        return True, None

    def get_adapter(self, mid: str):
        ok, reason = self.availability(mid)
        if not ok:
            raise ModelUnavailable(mid, reason or "unavailable")
        return self.entries[mid].adapter

    def resolve_auto(self) -> tuple[str, list[str]]:
        """Backend-controlled Auto policy. Never consults test metrics."""
        warnings: list[str] = []
        pref = self.settings.preferred_model
        ok, reason = self.availability(pref)
        if ok and self.entries[pref].selection_verified:
            return pref, warnings
        why = reason or "validation selection not recorded"
        for fb in self.settings.auto_fallback:
            if fb == pref:
                continue
            ok_fb, _ = self.availability(fb)
            if ok_fb:
                warnings.append(f"Auto fallback: preferred model {pref.upper()} is unavailable ({why}); "
                                f"used {fb.upper()} instead.")
                return fb, warnings
        raise ModelUnavailable("auto", f"no validated model available (preferred {pref}: {why})")

    def describe(self, mid: str) -> dict:
        e = self.entries[mid]
        ok, reason = self.availability(mid)
        m = e.manifest or {}
        test = (e.test_metrics or {}).get("metrics") or {}
        val = (e.validation_metrics or {}).get("metrics") or (m.get("validation_metrics") or {})
        return {
            "model_id": mid,
            **e.info,
            "available": ok,
            "unavailable_reason": reason,
            "version": e.version,
            "source_checkpoint": m.get("source_checkpoint"),
            "source_checkpoint_revision": m.get("source_checkpoint_revision"),
            "status": {
                "trained": e.trained,
                "selection": (m.get("status") or {}).get("selection"),
                "locked_test_evaluation": "completed" if e.test_metrics else "not_run",
                "runtime": e.adapter.status() if e.adapter else None,
            },
            "score_semantics": m.get("score_semantics"),
            "validation_macro_f1": val.get("macro_f1"),
            "test_metrics": ({k: test.get(k) for k in ("accuracy", "macro_f1", "weighted_f1",
                                                        "macro_precision", "macro_recall")}
                             if e.test_metrics else None),
            "test_split_rows": (e.test_metrics or {}).get("n_samples"),
            "artifact_size_bytes": m.get("artifact_size_bytes"),
            "training_runtime_seconds": m.get("training_runtime_seconds"),
            "is_preferred": mid == self.settings.preferred_model,
        }


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        log.exception("could not read %s", p)
        return None
