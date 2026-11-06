"""Runtime settings (environment variables; see .env.example).

Model/experiment facts (paths, label map, preprocessing, Auto policy defaults)
come from configs/config.yaml via the shared ``hinglish`` package, so the API
can never disagree with training about labels or normalization.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("HINGLISH_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
_PY_SRC = PROJECT_ROOT / "python" / "src"
if str(_PY_SRC) not in sys.path:
    sys.path.insert(0, str(_PY_SRC))
os.environ.setdefault("HINGLISH_PROJECT_ROOT", str(PROJECT_ROOT))

from hinglish.config import ProjectConfig, load_config  # noqa: E402


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v in (None, "") else int(v)


@dataclass
class Settings:
    cfg: ProjectConfig
    app_env: str = "development"
    database_url: str = ""
    cors_origins: list[str] = field(default_factory=list)
    allow_history_storage: bool = False
    auth_required: bool = False
    api_key: str | None = None
    max_text_chars: int = 500
    max_batch_rows: int = 500
    sync_batch_max_rows: int = 100
    max_upload_mb: int = 10
    preferred_model: str = "e4"
    auto_fallback: list[str] = field(default_factory=list)
    spark_service_url: str = "http://127.0.0.1:8090"
    transformer_device: str = "auto"
    load_models_on_startup: bool = True
    rate_limit_per_minute: int = 120
    log_level: str = "INFO"
    frontend_dir: Path | None = None
    batch_output_dir: Path = PROJECT_ROOT / "data" / "batch_outputs"

    @classmethod
    def from_env(cls) -> "Settings":
        cfg = load_config(PROJECT_ROOT)
        serving = cfg.get("serving", default={}) or {}
        default_db = f"sqlite:///{(PROJECT_ROOT / 'data' / 'hinglishsense.db').as_posix()}"
        fe = os.environ.get("FRONTEND_DIR", str(PROJECT_ROOT / "frontend"))
        return cls(
            cfg=cfg,
            app_env=os.environ.get("APP_ENV", "development"),
            database_url=os.environ.get("DATABASE_URL", default_db),
            cors_origins=[o.strip() for o in os.environ.get(
                "CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:8080").split(",") if o.strip()],
            allow_history_storage=_bool("ALLOW_HISTORY_STORAGE", False),
            auth_required=_bool("AUTH_REQUIRED", False),
            api_key=os.environ.get("API_KEY") or None,
            max_text_chars=_int("MAX_TEXT_CHARS", int(serving.get("max_text_chars", 500))),
            max_batch_rows=_int("MAX_BATCH_ROWS", int(serving.get("max_batch_rows", 500))),
            sync_batch_max_rows=_int("SYNC_BATCH_MAX_ROWS", 100),
            max_upload_mb=_int("MAX_UPLOAD_MB", int(serving.get("max_upload_mb", 10))),
            preferred_model=os.environ.get("PREFERRED_MODEL", serving.get("preferred_model", "e4")).lower(),
            auto_fallback=[m.strip().lower() for m in os.environ.get(
                "AUTO_FALLBACK", ",".join(serving.get("auto_fallback", []))).split(",") if m.strip()],
            spark_service_url=os.environ.get("SPARK_SERVICE_URL", serving.get("spark_service_url", "http://127.0.0.1:8090")),
            transformer_device=os.environ.get("TRANSFORMER_DEVICE", serving.get("transformer_device", "auto")),
            load_models_on_startup=_bool("LOAD_MODELS_ON_STARTUP", True),
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 120),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            frontend_dir=Path(fe) if fe and Path(fe).exists() else None,
            batch_output_dir=Path(os.environ.get("BATCH_OUTPUT_DIR", PROJECT_ROOT / "data" / "batch_outputs")),
        )

    def validate(self) -> None:
        if self.auth_required and not self.api_key:
            raise RuntimeError("AUTH_REQUIRED=true but API_KEY is not set")
        valid = {"e1", "e2", "e3", "e4"}
        if self.preferred_model not in valid or any(m not in valid for m in self.auto_fallback):
            raise RuntimeError("PREFERRED_MODEL / AUTO_FALLBACK must be among e1..e4")
        if self.app_env == "production" and not self.auth_required:
            raise RuntimeError("APP_ENV=production requires AUTH_REQUIRED=true (no open public inference)")
