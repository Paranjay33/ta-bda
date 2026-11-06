"""Persistence: SQLAlchemy 2.x models + session handling.

SQLite by default (single-user/local demo); PostgreSQL via DATABASE_URL for
hosted/multi-replica deployments. Schema changes go through Alembic
(backend/migrations). Model binaries are never stored here, only references.

Privacy: raw user text is NOT stored unless ALLOW_HISTORY_STORAGE=true, and
even then only the normalized text is kept.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine,
                        event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class ModelRegistryRow(Base):
    __tablename__ = "model_registry"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(String(8), index=True)
    model_name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(100))
    artifact_ref: Mapped[str] = mapped_column(String(500))
    source_checkpoint: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(40))
    manifest_path: Mapped[str] = mapped_column(String(500))
    validation_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    locked_test_macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    __table_args__ = (Index("ix_model_registry_exp_version", "experiment_id", "version", unique=True),)


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(16), default="single")  # single | compare | batch
    batch_job_id: Mapped[str | None] = mapped_column(String(32), ForeignKey("batch_jobs.id"), nullable=True, index=True)
    model_requested: Mapped[str] = mapped_column(String(8))
    model_used: Mapped[str] = mapped_column(String(8), index=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prediction: Mapped[str] = mapped_column(String(16), index=True)
    label_id: Mapped[int] = mapped_column(Integer)
    scores: Mapped[dict] = mapped_column(JSON)
    score_type: Mapped[str] = mapped_column(String(40))
    inference_ms: Mapped[float] = mapped_column(Float)
    input_chars: Mapped[int] = mapped_column(Integer)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # only when history storage enabled
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)


class BatchJob(Base):
    __tablename__ = "batch_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(16), index=True)  # queued | running | completed | failed
    model_requested: Mapped[str] = mapped_column(String(8))
    model_used: Mapped[str | None] = mapped_column(String(8), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    input_filename: Mapped[str | None] = mapped_column(String(200), nullable=True)
    text_column: Mapped[str | None] = mapped_column(String(100), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    output_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class ExperimentRow(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(String(8), index=True)
    model_version: Mapped[str] = mapped_column(String(100))
    split_manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preprocessing_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    manifest_path: Mapped[str] = mapped_column(String(500))
    metrics_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    training_status: Mapped[str] = mapped_column(String(40))
    test_status: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    __table_args__ = (Index("ix_experiments_exp_version", "experiment_id", "model_version", unique=True),)


class Database:
    def __init__(self, url: str):
        kwargs = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.url = url
        self.engine = create_engine(url, pool_pre_ping=True, future=True, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragma(dbapi_conn, _):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    def migrate(self) -> None:
        """Apply Alembic migrations (the schema source of truth)."""
        from pathlib import Path

        from alembic import command
        from alembic.config import Config

        here = Path(__file__).resolve().parent.parent
        acfg = Config(str(here / "alembic.ini"))
        acfg.set_main_option("script_location", str(here / "migrations"))
        acfg.set_main_option("sqlalchemy.url", self.url.replace("%", "%%"))
        acfg.attributes["configure_logger"] = False
        with self.engine.begin() as conn:
            acfg.attributes["connection"] = conn
            command.upgrade(acfg, "head")

    def healthy(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False
