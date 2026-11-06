"""initial schema: model_registry, batch_jobs, analysis_history, experiments

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("experiment_id", sa.String(8), nullable=False, index=True),
        sa.Column("model_name", sa.String(200), nullable=False),
        sa.Column("version", sa.String(100), nullable=False),
        sa.Column("artifact_ref", sa.String(500), nullable=False),
        sa.Column("source_checkpoint", sa.String(200)),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("manifest_path", sa.String(500), nullable=False),
        sa.Column("validation_macro_f1", sa.Float),
        sa.Column("locked_test_macro_f1", sa.Float),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_model_registry_exp_version", "model_registry", ["experiment_id", "version"], unique=True)

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, index=True),
        sa.Column("model_requested", sa.String(8), nullable=False),
        sa.Column("model_used", sa.String(8)),
        sa.Column("model_version", sa.String(100)),
        sa.Column("input_filename", sa.String(200)),
        sa.Column("text_column", sa.String(100)),
        sa.Column("row_count", sa.Integer, nullable=False),
        sa.Column("processed_count", sa.Integer, nullable=False),
        sa.Column("failed_count", sa.Integer, nullable=False),
        sa.Column("output_path", sa.String(500)),
        sa.Column("summary", sa.JSON),
        sa.Column("error", sa.String(300)),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime),
    )

    op.create_table(
        "analysis_history",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=False, index=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("batch_job_id", sa.String(32), sa.ForeignKey("batch_jobs.id"), index=True),
        sa.Column("model_requested", sa.String(8), nullable=False),
        sa.Column("model_used", sa.String(8), nullable=False, index=True),
        sa.Column("model_version", sa.String(100)),
        sa.Column("prediction", sa.String(16), nullable=False, index=True),
        sa.Column("label_id", sa.Integer, nullable=False),
        sa.Column("scores", sa.JSON, nullable=False),
        sa.Column("score_type", sa.String(40), nullable=False),
        sa.Column("inference_ms", sa.Float, nullable=False),
        sa.Column("input_chars", sa.Integer, nullable=False),
        sa.Column("normalized_text", sa.Text),
        sa.Column("session_id", sa.String(64), index=True),
        sa.Column("created_at", sa.DateTime, nullable=False, index=True),
    )

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("experiment_id", sa.String(8), nullable=False, index=True),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("split_manifest_sha256", sa.String(64)),
        sa.Column("preprocessing_sha256", sa.String(64)),
        sa.Column("manifest_path", sa.String(500), nullable=False),
        sa.Column("metrics_path", sa.String(500)),
        sa.Column("training_status", sa.String(40), nullable=False),
        sa.Column("test_status", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_experiments_exp_version", "experiments", ["experiment_id", "model_version"], unique=True)


def downgrade() -> None:
    op.drop_table("experiments")
    op.drop_table("analysis_history")
    op.drop_table("batch_jobs")
    op.drop_table("model_registry")
