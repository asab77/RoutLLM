"""Add one bounded adaptive execution summary table.

Revision ID: 0002
Revises: 0001
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "adaptive_execution_telemetry",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("initial_routed_model_id", sa.String(), nullable=False),
        sa.Column("returned_model_id", sa.String(), nullable=True),
        sa.Column("attempt_count", sa.BigInteger(), nullable=False),
        sa.Column("escalated", sa.Boolean(), nullable=False),
        sa.Column("validation_outcome", sa.String(32), nullable=False),
        sa.Column("terminal_outcome", sa.String(32), nullable=False),
        sa.Column("cumulative_known_cost_usd", sa.Numeric(asdecimal=True), nullable=False),
        sa.Column("cost_complete", sa.Boolean(), nullable=False),
        sa.Column("cumulative_latency_ms", sa.Double(), nullable=False),
        sa.Column("validator_version", sa.String(64), nullable=True),
        sa.Column("validation_duration_ms", sa.Double(), nullable=False),
        sa.Column("failure_codes", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "attempt_count > 0 AND attempt_count <= 3",
            name="ck_adaptive_attempt_count",
        ),
        sa.CheckConstraint(
            "cumulative_known_cost_usd >= 0",
            name="ck_adaptive_cost",
        ),
        sa.CheckConstraint(
            "cumulative_latency_ms >= 0",
            name="ck_adaptive_latency",
        ),
        sa.CheckConstraint(
            "validation_duration_ms >= 0",
            name="ck_adaptive_validation_duration",
        ),
    )
    op.create_index(
        "ix_adaptive_execution_request_id",
        "adaptive_execution_telemetry",
        ["request_id"],
    )
    op.create_index(
        "ix_adaptive_execution_created",
        "adaptive_execution_telemetry",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_table("adaptive_execution_telemetry")
