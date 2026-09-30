"""Add one bounded request-level chat activity summary table.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_activity",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_id", sa.String(128), nullable=False, unique=True),
        sa.Column("execution_mode", sa.String(16), nullable=False),
        sa.Column("category", sa.String(64), nullable=True),
        sa.Column("category_source", sa.String(16), nullable=True),
        sa.Column("initial_model_id", sa.String(), nullable=True),
        sa.Column("final_model_id", sa.String(), nullable=True),
        sa.Column("provider", sa.String(), nullable=True),
        sa.Column("routing_threshold_satisfied", sa.Boolean(), nullable=True),
        sa.Column("routing_fallback_used", sa.Boolean(), nullable=True),
        sa.Column("attempt_count", sa.BigInteger(), nullable=True),
        sa.Column("escalated", sa.Boolean(), nullable=True),
        sa.Column("validation_outcome", sa.String(32), nullable=True),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("input_tokens", sa.BigInteger(), nullable=True),
        sa.Column("output_tokens", sa.BigInteger(), nullable=True),
        sa.Column("latency_ms", sa.Double(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(asdecimal=True), nullable=True),
        sa.Column("cost_complete", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("attempt_count IS NULL OR (attempt_count > 0 AND attempt_count <= 3)", name="ck_chat_activity_attempt_count"),
        sa.CheckConstraint("input_tokens IS NULL OR input_tokens >= 0", name="ck_chat_activity_input_tokens"),
        sa.CheckConstraint("output_tokens IS NULL OR output_tokens >= 0", name="ck_chat_activity_output_tokens"),
        sa.CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="ck_chat_activity_latency"),
        sa.CheckConstraint("estimated_cost_usd IS NULL OR estimated_cost_usd >= 0", name="ck_chat_activity_cost"),
    )
    op.create_index(
        "ix_chat_activity_created_id",
        "chat_activity",
        [sa.text("created_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_table("chat_activity")
