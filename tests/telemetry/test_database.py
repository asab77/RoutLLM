from dataclasses import asdict
from decimal import Decimal
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from adaptive_llm_gateway.persistence.config import DatabaseSettings
from adaptive_llm_gateway.persistence.models import (
    AdaptiveExecutionTelemetry as AdaptiveExecutionRow,
    InferenceTelemetry,
)
from adaptive_llm_gateway.persistence.repository import PostgresTelemetryRepository
from adaptive_llm_gateway.telemetry.contracts import (
    AdaptiveExecutionTelemetry,
    AdaptiveTerminalOutcome,
    TelemetryEvent,
)
from adaptive_llm_gateway.validation import ValidationStatus


@pytest.mark.parametrize("url", ["", "sqlite:///test.db", "postgresql://user:secret@localhost/db",
    "postgresql+asyncpg://user:secret@localhost", "postgresql+asyncpg://localhost/db",
    "postgresql+asyncpg://user:secret@localhost:99999/db", "not a URL"])
def test_database_url_validation(url):
    with pytest.raises(ValidationError) as caught:
        DatabaseSettings(database_url=url)
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("timeout", [0, -1, 31, float("nan"), float("inf")])
def test_timeout_validation(timeout):
    with pytest.raises(ValidationError):
        DatabaseSettings(telemetry_timeout_seconds=timeout)


def test_environment_configuration_and_redaction(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:secret@localhost/db")
    monkeypatch.setenv("TELEMETRY_TIMEOUT_SECONDS", "3")
    settings = DatabaseSettings.from_environment()
    assert settings.telemetry_timeout_seconds == 3
    assert "secret" not in repr(settings)
    assert settings.database_url.get_secret_value().endswith("/db")


def test_unconfigured_database_is_explicitly_optional(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert DatabaseSettings.from_environment().database_url is None


def test_migrations_load_and_render_postgres_ddl(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:unused@localhost/test")
    output = StringIO()
    config = Config("alembic.ini", output_buffer=output)
    assert ScriptDirectory.from_config(config).get_current_head() == "0002"
    command.upgrade(config, "head", sql=True)
    sql = output.getvalue()
    assert "CREATE TABLE inference_telemetry" in sql
    assert "CREATE TABLE adaptive_execution_telemetry" in sql
    assert "NUMERIC" in sql and "TIMESTAMP WITH TIME ZONE" in sql
    assert "ix_telemetry_request_id" in sql
    assert "INSERT INTO alembic_version" in sql


def test_storage_schema_contains_no_raw_content():
    columns = InferenceTelemetry.__table__.columns
    assert not {"prompt", "system_prompt", "response", "text", "exception", "stack_trace"} & set(columns.keys())
    assert columns.estimated_cost_usd.type.asdecimal
    assert columns.estimated_cost_usd.type.scale is None
    assert columns.created_at.type.timezone
    assert not columns.request_id.unique
    adaptive_columns = AdaptiveExecutionRow.__table__.columns
    assert not {
        "prompt", "system_prompt", "response", "text", "exception",
        "stack_trace", "field_names", "allowed_labels", "provider_body",
    } & set(adaptive_columns.keys())
    assert adaptive_columns.cumulative_known_cost_usd.type.asdecimal
    assert adaptive_columns.cumulative_known_cost_usd.type.scale is None


def event():
    return TelemetryEvent(request_id="trace", model_id="fake-small", provider="fake", success=True,
        input_tokens=1, output_tokens=1, estimated_cost_usd=Decimal("0.000000000000000001"),
        latency_ms=1, max_output_tokens=2, temperature=1, prompt_characters=5, system_prompt_characters=0)


@pytest.mark.asyncio
async def test_repository_writes_exact_event_in_managed_transaction():
    sessions = MagicMock()
    transaction = sessions.begin.return_value
    session = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=session)
    transaction.__aexit__ = AsyncMock(return_value=False)
    value = event()
    await PostgresTelemetryRepository(sessions).record(value)
    row = session.add.call_args.args[0]
    assert all(getattr(row, name) == data for name, data in asdict(value).items())
    transaction.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_repository_writes_sanitized_adaptive_summary():
    sessions = MagicMock()
    transaction = sessions.begin.return_value
    session = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=session)
    transaction.__aexit__ = AsyncMock(return_value=False)
    value = AdaptiveExecutionTelemetry(
        request_id="trace",
        initial_routed_model_id="small",
        returned_model_id="large",
        attempt_count=2,
        escalated=True,
        validation_outcome=ValidationStatus.PASSED,
        terminal_outcome=AdaptiveTerminalOutcome.RETURNED,
        cumulative_known_cost_usd=Decimal("0.0012"),
        cost_complete=True,
        cumulative_latency_ms=12,
        validator_version="deterministic-v1",
        validation_duration_ms=1,
    )
    await PostgresTelemetryRepository(sessions).record_adaptive_execution(value)
    row = session.add.call_args.args[0]
    assert row.initial_routed_model_id == "small"
    assert row.returned_model_id == "large"
    assert row.validation_outcome == "passed"
    assert row.terminal_outcome == "returned"
    assert row.failure_codes is None


@pytest.mark.asyncio
async def test_repository_propagates_failure_to_transaction_manager():
    sessions = MagicMock()
    transaction = sessions.begin.return_value
    session = MagicMock()
    session.add.side_effect = RuntimeError("write error")
    transaction.__aenter__ = AsyncMock(return_value=session)
    transaction.__aexit__ = AsyncMock(return_value=False)
    with pytest.raises(RuntimeError):
        await PostgresTelemetryRepository(sessions).record(event())
    assert transaction.__aexit__.await_args.args[0] is RuntimeError


@pytest.mark.asyncio
async def test_query_session_closes_on_failure():
    sessions = MagicMock()
    manager = sessions.return_value
    session = MagicMock()
    session.execute = AsyncMock(side_effect=RuntimeError("query error"))
    manager.__aenter__ = AsyncMock(return_value=session)
    manager.__aexit__ = AsyncMock(return_value=False)
    with pytest.raises(RuntimeError):
        await PostgresTelemetryRepository(sessions).summary()
    assert manager.__aexit__.await_args.args[0] is RuntimeError
