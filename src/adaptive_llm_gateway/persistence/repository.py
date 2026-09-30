from dataclasses import asdict

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from adaptive_llm_gateway.telemetry.contracts import (
    AdaptiveExecutionTelemetry,
    ChatActivityRecord,
    TelemetryEvent,
    TelemetrySummary,
)

from .models import InferenceTelemetry as Row
from .models import AdaptiveExecutionTelemetry as AdaptiveRow
from .models import ChatActivity as ChatActivityRow


class PostgresTelemetryRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def record(self, event: TelemetryEvent) -> None:
        # begin commits on success, rolls back on failure, and closes the session.
        async with self.sessions.begin() as session:
            session.add(Row(**asdict(event)))

    async def record_adaptive_execution(
        self, event: AdaptiveExecutionTelemetry
    ) -> None:
        values = event.model_dump(mode="python")
        values["validation_outcome"] = event.validation_outcome.value
        values["terminal_outcome"] = event.terminal_outcome.value
        values["failure_codes"] = (
            ",".join(code.value for code in event.failure_codes) or None
        )
        async with self.sessions.begin() as session:
            session.add(AdaptiveRow(**values))

    async def record_chat_activity(self, event: ChatActivityRecord) -> None:
        values = event.model_dump(mode="python")
        values["execution_mode"] = event.execution_mode.value
        values["outcome"] = event.outcome.value
        async with self.sessions.begin() as session:
            session.add(ChatActivityRow(**values))

    async def list_chat_activity(
        self,
        *,
        limit: int,
        before: tuple | None,
    ) -> tuple[list[ChatActivityRecord], bool]:
        statement = select(ChatActivityRow)
        if before is not None:
            created_at, row_id = before
            statement = statement.where(or_(
                ChatActivityRow.created_at < created_at,
                and_(
                    ChatActivityRow.created_at == created_at,
                    ChatActivityRow.id < row_id,
                ),
            ))
        statement = statement.order_by(
            ChatActivityRow.created_at.desc(), ChatActivityRow.id.desc()
        ).limit(limit + 1)
        async with self.sessions() as session:
            rows = list((await session.scalars(statement)).all())
        has_more = len(rows) > limit
        return [
            ChatActivityRecord(
                id=row.id,
                request_id=row.request_id,
                execution_mode=row.execution_mode,
                category=row.category,
                category_source=row.category_source,
                initial_model_id=row.initial_model_id,
                final_model_id=row.final_model_id,
                provider=row.provider,
                routing_threshold_satisfied=row.routing_threshold_satisfied,
                routing_fallback_used=row.routing_fallback_used,
                attempt_count=row.attempt_count,
                escalated=row.escalated,
                validation_outcome=row.validation_outcome,
                outcome=row.outcome,
                error_category=row.error_category,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                latency_ms=row.latency_ms,
                estimated_cost_usd=row.estimated_cost_usd,
                cost_complete=row.cost_complete,
                created_at=row.created_at,
            )
            for row in rows[:limit]
        ], has_more

    async def summary(self) -> TelemetrySummary:
        statement = select(
            func.count().label("total_requests"),
            func.count().filter(Row.success.is_(True)).label("successful_requests"),
            func.count().filter(Row.success.is_(False)).label("failed_requests"),
            func.coalesce(func.sum(Row.estimated_cost_usd), 0).label("total_estimated_cost_usd"),
            func.avg(Row.latency_ms).label("average_latency_ms"),
            func.coalesce(func.sum(Row.input_tokens), 0).label("total_input_tokens"),
            func.coalesce(func.sum(Row.output_tokens), 0).label("total_output_tokens"),
        ).select_from(Row)
        async with self.sessions() as session:
            result = await session.execute(statement)
            return TelemetrySummary(**result.mappings().one())
