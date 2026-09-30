import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime
import logging
from uuid import UUID

from adaptive_llm_gateway.errors import InvalidActivityCursorError
from adaptive_llm_gateway.observability import JsonEventLogger

from .contracts import ChatActivityRecord, InferenceTelemetryRepository, TelemetrySummary

events = JsonEventLogger()


class TelemetryUnavailableError(RuntimeError):
    """Metrics are unavailable; never present missing data as zero activity."""


class TelemetryQueryService:
    def __init__(self, repository: InferenceTelemetryRepository | None, timeout: float = 2.0) -> None:
        self.repository = repository
        self.timeout = timeout

    async def summary(self) -> TelemetrySummary:
        if self.repository is None:
            raise TelemetryUnavailableError()
        try:
            async with asyncio.timeout(self.timeout):
                return await self.repository.summary()
        except Exception:
            events.emit(
                "dependency_degraded",
                level=logging.WARNING,
                dependency="postgresql",
                record_type="summary",
                outcome="unavailable",
            )
            raise TelemetryUnavailableError() from None


@dataclass(frozen=True)
class ChatActivityPage:
    items: list[ChatActivityRecord]
    next_cursor: str | None


def encode_activity_cursor(created_at: datetime, row_id: UUID) -> str:
    value = f"{created_at.isoformat()}|{row_id}".encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def decode_activity_cursor(cursor: str) -> tuple[datetime, UUID]:
    if not cursor or len(cursor) > 256:
        raise InvalidActivityCursorError("invalid activity cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        timestamp, raw_id = decoded.split("|", 1)
        created_at = datetime.fromisoformat(timestamp)
        row_id = UUID(raw_id)
    except (ValueError, UnicodeError):
        raise InvalidActivityCursorError("invalid activity cursor") from None
    if created_at.tzinfo is None:
        raise InvalidActivityCursorError("invalid activity cursor")
    return created_at, row_id


class ChatActivityQueryService:
    def __init__(self, repository: object | None, timeout: float = 2.0) -> None:
        self.repository = repository
        self.timeout = timeout

    async def page(self, *, limit: int, cursor: str | None) -> ChatActivityPage:
        if self.repository is None or not hasattr(self.repository, "list_chat_activity"):
            raise TelemetryUnavailableError()
        before = decode_activity_cursor(cursor) if cursor is not None else None
        try:
            async with asyncio.timeout(self.timeout):
                items, has_more = await self.repository.list_chat_activity(
                    limit=limit, before=before
                )
        except InvalidActivityCursorError:
            raise
        except Exception:
            events.emit(
                "dependency_degraded",
                level=logging.WARNING,
                dependency="postgresql",
                record_type="chat_activity",
                outcome="unavailable",
            )
            raise TelemetryUnavailableError() from None
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = encode_activity_cursor(last.created_at, last.id)
        return ChatActivityPage(items=items, next_cursor=next_cursor)
