import asyncio
import logging

from adaptive_llm_gateway.observability import JsonEventLogger

from .contracts import InferenceTelemetryRepository, TelemetrySummary

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
