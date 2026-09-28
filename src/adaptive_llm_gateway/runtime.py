"""Composition and resource lifecycle, separate from HTTP and domain logic."""

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service, configure_gateway
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from adaptive_llm_gateway.providers.vercel import create_http_client
from adaptive_llm_gateway.persistence.config import DatabaseSettings
from adaptive_llm_gateway.persistence.database import Database
from adaptive_llm_gateway.request_deadline import InferenceDeadlineSettings
from adaptive_llm_gateway.observability import Observability
from adaptive_llm_gateway.rate_limit import (
    DisabledRateLimiter,
    InferenceRateLimiter,
    RateLimitSettings,
    RedisRateLimiter,
    create_redis_client,
)

@asynccontextmanager
async def rate_limiter_service(
    observability: Observability | None = None,
) -> AsyncIterator[InferenceRateLimiter]:
    """Own one reusable Redis client for the application lifespan."""
    settings = RateLimitSettings.from_environment()
    if not settings.required:
        yield DisabledRateLimiter()
        return

    client = create_redis_client(settings)
    try:
        yield RedisRateLimiter(client, settings, observability)
    finally:
        await client.aclose()


@asynccontextmanager
async def application_service(
    observability: Observability | None = None,
) -> AsyncIterator[InferenceService]:
    database_settings = DatabaseSettings.from_environment()
    gateway_settings = GatewaySettings.from_environment()
    deadline_settings = InferenceDeadlineSettings.from_environment()
    http_client = create_http_client(gateway_settings)
    service = create_development_service(
        inference_deadline_seconds=deadline_settings.seconds
    )
    if observability is not None:
        service.observability = observability
    configure_gateway(service, gateway_settings, client=http_client)
    database = None
    try:
        if database_settings.database_url is None:
            service.observability.events.emit(
                "telemetry_disabled",
                level=logging.WARNING,
                dependency="postgresql",
                outcome="disabled",
            )
        else:
            database = Database(database_settings)
            service.telemetry = database.repository
            service.telemetry_timeout = database_settings.telemetry_timeout_seconds
            try:
                # Read-only probe checks connectivity and migrated schema; never creates it.
                async with asyncio.timeout(database_settings.telemetry_timeout_seconds):
                    await database.repository.summary()
            except Exception:
                service.observability.events.emit(
                    "telemetry_startup_probe_failed",
                    level=logging.WARNING,
                    dependency="postgresql",
                    outcome="unavailable",
                )
        yield service
    finally:
        try:
            if database is not None:
                await database.close()
        finally:
            await http_client.aclose()
