import asyncio
import os

import pytest
import pytest_asyncio
from pydantic import SecretStr
from redis.asyncio import Redis

from adaptive_llm_gateway.errors import RateLimitExceededError
from adaptive_llm_gateway.rate_limit import (
    InferenceKind,
    RateLimitSettings,
    RedisRateLimiter,
    client_identity_digest,
)
from adaptive_llm_gateway.request_deadline import RequestDeadline

pytestmark = pytest.mark.redis


@pytest_asyncio.fixture
async def redis_client():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("TEST_REDIS_URL is not configured")
    client = Redis.from_url(url)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


def settings(**overrides):
    values = {
        "required": True,
        "redis_url": SecretStr(os.environ.get("TEST_REDIS_URL", "redis://unused")),
        "hmac_secret": SecretStr("integration-secret-at-least-32-characters"),
        "client_capacity": 10,
        "client_window_seconds": 1,
        "global_capacity": 100,
        "global_window_seconds": 1,
        "explicit_weight": 1,
        "adaptive_weight": 1,
    }
    values.update(overrides)
    return RateLimitSettings(**values)


@pytest.mark.asyncio
async def test_client_isolation_global_sharing_and_private_keys(redis_client):
    configured = settings(client_capacity=2, global_capacity=3)
    limiter = RedisRateLimiter(redis_client, configured)
    await limiter.admit("203.0.113.10", InferenceKind.EXPLICIT, RequestDeadline(2))
    await limiter.admit("203.0.113.10", InferenceKind.EXPLICIT, RequestDeadline(2))
    await limiter.admit("203.0.113.11", InferenceKind.EXPLICIT, RequestDeadline(2))

    with pytest.raises(RateLimitExceededError):
        await limiter.admit("203.0.113.10", InferenceKind.EXPLICIT, RequestDeadline(2))
    with pytest.raises(RateLimitExceededError):
        await limiter.admit("203.0.113.12", InferenceKind.EXPLICIT, RequestDeadline(2))

    keys = [item.decode() for item in await redis_client.keys("*")]
    serialized = " ".join(keys)
    assert "203.0.113" not in serialized
    assert "private" not in serialized
    assert all(key.startswith("routellm:rate:{admission}:") for key in keys)
    ttls = await asyncio.gather(*(redis_client.pttl(key) for key in keys))
    assert all(ttl > 0 for ttl in ttls)


@pytest.mark.asyncio
async def test_concurrency_cannot_exceed_capacity(redis_client):
    limiter = RedisRateLimiter(
        redis_client, settings(client_capacity=3, global_capacity=3)
    )

    async def attempt():
        try:
            await limiter.admit(
                "198.51.100.20", InferenceKind.EXPLICIT, RequestDeadline(2)
            )
            return True
        except RateLimitExceededError:
            return False

    results = await asyncio.gather(*(attempt() for _ in range(20)))
    assert sum(results) == 3


@pytest.mark.asyncio
async def test_rejection_consumes_neither_scope(redis_client):
    configured = settings(client_capacity=10, global_capacity=1)
    limiter = RedisRateLimiter(redis_client, configured)
    await limiter.admit("198.51.100.1", InferenceKind.EXPLICIT, RequestDeadline(2))
    with pytest.raises(RateLimitExceededError):
        await limiter.admit("198.51.100.2", InferenceKind.EXPLICIT, RequestDeadline(2))

    digest = client_identity_digest(
        configured.hmac_secret.get_secret_value(), "198.51.100.2"
    )
    assert await redis_client.exists(
        f"routellm:rate:{{admission}}:client:{digest}"
    ) == 0


@pytest.mark.asyncio
async def test_ttl_recovers_capacity(redis_client):
    limiter = RedisRateLimiter(
        redis_client, settings(client_capacity=1, global_capacity=1)
    )
    await limiter.admit("192.0.2.1", InferenceKind.EXPLICIT, RequestDeadline(2))
    with pytest.raises(RateLimitExceededError):
        await limiter.admit("192.0.2.1", InferenceKind.EXPLICIT, RequestDeadline(2))
    await asyncio.sleep(1.05)
    await limiter.admit("192.0.2.1", InferenceKind.EXPLICIT, RequestDeadline(2))
