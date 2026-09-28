import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from redis.exceptions import ConnectionError as RedisConnectionError

from adaptive_llm_gateway import runtime
from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import (
    InferenceDeadlineExceededError,
    RateLimitExceededError,
    RateLimitUnavailableError,
)
from adaptive_llm_gateway.rate_limit import (
    InferenceKind,
    RateLimitSettings,
    RedisRateLimiter,
    client_identity_digest,
)
from adaptive_llm_gateway.request_deadline import RequestDeadline


def settings(**overrides):
    values = {
        "required": True,
        "redis_url": SecretStr("redis://127.0.0.1:6379/15"),
        "hmac_secret": SecretStr("test-secret-with-at-least-32-characters"),
    }
    values.update(overrides)
    return RateLimitSettings(**values)


def payload():
    return {
        "model_id": "fake-small",
        "prompt": "private prompt never sent to redis",
        "max_output_tokens": 2,
    }


class RecordingLimiter:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []

    async def admit(self, client_identity, kind, deadline):
        self.calls.append((client_identity, kind, deadline))
        if self.failure is not None:
            raise self.failure

    async def ready(self):
        return self.failure is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"redis_url": None},
        {"hmac_secret": None},
        {"redis_url": SecretStr("https://example.invalid")},
        {"hmac_secret": SecretStr("too-short")},
        {"adaptive_weight": 11},
    ],
)
def test_required_configuration_is_validated_and_redacted(overrides):
    with pytest.raises(ValidationError) as caught:
        settings(**overrides)
    assert "test-secret-with-at-least-32-characters" not in str(caught.value)


def test_hmac_identity_is_deterministic_distinct_and_secret_dependent():
    first = client_identity_digest("a" * 32, "203.0.113.8")
    assert first == client_identity_digest("a" * 32, "203.0.113.8")
    assert first != client_identity_digest("a" * 32, "203.0.113.9")
    assert first != client_identity_digest("b" * 32, "203.0.113.8")
    assert "203.0.113.8" not in first


def test_forwarding_headers_do_not_choose_application_identity():
    limiter = RecordingLimiter()
    with TestClient(
        create_app(create_development_service(), rate_limiter=limiter)
    ) as client:
        first = client.post(
            "/v1/inference", json=payload(), headers={"X-Forwarded-For": "1.2.3.4"}
        )
        second = client.post(
            "/v1/inference", json=payload(), headers={"X-Forwarded-For": "9.8.7.6"}
        )
    assert first.status_code == second.status_code == 200
    assert limiter.calls[0][0] == limiter.calls[1][0] == "testclient"


def test_adaptive_request_is_admitted_before_provider_execution():
    service = create_development_service()

    class AdmittedAdaptive:
        async def generate(self, request, *, deadline, **kwargs):
            response = await service.generate(
                "fake-small", request, deadline=deadline
            )
            decision = SimpleNamespace(
                selected_model_id="fake-small",
                threshold_satisfied=True,
                fallback_used=False,
                reason="quality_threshold_met",
            )
            return SimpleNamespace(
                response=response, routing_decision=decision, execution=None
            )

    limiter = RecordingLimiter()
    adaptive = AdaptiveRuntime(AdmittedAdaptive(), ("fake-small",))
    with TestClient(create_app(service, adaptive, limiter)) as client:
        response = client.post(
            "/v1/inference/adaptive",
            json={
                "prompt": "admitted adaptive request",
                "max_output_tokens": 2,
                "category": "qa",
                "quality_threshold": 0.8,
            },
        )
    assert response.status_code == 200
    assert limiter.calls[0][1] is InferenceKind.ADAPTIVE


def test_adaptive_rejection_precedes_routing_and_provider_work():
    service = create_development_service()
    adaptive_service = SimpleNamespace(generate=AsyncMock())
    adaptive = AdaptiveRuntime(adaptive_service, ("fake-small",))
    limiter = RecordingLimiter(RateLimitExceededError(retry_after_seconds=3))
    with TestClient(create_app(service, adaptive, limiter)) as client:
        response = client.post(
            "/v1/inference/adaptive",
            json={
                "prompt": "must not route",
                "max_output_tokens": 2,
                "category": "qa",
                "quality_threshold": 0.8,
            },
        )
    assert response.status_code == 429
    adaptive_service.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "failure,status,code",
    [
        (RateLimitExceededError(retry_after_seconds=7), 429, "rate_limit_exceeded"),
        (RateLimitUnavailableError("private redis endpoint"), 503, "rate_limit_unavailable"),
    ],
)
def test_rejection_is_sanitized_and_precedes_provider_work(failure, status, code):
    service = create_development_service()
    service.resolver.resolve = Mock(wraps=service.resolver.resolve)
    limiter = RecordingLimiter(failure)
    with TestClient(create_app(service, rate_limiter=limiter)) as client:
        response = client.post("/v1/inference", json=payload())
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "private redis endpoint" not in response.text
    service.resolver.resolve.assert_not_called()
    if status == 429:
        assert response.headers["Retry-After"] == "7"


def test_health_and_non_inference_routes_ignore_unavailable_redis():
    limiter = RecordingLimiter(RateLimitUnavailableError("unavailable"))
    with TestClient(
        create_app(create_development_service(), rate_limiter=limiter)
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
    assert limiter.calls == []


@pytest.mark.asyncio
async def test_explicit_and_adaptive_weights_are_forwarded_to_atomic_script():
    redis = AsyncMock()
    redis.eval.return_value = [1, 0, 8, 96]
    limiter = RedisRateLimiter(
        redis, settings(explicit_weight=2, adaptive_weight=4)
    )
    await limiter.admit("198.51.100.1", InferenceKind.EXPLICIT, RequestDeadline(1))
    await limiter.admit("198.51.100.1", InferenceKind.ADAPTIVE, RequestDeadline(1))
    assert redis.eval.await_args_list[0].args[4] == 2
    assert redis.eval.await_args_list[1].args[4] == 4


@pytest.mark.asyncio
async def test_redis_failure_and_timeout_fail_closed():
    unavailable = AsyncMock()
    unavailable.eval.side_effect = RedisConnectionError("private host")
    with pytest.raises(RateLimitUnavailableError):
        await RedisRateLimiter(unavailable, settings()).admit(
            "client", InferenceKind.EXPLICIT, RequestDeadline(1)
        )

    slow = AsyncMock()

    async def wait_forever(*args):
        await asyncio.Event().wait()

    slow.eval.side_effect = wait_forever
    with pytest.raises(RateLimitUnavailableError):
        await RedisRateLimiter(
            slow, settings(redis_timeout_seconds=0.01)
        ).admit("client", InferenceKind.EXPLICIT, RequestDeadline(1))


@pytest.mark.asyncio
async def test_redis_operation_is_bounded_by_remaining_deadline():
    slow = AsyncMock()
    cancelled = asyncio.Event()

    async def wait_forever(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    slow.eval.side_effect = wait_forever
    with pytest.raises(InferenceDeadlineExceededError):
        await RedisRateLimiter(
            slow, settings(redis_timeout_seconds=1)
        ).admit("client", InferenceKind.EXPLICIT, RequestDeadline(0.01))
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_already_expired_deadline_never_contacts_redis():
    redis = AsyncMock()
    deadline = RequestDeadline(0.001)
    await asyncio.sleep(0.002)
    with pytest.raises(InferenceDeadlineExceededError):
        await RedisRateLimiter(redis, settings()).admit(
            "client", InferenceKind.EXPLICIT, deadline
        )
    redis.eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_readiness_is_bounded_and_reports_dependency_state():
    healthy = AsyncMock()
    healthy.ping.return_value = True
    assert await RedisRateLimiter(healthy, settings()).ready() is True

    broken = AsyncMock()
    broken.ping.side_effect = RedisConnectionError("private")
    assert await RedisRateLimiter(broken, settings()).ready() is False


@pytest.mark.asyncio
async def test_lifespan_reuses_one_redis_client_and_closes_it_once(monkeypatch):
    client = AsyncMock()
    monkeypatch.setenv("RATE_LIMIT_REQUIRED", "true")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6379/15")
    monkeypatch.setenv(
        "RATE_LIMIT_HMAC_SECRET", "test-secret-with-at-least-32-characters"
    )
    # The production factory is synchronous; a plain closure also proves one call.
    calls = []

    def create(settings):
        calls.append(settings)
        return client

    monkeypatch.setattr(runtime, "create_redis_client", create)
    async with runtime.rate_limiter_service() as first:
        assert first is first
        assert len(calls) == 1
    client.aclose.assert_awaited_once()
