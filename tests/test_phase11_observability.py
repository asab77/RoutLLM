import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr, ValidationError

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import AdaptiveInferenceService
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import (
    CompletionRejectedError,
    GatewayError,
    GatewayErrorCategory,
    RateLimitExceededError,
    ResponseValidationError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.observability import (
    JsonEventLogger,
    Observability,
    ObservabilitySettings,
)
from adaptive_llm_gateway.providers.base import LLMProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.rate_limit import InferenceKind, RateLimitSettings, RedisRateLimiter
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.request_deadline import RequestDeadline
from adaptive_llm_gateway.routing.policy import ModelAcceptabilityPrediction
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.validation import (
    ValidationContract,
    ValidationFailure,
    ValidationFailureCode,
    ValidationResult,
    ValidationStatus,
)


TOKEN = "test-observability-token-at-least-32-characters"


class LogCapture:
    def __init__(self):
        self.messages = []

    def log(self, level, message):
        self.messages.append(message)


def observable(*, metrics_enabled=False, capture=None):
    return Observability(
        ObservabilitySettings(
            metrics_enabled=metrics_enabled,
            metrics_bearer_token=SecretStr(TOKEN) if metrics_enabled else None,
        ),
        event_logger=JsonEventLogger(capture) if capture is not None else None,
    )


def samples(metrics: Observability, name: str):
    result = []
    for family in text_string_to_metric_families(metrics.render().decode()):
        for sample in family.samples:
            if sample.name == name:
                result.append(sample)
    return result


def sample_value(metrics, name, **labels):
    matches = [
        item.value for item in samples(metrics, name)
        if all(item.labels.get(key) == value for key, value in labels.items())
    ]
    return sum(matches)


def configured_model(model_id="fixture"):
    return ModelConfig(
        model_id=model_id,
        provider="fixture-provider",
        provider_model_name=f"fixture/{model_id}",
        input_cost_per_1m_tokens="1",
        output_cost_per_1m_tokens="1",
        context_window=4096,
    )


def completion(model, text="ok", cost="0.25"):
    return InferenceResponse(
        text=text,
        model_id=model.model_id,
        provider=model.provider,
        input_tokens=1,
        output_tokens=1,
        latency_ms=2,
        estimated_cost_usd=cost,
    )


class OutcomeProvider(LLMProvider):
    def __init__(self, model, outcome):
        self.model = model
        self.outcome = outcome
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        outcome = self.outcome() if callable(self.outcome) else self.outcome
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def service_with_provider(model, outcome, metrics, telemetry=None):
    registry = ModelRegistry()
    registry.register(model)
    resolver = ProviderResolver()
    provider = OutcomeProvider(model, outcome)
    resolver.register(model.provider, lambda config: provider)
    return InferenceService(
        registry, resolver, telemetry=telemetry, observability=metrics
    ), provider


def test_metrics_configuration_requires_redacted_token():
    with pytest.raises(ValidationError) as caught:
        ObservabilitySettings(metrics_enabled=True)
    assert TOKEN not in str(caught.value)
    short_secret = "distinctive-short-secret"
    with pytest.raises(ValidationError) as caught:
        ObservabilitySettings(
            metrics_enabled=True, metrics_bearer_token=SecretStr(short_secret)
        )
    assert short_secret not in str(caught.value)


def test_http_metrics_count_duration_and_matched_routes_once():
    metrics = observable()
    app = create_app(observability=metrics)

    @app.get("/items/{item_id}")
    async def item(item_id: str):
        return {"item": item_id}

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/items/private-dynamic-value").status_code == 200

    assert sample_value(
        metrics,
        "routellm_http_requests_total",
        route="/health",
        method="GET",
        status_class="2xx",
        outcome="success",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_http_request_duration_seconds_count",
        route="/health",
        method="GET",
        status_class="2xx",
    ) == 1
    assert sample_value(
        metrics, "routellm_http_requests_in_progress", route="/health"
    ) == 0
    exposition = metrics.render().decode()
    assert 'route="/items/{item_id}"' in exposition
    assert "private-dynamic-value" not in exposition


def test_http_exception_produces_one_server_error_metric():
    metrics = observable()
    app = create_app(observability=metrics)

    @app.get("/controlled-error")
    async def controlled_error():
        raise RuntimeError("private exception body")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/controlled-error").status_code == 500
    assert sample_value(
        metrics,
        "routellm_http_requests_total",
        route="/controlled-error",
        status_class="5xx",
        outcome="server_error",
    ) == 1


def test_metrics_failure_cannot_fail_http_request():
    metrics = observable()
    metrics.http_requests.labels = Mock(
        side_effect=RuntimeError("simulated metrics failure")
    )
    with TestClient(create_app(observability=metrics)) as client:
        assert client.get("/health").status_code == 200


@pytest.mark.asyncio
async def test_cancellation_does_not_leave_in_progress_gauge_stuck():
    metrics = observable()
    app = create_app(observability=metrics)
    entered = asyncio.Event()

    @app.get("/blocking")
    async def blocking():
        entered.set()
        await asyncio.Event().wait()

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/blocking"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert sample_value(
            metrics, "routellm_http_requests_in_progress", route="/blocking"
        ) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sample_value(
        metrics, "routellm_http_requests_in_progress", route="/blocking"
    ) == 0
    assert sample_value(
        metrics,
        "routellm_http_requests_total",
        route="/blocking",
        status_class="cancelled",
        outcome="cancelled",
    ) == 1


@pytest.mark.asyncio
async def test_provider_attempt_latency_and_known_cost_are_counted_once():
    metrics = observable()
    model = configured_model()
    service, provider = service_with_provider(model, completion(model), metrics)
    await service.generate(model.model_id, InferenceRequest(prompt="private"))
    assert provider.calls == 1
    labels = {"provider": model.provider, "model": model.model_id, "outcome": "success"}
    assert sample_value(metrics, "routellm_provider_attempts_total", **labels) == 1
    assert sample_value(
        metrics, "routellm_provider_attempt_duration_seconds_count", **labels
    ) == 1
    assert sample_value(
        metrics,
        "routellm_estimated_cost_usd_total",
        request_mode="explicit",
        model=model.model_id,
    ) == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_provider_error_category_is_bounded_and_unknown_cost_is_absent():
    metrics = observable()
    model = configured_model()
    service, _ = service_with_provider(
        model, GatewayError(GatewayErrorCategory.TIMEOUT), metrics
    )
    with pytest.raises(GatewayError):
        await service.generate(model.model_id, InferenceRequest(prompt="private"))
    assert sample_value(
        metrics,
        "routellm_provider_attempts_total",
        provider=model.provider,
        model=model.model_id,
        outcome="error",
        error_category="gateway_timeout",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_estimated_cost_usd_total",
        request_mode="explicit",
        model=model.model_id,
    ) == 0


@pytest.mark.asyncio
async def test_rejected_known_completion_contributes_cost_once():
    metrics = observable()
    model = configured_model()
    rejected = CompletionRejectedError(
        GatewayErrorCategory.EMPTY_RESPONSE,
        completion=completion(model, text="", cost="0.40"),
    )
    service, _ = service_with_provider(model, rejected, metrics)
    with pytest.raises(CompletionRejectedError):
        await service.generate(model.model_id, InferenceRequest(prompt="private"))
    assert sample_value(
        metrics,
        "routellm_estimated_cost_usd_total",
        request_mode="explicit",
        model=model.model_id,
    ) == pytest.approx(0.40)


class FixedPredictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, features, candidates):
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=item.model_id,
                predicted_acceptability=self.probabilities[item.model_id],
            )
            for item in candidates
        )


class SequencedValidator:
    def __init__(self, results):
        self.results = iter(results)

    def validate(self, response, context):
        return next(self.results)


def failed_validation():
    return ValidationResult(
        status=ValidationStatus.FAILED,
        failures=(ValidationFailure(
            code=ValidationFailureCode.INVALID_JSON,
            recoverable_by_escalation=True,
        ),),
    )


def adaptive_service(metrics, validation_results, probabilities=None):
    models = tuple(configured_model(name) for name in ("a", "b", "c"))
    registry = ModelRegistry()
    outcomes = {}
    resolver = ProviderResolver()
    providers = {}
    for model in models:
        registry.register(model)
        outcomes[model.model_id] = completion(model, text=f'{{"model":"{model.model_id}"}}')
    resolver.register(
        "fixture-provider",
        lambda config: providers.setdefault(
            config.model_id, OutcomeProvider(config, outcomes[config.model_id])
        ),
    )
    inference = InferenceService(registry, resolver, observability=metrics)
    predictor = FixedPredictor(probabilities or {item.model_id: 0.9 for item in models})
    adaptive = AdaptiveInferenceService(
        inference,
        RoutingDecisionService(predictor),
        validator=SequencedValidator(validation_results),
    )
    return adaptive, models


@pytest.mark.asyncio
async def test_routing_validation_escalation_recovery_and_attempts_are_counted_once():
    metrics = observable()
    adaptive, models = adaptive_service(
        metrics,
        [failed_validation(), ValidationResult(status=ValidationStatus.PASSED)],
    )
    result = await adaptive.generate(
        InferenceRequest(prompt="private"),
        category="qa",
        quality_threshold=0.8,
        candidate_model_ids=tuple(item.model_id for item in models),
        validation=ValidationContract(format="json"),
        request_id="request-safe-id",
    )
    assert result.response.model_id == "b"
    assert sample_value(
        metrics,
        "routellm_routing_decisions_total",
        selected_model="a",
        reason="quality_threshold_met",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_validation_failures_total",
        model="a",
        reason="invalid_json",
        attempt="1",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_adaptive_requests_total",
        terminal_outcome="returned",
        escalated="true",
        recovered="true",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_adaptive_attempts_sum",
        terminal_outcome="returned",
    ) == 2
    assert sample_value(
        metrics,
        "routellm_estimated_cost_usd_total",
        request_mode="adaptive",
        model="a",
    ) == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_fallback_and_three_attempt_failure_have_one_terminal_record():
    fallback_metrics = observable()
    fallback, models = adaptive_service(
        fallback_metrics,
        [],
        probabilities={"a": 0.1, "b": 0.2, "c": 0.3},
    )
    await fallback.generate(
        InferenceRequest(prompt="private"),
        category="qa",
        quality_threshold=0.8,
        candidate_model_ids=tuple(item.model_id for item in models),
        request_id="fallback-id",
    )
    assert sample_value(
        fallback_metrics,
        "routellm_routing_decisions_total",
        selected_model="c",
        reason="no_model_met_threshold_fallback",
    ) == 1

    metrics = observable()
    adaptive, models = adaptive_service(
        metrics, [failed_validation(), failed_validation(), failed_validation()]
    )
    with pytest.raises(ResponseValidationError):
        await adaptive.generate(
            InferenceRequest(prompt="private"),
            category="qa",
            quality_threshold=0.8,
            candidate_model_ids=tuple(item.model_id for item in models),
            validation=ValidationContract(format="json"),
            request_id="terminal-id",
        )
    assert sample_value(
        metrics,
        "routellm_adaptive_requests_total",
        terminal_outcome="validation_failed",
        escalated="true",
        recovered="false",
    ) == 1
    assert sample_value(
        metrics,
        "routellm_adaptive_attempts_sum",
        terminal_outcome="validation_failed",
    ) == 3


@pytest.mark.asyncio
async def test_telemetry_success_failure_and_deadline_skip_are_counted_once():
    model = configured_model()
    metrics = observable()
    telemetry = AsyncMock()
    service, _ = service_with_provider(model, completion(model), metrics, telemetry)
    await service.generate(model.model_id, InferenceRequest(prompt="private"))
    assert sample_value(
        metrics, "routellm_telemetry_writes_total",
        record_type="inference", outcome="success"
    ) == 1

    telemetry.record.side_effect = RuntimeError("private database error")
    await service.generate(model.model_id, InferenceRequest(prompt="private"))
    assert sample_value(
        metrics, "routellm_telemetry_writes_total",
        record_type="inference", outcome="failure"
    ) == 1

    expired = RequestDeadline(0.001)
    await asyncio.sleep(0.002)
    await service._record(AsyncMock(), expired)
    assert sample_value(
        metrics, "routellm_telemetry_writes_total",
        record_type="inference", outcome="skipped_deadline"
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_code,scope", [(1, "client"), (2, "global"), (3, "both")])
async def test_rate_limit_scope_is_bounded_and_identity_is_not_a_label(scope_code, scope):
    metrics = observable()
    redis = AsyncMock()
    redis.eval.return_value = [0, 1000, scope_code]
    settings = RateLimitSettings(
        required=True,
        redis_url=SecretStr("redis://127.0.0.1:6379/15"),
        hmac_secret=SecretStr("rate-limit-secret-at-least-32-characters"),
    )
    limiter = RedisRateLimiter(redis, settings, metrics)
    with pytest.raises(RateLimitExceededError) as caught:
        await limiter.admit(
            "203.0.113.77", InferenceKind.EXPLICIT, RequestDeadline(1)
        )
    assert caught.value.scope == scope
    assert sample_value(
        metrics,
        "routellm_rate_limit_rejections_total",
        scope=scope,
        route="/v1/inference",
    ) == 1
    assert "203.0.113.77" not in metrics.render().decode()


def test_metrics_authentication_and_secret_privacy():
    with TestClient(create_app(observability=observable())) as disabled_client:
        assert disabled_client.get("/metrics").status_code == 404

    capture = LogCapture()
    metrics = observable(metrics_enabled=True, capture=capture)
    app = create_app(observability=metrics)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get(
            "/metrics", headers={"Authorization": "Bearer wrong-secret-value"}
        ).status_code == 401
        response = client.get(
            "/metrics", headers={"Authorization": f"Bearer {TOKEN}"}
        )
    assert response.status_code == 200
    assert "routellm_http_requests_total" in response.text
    assert TOKEN not in response.text
    assert TOKEN not in "\n".join(capture.messages)
    assert "wrong-secret-value" not in "\n".join(capture.messages)


class ReadyLimiter:
    def __init__(self, available):
        self.available = available

    async def admit(self, client_identity, kind, deadline):
        return None

    async def ready(self):
        return self.available


def test_health_and_readiness_dependency_semantics():
    service = service_with_provider(
        configured_model(), completion(configured_model()), observable()
    )[0]
    service.resolver.resolve = Mock(wraps=service.resolver.resolve)
    service.telemetry = AsyncMock()
    with TestClient(create_app(service, rate_limiter=ReadyLimiter(True))) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/health").status_code == 200
    service.resolver.resolve.assert_not_called()

    with TestClient(create_app(service, rate_limiter=ReadyLimiter(False))) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    service.telemetry.summary.assert_not_awaited()


def test_structured_logs_are_json_allowlisted_and_correlated():
    capture = LogCapture()
    metrics = observable(capture=capture)
    model = configured_model()
    service, _ = service_with_provider(
        model,
        completion(model, text="private provider response"),
        metrics,
    )
    app = create_app(service, observability=metrics)
    with TestClient(app) as client:
        response = client.post(
            "/v1/inference",
            json={
                "model_id": model.model_id,
                "prompt": "private prompt",
                "max_output_tokens": 2,
            },
            headers={
                "X-Request-ID": "safe-correlation-id",
                "Authorization": "Bearer private-authorization",
                "X-Forwarded-For": "203.0.113.88",
            },
        )
    assert response.status_code == 200
    assert "safe-correlation-id" not in metrics.render().decode()
    decoded = [json.loads(message) for message in capture.messages]
    assert all({"timestamp", "level", "event"} <= set(item) for item in decoded)
    assert any(item.get("request_id") == "safe-correlation-id" for item in decoded)
    serialized = "\n".join(capture.messages)
    for forbidden in (
        "private prompt",
        "private provider response",
        "private-authorization",
        "203.0.113.88",
        "validation",
        "rate-limit-secret",
    ):
        assert forbidden not in serialized

    metrics.events.emit(
        "allowlist_test",
        prompt="must disappear",
        response="must disappear",
        authorization="must disappear",
        validation_contract={"field": "secret"},
        provider_body="must disappear",
        route="/controlled",
    )
    payload = json.loads(capture.messages[-1])
    assert set(payload) == {"timestamp", "level", "event", "route"}
