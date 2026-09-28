import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from adaptive_llm_gateway import runtime
from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import AdaptiveInferenceService
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import configure_gateway, create_development_service
from adaptive_llm_gateway.errors import (
    GatewayError,
    GatewayErrorCategory,
    InferenceDeadlineExceededError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers.base import LLMProvider
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.request_deadline import (
    InferenceDeadlineSettings,
    RequestDeadline,
)
from adaptive_llm_gateway.routing.policy import ModelAcceptabilityPrediction
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.validation import (
    ValidationContract,
    ValidationFailure,
    ValidationFailureCode,
    ValidationResult,
    ValidationStatus,
)


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def model(model_id: str = "fixture", *, price: str = "1") -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider="budget",
        provider_model_name=f"fixture/{model_id}",
        input_cost_per_1m_tokens=price,
        output_cost_per_1m_tokens=price,
        context_window=4096,
    )


def response(config: ModelConfig, text: str = "{}") -> InferenceResponse:
    return InferenceResponse(
        text=text,
        model_id=config.model_id,
        provider=config.provider,
        input_tokens=1,
        output_tokens=1,
        latency_ms=1,
        estimated_cost_usd="0.000002",
    )


class BudgetProvider(LLMProvider):
    def __init__(self, config, outcomes, calls, budgets, clock=None) -> None:
        self.config = config
        self.outcomes = outcomes
        self.calls = calls
        self.budgets = budgets
        self.clock = clock

    async def generate(self, request):
        return response(self.config, self.outcomes[self.config.model_id])

    async def generate_with_timeout(self, request, *, timeout_seconds):
        self.calls.append(self.config.model_id)
        self.budgets.append(timeout_seconds)
        outcome = self.outcomes[self.config.model_id]
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return response(self.config, outcome)


def service_for(models, outcomes, calls=None, budgets=None, *, deadline=60):
    calls = calls if calls is not None else []
    budgets = budgets if budgets is not None else []
    registry = ModelRegistry()
    for item in models:
        registry.register(item)
    resolver = ProviderResolver()
    resolver.register(
        "budget", lambda config: BudgetProvider(config, outcomes, calls, budgets)
    )
    return InferenceService(
        registry, resolver, inference_deadline_seconds=deadline
    ), calls, budgets


class Predictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.inputs = []

    def predict(self, features, candidates):
        self.inputs.append(features)
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=item.model_id,
                predicted_acceptability=self.probabilities[item.model_id],
            )
            for item in candidates
        )


class AdvancingValidator:
    def __init__(self, clock, advances, *, pass_on=None):
        self.clock = clock
        self.advances = iter(advances)
        self.calls = 0
        self.pass_on = pass_on

    def validate(self, response, context):
        self.calls += 1
        self.clock.advance(next(self.advances))
        if self.calls == self.pass_on:
            return ValidationResult(status=ValidationStatus.PASSED)
        return ValidationResult(
            status=ValidationStatus.FAILED,
            failures=(ValidationFailure(
                code=ValidationFailureCode.INVALID_JSON,
                recoverable_by_escalation=True,
            ),),
        )


def adaptive_harness(clock, advances, *, pass_on=None):
    models = (model("a", price="1"), model("b", price="2"), model("c", price="3"))
    inference, calls, budgets = service_for(
        models, {item.model_id: "bad" for item in models}
    )
    predictor = Predictor({item.model_id: .9 for item in models})
    adaptive = AdaptiveInferenceService(
        inference,
        RoutingDecisionService(predictor),
        validator=AdvancingValidator(clock, advances, pass_on=pass_on),
    )
    return adaptive, models, calls, budgets, predictor


@pytest.mark.parametrize("value", [0, -1, 301, float("inf"), float("nan")])
def test_deadline_configuration_rejects_invalid_values(value):
    with pytest.raises(ValidationError):
        InferenceDeadlineSettings(seconds=value)


def test_deadline_configuration_reads_environment(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEADLINE_SECONDS", "12.5")
    assert InferenceDeadlineSettings.from_environment().seconds == 12.5


@pytest.mark.asyncio
async def test_explicit_success_uses_remaining_deadline_budget():
    config = model()
    service, calls, budgets = service_for((config,), {config.model_id: "ok"})
    clock = Clock()
    result = await service.generate(
        config.model_id,
        InferenceRequest(prompt="private"),
        deadline=RequestDeadline(5, clock=clock),
    )
    assert result.text == "ok"
    assert calls == [config.model_id]
    assert budgets == [5]


@pytest.mark.asyncio
async def test_expired_explicit_deadline_never_resolves_or_calls_provider():
    config = model()
    service, calls, _ = service_for((config,), {config.model_id: "unused"})
    clock = Clock()
    deadline = RequestDeadline(1, clock=clock)
    clock.advance(1)
    with pytest.raises(InferenceDeadlineExceededError):
        await service.generate(
            config.model_id, InferenceRequest(prompt="private"), deadline=deadline
        )
    assert calls == []


@pytest.mark.asyncio
async def test_telemetry_timeout_is_constrained_by_remaining_deadline(caplog):
    cleaned = asyncio.Event()

    class SlowTelemetry:
        async def record(self, event):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

    service = create_development_service(inference_deadline_seconds=.02)
    service.telemetry = SlowTelemetry()
    service.telemetry_timeout = 10
    result = await asyncio.wait_for(
        service.generate("fake-small", InferenceRequest(prompt="ok")), timeout=.2
    )
    assert result.text == "ok"
    assert cleaned.is_set()
    assert "telemetry_write_failed" in caplog.text


@pytest.mark.asyncio
async def test_telemetry_is_skipped_after_deadline_expires():
    clock = Clock()
    config = model()
    calls, budgets = [], []

    class ExpiringProvider(BudgetProvider):
        async def generate_with_timeout(self, request, *, timeout_seconds):
            calls.append(self.config.model_id)
            clock.advance(1)
            return response(self.config, "ok")

    registry = ModelRegistry()
    registry.register(config)
    resolver = ProviderResolver()
    resolver.register(
        "budget", lambda item: ExpiringProvider(item, {}, calls, budgets)
    )
    telemetry = AsyncMock()
    service = InferenceService(registry, resolver, telemetry=telemetry)
    result = await service.generate(
        config.model_id,
        InferenceRequest(prompt="private"),
        deadline=RequestDeadline(1, clock=clock),
    )
    assert result.text == "ok"
    telemetry.record.assert_not_awaited()


@pytest.mark.asyncio
async def test_adaptive_escalation_consumes_one_deadline_and_routes_once():
    clock = Clock()
    adaptive, models, calls, budgets, predictor = adaptive_harness(
        clock, [.2, .2, 0], pass_on=3
    )
    result = await adaptive.generate(
        InferenceRequest(prompt="private"),
        category="qa",
        quality_threshold=.8,
        candidate_model_ids=tuple(item.model_id for item in models),
        validation=ValidationContract(format="json"),
        deadline=RequestDeadline(1, clock=clock),
    )
    assert result.response.model_id == "c"
    assert calls == ["a", "b", "c"]
    assert budgets == [1, pytest.approx(.8), pytest.approx(.6)]
    assert len(predictor.inputs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "advances,expected_calls",
    [([1.0], ["a"]), ([.4, .6], ["a", "b"])],
)
async def test_deadline_stops_second_or_third_adaptive_attempt(
    advances, expected_calls
):
    clock = Clock()
    adaptive, models, calls, _, _ = adaptive_harness(clock, advances)
    with pytest.raises(InferenceDeadlineExceededError):
        await adaptive.generate(
            InferenceRequest(prompt="private"),
            category="qa",
            quality_threshold=.8,
            candidate_model_ids=tuple(item.model_id for item in models),
            validation=ValidationContract(format="json"),
            deadline=RequestDeadline(1, clock=clock),
        )
    assert calls == expected_calls


def test_whole_deadline_has_distinct_sanitized_http_error():
    class Slow(LLMProvider):
        async def generate(self, request):
            await asyncio.Event().wait()

    config = model()
    registry = ModelRegistry()
    registry.register(config)
    resolver = ProviderResolver()
    resolver.register("budget", lambda item: Slow())
    service = InferenceService(
        registry, resolver, inference_deadline_seconds=.01
    )
    with TestClient(create_app(service)) as client:
        result = client.post(
            "/v1/inference", json={"model_id": config.model_id, "prompt": "private"}
        )
    assert result.status_code == 504
    assert result.json()["error"]["code"] == "inference_deadline_exceeded"
    assert "private" not in result.text


def test_provider_timeout_remains_distinct_from_whole_deadline():
    async def handler(request):
        await asyncio.sleep(.02)

    settings = GatewaySettings(
        api_key=SecretStr("test-only-key"), timeout_seconds=.005
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = create_development_service(inference_deadline_seconds=1)
    configure_gateway(service, settings, client=client)
    with TestClient(create_app(service)) as api:
        result = api.post(
            "/v1/inference", json={"model_id": "gateway-nano", "prompt": "private"}
        )
    asyncio.run(client.aclose())
    assert result.status_code == 504
    assert result.json()["error"]["code"] == "gateway_timeout"


@pytest.mark.asyncio
async def test_client_cancellation_propagates_unchanged():
    config = model()
    service, _, _ = service_for(
        (config,), {config.model_id: asyncio.CancelledError()}
    )
    with pytest.raises(asyncio.CancelledError):
        await service.generate(config.model_id, InferenceRequest(prompt="private"))


@pytest.mark.asyncio
async def test_shared_http_client_handles_sequential_and_concurrent_calls():
    arrived = 0
    both_arrived = asyncio.Event()
    seen = []

    async def handler(request):
        nonlocal arrived
        arrived += 1
        seen.append(request)
        if arrived >= 3:
            both_arrived.set()
        if arrived >= 2:
            await asyncio.wait_for(both_arrived.wait(), timeout=1)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = create_development_service()
    configure_gateway(
        service,
        GatewaySettings(api_key=SecretStr("test-only-key")),
        client=client,
    )
    await service.generate("gateway-nano", InferenceRequest(prompt="one"))
    results = await asyncio.gather(*(
        service.generate("gateway-nano", InferenceRequest(prompt=value))
        for value in ("two", "three")
    ))
    assert [item.text for item in results] == ["ok", "ok"]
    assert len(seen) == 3
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_never_closes_injected_shared_client():
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    })
    service = create_development_service()
    configure_gateway(
        service,
        GatewaySettings(api_key=SecretStr("test-only-key")),
        client=client,
    )
    await service.generate("gateway-nano", InferenceRequest(prompt="private"))
    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_closes_shared_client_exactly_once(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(runtime, "create_http_client", lambda settings: client)
    async with runtime.application_service() as service:
        assert service.list_models()
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        GatewayError(GatewayErrorCategory.TIMEOUT),
        GatewayError(GatewayErrorCategory.RATE_LIMIT),
        GatewayError(GatewayErrorCategory.UPSTREAM),
    ],
)
async def test_provider_failures_are_never_retried(outcome):
    config = model()
    service, calls, _ = service_for((config,), {config.model_id: outcome})
    with pytest.raises(GatewayError) as caught:
        await service.generate(config.model_id, InferenceRequest(prompt="private"))
    assert caught.value is outcome
    assert calls == [config.model_id]
