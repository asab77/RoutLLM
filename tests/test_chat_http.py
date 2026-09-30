from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import (
    AdaptiveAttempt,
    AdaptiveExecution,
    AdaptiveInferenceResult,
)
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.application.chat import (
    DEFAULT_MODEL_ID_ENV,
    ChatConfiguration,
    build_chat_orchestration_service,
)
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import (
    InferenceDeadlineExceededError,
    RateLimitExceededError,
    ResponseValidationError,
)
from adaptive_llm_gateway.models import InferenceResponse, ModelConfig
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.rate_limit import InferenceKind
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.routing.policy import RoutingDecision, RoutingDecisionReason
from adaptive_llm_gateway.telemetry.contracts import ValidationTelemetry
from adaptive_llm_gateway.validation import ValidationStatus


def response(model_id: str = "fake-small", *, cost: str = "0.001", latency: float = 4) -> InferenceResponse:
    return InferenceResponse(
        text="answer",
        model_id=model_id,
        provider="fake",
        input_tokens=2,
        output_tokens=1,
        latency_ms=latency,
        estimated_cost_usd=cost,
    )


def decision(*, selected: str = "fake-small", fallback: bool = False) -> RoutingDecision:
    return RoutingDecision(
        selected_model_id=selected,
        selected_predicted_acceptability=0.7 if fallback else 0.9,
        selected_projected_cost_usd="0.001",
        quality_threshold=0.8,
        threshold_satisfied=not fallback,
        fallback_used=fallback,
        eligible_candidate_count=2,
        qualifying_candidate_count=0 if fallback else 1,
        reason=(
            RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK
            if fallback else RoutingDecisionReason.QUALITY_THRESHOLD_MET
        ),
    )


class RecordingLimiter:
    def __init__(self):
        self.calls = []

    async def admit(self, client_identity, kind, deadline):
        self.calls.append((client_identity, kind, deadline))

    async def ready(self):
        return True


class StubAdaptive:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def generate(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self.result


def chat_app(*, adaptive=None, limiter=None):
    service = create_development_service()
    runtime = AdaptiveRuntime(adaptive, ("fake-small", "fake-large")) if adaptive else None
    return create_app(
        service,
        runtime,
        rate_limiter=limiter,
        default_model_id="fake-small",
    )


def test_auto_uses_configured_direct_model_and_omits_adaptive_metadata():
    adaptive = StubAdaptive(
        AdaptiveInferenceResult(response=response(), routing_decision=decision())
    )
    app = chat_app(adaptive=adaptive)
    with TestClient(app) as client:
        result = client.post(
            "/v1/chat",
            json={"prompt": "hello world", "routing_mode": "auto"},
        )
    assert result.status_code == 200
    body = result.json()
    assert body["execution_mode"] == "direct"
    assert body["model_id"] == "fake-small"
    assert not ({"category", "category_source", "routing", "execution"} & body.keys())
    assert adaptive.calls == []


def test_auto_reuses_base_inference_telemetry():
    service = create_development_service()
    telemetry = AsyncMock()
    service.telemetry = telemetry
    with TestClient(create_app(service, default_model_id="fake-small")) as client:
        result = client.post(
            "/v1/chat",
            json={"prompt": "telemetry check", "routing_mode": "auto"},
            headers={"X-Request-ID": "chat-telemetry-1"},
        )
    assert result.status_code == 200
    telemetry.record.assert_awaited_once()
    event = telemetry.record.await_args.args[0]
    assert event.request_id == "chat-telemetry-1"
    assert event.model_id == "fake-small"


@pytest.mark.parametrize(
    "extra",
    [
        {"category": "coding"},
        {"validation": {"format": "text", "min_characters": 1}},
    ],
)
def test_auto_rejects_adaptive_only_fields(extra):
    with TestClient(chat_app()) as client:
        result = client.post(
            "/v1/chat",
            json={"prompt": "hello", "routing_mode": "auto", **extra},
        )
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "hello", "routing_mode": "manual"},
        {"prompt": "hello", "routing_mode": "manual", "category": "invalid"},
    ],
)
def test_manual_requires_a_canonical_category(body):
    with TestClient(chat_app()) as client:
        result = client.post("/v1/chat", json=body)
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "invalid_request"


def test_manual_delegates_with_frozen_threshold_and_preserves_routing():
    adaptive = StubAdaptive(
        AdaptiveInferenceResult(
            response=response(),
            routing_decision=decision(fallback=True),
        )
    )
    with TestClient(chat_app(adaptive=adaptive)) as client:
        result = client.post(
            "/v1/chat",
            json={"prompt": "write code", "routing_mode": "manual", "category": "coding"},
        )
    assert result.status_code == 200
    body = result.json()
    assert body["execution_mode"] == "adaptive"
    assert body["category"] == "coding"
    assert body["category_source"] == "manual"
    assert body["routing"]["selected_model_id"] == "fake-small"
    assert body["routing"]["fallback_used"] is True
    assert "execution" not in body
    assert adaptive.calls[0][1]["quality_threshold"] == 0.80
    assert "validation" not in adaptive.calls[0][1]


def test_manual_forwards_validation_and_maps_final_and_cumulative_metadata():
    contract = {"format": "json", "required_fields": ["answer"]}
    execution = AdaptiveExecution(
        attempts=(
            AdaptiveAttempt(
                attempt_number=1,
                model_id="fake-small",
                validation=ValidationTelemetry(
                    status=ValidationStatus.FAILED,
                    failure_codes=("invalid_json",),
                    duration_ms=1,
                ),
                latency_ms=4,
                estimated_cost_usd="0.001",
            ),
            AdaptiveAttempt(
                attempt_number=2,
                model_id="fake-large",
                validation=ValidationTelemetry(
                    status=ValidationStatus.PASSED,
                    duration_ms=1,
                ),
                latency_ms=9,
                estimated_cost_usd="0.004",
            ),
        ),
        escalated=True,
        validation_outcome=ValidationStatus.PASSED,
        total_estimated_cost_usd=Decimal("0.005"),
        total_latency_ms=13,
    )
    adaptive = StubAdaptive(
        AdaptiveInferenceResult(
            response=response("fake-large", cost="0.004", latency=9),
            routing_decision=decision(selected="fake-small"),
            execution=execution,
        )
    )
    with TestClient(chat_app(adaptive=adaptive)) as client:
        result = client.post(
            "/v1/chat",
            json={
                "prompt": "return json",
                "routing_mode": "manual",
                "category": "structured_json",
                "validation": contract,
            },
        )
    assert result.status_code == 200
    body = result.json()
    assert body["routing"]["selected_model_id"] == "fake-small"
    assert body["model_id"] == "fake-large"
    assert body["estimated_cost_usd"] == "0.004"
    assert body["latency_ms"] == 9
    assert body["execution"] == {
        "attempts": 2,
        "escalated": True,
        "validation_outcome": "passed",
        "total_estimated_cost_usd": "0.005",
        "total_latency_ms": 13.0,
    }
    assert adaptive.calls[0][1]["validation"].model_dump(
        mode="json", exclude_defaults=True
    ) == contract


def test_manual_validation_failure_preserves_typed_error():
    class FailedAdaptive:
        async def generate(self, request, **kwargs):
            raise ResponseValidationError("private validator detail")

    with TestClient(chat_app(adaptive=FailedAdaptive())) as client:
        result = client.post(
            "/v1/chat",
            json={
                "prompt": "return json",
                "routing_mode": "manual",
                "category": "structured_json",
                "validation": {"format": "json"},
            },
        )
    assert result.status_code == 502
    assert result.json()["error"] == {
        "code": "response_validation_failed",
        "message": "No response passed the configured validation checks.",
    }
    assert "private" not in result.text


def test_chat_uses_one_mode_appropriate_admission_and_propagates_request_id():
    adaptive = StubAdaptive(
        AdaptiveInferenceResult(response=response(), routing_decision=decision())
    )
    limiter = RecordingLimiter()
    with TestClient(chat_app(adaptive=adaptive, limiter=limiter)) as client:
        direct = client.post(
            "/v1/chat",
            json={"prompt": "hello", "routing_mode": "auto"},
            headers={"X-Request-ID": "chat-direct-1"},
        )
        manual = client.post(
            "/v1/chat",
            json={"prompt": "hello", "routing_mode": "manual", "category": "qa"},
            headers={"X-Request-ID": "chat-manual-1"},
        )
    assert direct.json()["request_id"] == direct.headers["X-Request-ID"] == "chat-direct-1"
    assert manual.json()["request_id"] == manual.headers["X-Request-ID"] == "chat-manual-1"
    assert [call[1] for call in limiter.calls] == [InferenceKind.EXPLICIT, InferenceKind.ADAPTIVE]
    assert adaptive.calls[0][1]["request_id"] == "chat-manual-1"


def test_malformed_request_id_remains_sanitized_400():
    with TestClient(chat_app()) as client:
        result = client.post(
            "/v1/chat",
            json={"prompt": "hello", "routing_mode": "auto"},
            headers={"X-Request-ID": "bad id"},
        )
    assert result.status_code == 400
    assert result.json()["error"]["code"] == "invalid_request_id"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RateLimitExceededError(retry_after_seconds=2), 429, "rate_limit_exceeded"),
        (InferenceDeadlineExceededError("private deadline detail"), 504, "inference_deadline_exceeded"),
    ],
)
def test_chat_admission_failures_are_sanitized_before_execution(error, status, code):
    class RejectingLimiter(RecordingLimiter):
        async def admit(self, client_identity, kind, deadline):
            raise error

    with TestClient(chat_app(limiter=RejectingLimiter())) as client:
        result = client.post(
            "/v1/chat", json={"prompt": "hello", "routing_mode": "auto"}
        )
    assert result.status_code == status
    assert result.json()["error"]["code"] == code
    assert "private" not in result.text


def test_provider_unavailable_remains_sanitized_503():
    registry = ModelRegistry()
    registry.register(ModelConfig(
        model_id="configured",
        provider="missing-provider",
        provider_model_name="missing",
        input_cost_per_1m_tokens="1",
        output_cost_per_1m_tokens="1",
        context_window=1024,
    ))
    service = InferenceService(registry, ProviderResolver())
    with TestClient(create_app(service, default_model_id="configured")) as client:
        result = client.post(
            "/v1/chat", json={"prompt": "hello", "routing_mode": "auto"}
        )
    assert result.status_code == 503
    assert result.json()["error"] == {
        "code": "provider_unavailable",
        "message": "No adapter is available for this model.",
    }


def test_production_lifespan_requires_default_model_configuration(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ID_ENV, raising=False)
    with pytest.raises(ValueError, match=DEFAULT_MODEL_ID_ENV):
        with TestClient(create_app()):
            pass


def test_default_model_configuration_is_required(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ID_ENV, raising=False)
    with pytest.raises(ValueError, match=DEFAULT_MODEL_ID_ENV):
        ChatConfiguration.from_environment()


def test_default_model_configuration_rejects_unknown_and_disabled_models():
    service = create_development_service()
    with pytest.raises(ValueError, match="not registered"):
        build_chat_orchestration_service(
            service, ChatConfiguration(default_model_id="missing"), None
        )

    registry = ModelRegistry()
    registry.register(ModelConfig(
        model_id="disabled",
        provider="fake",
        provider_model_name="echo-v1",
        input_cost_per_1m_tokens="1",
        output_cost_per_1m_tokens="1",
        context_window=1024,
        enabled=False,
    ))
    service.registry = registry
    with pytest.raises(ValueError, match="disabled"):
        build_chat_orchestration_service(
            service, ChatConfiguration(default_model_id="disabled"), None
        )


def test_valid_enabled_default_model_configuration_succeeds():
    service = create_development_service()
    chat = build_chat_orchestration_service(
        service, ChatConfiguration(default_model_id="fake-small"), None
    )
    assert chat.default_model_id == "fake-small"
