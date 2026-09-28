"""Offline Phase 10 bounded validation-escalation tests."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import AdaptiveInferenceService
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import (
    CompletionRejectedError,
    GatewayError,
    GatewayErrorCategory,
    ProviderFailureError,
    ResponseValidationError,
)
from adaptive_llm_gateway.models import (
    InferenceRequest,
    InferenceResponse,
    ModelConfig,
    TerminationReason,
)
from adaptive_llm_gateway.providers.base import LLMProvider
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.registry import ModelRegistry
from adaptive_llm_gateway.routing.policy import ModelAcceptabilityPrediction
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.telemetry.contracts import AdaptiveExecutionTelemetry
from adaptive_llm_gateway.validation import (
    ValidationContract,
    ValidationFailure,
    ValidationFailureCode,
    ValidationResult,
    ValidationStatus,
)


def configured_model(model_id: str, price: str) -> ModelConfig:
    return ModelConfig(
        model_id=model_id,
        provider="scenario",
        provider_model_name=f"fixture/{model_id}",
        input_cost_per_1m_tokens=price,
        output_cost_per_1m_tokens=price,
        context_window=10_000,
    )


def completion(model: ModelConfig, text: str, *, latency: float = 1,
               cost: str = "0.001", termination=TerminationReason.COMPLETE):
    return InferenceResponse(
        text=text,
        model_id=model.model_id,
        provider=model.provider,
        input_tokens=2,
        output_tokens=1,
        latency_ms=latency,
        estimated_cost_usd=Decimal(cost),
        termination_reason=termination,
        provider_termination_reason=("length" if termination is TerminationReason.LENGTH else "stop"),
    )


class ScenarioProvider(LLMProvider):
    def __init__(self, model, outcomes, calls):
        self.model = model
        self.outcomes = outcomes
        self.calls = calls

    async def generate(self, request):
        self.calls.append((self.model.model_id, request))
        outcome = self.outcomes[self.model.model_id]
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(self.model)
        return completion(self.model, outcome)


class Predictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = []

    def predict(self, features, candidates):
        self.calls.append((features, tuple(candidates)))
        return tuple(
            ModelAcceptabilityPrediction(
                model_id=model.model_id,
                predicted_acceptability=self.probabilities[model.model_id],
            )
            for model in candidates
        )


class Telemetry:
    def __init__(self):
        self.attempts = []
        self.executions: list[AdaptiveExecutionTelemetry] = []

    async def record(self, event):
        self.attempts.append(event)

    async def record_adaptive_execution(self, event):
        self.executions.append(event)


def harness(probabilities, outcomes, *, prices=None, max_attempts=3,
            validator=None, telemetry=None):
    prices = prices or {model_id: str(index + 1) for index, model_id in enumerate(probabilities)}
    models = tuple(configured_model(model_id, prices[model_id]) for model_id in probabilities)
    registry = ModelRegistry()
    for model in models:
        registry.register(model)
    calls = []
    resolver = ProviderResolver()
    resolver.register(
        "scenario",
        lambda model: ScenarioProvider(model, outcomes, calls),
    )
    telemetry = telemetry or Telemetry()
    inference = InferenceService(registry, resolver, telemetry=telemetry)
    predictor = Predictor(probabilities)
    adaptive = AdaptiveInferenceService(
        inference,
        RoutingDecisionService(predictor),
        validator=validator,
        max_validation_attempts=max_attempts,
    )
    return adaptive, models, calls, predictor, telemetry


async def run(adaptive, models, *, threshold=.8, contract=None):
    return await adaptive.generate(
        InferenceRequest(prompt="private prompt", max_output_tokens=8),
        category="qa",
        quality_threshold=threshold,
        candidate_model_ids=tuple(model.model_id for model in models),
        validation=contract or ValidationContract(format="json"),
        request_id="correlation-1",
    )


@pytest.mark.asyncio
async def test_initial_pass_routes_and_predicts_once():
    adaptive, models, calls, predictor, telemetry = harness(
        {"cheap": .9, "strong": .95}, {"cheap": "{}", "strong": "{}"}
    )
    result = await run(adaptive, models)
    assert [model_id for model_id, _ in calls] == ["cheap"]
    assert len(predictor.calls) == 1
    assert result.routing_decision.selected_model_id == "cheap"
    assert result.response.model_id == "cheap"
    assert result.execution.escalated is False
    assert telemetry.executions[0].returned_model_id == "cheap"


@pytest.mark.asyncio
async def test_one_escalation_recovers_with_next_cheapest_qualifier():
    adaptive, models, calls, predictor, _ = harness(
        {"cheap": .9, "middle": .81, "strong": .99},
        {"cheap": "bad", "middle": "{}", "strong": "{}"},
        prices={"cheap": "1", "middle": "2", "strong": "10"},
    )
    result = await run(adaptive, models)
    assert [item[0] for item in calls] == ["cheap", "middle"]
    assert result.response.model_id == "middle"
    assert result.execution.escalated is True
    assert len(result.execution.attempts) == 2
    assert len(predictor.calls) == 1


@pytest.mark.asyncio
async def test_two_escalations_recover_with_three_unique_calls():
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .9, "c": .9},
        {"a": "bad", "b": "bad", "c": "{}"},
    )
    result = await run(adaptive, models)
    assert [item[0] for item in calls] == ["a", "b", "c"]
    assert len({item[0] for item in calls}) == 3
    assert result.response.model_id == "c"
    assert len(result.execution.attempts) == 3


@pytest.mark.asyncio
async def test_max_attempts_exhausted_without_fourth_call():
    adaptive, models, calls, _, telemetry = harness(
        {"a": .9, "b": .9, "c": .9, "d": .9},
        {key: "bad" for key in "abcd"},
    )
    with pytest.raises(ResponseValidationError):
        await run(adaptive, models)
    assert [item[0] for item in calls] == ["a", "b", "c"]
    assert telemetry.executions[0].attempt_count == 3
    assert telemetry.executions[0].terminal_outcome == "validation_failed"


@pytest.mark.asyncio
async def test_no_unused_qualifying_candidate_stops():
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .79}, {"a": "bad", "b": "{}"}
    )
    with pytest.raises(ResponseValidationError):
        await run(adaptive, models)
    assert [item[0] for item in calls] == ["a"]


@pytest.mark.asyncio
async def test_initial_routing_fallback_never_escalates_below_threshold():
    adaptive, models, calls, _, _ = harness(
        {"cheap": .4, "strong": .6}, {"cheap": "{}", "strong": "bad"}
    )
    with pytest.raises(ResponseValidationError):
        await run(adaptive, models, threshold=.9)
    assert [item[0] for item in calls] == ["strong"]


@pytest.mark.asyncio
async def test_initial_provider_failure_is_preserved():
    failure = ProviderFailureError("controlled")
    adaptive, models, calls, _, telemetry = harness(
        {"a": .9, "b": .9}, {"a": failure, "b": "{}"}
    )
    with pytest.raises(ProviderFailureError) as caught:
        await run(adaptive, models)
    assert caught.value is failure
    assert [item[0] for item in calls] == ["a"]
    assert telemetry.executions[0].terminal_outcome == "provider_failure"


@pytest.mark.asyncio
async def test_escalated_provider_failure_stops_before_third_attempt():
    failure = ProviderFailureError("controlled")
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .9, "c": .9},
        {"a": "bad", "b": failure, "c": "{}"},
    )
    with pytest.raises(ProviderFailureError):
        await run(adaptive, models)
    assert [item[0] for item in calls] == ["a", "b"]


@pytest.mark.asyncio
async def test_nonrecoverable_validation_failure_stops_immediately():
    class Nonrecoverable:
        def validate(self, response, context):
            return ValidationResult(
                status=ValidationStatus.FAILED,
                failures=(ValidationFailure(
                    code=ValidationFailureCode.INVALID_JSON,
                    recoverable_by_escalation=False,
                ),),
            )
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .9}, {"a": "bad", "b": "{}"},
        validator=Nonrecoverable(),
    )
    with pytest.raises(ResponseValidationError):
        await run(adaptive, models)
    assert [item[0] for item in calls] == ["a"]


@pytest.mark.asyncio
async def test_deterministic_tie_order_uses_model_id():
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "c": .9, "b": .9},
        {"a": "bad", "b": "{}", "c": "{}"},
        prices={"a": "1", "b": "2", "c": "2"},
    )
    result = await run(adaptive, models)
    assert [item[0] for item in calls] == ["a", "b"]
    assert result.response.model_id == "b"


@pytest.mark.asyncio
async def test_contract_absent_preserves_single_attempt_and_payload():
    adaptive, models, calls, predictor, telemetry = harness(
        {"a": .9, "b": .9}, {"a": "bad", "b": "{}"}
    )
    request = InferenceRequest(prompt="private prompt", max_output_tokens=8)
    result = await adaptive.generate(
        request,
        category="qa",
        quality_threshold=.8,
        candidate_model_ids=tuple(model.model_id for model in models),
        request_id="correlation-1",
    )
    assert [item[0] for item in calls] == ["a"]
    assert calls[0][1] is request
    assert len(predictor.calls) == 1
    assert result.execution is None
    assert telemetry.executions == []


@pytest.mark.asyncio
async def test_typed_length_completion_escalates_without_string_parsing():
    def length(model):
        normalized = completion(
            model, "private partial", latency=2, cost="0.002",
            termination=TerminationReason.LENGTH,
        )
        return CompletionRejectedError(
            GatewayErrorCategory.OUTPUT_BUDGET_EXHAUSTION,
            completion=normalized,
            diagnostics={"safe": "metadata"},
        )
    adaptive, models, calls, _, telemetry = harness(
        {"a": .9, "b": .9}, {"a": length(configured_model("a", "1")), "b": "{}"}
    )
    result = await run(adaptive, models)
    assert [item[0] for item in calls] == ["a", "b"]
    assert result.execution.attempts[0].validation.failure_codes == ("output_truncated", "invalid_json")
    assert telemetry.attempts[0].estimated_cost_usd == Decimal("0.002")


@pytest.mark.asyncio
async def test_accounting_sums_failed_and_returned_attempts():
    adaptive, models, _, _, telemetry = harness(
        {"a": .9, "b": .9},
        {
            "a": lambda model: completion(model, "bad", latency=2, cost="0.003"),
            "b": lambda model: completion(model, "{}", latency=5, cost="0.007"),
        },
    )
    result = await run(adaptive, models)
    assert result.execution.total_estimated_cost_usd == Decimal("0.010")
    assert result.execution.total_latency_ms == 7
    summary = telemetry.executions[0]
    assert summary.cumulative_known_cost_usd == Decimal("0.010")
    assert summary.cost_complete is True
    assert summary.cumulative_latency_ms == 7


def test_http_execution_metadata_terminal_error_and_privacy():
    telemetry = Telemetry()
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .9},
        {"a": "private failed response", "b": "{}"},
        telemetry=telemetry,
    )
    runtime = AdaptiveRuntime(adaptive, tuple(model.model_id for model in models))
    app = create_app(adaptive._inference_service, runtime)
    payload = {
        "prompt": "private prompt",
        "category": "qa",
        "quality_threshold": .8,
        "validation": {"format": "json", "required_fields": ["private_field"]},
    }
    with TestClient(app) as client:
        failed = client.post("/v1/inference/adaptive", json=payload)
    assert failed.status_code == 502
    assert failed.json()["error"]["code"] == "response_validation_failed"
    assert "private" not in failed.text
    assert all("private" not in event.model_dump_json() for event in telemetry.executions)
    assert all("private" not in repr(event) for event in telemetry.attempts)
    assert len(calls) == 2


@pytest.mark.parametrize("value", [0, 4, True, 1.5])
def test_attempt_limit_is_trusted_and_bounded(value):
    with pytest.raises(ValueError):
        harness({"a": .9}, {"a": "{}"}, max_attempts=value)


@pytest.mark.asyncio
async def test_configured_one_attempt_limit_stops_after_initial_failure():
    adaptive, models, calls, _, _ = harness(
        {"a": .9, "b": .9}, {"a": "bad", "b": "{}"}, max_attempts=1
    )
    with pytest.raises(ResponseValidationError):
        await run(adaptive, models)
    assert [item[0] for item in calls] == ["a"]
