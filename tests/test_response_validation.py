"""Offline checks for the Phase 10 deterministic validation boundary."""
import asyncio
import json
from dataclasses import asdict

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from adaptive_llm_gateway.api.schemas import AdaptiveInferencePayload, InferencePayload
from adaptive_llm_gateway.application.adaptive import AdaptiveInferenceService
from adaptive_llm_gateway.errors import CompletionRejectedError, GatewayError
from adaptive_llm_gateway.models import InferenceResponse, TerminationReason
from adaptive_llm_gateway.routing.service import RoutingDecisionService
from adaptive_llm_gateway.telemetry.contracts import ValidationTelemetry
from adaptive_llm_gateway.validation import (
    DeterministicResponseValidator, ValidationContext, ValidationContract, ValidationResult,
    ValidationStatus,
)
from adaptive_llm_gateway.validation.contracts import MAX_INPUT_CHARACTERS, MAX_JSON_DEPTH, MAX_FAILURES
from test_adaptive_http import configured_app, adaptive_payload, model
from test_adaptive_inference import FixedPredictor, inference_service
from test_gateway import adapter, valid_body


def response(text="ok", termination=TerminationReason.COMPLETE):
    return InferenceResponse(text=text, model_id="test", provider="fake", input_tokens=1,
                             output_tokens=1, latency_ms=0, estimated_cost_usd="0",
                             termination_reason=termination)


def validate(text, contract=None, termination=TerminationReason.COMPLETE):
    return DeterministicResponseValidator().validate(
        response(text, termination), ValidationContext(
            contract=ValidationContract(**contract) if contract is not None else None))


@pytest.mark.parametrize("text,contract,code", [
    ("ok", {"format": "text"}, None),
    ("a", {"format": "text"}, None),
    (" \n", {"format": "text"}, "empty_response"),
    (" a ", {"format": "text", "min_characters": 2}, "minimum_characters"),
    (' {"a":1} \n', {"format": "json"}, None),
    ('{"a":', {"format": "json"}, "invalid_json"),
    ('```json\n{}\n```', {"format": "json"}, "invalid_json"),
    ('Here: {}', {"format": "json"}, "invalid_json"),
    ('\u00a0{}', {"format": "json"}, "invalid_json"),
    ('{} {}', {"format": "json"}, "invalid_json"),
    ('{"a":1,"a":2}', {"format": "json"}, "invalid_json"),
    ('{"nested":{"a":1,"a":2}}', {"format": "json"}, "invalid_json"),
    ('NaN', {"format": "json"}, "invalid_json"),
    ('Infinity', {"format": "json"}, "invalid_json"),
    ('-Infinity', {"format": "json"}, "invalid_json"),
    ('[]', {"format": "json", "root_type": "object"}, "root_type"),
    ('{}', {"format": "json", "required_fields": ["private_field"]}, "missing_field"),
    ('[]', {"format": "json", "required_fields": ["private_field"]}, "root_type"),
    ('{"a":true}', {"format": "json", "field_types": {"a": "integer"}}, "field_type"),
    ('{"a":true}', {"format": "json", "field_types": {"a": "number"}}, "field_type"),
    ('{"a":"1"}', {"format": "json", "field_types": {"a": "integer"}}, "field_type"),
    ('{"a":1.5}', {"format": "json", "field_types": {"a": "integer"}}, "field_type"),
    ('{"a":1.0}', {"format": "json", "field_types": {"a": "integer"}}, None),
    ('{}', {"format": "json", "field_types": {"a": "integer"}}, None),
    (' YES \n', {"format": "label", "allowed_labels": ["YES", "NO"]}, None),
    ('yes', {"format": "label", "allowed_labels": ["YES", "NO"]}, "label_not_allowed"),
])
def test_checks(text, contract, code):
    result = validate(text, contract)
    assert result.status == ("failed" if code else "passed")
    assert [item.code for item in result.failures] == ([code] if code else [])
    assert result.duration_ms >= 0


@pytest.mark.parametrize("value,kind", [("{}", "object"), ("[]", "array"), ('"x"', "string"),
    ("1e1000", "number"), ("1", "integer"), ("false", "boolean"), ("null", "null")])
def test_json_types(value, kind):
    assert validate(value, {"format": "json", "root_type": kind}).status == "passed"
    assert validate('{"a":' + value + '}', {"format": "json", "field_types": {"a": kind}}).status == "passed"


def test_not_run_and_truncation():
    assert validate("", termination=TerminationReason.LENGTH).status == "not_run"
    assert validate("ok", {"format": "text"}, TerminationReason.LENGTH).failures[0].code == "output_truncated"
    assert validate("ok", {"format": "text"}, TerminationReason.UNKNOWN).status == "passed"


def test_bounds_and_quoted_brackets():
    assert validate("x" * MAX_INPUT_CHARACTERS, {"format": "text"}).status == "passed"
    assert validate("x" * (MAX_INPUT_CHARACTERS + 1), {"format": "text"}).failures[0].code == "input_size_limit"
    assert validate("[" * MAX_JSON_DEPTH + "0" + "]" * MAX_JSON_DEPTH, {"format": "json"}).status == "passed"
    assert validate("[" * (MAX_JSON_DEPTH + 1), {"format": "json"}).failures[0].code == "json_depth_limit"
    assert validate(json.dumps('"' + '[' * 100 + '\\'), {"format": "json"}).status == "passed"
    result = validate("{}", {"format": "json", "required_fields": [str(i) for i in range(64)]})
    assert len(result.failures) == MAX_FAILURES


INVALID_CONTRACTS = [
    {"format": "xml"}, {"format": "text", "root_type": "object"},
    {"format": "json", "root_type": "array", "required_fields": ["a"]},
    {"format": "label"}, {"format": "text", "allowed_labels": ["x"]},
    {"format": "label", "allowed_labels": [" x "]},
    {"format": "label", "allowed_labels": ["x", "x"]},
    {"format": "json", "required_fields": ["x", "x"]},
    {"format": "json", "field_types": {"a": "date"}},
    {"format": "text", "min_characters": True},
    {"format": "text", "min_characters": MAX_INPUT_CHARACTERS + 1},
    {"format": "text", "regex": "private_regex"},
    {"format": "json", "required_fields": [str(i) for i in range(65)]},
    {"format": "json", "field_types": {str(i): "string" for i in range(65)}},
    {"format": "label", "allowed_labels": [str(i) for i in range(65)]},
    {"format": "label", "allowed_labels": ["x" * 129]},
]


@pytest.mark.parametrize("contract", INVALID_CONTRACTS)
def test_contract_rejected_and_http_sanitized(contract):
    with pytest.raises(ValidationError):
        ValidationContract(**contract)
    calls = []
    app = configured_app({"a": .9}, [model("a")], calls)
    with TestClient(app) as client:
        result = client.post("/v1/inference/adaptive", json=adaptive_payload(validation=contract))
    assert result.status_code == 422
    assert result.json()["error"] == {"code": "invalid_request", "message": "Request body does not match the inference schema."}
    assert calls == []


def test_sanitized_results_telemetry_and_internal_error():
    result = validate('{"private_field":"private_response"}', {
        "format": "json", "required_fields": ["private_missing"], "field_types": {"private_field": "integer"}})
    encoded = ValidationTelemetry.from_result(result).model_dump_json()
    assert "private" not in encoded
    class Broken(DeterministicResponseValidator):
        def _check(self, response, context):
            raise RuntimeError("private_parser_secret")
    result = Broken().validate(response(), ValidationContext(contract=ValidationContract(format="text")))
    assert result.status == "error"
    assert "private" not in result.model_dump_json()
    assert ValidationTelemetry.from_result(result).failure_codes == ("validator_error",)


def test_runtime_parity_and_injection():
    models, calls = [model("a"), model("b", price="2")], []
    predictor = FixedPredictor({"a": .9, "b": .95})
    events = []
    class Repository:
        async def record(self, event):
            events.append(event)
    service = inference_service(models, calls, telemetry=Repository())
    class SpyValidator:
        def __init__(self):
            self.contexts = []
        def validate(self, response, context):
            self.contexts.append(context)
            return ValidationResult(status=ValidationStatus.PASSED)
    validator = SpyValidator()
    adaptive = AdaptiveInferenceService(service, RoutingDecisionService(predictor), validator=validator)
    basic = AdaptiveInferencePayload(**adaptive_payload())
    constrained = AdaptiveInferencePayload(**adaptive_payload(validation={"format": "json", "required_fields": ["private_field"]}))
    assert basic.to_domain() == constrained.to_domain()
    async def run():
        results = []
        for payload in (basic, constrained):
            results.append(await adaptive.generate(payload.to_domain(), category=payload.category,
                quality_threshold=payload.quality_threshold, candidate_model_ids=["a", "b"],
                validation=payload.validation, request_id="same-id"))
        return results
    first, second = asyncio.run(run())
    assert first.routing_decision == second.routing_decision
    assert first.routing_decision.quality_threshold == .8
    assert predictor.calls[0] == predictor.calls[1]
    assert not predictor.calls[1][0].requests_structured_output
    assert calls == ["a", "a"]
    assert len(validator.contexts) == 1
    assert validator.contexts[0].contract == constrained.validation
    assert all(event.request_id == "same-id" for event in events)
    assert "private_field" not in str([asdict(event) for event in events])


def test_http_validation_enforcement_and_legacy_response():
    calls = []
    app = configured_app({"a": .9, "b": .95}, [model("a"), model("b")], calls)
    with TestClient(app) as client:
        before = client.post("/v1/inference/adaptive", json=adaptive_payload(), headers={"X-Request-ID": "same"})
        after = client.post("/v1/inference/adaptive", json=adaptive_payload(validation={"format": "text"}), headers={"X-Request-ID": "same"})
    assert before.status_code == after.status_code == 200
    after_data = after.json()
    execution = after_data.pop("execution")
    assert before.json() == after_data
    assert execution["attempts"] == 1 and execution["escalated"] is False
    assert len(calls) == 2
    with pytest.raises(ValidationError):
        InferencePayload(model_id="a", prompt="x", validation={"format": "json"})


@pytest.mark.asyncio
@pytest.mark.parametrize("text,finish,category", [(" ", "stop", "gateway_empty_response"),
    ("private_partial", "length", "output_budget_exhaustion")])
async def test_typed_completion_legacy_error_and_wire_isolation(text, finish, category):
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        body = valid_body()
        body["choices"][0] = {"message": {"content": text}, "finish_reason": finish}
        return httpx.Response(200, json=body)
    payload = AdaptiveInferencePayload(**adaptive_payload(validation={"format": "json", "required_fields": ["private_field"]}))
    with pytest.raises(CompletionRejectedError) as caught:
        await adapter(handler).generate(payload.to_domain())
    error = caught.value
    assert isinstance(error, GatewayError)
    assert str(error) == category and error.category == category
    assert error.completion.text == text
    assert error.completion.input_tokens == 10
    assert "private" not in str(error.diagnostics) and "private" not in repr(error)
    assert len(seen) == 1 and "validation" not in seen[0]
    assert "private_field" not in json.dumps(seen)
    result = DeterministicResponseValidator().validate(error.completion, ValidationContext(contract=payload.validation))
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_rejected_completion_service_telemetry_and_public_error_remain_private():
    from adaptive_llm_gateway.api.app import create_app
    from adaptive_llm_gateway.application.service import InferenceService
    from adaptive_llm_gateway.providers.gateway_config import REAL_MODELS
    from adaptive_llm_gateway.providers.resolver import ProviderResolver
    from adaptive_llm_gateway.registry import ModelRegistry

    events = []
    class Repository:
        async def record(self, event):
            events.append(event)
    def handler(request):
        body = valid_body()
        body["choices"][0] = {"message": {"content": "private_response"}, "finish_reason": "length"}
        return httpx.Response(200, json=body)
    registry = ModelRegistry()
    registry.register(REAL_MODELS[0])
    resolver = ProviderResolver()
    resolver.register("vercel", lambda model: adapter(handler, model=model))
    service = InferenceService(registry, resolver, telemetry=Repository())
    with TestClient(create_app(service)) as client:
        result = client.post("/v1/inference", json={"model_id": REAL_MODELS[0].model_id, "prompt": "private_prompt"})
    assert result.status_code == 502
    assert result.json()["error"]["code"] == "output_budget_exhaustion"
    assert "private" not in result.text
    assert len(events) == 1 and not events[0].success
    assert events[0].error_category == "output_budget_exhaustion"
    assert "private" not in str(asdict(events[0]))
