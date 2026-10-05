import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from adaptive_llm_gateway.benchmarks.execution_readiness import (
    gemini_confirmation_dry_run,
    load_pricing_readiness,
)
from adaptive_llm_gateway.benchmarks.models import (
    BenchmarkDataset, BenchmarkResult, BenchmarkTask,
)
from adaptive_llm_gateway.benchmarks.protocol_audit import (
    response_output_budget_exhaustion,
)
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    ROUTING_BENCHMARK_MODELS,
    ROUTING_BENCHMARK_MODELS_V15,
    load_execution_protocol,
)
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory
from adaptive_llm_gateway.models import (
    InferenceRequest, InferenceResponse, ModelConfig, TerminationReason,
)
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings, REAL_MODELS
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider


ROOT = Path("benchmarks/protocols/routing-benchmark-v1.2")
PROTOCOL_14 = ROOT / "protocol.json"
PROTOCOL_15 = ROOT / "protocol-1.5.json"
READINESS_15 = ROOT / "execution-readiness-1.5.json"
CONFIRMATION = ROOT / "gemini-budget-confirmation-1.5.json"
DATASET = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")


def _adapter(body):
    return VercelGatewayProvider(
        REAL_MODELS[0], GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
    )


def test_protocol_14_is_immutable_and_15_has_only_the_approved_reserve_change():
    old, old_models, old_sha = load_execution_protocol(PROTOCOL_14)
    new, new_models, new_sha = load_execution_protocol(PROTOCOL_15)
    assert old_sha == "f132c1eec6b1a231f6ce5ea50702b59a03e76af7d44359ba614fa0ed4ae6e565"
    assert new_sha == "fd9f60ca01e912418ba9b54804ea9a6f92cd3b326fc17d489bb7e9e51cd5521d"
    assert old_models == ROUTING_BENCHMARK_MODELS
    assert new_models == ROUTING_BENCHMARK_MODELS_V15
    old_by_id = {model.model_id: model for model in old_models}
    new_by_id = {model.model_id: model for model in new_models}
    assert new_by_id["candidate-gemini-3-flash"].output_token_policy.model_dump() == {
        "category_overrides": (),
        "reasoning_headroom_tokens": 384,
        "expected_reasoning_tokens": 256,
    }
    for model_id in set(old_by_id) - {"candidate-gemini-3-flash"}:
        assert new_by_id[model_id] == old_by_id[model_id]
    assert old["dataset_sha256"] == new["dataset_sha256"]


@pytest.mark.protected_final_data
def test_protocol_15_pricing_and_confirmation_dry_run_are_strict():
    protocol, models, digest = load_execution_protocol(PROTOCOL_15)
    readiness = load_pricing_readiness(
        READINESS_15, protocol=protocol, protocol_sha256=digest,
        candidate_models=models)
    assert readiness.status == "READY_FOR_PAID_EXECUTION"
    report = gemini_confirmation_dry_run(
        protocol_path=PROTOCOL_15, readiness_path=READINESS_15,
        dataset_path=DATASET, split_path=SPLIT,
        confirmation_path=CONFIRMATION)
    assert report["status"] == "READY_FOR_CONFIRMATION"
    assert report["maximum_candidate_calls"] == 5
    assert report["reasoning_reserve_tokens"] == 384
    assert report["astra_calls"] == report["provider_calls_made"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("raw", "normalized"), (
    ("stop", TerminationReason.COMPLETE),
    ("tool_calls", TerminationReason.TOOL_CALL),
    ("content_filter", TerminationReason.CONTENT_FILTER),
    ("provider_custom", TerminationReason.OTHER),
))
async def test_provider_normalizes_and_persists_sanitized_termination(raw, normalized):
    response = await _adapter({
        "choices": [{"message": {"content": "answer"}, "finish_reason": raw}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }).generate(InferenceRequest(prompt="hello"))
    assert response.termination_reason is normalized
    assert response.provider_termination_reason == raw


@pytest.mark.asyncio
async def test_authoritative_length_is_an_execution_failure_without_reasoning_leakage():
    body = {
        "choices": [{"message": {
            "content": "def answer():\n    return len",
            "reasoning_content": "private hidden reasoning",
        }, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 60,
                  "completion_tokens_details": {"reasoning_tokens": 48}},
    }
    with pytest.raises(GatewayError) as error:
        await _adapter(body).generate(InferenceRequest(prompt="hello", max_output_tokens=64))
    assert error.value.category is GatewayErrorCategory.OUTPUT_BUDGET_EXHAUSTION
    assert error.value.diagnostics["termination_reason"] == "length"
    serialized = repr(error.value.diagnostics)
    assert "private hidden reasoning" not in serialized
    assert "def answer" not in serialized


def test_fallback_heuristic_requires_missing_metadata_near_limit_and_visible_cutoff():
    base = dict(model_id="fixture", provider="fake", input_tokens=3,
                latency_ms=1, estimated_cost_usd="0")
    truncated = InferenceResponse(
        text="def answer():\n    return len", output_tokens=96, **base)
    complete = InferenceResponse(
        text="def answer():\n    return len([])", output_tokens=96,
        termination_reason=TerminationReason.COMPLETE,
        provider_termination_reason="stop", **base)
    assert response_output_budget_exhaustion(truncated, 100)
    assert not response_output_budget_exhaustion(complete, 100)


@pytest.mark.asyncio
async def test_runner_classifies_fallback_exhaustion_instead_of_quality_data(tmp_path):
    class TruncatedProvider:
        async def generate(self, request):
            return InferenceResponse(
                text="def answer():\n    return len", model_id="truncated",
                provider="truncated-provider", input_tokens=5, output_tokens=96,
                latency_ms=1, estimated_cost_usd="0.0001")

    service = create_development_service()
    service.registry.register(ModelConfig(
        model_id="truncated", provider="truncated-provider",
        provider_model_name="fixture/truncated", input_cost_per_1m_tokens="1",
        output_cost_per_1m_tokens="1", context_window=4096))
    service.resolver.register("truncated-provider", lambda model: TruncatedProvider())
    dataset = BenchmarkDataset(name="fixture", version="1", tasks=(
        BenchmarkTask(task_id="coding-case", category="coding", prompt="Write code.",
                      max_output_tokens=100),))
    run = await BenchmarkRunner(service, FileBenchmarkRepository(tmp_path)).run(
        dataset, ["truncated"])
    path, = (tmp_path / str(run.run_id) / "results").glob("*.json")
    result = BenchmarkResult.model_validate_json(path.read_bytes())
    assert not result.success and result.response is None
    assert result.error_category == "output_budget_exhaustion"
    assert result.error_details["fallback_heuristic"] is True


def test_correction_artifacts_are_hash_identified_and_contain_no_reasoning_text():
    expected = {
        PROTOCOL_15: "fd9f60ca01e912418ba9b54804ea9a6f92cd3b326fc17d489bb7e9e51cd5521d",
        READINESS_15: "0674ff0ea77271099b93932985e202c9852610a1215f9955d549ba53f1982174",
        CONFIRMATION: "b80ef76b8238681e935a1f289492247a90b92ba0095c81cb21d6818f9dc1a436",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    combined = b"".join(path.read_bytes() for path in expected)
    assert b"reasoning_content" not in combined
    assert b"reasoning_text" not in combined
    assert b"chain_of_thought" not in combined
