import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from adaptive_llm_gateway.benchmarks.execution_readiness import (
    READY_STATUS,
    gemini_confirmation_dry_run,
    load_pricing_readiness,
)
from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkTask
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    ROUTING_BENCHMARK_MODELS_V15,
    ROUTING_BENCHMARK_MODELS_V16,
    load_execution_protocol,
)
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory
from adaptive_llm_gateway.models import (
    InferenceRequest,
    InferenceResponse,
    ReasoningEffort,
)
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS, GatewaySettings
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider
from adaptive_llm_gateway.routing.features import ProductionRequestFeatureExtractor
from adaptive_llm_gateway.routing.quality_features import canonical_from_production


ROOT = Path("benchmarks/protocols/routing-benchmark-v1.2")
PROTOCOL_14 = ROOT / "protocol.json"
PROTOCOL_15 = ROOT / "protocol-1.5.json"
PROTOCOL_16 = ROOT / "protocol-1.6.json"
READINESS_16 = ROOT / "execution-readiness-1.6.json"
CONFIRMATION_16 = ROOT / "gemini-minimal-confirmation-1.6.json"
COSTS_16 = ROOT / "gemini-minimal-costs-1.6.json"
DATASET = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")
SPECIFICATION = Path("benchmarks/specifications/summarization-propositions-v1.0.0.json")
EVALUATOR = ROOT / "evaluator-manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gemini(models):
    return next(model for model in models if model.model_id == "candidate-gemini-3-flash")


def test_protocol_16_changes_only_gemini_reasoning_effort():
    protocol, models, digest = load_execution_protocol(PROTOCOL_16)
    assert (protocol["version"], digest) == (
        "1.6.0",
        "10ca97aafd313c29778103fe0f6f7145243a3e8795d745e1b51a9b0a582c42b3",
    )
    assert gemini(models).reasoning_effort is ReasoningEffort.MINIMAL
    assert gemini(models).output_token_policy.reasoning_headroom_tokens == 384
    assert [
        model for model in models if model.model_id != "candidate-gemini-3-flash"
    ] == [
        model for model in ROUTING_BENCHMARK_MODELS_V15
        if model.model_id != "candidate-gemini-3-flash"
    ]
    old = gemini(ROUTING_BENCHMARK_MODELS_V15)
    new = gemini(ROUTING_BENCHMARK_MODELS_V16)
    assert new.model_copy(update={"reasoning_effort": old.reasoning_effort}) == old


def test_protocol_16_allowances_and_protected_hashes_are_exact():
    _, models, _ = load_execution_protocol(PROTOCOL_16)
    model = gemini(models)
    expected = {
        "coding-easy-001": 512,
        "coding-medium-001": 576,
        "coding-hard-001": 640,
        "summarization-easy-001": 480,
        "summarization-medium-001": 512,
    }
    dataset = json.loads(DATASET.read_bytes())
    tasks = {item["task_id"]: item for item in dataset["tasks"]}
    assert {
        task_id: model.provider_output_allowance(
            tasks[task_id]["max_output_tokens"], category=tasks[task_id]["category"])
        for task_id in expected
    } == expected
    assert sha(DATASET) == "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005"
    assert sha(SPLIT) == "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
    assert sha(SPECIFICATION) == "b5568b26d7fdce018e3cfef9020780645e12e02f488ab495b14a1e621dd62c1e"
    assert sha(EVALUATOR) == "7dfb5c29426520edcb2c64491c1d5f62fddfc3bdbb3869599378da781f35d413"
    assert sha(PROTOCOL_14) == "f132c1eec6b1a231f6ce5ea50702b59a03e76af7d44359ba614fa0ed4ae6e565"
    assert sha(PROTOCOL_15) == "fd9f60ca01e912418ba9b54804ea9a6f92cd3b326fc17d489bb7e9e51cd5521d"


def test_protocol_16_readiness_confirmation_and_cost_plan():
    protocol, models, digest = load_execution_protocol(PROTOCOL_16)
    readiness = load_pricing_readiness(
        READINESS_16, protocol=protocol, protocol_sha256=digest,
        candidate_models=models)
    assert readiness.status == READY_STATUS
    report = gemini_confirmation_dry_run(
        protocol_path=PROTOCOL_16,
        readiness_path=READINESS_16,
        dataset_path=DATASET,
        split_path=SPLIT,
        confirmation_path=CONFIRMATION_16,
    )
    assert report == {
        "status": "READY_FOR_CONFIRMATION",
        "protocol_sha256": digest,
        "pricing_readiness_version": "1.0.0",
        "case_ids": [
            "coding-easy-001", "coding-hard-001", "coding-medium-001",
            "summarization-easy-001", "summarization-medium-001",
        ],
        "candidate_id": "candidate-gemini-3-flash",
        "reasoning_effort": "minimal",
        "reasoning_reserve_tokens": 384,
        "maximum_candidate_calls": 5,
        "astra_calls": 0,
        "provider_calls_made": 0,
    }
    manifest = json.loads(CONFIRMATION_16.read_bytes())
    assert [item["task_id"] for item in manifest["cases"]] == [
        "coding-easy-001", "coding-medium-001", "coding-hard-001",
        "summarization-easy-001", "summarization-medium-001",
    ]
    assert manifest["candidate_retries"] == manifest["astra_calls"] == 0
    costs = json.loads(COSTS_16.read_bytes())
    assert costs["five_case_confirmation"]["expected_cost_usd"] == "0.00486650"
    assert costs["five_case_confirmation"]["worst_case_cost_usd"] == "0.00835550"
    assert costs["full_train"]["total_expected_cost_usd"] == "2.38771000"
    assert costs["full_train"]["total_worst_case_cost_usd"] == "4.57766875"


@pytest.mark.asyncio
async def test_gateway_request_uses_one_minimal_reasoning_setting():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })

    adapter = VercelGatewayProvider(
        gemini(ROUTING_BENCHMARK_MODELS_V16),
        GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(handler),
    )
    await adapter.generate(InferenceRequest(prompt="test", max_output_tokens=512, temperature=0))
    payload = captured[0]
    assert payload["reasoning"] == {"effort": "minimal"}
    assert "low" not in json.dumps(payload)
    assert set(payload["providerOptions"]) == {"gateway"}
    assert payload["max_tokens"] == 512


@pytest.mark.asyncio
async def test_length_remains_output_budget_exhaustion():
    adapter = VercelGatewayProvider(
        gemini(ROUTING_BENCHMARK_MODELS_V16),
        GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 508},
        })),
    )
    with pytest.raises(GatewayError) as error:
        await adapter.generate(InferenceRequest(prompt="test", max_output_tokens=512))
    assert error.value.category is GatewayErrorCategory.OUTPUT_BUDGET_EXHAUSTION
    assert error.value.diagnostics["termination_reason"] == "length"


@pytest.mark.asyncio
async def test_runner_passes_minimal_model_to_provider_and_feature_contract(tmp_path):
    captured_models = []

    class Provider:
        def __init__(self, model):
            captured_models.append(model)

        async def generate(self, request):
            return InferenceResponse(
                text="answer", model_id=captured_models[-1].model_id,
                provider="vercel", input_tokens=3, output_tokens=1,
                latency_ms=1, estimated_cost_usd="0.00001",
            )

    service = create_development_service()
    service.registry.register(gemini(CANDIDATE_MODELS))
    service.resolver.register("vercel", Provider)
    model = gemini(ROUTING_BENCHMARK_MODELS_V16)
    dataset = BenchmarkDataset(name="fixture", version="1", tasks=(
        BenchmarkTask(task_id="coding-case", category="coding", prompt="Return code.",
                      max_output_tokens=128, temperature=0),
    ))
    await BenchmarkRunner(
        service, FileBenchmarkRepository(tmp_path),
        model_configurations={model.model_id: model},
    ).run(dataset, [model.model_id])
    assert captured_models[0].reasoning_effort is ReasoningEffort.MINIMAL
    request_features = ProductionRequestFeatureExtractor().extract(
        dataset.tasks[0].to_request(), category_hint="coding")
    assert canonical_from_production(
        request_features, model).reasoning_effort is ReasoningEffort.MINIMAL


def test_protocol_16_artifacts_are_hash_identified_and_safe():
    expected = {
        PROTOCOL_16: "10ca97aafd313c29778103fe0f6f7145243a3e8795d745e1b51a9b0a582c42b3",
        READINESS_16: "ff1102fca533ad8c57c48ac8e32bf16cf13b46b3e4b3a8c0c40b15d7050aac81",
        CONFIRMATION_16: "066f3cae5118870064a855c759fc64d72f49000b7136fa44cd5b8a17a6e2d41e",
        COSTS_16: "cc9d07f239098f597f6c950bcac497d67bc732ef125d77ed7d70586ddd4b56dd",
    }
    for path, digest in expected.items():
        assert sha(path) == digest
    combined = b"".join(path.read_bytes() for path in expected)
    assert b"reasoning_content" not in combined
    assert b"authorization_header" not in combined.lower()
    assert b"bearer " not in combined.lower()
    assert b"api_key" not in combined.lower()
