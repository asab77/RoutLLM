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
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    ROUTING_BENCHMARK_MODELS_V16,
    ROUTING_BENCHMARK_MODELS_V17,
    load_execution_protocol,
)
from adaptive_llm_gateway.errors import GatewayError, GatewayErrorCategory
from adaptive_llm_gateway.models import (
    InferenceRequest,
    OutputTokenAccounting,
    ReasoningControlMechanism,
    ReasoningEffort,
)
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider


ROOT = Path("benchmarks/protocols/routing-benchmark-v1.2")
PROTOCOL_14 = ROOT / "protocol.json"
PROTOCOL_15 = ROOT / "protocol-1.5.json"
PROTOCOL_16 = ROOT / "protocol-1.6.json"
PROTOCOL_17 = ROOT / "protocol-1.7.json"
READINESS_17 = ROOT / "execution-readiness-1.7.json"
CONFIRMATION_17 = ROOT / "gemini-native-minimal-confirmation-1.7.json"
COSTS_17 = ROOT / "gemini-native-minimal-costs-1.7.json"
DATASET = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gemini(models):
    return next(model for model in models if model.model_id == "candidate-gemini-3-flash")


def test_protocol_17_changes_only_gemini_reasoning_control():
    protocol, models, digest = load_execution_protocol(PROTOCOL_17)
    assert (protocol["version"], digest) == (
        "1.7.0",
        "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8",
    )
    model = gemini(models)
    assert model.reasoning_effort is ReasoningEffort.MINIMAL
    assert model.reasoning_control is ReasoningControlMechanism.GOOGLE_PROVIDER_NATIVE
    assert model.capabilities.output_token_accounting is OutputTokenAccounting.REASONING_AND_VISIBLE
    assert model.output_token_policy.reasoning_headroom_tokens == 384
    assert model.model_copy(update={
        "reasoning_control": ReasoningControlMechanism.GATEWAY_SHARED,
    }) == gemini(ROUTING_BENCHMARK_MODELS_V16)
    assert [m for m in models if m.model_id != model.model_id] == [
        m for m in ROUTING_BENCHMARK_MODELS_V16 if m.model_id != model.model_id
    ]
    frozen = next(item for item in protocol["candidates"]
                  if item["candidate_id"] == model.model_id)
    assert frozen["reasoning_effort"] == "minimal"
    assert frozen["reasoning_control"] == "google_provider_native"


@pytest.mark.asyncio
async def test_protocol_17_serializes_only_google_native_minimal():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })

    provider = VercelGatewayProvider(
        gemini(ROUTING_BENCHMARK_MODELS_V17),
        GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(handler),
    )
    await provider.generate(InferenceRequest(
        prompt="test", system_prompt="system", max_output_tokens=512, temperature=0))
    assert captured == [{
        "model": "google/gemini-3-flash",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "test"},
        ],
        "stream": False,
        "max_tokens": 512,
        "providerOptions": {
            "gateway": {"only": ["google"]},
            "google": {"thinkingConfig": {"thinkingLevel": "minimal"}},
        },
        "temperature": 0,
    }]
    assert "reasoning" not in captured[0]


@pytest.mark.asyncio
async def test_protocol_17_preserves_authoritative_length_handling():
    provider = VercelGatewayProvider(
        gemini(ROUTING_BENCHMARK_MODELS_V17),
        GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 508},
        })),
    )
    with pytest.raises(GatewayError) as error:
        await provider.generate(InferenceRequest(prompt="test", max_output_tokens=512))
    assert error.value.category is GatewayErrorCategory.OUTPUT_BUDGET_EXHAUSTION
    assert error.value.diagnostics["termination_reason"] == "length"


def test_protocol_17_readiness_manifest_allowances_and_costs():
    protocol, models, digest = load_execution_protocol(PROTOCOL_17)
    readiness = load_pricing_readiness(
        READINESS_17, protocol=protocol, protocol_sha256=digest,
        candidate_models=models)
    assert readiness.status == READY_STATUS
    report = gemini_confirmation_dry_run(
        protocol_path=PROTOCOL_17,
        readiness_path=READINESS_17,
        dataset_path=DATASET,
        split_path=SPLIT,
        confirmation_path=CONFIRMATION_17,
    )
    assert report["status"] == "READY_FOR_CONFIRMATION"
    assert report["reasoning_effort"] == "minimal"
    assert report["reasoning_control"] == "google_provider_native"
    assert report["reasoning_reserve_tokens"] == 384
    assert report["maximum_candidate_calls"] == 5
    assert report["astra_calls"] == report["provider_calls_made"] == 0
    manifest = json.loads(CONFIRMATION_17.read_bytes())
    assert [item["task_id"] for item in manifest["cases"]] == [
        "coding-easy-001", "coding-medium-001", "coding-hard-001",
        "summarization-easy-001", "summarization-medium-001",
    ]
    assert [item["provider_output_allowance"] for item in manifest["cases"]] == [
        512, 576, 640, 480, 512,
    ]
    assert manifest["candidate_retries"] == manifest["astra_calls"] == 0
    assert manifest["shared_reasoning_field"] == "omitted"
    costs = json.loads(COSTS_17.read_bytes())
    assert costs["five_case_confirmation"]["expected_cost_usd"] == "0.00486650"
    assert costs["five_case_confirmation"]["worst_case_cost_usd"] == "0.00835550"
    assert costs["full_train"]["total_expected_cost_usd"] == "2.38771000"
    assert costs["full_train"]["total_worst_case_cost_usd"] == "4.57766875"


def test_protocol_17_artifacts_and_historical_inputs_are_hash_identified():
    expected = {
        PROTOCOL_14: "f132c1eec6b1a231f6ce5ea50702b59a03e76af7d44359ba614fa0ed4ae6e565",
        PROTOCOL_15: "fd9f60ca01e912418ba9b54804ea9a6f92cd3b326fc17d489bb7e9e51cd5521d",
        PROTOCOL_16: "10ca97aafd313c29778103fe0f6f7145243a3e8795d745e1b51a9b0a582c42b3",
        ROOT / "execution-readiness-1.5.json":
            "0674ff0ea77271099b93932985e202c9852610a1215f9955d549ba53f1982174",
        ROOT / "execution-readiness-1.6.json":
            "ff1102fca533ad8c57c48ac8e32bf16cf13b46b3e4b3a8c0c40b15d7050aac81",
        ROOT / "gemini-budget-confirmation-1.5.json":
            "b80ef76b8238681e935a1f289492247a90b92ba0095c81cb21d6818f9dc1a436",
        ROOT / "gemini-minimal-confirmation-1.6.json":
            "066f3cae5118870064a855c759fc64d72f49000b7136fa44cd5b8a17a6e2d41e",
        ROOT / "gemini-budget-costs-1.5.json":
            "674a5aeea3bef08b23b57ba1a7c24cee948b8787ff2adb7a5c2b48007605e4bc",
        ROOT / "gemini-minimal-costs-1.6.json":
            "cc9d07f239098f597f6c950bcac497d67bc732ef125d77ed7d70586ddd4b56dd",
        PROTOCOL_17: "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8",
        READINESS_17: "3acf8a22b1b679012d23738c94122163cddcf6cf10d7df4f8c395d2855aca9ef",
        CONFIRMATION_17: "035ec434cb5ad1462ba133e2a197f5682945d0c30e034068f66766d823f8a632",
        COSTS_17: "bd2781a4f78cf07c68cd06a1d4f8d567d48132583d37a4152358200e25c4f5e1",
        Path("artifacts/routing-benchmark-v1/development-manifest.json"):
            "93daa569d7149666328ffad28eb2db5fbd1342700235d47be06bf001fef61c0e",
        Path("artifacts/routing-benchmark-v1/final-manifest.json"):
            "cca8f7f57df723f6f92624394c382544f972165d4087c69d4f3b1e402b43ef65",
    }
    assert {path: sha(path) for path in expected} == expected
    combined = b"".join(path.read_bytes() for path in (
        PROTOCOL_17, READINESS_17, CONFIRMATION_17, COSTS_17))
    assert b"reasoning_content" not in combined
    assert b"authorization_header" not in combined.lower()
    assert b"bearer " not in combined.lower()
    assert b"api_key" not in combined.lower()
