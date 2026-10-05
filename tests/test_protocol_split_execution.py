import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from adaptive_llm_gateway.benchmarks.__main__ import (
    _model_overrides_from_validated_contract,
)
from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    BENCHMARK_NAME,
    ROUTING_BENCHMARK_MODELS,
    execution_models_from_protocol,
    load_execution_protocol,
    select_execution_dataset,
)
from adaptive_llm_gateway.models import (
    InferenceRequest,
    OutputTokenAccounting,
    ReasoningControlMechanism,
    ReasoningEffort,
)
from adaptive_llm_gateway.providers.gateway_config import GatewaySettings
from adaptive_llm_gateway.providers.vercel import VercelGatewayProvider


DATASET = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")
PROTOCOL_ROOT = Path("benchmarks/protocols/routing-benchmark-v1.2")
PROTOCOL_17 = PROTOCOL_ROOT / "protocol-1.7.json"
MODEL_IDS = [
    "candidate-nemotron-3.5-lightning",
    "candidate-gpt-6-luna",
    "candidate-gemini-3-flash",
    "candidate-claude-sonnet-5",
]
def selected_dataset(part: str | None):
    dataset = load_dataset(DATASET)
    if part is None:
        return dataset
    arguments = {}
    return select_execution_dataset(
        dataset,
        json.loads(SPLIT.read_bytes()),
        split=part,
        **arguments,
    )


def protocol_17_overrides():
    _, models, _ = load_execution_protocol(PROTOCOL_17)
    return _model_overrides_from_validated_contract(
        protocol_models=models,
        routing_benchmark_validated=True,
        selected_model_ids=MODEL_IDS,
    )


@pytest.mark.parametrize("part", [None, "train", "development"])
@pytest.mark.protected_final_data
def test_protocol_17_overrides_survive_full_and_split_selection(part):
    dataset = selected_dataset(part)
    overrides = protocol_17_overrides()
    gemini = overrides["candidate-gemini-3-flash"]

    if part is not None:
        assert dataset.name != BENCHMARK_NAME
    assert gemini.reasoning_control is ReasoningControlMechanism.GOOGLE_PROVIDER_NATIVE
    assert gemini.reasoning_effort is ReasoningEffort.MINIMAL
    assert (
        gemini.capabilities.output_token_accounting
        is OutputTokenAccounting.REASONING_AND_VISIBLE
    )
    assert gemini.output_token_policy.reasoning_headroom_tokens == 384


@pytest.mark.protected_final_data
def test_renamed_subset_cannot_remove_validated_protocol_overrides():
    development = selected_dataset("development")
    assert development.name == f"{BENCHMARK_NAME}-development"
    assert set(protocol_17_overrides()) == set(MODEL_IDS)


@pytest.mark.parametrize("part", ["train", "development"])
@pytest.mark.asyncio
@pytest.mark.protected_final_data
async def test_each_split_serializes_google_native_minimal_without_shared_reasoning(part):
    selected_dataset(part)  # Exercise validation, selection, and the FINAL gate.
    gemini = protocol_17_overrides()["candidate-gemini-3-flash"]
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        })

    provider = VercelGatewayProvider(
        gemini,
        GatewaySettings(api_key=SecretStr("test-only-key")),
        transport=httpx.MockTransport(handler),
    )
    await provider.generate(InferenceRequest(prompt="test", max_output_tokens=416))
    payload = captured[0]
    assert payload["providerOptions"] == {
        "gateway": {"only": ["google"]},
        "google": {"thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    assert "reasoning" not in payload


def test_protocol_17_preserves_non_gemini_candidate_configurations():
    _, expected, _ = load_execution_protocol(PROTOCOL_17)
    overrides = protocol_17_overrides()
    for model in expected:
        if model.model_id != "candidate-gemini-3-flash":
            assert overrides[model.model_id] == model


@pytest.mark.parametrize(
    "protocol_name",
    ["protocol.json", "protocol-1.5.json", "protocol-1.6.json", "protocol-1.7.json"],
)
def test_historical_supported_protocol_contracts_remain_authoritative(protocol_name):
    _, models, _ = load_execution_protocol(PROTOCOL_ROOT / protocol_name)
    overrides = _model_overrides_from_validated_contract(
        protocol_models=models,
        routing_benchmark_validated=True,
        selected_model_ids=MODEL_IDS,
    )
    assert overrides == {model.model_id: model for model in models}


def test_validated_historical_default_contract_is_preserved_without_protocol():
    overrides = _model_overrides_from_validated_contract(
        protocol_models=None,
        routing_benchmark_validated=True,
        selected_model_ids=MODEL_IDS,
    )
    assert overrides == {model.model_id: model for model in ROUTING_BENCHMARK_MODELS}


def test_unvalidated_dataset_gets_no_benchmark_overrides():
    assert _model_overrides_from_validated_contract(
        protocol_models=None,
        routing_benchmark_validated=False,
        selected_model_ids=MODEL_IDS,
    ) == {}


def test_unsupported_protocol_contract_still_fails_closed():
    protocol = json.loads(PROTOCOL_17.read_bytes())
    protocol["version"] = "9.9.9"
    with pytest.raises(ValueError, match="identity/version mismatch"):
        execution_models_from_protocol(protocol)
