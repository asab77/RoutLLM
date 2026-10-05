import hashlib
import json
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks.execution_readiness import (
    READY_STATUS,
    corrected_pilot_dry_run,
    load_pricing_readiness,
)
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    ROUTING_BENCHMARK_MODELS,
    execution_models_from_protocol,
    load_execution_protocol,
)
from adaptive_llm_gateway.models import OutputTokenAccounting


PROTOCOL_V13 = Path("benchmarks/protocols/routing-benchmark-v1/protocol.json")
PROTOCOL_V14 = Path("benchmarks/protocols/routing-benchmark-v1.2/protocol.json")
READINESS = Path("benchmarks/protocols/routing-benchmark-v1.2/execution-readiness.json")
DATASET = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")
PILOT = Path("benchmarks/protocols/routing-benchmark-v1/corrected-pilot-rerun-manifest.json")
COST = Path("benchmarks/protocols/routing-benchmark-v1.2/cost-estimate.json")
EVALUATOR = Path("benchmarks/protocols/routing-benchmark-v1.2/evaluator-manifest.json")
SPECIFICATION = Path("benchmarks/specifications/summarization-propositions-v1.0.0.json")


def test_protocol_14_and_historical_13_are_hash_locked_and_supported():
    protocol_14, models_14, digest_14 = load_execution_protocol(PROTOCOL_V14)
    protocol_13, models_13, digest_13 = load_execution_protocol(PROTOCOL_V13)

    assert (protocol_14["protocol"], protocol_14["version"], digest_14) == (
        "routellm-routing-benchmark-v1.2",
        "1.4.0",
        "f132c1eec6b1a231f6ce5ea50702b59a03e76af7d44359ba614fa0ed4ae6e565",
    )
    assert (protocol_13["protocol"], protocol_13["version"], digest_13) == (
        "routellm-routing-benchmark-v1",
        "1.3.0",
        "8a569e225734c52480d5b31d11db9d78a8b2c0417d621b6d7234d696b0859204",
    )
    assert models_14 == models_13 == ROUTING_BENCHMARK_MODELS


@pytest.mark.parametrize(
    ("field", "value"),
    (("protocol", "unknown-routing-benchmark"), ("version", "99.0.0")),
)
def test_unknown_protocol_contracts_fail_closed(field, value):
    protocol = json.loads(PROTOCOL_V14.read_bytes())
    protocol[field] = value
    with pytest.raises(ValueError, match="identity/version mismatch"):
        execution_models_from_protocol(protocol)


def test_protocol_hash_integrity_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL_V14.read_bytes())
    protocol["service_tier"] = "tampered"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        load_execution_protocol(path)


def test_protocol_14_loads_exact_candidate_and_token_contracts():
    protocol, models, _ = load_execution_protocol(PROTOCOL_V14)
    by_id = {model.model_id: model for model in models}
    assert [(model.model_id, model.provider_model_name) for model in models] == [
        (item["candidate_id"], item["upstream_model_slug"])
        for item in protocol["candidates"]
    ]
    expected = {
        "candidate-nemotron-3.5-lightning": (OutputTokenAccounting.VISIBLE_ONLY, 0),
        "candidate-gpt-6-luna": (OutputTokenAccounting.REASONING_AND_VISIBLE, 128),
        "candidate-gemini-3-flash": (OutputTokenAccounting.REASONING_AND_VISIBLE, 256),
        "candidate-claude-sonnet-5": (OutputTokenAccounting.REASONING_AND_VISIBLE, 128),
    }
    assert {
        model_id: (
            model.capabilities.output_token_accounting,
            model.output_token_policy.reasoning_headroom_tokens,
        )
        for model_id, model in by_id.items()
    } == expected
    assert {model_id: model.provider_output_allowance(32, category="classification")
            for model_id, model in by_id.items()} == {
        "candidate-nemotron-3.5-lightning": 32,
        "candidate-gpt-6-luna": 160,
        "candidate-gemini-3-flash": 288,
        "candidate-claude-sonnet-5": 160,
    }


def test_pricing_readiness_is_exact_and_stale_status_fails_closed(tmp_path):
    protocol, models, digest = load_execution_protocol(PROTOCOL_V14)
    record = load_pricing_readiness(
        READINESS, protocol=protocol, protocol_sha256=digest, candidate_models=models)
    assert record.status == READY_STATUS
    assert len(record.models) == 5

    stale = json.loads(READINESS.read_bytes())
    stale["status"] = "REQUIRES_REVERIFICATION"
    stale_path = tmp_path / "stale.json"
    stale_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="not ready"):
        load_pricing_readiness(
            stale_path, protocol=protocol, protocol_sha256=digest,
            candidate_models=models)


@pytest.mark.protected_final_data
def test_corrected_pilot_dry_run_reaches_paid_boundary_without_crossing_it():
    report = corrected_pilot_dry_run(
        protocol_path=PROTOCOL_V14,
        readiness_path=READINESS,
        dataset_path=DATASET,
        split_path=SPLIT,
        evaluator_path=EVALUATOR,
        specification_path=SPECIFICATION,
        pilot_path=PILOT,
        cost_path=COST,
    )
    assert report["status"] == READY_STATUS
    assert (report["pilot_tasks"], report["train_tasks"],
            report["development_tasks"], report["final_tasks"]) == (21, 21, 0, 0)
    assert report["candidate_count"] == 4
    assert report["evaluator_version"] == "1.3.0"
    assert report["maximum_candidate_attempts"] == 84
    assert report["maximum_astra_calls"] == 12
    assert report["projected_total_cost_usd"] == "0.3748380900"
    assert report["projected_worst_case_cost_usd"] == "0.58863"
    assert report["authorized_ceiling_usd"] == "0.60"
    assert report["provider_calls_made"] == 0
    assert report["pricing_readiness_sha256"] == hashlib.sha256(
        READINESS.read_bytes()).hexdigest()
