import hashlib
import inspect
import json
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkTask
from adaptive_llm_gateway.benchmarks.repository import FileBenchmarkRepository
from adaptive_llm_gateway.benchmarks.runner import BenchmarkRunner
from adaptive_llm_gateway.benchmarks.protocol_audit import (
    is_output_budget_exhaustion, output_budget_diagnostic,
)
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    DEFAULT_DATASET, DEFAULT_PROTOCOL_DIR, PROTOCOL_VERSION,
    ROUTING_BENCHMARK_MODELS, build_corrected_rerun_manifest,
    build_pilot_manifest, build_protocol, build_split_manifest,
    execution_models_from_protocol,
)
from adaptive_llm_gateway.evaluation.semantic_fixtures import ADVERSARIAL_SEMANTIC_FIXTURES
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.models import (
    OutputTokenAccounting, OutputTokenPolicy,
)

DATASET_SHA = "70152d1ac15e827a0ff940cf5becf76701578d0a62d773b4ad6e3823c3fa27ca"
SPLIT_SHA = "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
OLD_PROTOCOL_SHA = "8db036771695c54d624d510534049e0669717576b4cadbeed38330c83582912b"
OLD_PILOT = Path("artifacts/routing-benchmark-v1/pilot-runs/1aa850f6-0862-4704-8f0b-c246c9d990ec")
OLD_PILOT_TREE_SHA = "9c194b025d3acfd6f12a2ce1f0f0f4d673757252665f28fd0b2d518ad815425a"
FAILED_ASTRA_VALIDATION = Path(
    "artifacts/routing-benchmark-v1/protocol-correction/astra-1.1-live-validation.json")
FAILED_ASTRA_VALIDATION_SHA = "50fa94804e9f2104c7fb60804cb22bfc5e2289a89ed773df50ebadef4ea556fe"


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_dataset_content_and_split_are_unchanged():
    assert hashlib.sha256(DEFAULT_DATASET.read_bytes()).hexdigest() == DATASET_SHA
    assert hashlib.sha256((DEFAULT_PROTOCOL_DIR / "split-manifest.json").read_bytes()).hexdigest() == SPLIT_SHA


@pytest.mark.local_evidence
def test_historical_pilot_and_failed_astra_evidence_are_immutable():
    if not OLD_PILOT.exists() or not FAILED_ASTRA_VALIDATION.exists():
        pytest.skip("historical local evidence is unavailable")
    assert directory_sha256(OLD_PILOT) == OLD_PILOT_TREE_SHA
    assert hashlib.sha256(FAILED_ASTRA_VALIDATION.read_bytes()).hexdigest() == FAILED_ASTRA_VALIDATION_SHA


def test_typed_allowance_separates_visible_requirement_from_provider_limit():
    models = {model.model_id: model for model in ROUTING_BENCHMARK_MODELS}
    visible = 32
    assert models["candidate-nemotron-3.5-lightning"].provider_output_allowance(
        visible, category="classification") == 32
    assert models["candidate-gpt-6-luna"].provider_output_allowance(
        visible, category="classification") == 160
    assert models["candidate-gemini-3-flash"].provider_output_allowance(
        visible, category="classification") == 288
    assert models["candidate-claude-sonnet-5"].provider_output_allowance(
        visible, category="classification") == 160
    assert all(model.output_token_policy.reasoning_headroom_tokens <= 256
               for model in models.values())


def test_coding_headroom_is_complete_bounded_and_category_independent():
    for model in ROUTING_BENCHMARK_MODELS:
        expected = 256 + model.output_token_policy.reasoning_headroom_tokens
        assert model.provider_output_allowance(256, category="coding") == expected
        assert model.provider_output_allowance(256, category="reasoning") == expected
    assert not any(model.output_token_policy.category_overrides
                   for model in ROUTING_BENCHMARK_MODELS)


@pytest.mark.asyncio
async def test_runner_records_visible_requirement_and_corrected_provider_allowance(tmp_path):
    service = create_development_service()
    base = service.registry.get("fake-small")
    corrected = base.model_copy(update={
        "capabilities": base.capabilities.model_copy(update={
            "output_token_accounting": OutputTokenAccounting.REASONING_AND_VISIBLE}),
        "output_token_policy": OutputTokenPolicy(
            reasoning_headroom_tokens=32, expected_reasoning_tokens=8),
    })
    task = BenchmarkTask(task_id="classification-test", category="classification",
                         prompt="Return OK.", max_output_tokens=8)
    run = await BenchmarkRunner(
        service, FileBenchmarkRepository(tmp_path),
        model_configurations={base.model_id: corrected},
    ).run(BenchmarkDataset(name="fixture", version="1", tasks=(task,)), [base.model_id])
    assert run.configuration["task_visible_output_requirements"] == {task.task_id: 8}
    assert run.configuration["effective_max_output_tokens"] == {
        task.task_id: {base.model_id: 40}}


def test_allowance_policy_is_typed_deterministic_and_has_no_slug_branching():
    combined = [model for model in ROUTING_BENCHMARK_MODELS
                if model.capabilities.output_token_accounting
                is OutputTokenAccounting.REASONING_AND_VISIBLE]
    assert len(combined) == 3
    assert all(model.provider_output_allowance(64, category="qa")
               == model.provider_output_allowance(64, category="qa") for model in combined)
    from adaptive_llm_gateway.models.schemas import ModelConfig
    source = inspect.getsource(ModelConfig.provider_output_allowance)
    assert all(value not in source for value in ("gemini", "luna", "sonnet", "nemotron"))


@pytest.mark.local_evidence
def test_all_seven_stored_missing_cases_reproduce_as_budget_exhaustion():
    if not OLD_PILOT.exists():
        pytest.skip("historical pilot artifacts are local generated evidence")
    failures = []
    for path in (OLD_PILOT / "results").glob("*.json"):
        result = json.loads(path.read_text())
        if not result["success"]:
            failures.append(result)
    assert len(failures) == 7
    assert all(is_output_budget_exhaustion(item["error_details"]) for item in failures)
    diagnostics = [output_budget_diagnostic(item["error_details"]) for item in failures]
    assert {item["classification"] for item in diagnostics} == {"OUTPUT_BUDGET_EXHAUSTION"}
    assert {item["visible_tokens"] for item in diagnostics} == {0}


def test_corrected_protocol_version_and_rerun_identity():
    dataset = load_dataset(DEFAULT_DATASET)
    split = build_split_manifest(dataset)
    pilot = build_pilot_manifest(dataset, split)
    protocol = build_protocol(DATASET_SHA, SPLIT_SHA, "c" * 64, "d" * 64)
    encoded = (json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    protocol_sha = hashlib.sha256(encoded).hexdigest()
    assert protocol["version"] == PROTOCOL_VERSION == "1.3.0"
    assert execution_models_from_protocol(protocol) == ROUTING_BENCHMARK_MODELS
    assert protocol_sha != OLD_PROTOCOL_SHA
    rerun = build_corrected_rerun_manifest(pilot, protocol_sha)
    assert rerun["tasks"] == pilot["tasks"]
    assert rerun["task_count"] == 21 and rerun["candidate_count"] == 4
    assert rerun["candidate_call_limit"] == 84
    assert rerun["development_tasks"] == rerun["final_tasks"] == 0


def test_semantic_boundary_fixtures_cover_both_sides_of_each_dimension():
    supported = {item.phenomenon for item in ADVERSARIAL_SEMANTIC_FIXTURES
                 if item.expected == "supported"}
    unsupported = {item.phenomenon for item in ADVERSARIAL_SEMANTIC_FIXTURES
                   if item.expected == "unsupported"}
    assert {"lexical_paraphrase", "semantic_compression", "descriptive_characterization",
            "temporal_compression", "attribution_preservation", "equivalent_quantity",
            "equivalent_certainty", "borderline_compression"} <= supported
    assert {"invented_cause", "invented_motive", "invented_diagnosis", "invented_actor",
            "invented_event", "changed_quantity", "changed_time", "attribution_swap",
            "certainty_strengthening", "contradiction", "interpretive_diagnosis"} <= unsupported
