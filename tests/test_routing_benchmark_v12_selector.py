import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from adaptive_llm_gateway.benchmarks.models import load_dataset
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    BENCHMARK_NAME,
    CATEGORIES,
    ROUTING_V12_DATASET_VERSION,
    load_execution_protocol,
    select_execution_dataset,
    validate_routing_benchmark,
)


DATASET_V11 = Path("benchmarks/datasets/routing-benchmark-v1.json")
DATASET_V12 = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")
PROTOCOL_17 = Path("benchmarks/protocols/routing-benchmark-v1.2/protocol-1.7.json")


@pytest.fixture(scope="module")
def dataset_v12():
    return load_dataset(DATASET_V12)


@pytest.fixture(scope="module")
def split():
    return json.loads(SPLIT.read_bytes())


def select(dataset, split, part):
    return select_execution_dataset(dataset, split, split=part)


@pytest.mark.protected_final_data
def test_v12_is_an_explicit_supported_dataset(dataset_v12):
    validate_routing_benchmark(dataset_v12)
    assert (dataset_v12.name, dataset_v12.version) == (
        BENCHMARK_NAME, ROUTING_V12_DATASET_VERSION)


@pytest.mark.parametrize(("part", "count", "per_category"), [
    ("train", 140, 20),
    ("development", 42, 6),
])
@pytest.mark.protected_final_data
def test_v12_split_selection_is_exact(dataset_v12, split, part, count, per_category):
    selected = select(dataset_v12, split, part)
    assert len(selected.tasks) == count
    assert Counter(task.category for task in selected.tasks) == Counter(
        {category: per_category for category in CATEGORIES})
    selected_ids = {task.task_id for task in selected.tasks}
    assert all(entry["split"] == part for entry in split["entries"]
               if entry["task_id"] in selected_ids)
    assert selected_ids == {
        entry["task_id"] for entry in split["entries"] if entry["split"] == part}


@pytest.mark.protected_final_data
def test_v12_selected_splits_have_no_cross_contamination(dataset_v12, split):
    selected = {part: {task.task_id for task in select(dataset_v12, split, part).tasks}
                for part in ("train", "development")}
    assert selected["train"].isdisjoint(selected["development"])


def test_v12_family_separation_remains_intact(split):
    family_splits = defaultdict(set)
    for entry in split["entries"]:
        family_splits[entry["task_family_id"]].add(entry["split"])
    assert family_splits and all(len(parts) == 1 for parts in family_splits.values())


@pytest.mark.parametrize("version", ["9.9.9", "v1.2", "1.2"])
@pytest.mark.protected_final_data
def test_unknown_or_malformed_versions_fail_closed(dataset_v12, version):
    with pytest.raises(ValueError, match="identity/version mismatch"):
        validate_routing_benchmark(dataset_v12.model_copy(update={"version": version}))


@pytest.mark.protected_final_data
def test_identity_mismatch_fails_closed(dataset_v12):
    with pytest.raises(ValueError, match="identity/version mismatch"):
        validate_routing_benchmark(dataset_v12.model_copy(update={"name": "other-benchmark"}))


@pytest.mark.protected_final_data
def test_supported_version_with_changed_content_hash_fails_closed(dataset_v12):
    tasks = list(dataset_v12.tasks)
    tasks[0] = tasks[0].model_copy(update={"prompt": tasks[0].prompt + " "})
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_routing_benchmark(dataset_v12.model_copy(update={"tasks": tuple(tasks)}))


@pytest.mark.protected_final_data
def test_historical_v11_selection_remains_supported(split):
    historical = load_dataset(DATASET_V11)
    validate_routing_benchmark(historical)
    assert len(select(historical, split, "train").tasks) == 140


def test_protocol_17_identity_and_gemini_contract_are_unchanged():
    protocol, candidates, digest = load_execution_protocol(PROTOCOL_17)
    assert digest == "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8"
    assert protocol["dataset_sha256"] == (
        "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005")
    gemini = next(model for model in candidates
                  if model.model_id == "candidate-gemini-3-flash")
    assert gemini.reasoning_control.value == "google_provider_native"
    assert gemini.reasoning_effort.value == "minimal"
    assert gemini.capabilities.output_token_accounting.value == "reasoning_and_visible"
    assert gemini.output_token_policy.reasoning_headroom_tokens == 384
    assert hashlib.sha256(PROTOCOL_17.read_bytes()).hexdigest() == digest
