import asyncio
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from adaptive_llm_gateway.benchmarks.features import extract_request_features
from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkTask, load_dataset
from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    BENCHMARK_NAME, BENCHMARK_VERSION, CATEGORIES, DEFAULT_DATASET,
    DEFAULT_PROTOCOL_DIR, DIFFICULTY_COUNTS, EVALUATOR_TYPES,
    PROTOCOL_VERSION, ROUTING_BENCHMARK_MODELS,
    build_artifacts, build_cost_estimate, build_evaluator_manifest,
    build_pilot_manifest, build_provenance_manifest, build_protocol,
    build_split_manifest, canonicalize_content, coding_functional_audit,
    construct_dataset, content_quality_audit, duplicate_audit, edit_similarity, ground_truth_audit,
    load_training_dataset, ngram_jaccard, normalize_content,
    select_execution_dataset, validate_pilot_manifest,
    validate_routing_benchmark, validate_split_manifest,
)
from adaptive_llm_gateway.evaluation.evaluators import evaluator_for


@pytest.fixture(scope="module")
def dataset():
    return construct_dataset()


@pytest.fixture(scope="module")
def split(dataset):
    return build_split_manifest(dataset)


@pytest.fixture(scope="module")
def pilot(dataset, split):
    return build_pilot_manifest(dataset, split)


def passing_coding_audit():
    return {"status": "PASS", "canonical_passed": 32, "incorrect_rejected": 32,
            "tasks": 32, "failures": [], "sandbox": {"type": "test-fixture"}}


def test_identity_and_schema_metadata_validate(dataset):
    validate_routing_benchmark(dataset)
    assert dataset.name == BENCHMARK_NAME and dataset.version == BENCHMARK_VERSION
    assert all(task.task_family_id and task.source_type and task.source_id
               and task.evaluator_type and task.family_variant for task in dataset.tasks)


def test_task_family_is_required_for_routing_v1(dataset):
    tasks = list(dataset.tasks)
    tasks[0] = tasks[0].model_copy(update={"task_family_id": None})
    with pytest.raises(ValueError, match="provenance|families"):
        validate_routing_benchmark(dataset.model_copy(update={"tasks": tuple(tasks)}))


def test_historical_foundations_remain_compatible():
    for path in (Path("benchmarks/datasets/foundation-v1.json"),
                 Path("benchmarks/datasets/foundation-v2.json"),
                 Path("benchmarks/datasets/foundation-v3.json")):
        historical = load_dataset(path)
        assert historical.tasks and all(task.task_family_id is None for task in historical.tasks)


def test_evaluation_metadata_never_enters_request_or_feature_vector(dataset):
    task = dataset.tasks[0]
    request = task.to_request()
    assert not hasattr(request, "task_family_id") and not hasattr(request, "difficulty")
    features = extract_request_features(task).model_dump()
    forbidden = {"task_family_id", "source_type", "source_id", "generation_seed",
                 "evaluator_type", "family_variant", "difficulty"}
    assert forbidden.isdisjoint(features)


def test_exact_counts_and_category_balance(dataset):
    assert len(dataset.tasks) == 224
    assert Counter(task.category for task in dataset.tasks) == Counter({category: 32 for category in CATEGORIES})


def test_exact_difficulty_distribution(dataset):
    for category in CATEGORIES:
        assert Counter(task.difficulty for task in dataset.tasks if task.category == category) == Counter(DIFFICULTY_COUNTS)
    assert Counter(task.difficulty for task in dataset.tasks) == Counter({"easy": 56, "medium": 91, "hard": 77})


def test_difficulty_is_content_derived_not_family_position(dataset):
    import adaptive_llm_gateway.benchmarks.routing_benchmark_v1 as module
    assert not hasattr(module, "FAMILY_PLAN")
    longest = [task.difficulty for task in dataset.tasks
               if task.task_family_id and task.task_family_id.endswith("longest-nondecreasing")]
    nested = [task.difficulty for task in dataset.tasks
              if task.task_family_id and task.task_family_id.endswith("nested-group-total")]
    assert longest == ["hard", "hard"] and nested == ["easy", "easy"]


def test_family_counts_and_maximum(dataset):
    for category in CATEGORIES:
        counts = Counter(task.task_family_id for task in dataset.tasks if task.category == category)
        assert len(counts) == 16 and max(counts.values()) == 2


def test_split_totals_and_per_category(dataset, split):
    validate_split_manifest(dataset, split)
    assert Counter(item["split"] for item in split["entries"]) == {"train": 140, "development": 42, "final": 42}
    assert Counter((item["internal_category"], item["split"]) for item in split["entries"]) == Counter(
        {(category, part): count for category in CATEGORIES
         for part, count in {"train": 20, "development": 6, "final": 6}.items()})


def test_no_family_crosses_splits(split):
    family_splits = defaultdict(set)
    for item in split["entries"]:
        family_splits[item["task_family_id"]].add(item["split"])
    assert all(len(parts) == 1 for parts in family_splits.values())


def test_split_is_deterministic_and_outcome_free(dataset):
    assert build_split_manifest(dataset) == build_split_manifest(dataset)
    assert build_split_manifest(dataset)["outcome_features_used"] is False
    serialized = json.dumps(build_split_manifest(dataset))
    assert "acceptable" not in serialized and "quality_score" not in serialized


def test_ground_truth_positive_negative_and_oracles(dataset):
    audit = ground_truth_audit(dataset)
    assert audit["status"] == "PASS" and not audit["failures"]
    assert audit["counts"] == {"coding_static": 32, "objective_positive_negative": 160,
                               "qa_normalization": 32, "reasoning_oracles": 32,
                               "semantic_contracts": 32}


def test_all_evaluator_paths_are_explicit(dataset):
    manifest = build_evaluator_manifest(dataset)
    assert len(manifest["entries"]) == 224
    by_id = {task.task_id: task for task in dataset.tasks}
    assert all(entry["evaluator_type"] == EVALUATOR_TYPES[by_id[entry["task_id"]].category]
               for entry in manifest["entries"])
    assert sum(entry["semantic_judge_required"] for entry in manifest["entries"]) == 32
    assert sum(entry["functional_execution_required"] for entry in manifest["entries"]) == 32


@pytest.mark.docker_sandbox
def test_coding_oracles_pass_and_incorrect_solutions_fail(dataset):
    audit = asyncio.run(coding_functional_audit(dataset))
    assert audit["status"] == "PASS"
    assert audit["canonical_passed"] == audit["incorrect_rejected"] == 32


def test_semantic_contracts_are_complete_and_local(dataset):
    summaries = [task for task in dataset.tasks if task.category == "summarization"]
    assert len(summaries) == 32
    for task in summaries:
        metadata = task.evaluation_metadata
        assert metadata["required_facts"] and metadata["semantic_requirements"]
        assert metadata["judge_contract"] == {"judge_version": "1.0.0",
                                               "prompt_version": "summary-rubric-1.0.0",
                                               "factual_consistency_veto": True}
        assert metadata["deterministic_constraints"]["max_words"] * 1.5 <= task.max_output_tokens


def test_coding_variants_have_independent_behaviors_and_fixtures(dataset):
    families = defaultdict(list)
    for task in dataset.tasks:
        if task.category == "coding":
            families[task.task_family_id].append(task)
    assert len(families) == 16
    for pair in families.values():
        left, right = sorted(pair, key=lambda task: task.family_variant)
        assert left.evaluation_metadata["function_name"] != right.evaluation_metadata["function_name"]
        assert left.evaluation_metadata["functional_tests"] != right.evaluation_metadata["functional_tests"]


def test_ascii_slug_semantics_are_explicit_and_executed(dataset):
    tasks = [task for task in dataset.tasks if task.task_family_id and task.task_family_id.endswith("slug-validation")]
    assert len(tasks) == 2
    assert all("ASCII" in task.prompt or "a-z" in task.prompt for task in tasks)
    alternate = next(task for task in tasks if "ASCII digits 0-9" in task.prompt)
    cases = alternate.evaluation_metadata["functional_tests"]
    assert any(case["args"] == ["item-٣"] and case["expected"] == [] for case in cases)


def test_summarization_pairs_have_independent_sources_and_facts(dataset):
    families = defaultdict(list)
    for task in dataset.tasks:
        if task.category == "summarization":
            families[task.task_family_id].append(task)
    assert len(families) == 16
    for pair in families.values():
        left, right = sorted(pair, key=lambda task: task.family_variant)
        assert left.prompt != right.prompt
        assert left.evaluation_metadata["required_facts"] != right.evaluation_metadata["required_facts"]


def test_qa_equivalence_and_malformed_prompt_correction(dataset):
    qa = [task for task in dataset.tasks if task.category == "qa"]
    assert len(qa) == 32
    assert all("before nothing" not in task.prompt for task in qa)
    for task in qa:
        accepted = str(task.evaluation_metadata["accepted_answers"][0])
        positive = f"  {accepted.upper()}!  "
        assert evaluator_for(task).score(task, positive)[0] == 1.0
        assert evaluator_for(task).score(task, "definitely-wrong")[0] == 0.0


def test_content_quality_gate_and_pilot_acceptance_targets(dataset, split, pilot):
    audit = content_quality_audit(dataset, split, pilot)
    assert audit["status"] == "PASS"
    assert audit["family_variation"]["TOO_COSMETIC"] == 0
    assert audit["malformed_prompts"] == []
    assert audit["trivial_no_routing_signal"] == 0
    assert audit["pilot"] == {"tasks": 21, "trivial_no_routing_signal": 0,
                              "high_ambiguity": 0, "high_evaluator_risk": 0}
    assert audit["final_test"] == {"tasks": 42, "status": "SUITABLE_TO_FREEZE"}
    assert audit["coding_needs_review"] == audit["summarization_needs_review"] == 0


def test_normalized_exact_and_canonical_duplicate_primitives():
    assert normalize_content(" A  12! ") == "a 12"
    assert canonicalize_content("Box 12") == canonicalize_content("Box 19")
    assert ngram_jaccard("a b c d", "a b c e", 3) == pytest.approx(1 / 3)
    assert edit_similarity("one two three", "one two four") == pytest.approx(2 / 3)


def test_duplicate_audit_is_deterministic_and_historical(dataset):
    paths = [Path("benchmarks/datasets/foundation-v1.json"),
             Path("benchmarks/datasets/foundation-v2.json"),
             Path("benchmarks/datasets/foundation-v3.json")]
    first = duplicate_audit(dataset, paths)
    second = duplicate_audit(dataset, paths)
    assert first == second and first["unresolved_count"] == 0
    assert set(first["historical_datasets"]) == {"routellm-foundation", "routellm-foundation-v2", "routellm-foundation-v3"}
    assert all(flag["disposition"] in {"same_family", "acceptable_thematic_overlap"} for flag in first["flags"])


def test_exact_duplicate_is_rejected_by_audit(dataset, tmp_path):
    tasks = list(dataset.tasks)
    duplicate = tasks[0].model_copy(update={"task_id": "classification-easy-999",
                                             "task_family_id": "rbv1-classification-16-compliance-hierarchy"})
    mutated = BenchmarkDataset(name=BENCHMARK_NAME, version=BENCHMARK_VERSION,
                               tasks=tuple(tasks + [duplicate]))
    audit = duplicate_audit(mutated, [])
    assert audit["unresolved_count"] >= 1
    assert any(flag["disposition"] == "true_duplicate_reject" for flag in audit["flags"])


def test_pilot_is_exact_balanced_train_only_and_multifamily(dataset, split, pilot):
    validate_pilot_manifest(pilot, split)
    assert len(pilot["tasks"]) == 21
    assert Counter(item["internal_category"] for item in pilot["tasks"]) == Counter({category: 3 for category in CATEGORIES})
    assert {item["split"] for item in pilot["tasks"]} == {"train"}
    assert pilot["expected_candidate_calls"] == 84
    assert pilot["maximum_expected_semantic_judge_calls"] == 12


def test_invalid_pilot_fails(dataset, split, pilot):
    invalid = {**pilot, "tasks": pilot["tasks"][:-1]}
    with pytest.raises(ValueError, match="21"):
        validate_pilot_manifest(invalid, split)


def test_token_budgets_fit_and_have_no_candidate_hacks(dataset):
    validate_routing_benchmark(dataset)
    serialized = json.dumps(dataset.model_dump(mode="json"))
    assert "candidate-nemotron" not in serialized and "candidate-gemini" not in serialized


def test_protocol_freezes_four_candidates_and_pricing_status(dataset, split):
    evaluator = build_evaluator_manifest(dataset)
    provenance = build_provenance_manifest(dataset)
    protocol = build_protocol("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    assert len(protocol["candidates"]) == 4
    assert protocol["version"] == PROTOCOL_VERSION
    assert protocol["pricing"]["status"] == "REQUIRES_REVERIFICATION"
    assert all(item["upstream_provider_pin"] for item in protocol["candidates"])
    assert protocol["final_evaluation_gate"]["explicit_flag_required"]
    assert protocol["output_allowance_semantics"]["candidate_slug_branching"] is False
    assert all(not model.output_token_policy.category_overrides
               for model in ROUTING_BENCHMARK_MODELS)


def test_cost_estimate_has_expected_call_counts(dataset, pilot):
    estimate = build_cost_estimate(dataset, pilot)
    assert estimate["pilot"]["candidate_calls"] == 84
    assert estimate["pilot"]["semantic_judge_calls"] == 12
    assert estimate["full"]["candidate_calls"] == 896
    assert estimate["full"]["semantic_judge_calls"] == 128
    assert estimate["pricing_status"] == "REQUIRES_REVERIFICATION"
    assert Decimal(estimate["pilot"]["worst_case_authorized_allowance"]["total_cost_usd"]) \
        >= Decimal(estimate["pilot"]["expected_total_cost_usd"])


def test_training_loader_and_normal_development_path_exclude_final(dataset, split, tmp_path):
    dataset_path = tmp_path / "dataset.json"
    split_path = tmp_path / "split.json"
    dataset_path.write_text(json.dumps(dataset.model_dump(mode="json")))
    split_path.write_text(json.dumps(split))
    training = load_training_dataset(dataset_path, split_path)
    assert len(training.tasks) == 140
    development = select_execution_dataset(dataset, split, split="development")
    assert len(development.tasks) == 42
    final_ids = {entry["task_id"] for entry in split["entries"] if entry["split"] == "final"}
    assert final_ids.isdisjoint(task.task_id for task in training.tasks + development.tasks)


def test_final_execution_requires_explicit_gate_and_identities(dataset, split):
    with pytest.raises(ValueError, match="explicit"):
        select_execution_dataset(dataset, split, split="final")
    with pytest.raises(ValueError, match="identities"):
        select_execution_dataset(dataset, split, split="final", allow_final_evaluation=True)
    final = select_execution_dataset(dataset, split, split="final", allow_final_evaluation=True,
                                     predictor_sha256="a" * 64, policy_sha256="b" * 64)
    assert len(final.tasks) == 42


def test_normal_benchmark_cli_rejects_final_execution(monkeypatch, capsys, tmp_path):
    from adaptive_llm_gateway.benchmarks.__main__ import main
    monkeypatch.setattr("sys.argv", ["benchmark", "--dataset", str(DEFAULT_DATASET),
                                    "--split", "final", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "final-evaluation gate" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_invalid_dataset_count_fails(dataset):
    invalid = dataset.model_copy(update={"tasks": dataset.tasks[:-1]})
    with pytest.raises(ValueError, match="224"):
        validate_routing_benchmark(invalid)


def test_family_leakage_fails(dataset, split):
    invalid = json.loads(json.dumps(split))
    family = invalid["entries"][0]["task_family_id"]
    pair = [entry for entry in invalid["entries"] if entry["task_family_id"] == family]
    pair[0]["split"] = "development"
    with pytest.raises(ValueError):
        validate_split_manifest(dataset, invalid)


def test_missing_ground_truth_fails_closed(dataset):
    task = next(task for task in dataset.tasks if task.category == "qa")
    broken = task.model_copy(update={"evaluation_metadata": {"difficulty": task.difficulty}})
    mutated = dataset.model_copy(update={"tasks": (broken,) + dataset.tasks[1:]})
    with pytest.raises((KeyError, ValueError)):
        ground_truth_audit(mutated)


def test_missing_evaluator_fails(dataset):
    task = dataset.tasks[0].model_copy(update={"evaluator_type": "wrong"})
    invalid = dataset.model_copy(update={"tasks": (task,) + dataset.tasks[1:]})
    with pytest.raises(ValueError, match="evaluator"):
        validate_routing_benchmark(invalid)


def test_repeated_build_is_byte_identical(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_artifacts(output_root=first / "artifacts", dataset_path=first / "dataset.json",
                    protocol_dir=first / "protocol", coding_audit=passing_coding_audit())
    build_artifacts(output_root=second / "artifacts", dataset_path=second / "dataset.json",
                    protocol_dir=second / "protocol", coding_audit=passing_coding_audit())
    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files


@pytest.mark.local_evidence
def test_canonical_hashes_match_written_artifacts():
    dataset = load_dataset(DEFAULT_DATASET)
    identities = json.loads(Path("artifacts/routing-benchmark-v1/identities.json").read_text())
    assert dataset.sha256 == identities["dataset_content_sha256"]
    names = {"split-manifest.json": "split_manifest_sha256",
             "evaluator-manifest.json": "evaluator_manifest_sha256",
             "provenance-manifest.json": "provenance_manifest_sha256",
             "protocol.json": "benchmark_protocol_sha256"}
    import hashlib
    for filename, key in names.items():
        assert hashlib.sha256((DEFAULT_PROTOCOL_DIR / filename).read_bytes()).hexdigest() == identities[key]


def test_security_has_no_secrets_or_provider_outputs(dataset):
    serialized = json.dumps(dataset.model_dump(mode="json")).casefold()
    forbidden = ("authorization: bearer", "begin private key", "api_key=",
                 "raw_provider_output", "provider_payload", "reasoning_text")
    assert not any(value in serialized for value in forbidden)
