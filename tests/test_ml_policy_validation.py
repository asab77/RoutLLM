import hashlib
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from adaptive_llm_gateway.routing.ml_diagnostics import (
    CATEGORY_CANDIDATE_INTERACTION,
    FULL_ADDITIVE,
)
from adaptive_llm_gateway.routing.ml_experiment import THRESHOLDS, select_cost_aware_candidate
from adaptive_llm_gateway.routing.ml_features import (
    FORBIDDEN_PREDICTIVE_FIELDS,
    MLExperimentRow,
    build_outer_folds,
    load_ml_dataset,
)
from adaptive_llm_gateway.routing.ml_policy_validation import (
    NO_CATEGORY_REPRESENTATION,
    PROTECTED_PHASE_7_FILES,
    compare_policy_points,
    run_policy_validation,
)

RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
RESULT_ROOT = Path("benchmark-results")
PHASE_7 = RESULT_ROOT / str(RUN_ID) / "phase-7"


@pytest.fixture(scope="module")
def dataset():
    return load_ml_dataset(RESULT_ROOT, RUN_ID)


@pytest.fixture(scope="module")
def validation():
    return run_policy_validation(RESULT_ROOT, RUN_ID)


def _synthetic_row(candidate_id: str, output_cost: str, realized_cost: str):
    return MLExperimentRow(
        task_id="synthetic-task",
        candidate_id=candidate_id,
        category="qa",
        features={
            "category": "qa",
            "candidate_id": candidate_id,
            "upstream_provider_pin": "offline",
            "reasoning_effort": "none",
            "prompt_characters": 40.0,
            "approximate_input_tokens": 10.0,
            "requested_max_output_tokens": 20.0,
            "constraint_indicator_count": 0.0,
            "reasoning_indicator_count": 0.0,
            "configured_input_cost_per_1m_tokens": 1.0,
            "configured_output_cost_per_1m_tokens": float(output_cost),
            "context_window": 4096.0,
            "effective_max_output_tokens": 20.0,
            "contains_code": False,
            "requests_structured_output": False,
            "supports_temperature": True,
        },
        label_status="valid",
        acceptable=True,
        analysis_difficulty="easy",
        realized_cost_usd=Decimal(realized_cost),
        latency_ms=1,
    )


@pytest.mark.local_evidence
def test_exact_phase_7b_folds_are_reused(dataset, validation):
    _, results = validation
    stored = json.loads((PHASE_7 / "fold-assignments.json").read_text())["folds"]
    generated = [item.model_dump(mode="json") for item in build_outer_folds(dataset)]
    assert generated == stored
    assert results["integrity"]["phase_7b_folds_reused_exactly"] is True


@pytest.mark.local_evidence
def test_full_frontier_evaluates_all_56_requests(validation):
    _, results = validation
    for variant in (FULL_ADDITIVE, CATEGORY_CANDIDATE_INTERACTION):
        for summary in results["frontier"]["variants"][variant].values():
            assert summary["total_requests"] == 56
            assert summary["selected_rows"] == 56


@pytest.mark.local_evidence
def test_threshold_grid_is_frozen(validation):
    _, results = validation
    assert THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    assert results["frontier"]["threshold_grid"] == list(THRESHOLDS)


def test_selector_uses_projected_pre_generation_cost():
    task_id = "synthetic-task"
    rows = [_synthetic_row("cheap", "1", "9"), _synthetic_row("expensive", "5", "1")]
    records = [{"candidate_id": row.candidate_id, "predicted_probability": 0.8}
               for row in rows]
    index = {(row.task_id, row.candidate_id): row for row in rows}
    selected, fallback = select_cost_aware_candidate(task_id, records, index, 0.5)
    expected = min(rows, key=lambda row: (row.projected_cost_usd, row.candidate_id))
    assert selected["candidate_id"] == expected.candidate_id
    assert fallback is False


def test_realized_cost_cannot_change_selection():
    task_id = "synthetic-task"
    rows = [_synthetic_row("cheap", "1", "9"), _synthetic_row("expensive", "5", "1")]
    records = [{"candidate_id": row.candidate_id, "predicted_probability": 0.8}
               for row in rows]
    first_index = {(row.task_id, row.candidate_id): row for row in rows}
    first = select_cost_aware_candidate(task_id, records, first_index, 0.5)[0]
    changed = [rows[0].model_copy(update={"realized_cost_usd": Decimal("999")}),
               rows[1].model_copy(update={"realized_cost_usd": Decimal("0")})]
    second_index = {(row.task_id, row.candidate_id): row for row in changed}
    second = select_cost_aware_candidate(task_id, records, second_index, 0.5)[0]
    assert first["candidate_id"] == second["candidate_id"]


@pytest.mark.local_evidence
def test_missing_selected_labels_are_visible_and_not_negative(validation):
    _, results = validation
    summary = results["frontier"]["variants"][CATEGORY_CANDIDATE_INTERACTION]["0.50"]
    assert summary["missing_selected_labels"] == 1
    assert summary["acceptable"] + summary["unacceptable"] == summary["valid_selected_labels"]
    assert summary["valid_selected_labels"] + summary["missing_selected_labels"] == 56


def test_pareto_logic_detects_dominance():
    better = {
        "valid_selected_labels": 56,
        "acceptable_rate_among_valid": 0.8,
        "total_realized_cost_usd": "1.0",
    }
    worse = {
        "valid_selected_labels": 56,
        "acceptable_rate_among_valid": 0.7,
        "total_realized_cost_usd": "2.0",
    }
    assert compare_policy_points(better, worse) == "first_dominates"
    assert compare_policy_points(worse, better) == "second_dominates"


def test_pareto_logic_is_coverage_aware():
    first = {
        "valid_selected_labels": 55,
        "acceptable_rate_among_valid": 0.9,
        "total_realized_cost_usd": "1.0",
    }
    second = {
        "valid_selected_labels": 56,
        "acceptable_rate_among_valid": 0.7,
        "total_realized_cost_usd": "2.0",
    }
    assert compare_policy_points(first, second) == "incomparable_coverage"


@pytest.mark.local_evidence
def test_fold_accounting_covers_each_request_once(validation):
    _, results = validation
    for variant in results["stability"]["variants"].values():
        for threshold in variant.values():
            assert sum(item["requests"] for item in threshold["folds"]) == 56
            assert sum(item["valid_selected_labels"] + item["missing_selected_labels"]
                       for item in threshold["folds"]) == 56


@pytest.mark.local_evidence
def test_each_outer_fold_contains_exactly_14_requests(validation):
    _, results = validation
    assert results["stability"]["fold_request_count"] == 14
    for variant in results["stability"]["variants"].values():
        for threshold in variant.values():
            assert all(item["requests"] == 14 for item in threshold["folds"])


@pytest.mark.local_evidence
def test_category_routing_accounts_for_all_seven_categories(validation):
    _, results = validation
    categories = results["categories"]["categories"]
    assert set(categories) == {
        "classification", "coding", "extraction", "json", "qa", "reasoning",
        "summarization",
    }
    assert len(categories) * results["categories"]["category_request_count"] == 56


@pytest.mark.local_evidence
def test_each_category_contains_exactly_eight_requests(validation):
    _, results = validation
    for category in results["categories"]["categories"].values():
        assert all(item["requests"] == 8 for item in category["thresholds"].values())


def test_no_category_representation_excludes_category():
    assert "category" not in NO_CATEGORY_REPRESENTATION.features


def test_no_category_representation_excludes_interactions():
    assert "category_candidate" not in NO_CATEGORY_REPRESENTATION.features
    assert all("category_candidate" not in name for name in NO_CATEGORY_REPRESENTATION.features)


def test_no_category_representation_has_no_forbidden_fields():
    assert not set(NO_CATEGORY_REPRESENTATION.features) & FORBIDDEN_PREDICTIVE_FIELDS


@pytest.mark.local_evidence
def test_rule_vs_interaction_disagreement_accounting(validation):
    _, results = validation
    comparisons = results["rules"]["ml_vs_rule"][
        CATEGORY_CANDIDATE_INTERACTION]["threshold_selected"]
    for comparison in comparisons.values():
        assert comparison["agreement_count"] + comparison["disagreement_count"] == 56
        assert sum(comparison["disagreement_outcomes"].values()) == comparison["disagreement_count"]


@pytest.mark.local_evidence
def test_oracle_opportunity_accounting(validation):
    _, results = validation
    for threshold in results["oracle"]["thresholds"].values():
        assert threshold["requests"] == 56
        assert sum(threshold["counts"].values()) == 56
        assert len(threshold["details"]) == 56


@pytest.mark.local_evidence
def test_artifacts_are_deterministic_and_protected_artifacts_unchanged(validation):
    paths, _ = validation
    new_before = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for name, path in paths.items()}
    protected_before = {
        name: hashlib.sha256((PHASE_7 / name).read_bytes()).hexdigest()
        for name in PROTECTED_PHASE_7_FILES
    }
    run_policy_validation(RESULT_ROOT, RUN_ID)
    new_after = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for name, path in paths.items()}
    protected_after = {
        name: hashlib.sha256((PHASE_7 / name).read_bytes()).hexdigest()
        for name in PROTECTED_PHASE_7_FILES
    }
    assert new_before == new_after
    assert protected_before == protected_after
