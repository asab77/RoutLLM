from collections import Counter

import pytest

from adaptive_llm_gateway.routing.phase9_retraining import (
    build_grouped_folds, fit_candidate, generate_oof, load_training_evidence,
)
from adaptive_llm_gateway.routing.quality_features import CANONICAL_PREDICTIVE_FEATURES


@pytest.mark.local_evidence
def test_phase9_training_evidence_preserves_missing_labels():
    dataset, _, provenance = load_training_evidence()
    assert len(dataset.rows) == 560
    assert Counter(row.label_status for row in dataset.rows) == Counter(
        {"valid": 557, "missing": 3})
    assert all(row.acceptable is None for row in dataset.rows
               if row.label_status == "missing")
    assert len(provenance["missing_labels"]) == 3


@pytest.mark.local_evidence
def test_phase9_folds_are_grouped_and_category_balanced():
    dataset, _, _ = load_training_evidence()
    folds = build_grouped_folds(dataset)
    assert len(folds) == 5
    assert all(item["request_count"] == 28 for item in folds)
    assert all(set(item["category_counts"].values()) == {4} for item in folds)
    task_ids = [task for item in folds for task in item["task_ids"]]
    assert len(task_ids) == len(set(task_ids)) == 140


@pytest.mark.local_evidence
def test_phase9_oof_covers_every_row_without_group_leakage():
    dataset, _, _ = load_training_evidence()
    records, audits = generate_oof(dataset, build_grouped_folds(dataset))
    assert len(records) == 560
    assert len({(item["task_id"], item["candidate_id"]) for item in records}) == 560
    assert all(item["group_overlap"] == 0 for item in audits)
    assert sum(item["test_rows"] for item in audits) == 560


@pytest.mark.local_evidence
def test_phase9_retains_canonical_leakage_free_contract_and_fits_valid_rows(tmp_path):
    assert len(CANONICAL_PREDICTIVE_FEATURES) == 16
    forbidden = {
        "provider_pin", "latency_ms", "actual_token_usage", "actual_cost",
        "provider_outcome", "judge_output", "benchmark_label", "quality_score",
    }
    assert forbidden.isdisjoint(CANONICAL_PREDICTIVE_FEATURES)
    dataset, _, _ = load_training_evidence()
    metadata = fit_candidate(dataset, tmp_path)
    assert metadata["valid_training_rows"] == 557
    assert metadata["missing_label_rows"] == 3
    assert (tmp_path / "predictor.pkl").is_file()
    assert (tmp_path / "metadata.json").is_file()
