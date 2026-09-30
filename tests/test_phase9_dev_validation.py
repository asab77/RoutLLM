from collections import Counter

import pytest

from adaptive_llm_gateway.routing.phase9_dev_validation import (
    PREDICTOR_SHA256, THRESHOLD, _predict, analyze, load_dev_evidence,
)


@pytest.mark.local_evidence
def test_dev_evidence_preserves_all_missing_labels():
    dataset, _, manifest, _ = load_dev_evidence()
    assert manifest["configuration"]["execution_split"] == "development"
    assert len(dataset.rows) == 168
    assert Counter(row.label_status for row in dataset.rows) == Counter(
        {"valid": 140, "missing": 28})
    assert all(row.acceptable is None for row in dataset.rows
               if row.label_status == "missing")


@pytest.mark.local_evidence
def test_frozen_predictor_generates_one_prediction_per_dev_pair():
    dataset, _, _, _ = load_dev_evidence()
    records = _predict(dataset)
    assert len(records) == 168
    assert len({(item["task_id"], item["candidate_id"]) for item in records}) == 168


@pytest.mark.local_evidence
def test_dev_analysis_uses_only_frozen_threshold_and_predictor():
    report = analyze()
    assert THRESHOLD == 0.80
    assert report["source"]["predictor_sha256"] == PREDICTOR_SHA256
    assert report["source"]["threshold"] == 0.80
    assert report["source"]["final_evaluation_explicitly_unlocked"] is False
    assert report["routing_threshold_0_80"]["requests_routed"] == 42
    assert report["baselines"]["ORACLE_CHEAPEST_ACCEPTABLE"][
        "abstain_no_acceptable"] == 2
