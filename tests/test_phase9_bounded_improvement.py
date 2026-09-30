import pytest

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.routing.bounded_features import (
    BOUNDED_FEATURE_NAMES,
    extract_bounded_structural_features,
)
from adaptive_llm_gateway.routing.phase9_bounded_improvement import analyze


def test_numeric_quantity_count_handles_signed_decimal_and_percent_values():
    result = extract_bounded_structural_features(InferenceRequest(
        prompt="Compare 12, -3.5, and 8%. Code P1 is not a numeric quantity."
    ))
    assert result.numeric_quantity_count == 3


def test_conditional_operator_count_uses_general_request_language():
    result = extract_bounded_structural_features(InferenceRequest(
        prompt="If approved, proceed; otherwise stop unless an exception applies."
    ))
    assert result.conditional_operator_count == 3


def test_symbolic_math_operator_count_handles_compound_operators_once():
    result = extract_bounded_structural_features(InferenceRequest(
        prompt="Return whether a >= b + c / 2."
    ))
    assert result.symbolic_math_operator_count == 3


def test_comparison_structure_count_uses_pre_generation_text_only():
    result = extract_bounded_structural_features(InferenceRequest(
        prompt="Choose the highest value below 10 after filtering.",
        system_prompt="Use at most one result.",
    ))
    assert result.comparison_structure_count == 4


@pytest.mark.local_evidence
def test_bounded_experiment_is_train_only_and_rejects_the_augmentation(tmp_path):
    report = analyze(tmp_path)
    assert tuple(report["features"]["tested"]) == BOUNDED_FEATURE_NAMES
    assert report["source"]["training_requests"] == 140
    assert report["source"]["valid_training_rows"] == 557
    assert report["source"]["development_labels_used"] == 0
    assert report["source"]["final_labels_used"] == 0
    assert report["decision"] == "CURRENT_ROUTER_RETAINED"
    assert report["acceptance"]["predictor_artifact_created"] is False
    assert report["acceptance"]["feature_contract_version_created"] is False
    assert (tmp_path / "analysis.json").is_file()
