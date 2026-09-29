import hashlib
import inspect
import json
from decimal import Decimal

import pytest

from adaptive_llm_gateway.api.schemas import AdaptiveInferencePayload, InferencePayload
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRoutingConfig
from adaptive_llm_gateway.routing.calibration_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FINE_THRESHOLDS,
    POLICY_CONCLUSIONS,
    frontier,
    grouped_bootstrap,
    load_grouped_oof_evidence,
    reliability_table,
    route_at_threshold,
    run_calibration_analysis,
)
from adaptive_llm_gateway.routing.ml_experiment import THRESHOLDS
from adaptive_llm_gateway.routing.provider_pin_ablation import (
    INTERACTION_NO_PROVIDER_PIN,
    NO_PIN_REPRESENTATION,
)


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    path = tmp_path_factory.mktemp("phase8g") / "report.json"
    report = run_calibration_analysis(report_path=path)
    return report, path


@pytest.fixture(scope="module")
def evidence():
    return load_grouped_oof_evidence()


def test_frozen_hashes_and_formulation_provenance(analysis):
    report, _ = analysis
    provenance = report["provenance"]
    assert provenance["foundation_v2_sha256"] == "dcab5e9b6f1b8bc063555637e06db0d32ec51056959a7586f2cc411949adc964"
    assert provenance["foundation_v3_sha256"] == "64eda8e7388233f645906412a2524982ebfb9c848c42ee24a468ba04c31d16e6"
    assert provenance["foundation_v3_protocol_sha256"] == "973c46e2dfd7c5739f623ff1c7f2378dc3e77c80e5815e23c975816700d4b861"
    assert provenance["phase_7a_design_sha256"] == "abb18acee90efd773a55eedd0f75ae69cb487b8b2f667c8d1aae336b0e8e434b"
    assert provenance["formulation"] == INTERACTION_NO_PROVIDER_PIN
    assert provenance["provider_pin_present"] is False
    assert "upstream_provider_pin" not in NO_PIN_REPRESENTATION.features


def test_grouped_oof_source_is_reproduced_without_full_fit_evidence(analysis):
    provenance = analysis[0]["provenance"]
    assert "grouped OOF" in provenance["prediction_source"]
    assert provenance["maximum_regeneration_probability_difference"] == 0
    assert provenance["fold_isolation_verified"] is True
    assert provenance["final_full_data_artifact_used_as_evidence"] is False


def test_exact_oof_shape_and_groups(analysis):
    provenance = analysis[0]["provenance"]
    assert provenance["prediction_rows"] == 224
    assert provenance["valid_labels"] == 216
    assert provenance["missing_labels"] == 8
    assert provenance["request_groups"] == 56
    assert provenance["outer_folds"] == 4
    assert provenance["grouping_key"] == "task_id"


def test_approved_baseline_metrics_reproduce(analysis):
    baseline = analysis[0]["baseline_reproduction"]
    assert baseline["status"] == "PASS"
    metrics = baseline["predictive_metrics"]
    assert metrics["log_loss"] == pytest.approx(0.407419898387, abs=1e-12)
    assert metrics["brier_score"] == pytest.approx(0.129855392134, abs=1e-12)
    assert metrics["roc_auc"] == pytest.approx(0.888299260548, abs=1e-12)
    assert metrics["average_precision"] == pytest.approx(0.900570343522, abs=1e-12)
    assert baseline["top_1_acceptable_rate"] == pytest.approx(43 / 54)
    assert baseline["pairwise_ranking_accuracy"] == pytest.approx(115 / 126)


def test_global_calibration_metrics_and_slope_are_finite(analysis):
    calibration = analysis[0]["calibration"]
    assert calibration["global_metrics"]["brier_score"] == pytest.approx(0.129855392134)
    assert calibration["global_metrics"]["log_loss"] == pytest.approx(0.407419898387)
    assert calibration["slope_intercept"]["slope"] == pytest.approx(1.17271272947)
    assert calibration["slope_intercept"]["intercept"] == pytest.approx(-0.040298991467)


@pytest.mark.parametrize(("name", "ece", "mce"), [
    ("equal_width_10_bins", 0.032252514199, 0.158916141277),
    ("equal_frequency_10_bins", 0.05652680565, 0.173322383352),
])
def test_ece_mce_are_deterministic(analysis, name, ece, mce):
    value = analysis[0]["calibration"][name]
    assert value["ece"] == pytest.approx(ece, abs=1e-12)
    assert value["mce"] == pytest.approx(mce, abs=1e-12)
    assert sum(item["count"] for item in value["bins"]) == 216


def test_equal_width_empty_bins_are_retained():
    records = [
        {"label_status": "valid", "target": False, "predicted_probability": 0.05},
        {"label_status": "valid", "target": True, "predicted_probability": 0.95},
    ]
    result = reliability_table(records, bins=5, strategy="equal_width")
    assert result["reported_bins"] == 5
    assert result["empty_bins"] == 3
    assert sum(item["count"] for item in result["bins"]) == 2


def test_quantile_bins_partition_every_valid_row():
    records = [
        {"label_status": "valid", "target": index % 2 == 0,
         "predicted_probability": index / 22}
        for index in range(22)
    ]
    result = reliability_table(records, bins=10, strategy="equal_frequency")
    assert result["reported_bins"] == 10
    assert sum(item["count"] for item in result["bins"]) == 22
    assert max(item["count"] for item in result["bins"]) <= 3


def test_candidate_diagnostics_preserve_actual_counts(analysis):
    diagnostics = analysis[0]["candidate_diagnostics"]
    assert {name: value["rows"] for name, value in diagnostics.items()} == {
        "candidate-claude-sonnet-5": 54,
        "candidate-gemini-3-flash": 51,
        "candidate-gpt-6-luna": 55,
        "candidate-nemotron-3.5-lightning": 56,
    }
    assert all("at most 56 request groups" in value["warning"] for value in diagnostics.values())


def test_category_diagnostics_are_canonical_and_small_sample_limited(analysis):
    diagnostics = analysis[0]["category_diagnostics"]
    assert set(diagnostics) == {
        "classification", "coding", "extraction", "structured_json", "qa",
        "reasoning", "summarization",
    }
    assert all(value["independent_requests"] == 8 for value in diagnostics.values())
    assert all("diagnostic only" in value["warning"] for value in diagnostics.values())
    assert sum(value["valid_candidate_labels"] for value in diagnostics.values()) == 216


def test_threshold_grids_are_frozen_and_deterministic(analysis):
    routing = analysis[0]["routing"]
    assert tuple(routing["coarse_threshold_grid"]) == THRESHOLDS
    assert tuple(routing["fine_analysis_only_grid"]) == FINE_THRESHOLDS
    assert FINE_THRESHOLDS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def test_routing_missing_labels_keep_cost_and_leave_quality_denominator(analysis):
    for summary in analysis[0]["routing"]["coarse"].values():
        assert summary["requests_routed"] == 56
        assert summary["valid_selected_labels"] + summary["missing_selected_labels"] == 56
        assert summary["acceptable_selections"] + summary["unacceptable_selections"] == summary["valid_selected_labels"]
        assert Decimal(summary["total_projected_cost_usd"]) > 0


def test_canonical_policy_and_projected_cost_are_reused(evidence):
    dataset, records, _, _ = evidence
    selections = route_at_threshold(dataset, records, 0.8)
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    assert all(
        item["projected_cost_usd"]
        == row_index[(item["task_id"], item["candidate_id"])].projected_cost_usd
        for item in selections
    )
    source = inspect.getsource(route_at_threshold)
    assert "CostAwareRoutingPolicy" in source
    assert "CandidatePrediction" in source


def test_distribution_fallback_and_threshold_counts_are_complete(analysis):
    for summary in analysis[0]["routing"]["fine"].values():
        assert sum(summary["selected_model_distribution"].values()) == 56
        assert summary["fallback_count"] + summary["threshold_met_count"] == 56


def test_grouped_bootstrap_is_deterministic_and_request_level(evidence):
    dataset, records, _, _ = evidence
    selections = {"0.80": route_at_threshold(dataset, records, 0.8)}
    first = grouped_bootstrap(selections, seed=31, replicates=100)
    second = grouped_bootstrap(selections, seed=31, replicates=100)
    assert first == second
    assert first["unit"] == "independent request group"
    assert first["replicates"] == 100


def test_analysis_bootstrap_contract_and_intervals(analysis):
    uncertainty = analysis[0]["grouped_uncertainty"]
    assert uncertainty["seed"] == BOOTSTRAP_SEED
    assert uncertainty["replicates"] == BOOTSTRAP_REPLICATES
    assert set(uncertainty["thresholds"]) == {f"{value:.2f}" for value in THRESHOLDS}
    for metrics in uncertainty["thresholds"].values():
        assert all(len(interval) == 2 and interval[0] <= interval[1] for interval in metrics.values())


def test_baselines_are_matched_and_oracle_is_analysis_only(analysis):
    baselines = analysis[0]["baselines"]
    assert {"ALWAYS_CHEAPEST", "RULE_BASED_V1", "FOLD_LOCAL_ALWAYS_STRONGEST", "ORACLE_CHEAPEST_ACCEPTABLE"} <= set(baselines["policies"])
    assert baselines["policies"]["FOLD_LOCAL_ALWAYS_STRONGEST"]["requests_routed"] == 56
    assert baselines["policies"]["ALWAYS_CHEAPEST"]["requests_routed"] == 56
    assert baselines["policies"]["RULE_BASED_V1"]["requests_routed"] == 56
    assert baselines["policies"]["ORACLE_CHEAPEST_ACCEPTABLE"]["not_deployable"] is True
    assert "not deployable" in baselines["oracle_warning"]


def test_frontier_requires_comparable_coverage_and_selects_no_winner(analysis):
    result = analysis[0]["frontier"]
    assert "equal valid-selected-label counts" in result["coverage_comparison_rule"]
    assert result["automatic_winner_selected"] is False
    assert result["non_dominated_thresholds"] == ["0.50", "0.55", "0.60", "0.65", "0.95"]


def test_frontier_dominance_logic_unit_case():
    data = {
        "a": {"valid_selected_labels": 10, "average_projected_cost_per_request_usd": "1", "observed_acceptable_rate_among_valid": 0.8},
        "b": {"valid_selected_labels": 10, "average_projected_cost_per_request_usd": "2", "observed_acceptable_rate_among_valid": 0.7},
        "c": {"valid_selected_labels": 9, "average_projected_cost_per_request_usd": "0.5", "observed_acceptable_rate_among_valid": 0.9},
    }
    assert frontier(data)["non_dominated_thresholds"] == ["a", "c"]


def test_threshold_stability_flags_descriptive_brittle_regions(analysis):
    stability = analysis[0]["threshold_stability"]
    assert stability["brittle_intervals"] == ["0.50->0.55", "0.65->0.70", "0.85->0.90"]
    assert all("flagged_brittle" in item for item in stability["adjacent_fine_thresholds"])


def test_platt_comparison_has_no_group_leakage_and_same_rows(analysis):
    experiment = analysis[0]["post_hoc_calibration"]
    assert experiment["status"] == "EXPLORATORY_ONLY"
    assert experiment["raw_metrics"]["rows"] == experiment["cross_fitted_platt_metrics"]["rows"] == 216
    assert all(item["group_overlap"] == 0 for item in experiment["outer_group_audit"])
    assert sum(item["evaluation_rows"] for item in experiment["outer_group_audit"]) == 224
    assert set(experiment["routing_on_same_evaluation_predictions"]) == {f"{value:.2f}" for value in THRESHOLDS}


def test_platt_does_not_improve_proper_losses_or_ece(analysis):
    experiment = analysis[0]["post_hoc_calibration"]
    assert experiment["cross_fitted_platt_metrics"]["log_loss"] > experiment["raw_metrics"]["log_loss"]
    assert experiment["cross_fitted_platt_metrics"]["brier_score"] > experiment["raw_metrics"]["brier_score"]
    assert experiment["cross_fitted_platt_equal_width_ece"] > experiment["raw_equal_width_ece"]
    assert experiment["cross_fitted_platt_equal_frequency_ece"] > experiment["raw_equal_frequency_ece"]
    assert experiment["production_predictor_replaced"] is False


def test_isotonic_skip_reason_and_feasibility_are_explicit(analysis):
    experiment = analysis[0]["post_hoc_calibration"]
    assert experiment["isotonic"].startswith("SKIPPED:")
    assert "56 independent request groups" in experiment["isotonic"]
    assert "exploratory" in experiment["feasibility_conclusion"]


def test_policy_conclusion_is_allowed_and_restrained(analysis):
    decision = analysis[0]["policy_decision"]
    assert decision["conclusion"] in POLICY_CONCLUSIONS
    assert decision["conclusion"] == "MORE_DATA_REQUIRED_BEFORE_POLICY_FREEZE"
    assert decision["Q5_production_default_now"] == "No."
    assert decision["Q6_api_caller_threshold"].startswith("Yes")
    assert decision["Q7_coarse_quality_modes"].startswith("No")


def test_quality_modes_remain_analysis_only(analysis):
    quality_modes = analysis[0]["quality_modes"]
    assert quality_modes["implemented"] is False
    assert quality_modes["conclusion"] == "NOT_ENOUGH_EVIDENCE_YET"


def test_production_api_and_config_contracts_are_unchanged():
    assert InferencePayload.model_fields["model_id"].is_required()
    assert not AdaptiveInferencePayload.model_fields["quality_threshold"].is_required()
    assert AdaptiveInferencePayload.model_fields["category"].is_required()
    assert "quality_threshold" not in AdaptiveRoutingConfig.model_fields
    assert "quality_mode" not in AdaptiveInferencePayload.model_fields


def test_analysis_has_no_provider_telemetry_or_full_fit_predictor_dependency():
    import adaptive_llm_gateway.routing.calibration_analysis as module

    source = inspect.getsource(module)
    assert "SklearnQualityPredictor" not in source
    assert "from .predictor" not in source
    assert "provider.generate" not in source
    assert "telemetry" not in source.lower()
    assert "upstream_provider_pin" in source  # only the explicit leakage guard


def test_generated_artifact_schema_and_privacy(analysis):
    report, path = analysis
    assert json.loads(path.read_text()) == report
    assert set(report) >= {
        "provenance", "calibration", "candidate_diagnostics", "category_diagnostics",
        "routing", "grouped_uncertainty", "baselines", "frontier",
        "post_hoc_calibration", "policy_decision", "limitations",
    }
    text = path.read_text().lower()
    for forbidden in ("raw_prompt", "raw_response", "authorization", "api_key", "provider_payload"):
        assert forbidden not in text


def test_deterministic_analysis_rerun_is_byte_identical(analysis, tmp_path):
    _, first_path = analysis
    second_path = tmp_path / "second.json"
    first = first_path.read_bytes()
    run_calibration_analysis(report_path=second_path)
    assert hashlib.sha256(first).digest() == hashlib.sha256(second_path.read_bytes()).digest()
