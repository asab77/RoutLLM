"""Phase 8G grouped-OOF calibration and quality/cost policy analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .analysis import (
    ALWAYS_CHEAPEST,
    FOUNDATION_V2_SHA256,
    FOUNDATION_V3_PROTOCOL_SHA256,
    FOUNDATION_V3_SHA256,
    ORACLE_CHEAPEST_ACCEPTABLE,
    RULE_BASED_V1,
    AlwaysCheapestPolicy,
    RuleBasedV1Policy,
    load_frozen_foundation_v3,
)
from .ml_experiment import DESIGN_SHA256, FOLD_LOCAL_ALWAYS_STRONGEST, THRESHOLDS, fold_local_strongest_decisions
from .ml_features import load_ml_dataset
from .ml_diagnostics import _rank_variant
from .policy import CandidatePrediction, CostAwareRoutingPolicy
from .provider_pin_ablation import (
    INTERACTION_NO_PROVIDER_PIN,
    NO_PIN_REPRESENTATION,
    generate_predictions,
)

RUN_ID = UUID("61707aba-5ab2-4c16-8ec3-eed74555d69c")
ROOT = Path("benchmark-results")
OOF_RELATIVE_PATH = Path("phase-8c0/provider-pin-ablation-predictions.jsonl")
SUMMARY_RELATIVE_PATH = Path("phase-8c0/provider-pin-ablation.json")
DEFAULT_REPORT_PATH = Path("artifacts/routing-analysis/phase-8g-calibration.json")
FINE_THRESHOLDS = tuple(round(value / 100, 2) for value in range(50, 96, 5))
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_REPLICATES = 2000
POLICY_CONCLUSIONS = (
    "SUPPORTED_GLOBAL_DEFAULT",
    "KEEP_CALLER_THRESHOLD_NO_DEFAULT",
    "CALIBRATION_REQUIRED_BEFORE_DEFAULT",
    "MORE_DATA_REQUIRED_BEFORE_POLICY_FREEZE",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 12)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def load_grouped_oof_evidence(root: Path = ROOT, run_id: UUID = RUN_ID):
    """Validate stored Phase 8C-0 OOF evidence by deterministic regeneration."""
    base = root / str(run_id)
    frozen_paths = {
        "foundation_v2_sha256": Path("benchmarks/datasets/foundation-v2.json"),
        "foundation_v3_sha256": Path("benchmarks/datasets/foundation-v3.json"),
        "foundation_v3_protocol_sha256": Path("benchmarks/protocols/foundation-v3.json"),
        "phase_7a_design_sha256": base / "phase-7/ml-router-design.json",
    }
    expected_hashes = {
        "foundation_v2_sha256": FOUNDATION_V2_SHA256,
        "foundation_v3_sha256": FOUNDATION_V3_SHA256,
        "foundation_v3_protocol_sha256": FOUNDATION_V3_PROTOCOL_SHA256,
        "phase_7a_design_sha256": DESIGN_SHA256,
    }
    actual_hashes = {name: _sha256(path) for name, path in frozen_paths.items()}
    if actual_hashes != expected_hashes:
        raise ValueError("frozen calibration-analysis input hash mismatch")
    prediction_path = base / OOF_RELATIVE_PATH
    summary_path = base / SUMMARY_RELATIVE_PATH
    stored_all = tuple(
        json.loads(line) for line in prediction_path.read_text().splitlines() if line
    )
    stored = tuple(
        item for item in stored_all if item["variant"] == INTERACTION_NO_PROVIDER_PIN
    )
    dataset = load_ml_dataset(root, run_id)
    regenerated_all, audits, _, folds = generate_predictions(dataset)
    regenerated = tuple(
        item for item in regenerated_all
        if item["variant"] == INTERACTION_NO_PROVIDER_PIN
    )
    key = lambda item: (str(item["task_id"]), str(item["candidate_id"]))
    stored = tuple(sorted(stored, key=key))
    regenerated = tuple(sorted(regenerated, key=key))
    if len(stored_all) != 448 or len(stored) != 224 or len(regenerated) != 224:
        raise ValueError("Phase 8C-0 OOF prediction shape mismatch")
    if [key(item) for item in stored] != [key(item) for item in regenerated]:
        raise ValueError("stored and regenerated OOF associations differ")
    maximum_delta = max(
        abs(float(actual["predicted_probability"]) - float(expected["predicted_probability"]))
        for actual, expected in zip(stored, regenerated)
    )
    if maximum_delta > 1e-15:
        raise ValueError("stored no-provider-pin OOF probabilities did not reproduce")
    if "upstream_provider_pin" in NO_PIN_REPRESENTATION.features:
        raise ValueError("provider pin leaked into the accepted formulation")
    for audit in audits[INTERACTION_NO_PROVIDER_PIN]:
        if set(audit["train_task_ids"]) & set(audit["test_task_ids"]):
            raise ValueError("grouped OOF fold leakage detected")
    task_ids = {str(item["task_id"]) for item in stored}
    if len(task_ids) != 56:
        raise ValueError("OOF evidence must contain 56 request groups")
    valid = sum(item["label_status"] == "valid" for item in stored)
    missing = len(stored) - valid
    if (valid, missing) != (216, 8):
        raise ValueError("OOF valid/missing label counts changed")
    summary = json.loads(summary_path.read_text())
    if summary.get("analysis") != "PHASE_8C_0_PROVIDER_PIN_ABLATION":
        raise ValueError("Phase 8C-0 provenance mismatch")
    return dataset, stored, folds, {
        "prediction_source": "stored grouped OOF Phase 8C-0 predictions validated by deterministic regeneration",
        "formulation": INTERACTION_NO_PROVIDER_PIN,
        "provider_pin_present": False,
        "prediction_rows": len(stored),
        "valid_labels": valid,
        "missing_labels": missing,
        "request_groups": len(task_ids),
        "outer_folds": len(folds),
        "grouping_key": "task_id",
        "fold_isolation_verified": True,
        "maximum_regeneration_probability_difference": maximum_delta,
        "oof_prediction_sha256": _sha256(prediction_path),
        "phase_8c0_summary_sha256": _sha256(summary_path),
        "final_full_data_artifact_used_as_evidence": False,
        **actual_hashes,
    }


def predictive_metrics(records) -> dict[str, object]:
    valid = [item for item in records if item["label_status"] == "valid"]
    target = np.asarray([int(bool(item["target"])) for item in valid], dtype=int)
    probability = np.asarray([float(item["predicted_probability"]) for item in valid])
    return {
        "rows": len(valid),
        "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
    }


def reliability_table(records, *, bins: int, strategy: str) -> dict[str, object]:
    valid = [item for item in records if item["label_status"] == "valid"]
    probability = np.asarray([float(item["predicted_probability"]) for item in valid])
    target = np.asarray([int(bool(item["target"])) for item in valid], dtype=int)
    memberships: list[np.ndarray]
    bounds: list[tuple[float, float]]
    if strategy == "equal_width":
        edges = np.linspace(0, 1, bins + 1)
        indexes = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
        memberships = [np.flatnonzero(indexes == index) for index in range(bins)]
        bounds = [(float(edges[index]), float(edges[index + 1])) for index in range(bins)]
    elif strategy == "equal_frequency":
        order = np.argsort(probability, kind="stable")
        memberships = [chunk for chunk in np.array_split(order, min(bins, len(order)))]
        bounds = [
            (float(probability[chunk].min()), float(probability[chunk].max()))
            for chunk in memberships
        ]
    else:
        raise ValueError("reliability strategy must be equal_width or equal_frequency")
    table = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, (members, (lower, upper)) in enumerate(zip(memberships, bounds)):
        count = len(members)
        predicted = float(probability[members].mean()) if count else None
        observed = float(target[members].mean()) if count else None
        gap = abs(predicted - observed) if count else None
        if gap is not None:
            weighted_gap += count / len(valid) * gap
            maximum_gap = max(maximum_gap, gap)
        table.append({
            "bin": index + 1,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": count,
            "mean_predicted_probability": predicted,
            "observed_acceptable_fraction": observed,
            "absolute_calibration_gap": gap,
        })
    return {
        "strategy": strategy,
        "requested_bins": bins,
        "reported_bins": len(table),
        "empty_bins": sum(item["count"] == 0 for item in table),
        "ece": weighted_gap,
        "mce": maximum_gap,
        "bins": table,
    }


def calibration_slope_intercept(records) -> dict[str, float]:
    valid = [item for item in records if item["label_status"] == "valid"]
    probability = np.clip(
        np.asarray([float(item["predicted_probability"]) for item in valid]),
        1e-12,
        1 - 1e-12,
    )
    target = np.asarray([int(bool(item["target"])) for item in valid], dtype=int)
    logit = np.log(probability / (1 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit, target)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def subgroup_diagnostics(records, dataset) -> tuple[dict, dict]:
    rows = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    candidates = {}
    for candidate_id in sorted({str(item["candidate_id"]) for item in records}):
        selected = [item for item in records if item["candidate_id"] == candidate_id]
        valid = [item for item in selected if item["label_status"] == "valid"]
        metric = predictive_metrics(valid)
        reliability = reliability_table(valid, bins=5, strategy="equal_frequency")
        candidates[candidate_id] = {
            **metric,
            "mean_predicted_probability": float(np.mean([
                float(item["predicted_probability"]) for item in valid
            ])),
            "observed_positive_rate": metric["positive"] / metric["rows"],
            "ece_equal_frequency_5_bins": reliability["ece"],
            "reliability": reliability,
            "warning": "candidate diagnostic uses at most 56 request groups",
        }
    categories = {}
    category_names = sorted({row.category for row in dataset.rows})
    for category in category_names:
        selected = [
            item for item in records
            if rows[(str(item["task_id"]), str(item["candidate_id"]))].category == category
            and item["label_status"] == "valid"
        ]
        probability = [float(item["predicted_probability"]) for item in selected]
        target = [int(bool(item["target"])) for item in selected]
        canonical = "structured_json" if category == "json" else category
        categories[canonical] = {
            "independent_requests": 8,
            "valid_candidate_labels": len(selected),
            "positive": sum(target),
            "positive_rate": sum(target) / len(target),
            "mean_predicted_probability": float(np.mean(probability)),
            "brier_score": float(brier_score_loss(target, probability)),
            "absolute_mean_calibration_gap": abs(float(np.mean(probability)) - sum(target) / len(target)),
            "warning": "diagnostic only: category contains eight independent requests",
        }
    return candidates, categories


def route_at_threshold(dataset, records, threshold: float):
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["task_id"])].append(record)
    policy = CostAwareRoutingPolicy()
    selections = []
    for task_id, candidates in sorted(grouped.items()):
        decision = policy.route((
            CandidatePrediction(
                model_id=str(item["candidate_id"]),
                predicted_acceptability=float(item["predicted_probability"]),
                projected_cost_usd=row_index[(task_id, str(item["candidate_id"]))].projected_cost_usd,
            ) for item in candidates
        ), threshold)
        row = row_index[(task_id, decision.selected_model_id)]
        selections.append({
            "task_id": task_id,
            "candidate_id": decision.selected_model_id,
            "projected_cost_usd": row.projected_cost_usd,
            "label_status": row.label_status,
            "acceptable": row.acceptable,
            "quality_score": row.quality_score,
            "fallback_used": decision.fallback_used,
            "threshold_satisfied": decision.threshold_satisfied,
        })
    return selections


def summarize_selections(selections, *, reference_cost: Decimal | None = None):
    valid = [item for item in selections if item["label_status"] == "valid"]
    missing = len(selections) - len(valid)
    acceptable = sum(item["acceptable"] is True for item in valid)
    total_cost = sum((item["projected_cost_usd"] for item in selections), Decimal(0))
    fallback_count = sum(bool(item.get("fallback_used")) for item in selections)
    threshold_met = sum(bool(item.get("threshold_satisfied")) for item in selections)
    result = {
        "requests_routed": len(selections),
        "valid_selected_labels": len(valid),
        "missing_selected_labels": missing,
        "acceptable_selections": acceptable,
        "unacceptable_selections": len(valid) - acceptable,
        "observed_acceptable_rate_among_valid": acceptable / len(valid) if valid else None,
        "mean_observed_quality_among_valid": float(np.mean([
            item["quality_score"] for item in valid
        ])) if valid else None,
        "total_projected_cost_usd": str(total_cost),
        "average_projected_cost_per_request_usd": str(total_cost / Decimal(len(selections))) if selections else None,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(selections) if selections else None,
        "threshold_met_count": threshold_met,
        "threshold_met_rate": threshold_met / len(selections) if selections else None,
        "selected_model_distribution": dict(sorted(Counter(
            str(item["candidate_id"]) for item in selections
        ).items())),
    }
    if reference_cost is not None:
        result["projected_cost_reduction_vs_matched_strongest_usd"] = str(reference_cost - total_cost)
        result["projected_cost_reduction_vs_matched_strongest_fraction"] = float(
            (reference_cost - total_cost) / reference_cost
        )
    return result


def baseline_analysis(root, run_id, dataset, folds):
    frozen = load_frozen_foundation_v3(root, run_id)
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    request_index = {request.task_id: request for request in frozen.requests}

    def selections_from_decisions(decisions):
        output = []
        for decision in decisions:
            if decision.candidate_id is None:
                continue
            row = row_index[(decision.task_id, str(decision.candidate_id))]
            output.append({
                "task_id": decision.task_id,
                "candidate_id": str(decision.candidate_id),
                "projected_cost_usd": row.projected_cost_usd,
                "label_status": row.label_status,
                "acceptable": row.acceptable,
                "quality_score": row.quality_score,
            })
        return output

    cheapest_policy = AlwaysCheapestPolicy()
    rule_policy = RuleBasedV1Policy()
    cheapest = [cheapest_policy.select(request) for request in frozen.requests]
    rule = [rule_policy.select(request) for request in frozen.requests]
    strongest, strongest_audit = fold_local_strongest_decisions(frozen, folds)
    oracle = []
    for task_id in sorted(request_index):
        acceptable = [
            row for row in dataset.rows
            if row.task_id == task_id and row.label_status == "valid" and row.acceptable
        ]
        if acceptable:
            selected = min(acceptable, key=lambda row: (row.projected_cost_usd, row.candidate_id))
            oracle.append(type(strongest[0])(
                policy=ORACLE_CHEAPEST_ACCEPTABLE,
                task_id=task_id,
                candidate_id=selected.candidate_id,
                reason="retrospective lowest projected cost among observed acceptable candidates",
            ))
    summaries = {
        ALWAYS_CHEAPEST: summarize_selections(selections_from_decisions(cheapest)),
        RULE_BASED_V1: summarize_selections(selections_from_decisions(rule)),
        FOLD_LOCAL_ALWAYS_STRONGEST: summarize_selections(selections_from_decisions(strongest)),
        ORACLE_CHEAPEST_ACCEPTABLE: {
            **summarize_selections(selections_from_decisions(oracle)),
            "analysis_only": True,
            "not_deployable": True,
            "requests_without_observed_acceptable_candidate": 56 - len(oracle),
        },
    }
    strongest_cost = Decimal(summaries[FOLD_LOCAL_ALWAYS_STRONGEST]["total_projected_cost_usd"])
    return summaries, strongest_cost, strongest_audit


def grouped_bootstrap(selections_by_threshold, *, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    rng = np.random.default_rng(seed)
    output = {}
    for key, selections in selections_by_threshold.items():
        index = {item["task_id"]: item for item in selections}
        task_ids = np.asarray(sorted(index), dtype=object)
        rates, costs, fallbacks = [], [], []
        for _ in range(replicates):
            sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
            rows = [index[str(task_id)] for task_id in sampled]
            valid = [item for item in rows if item["label_status"] == "valid"]
            rates.append(sum(item["acceptable"] is True for item in valid) / len(valid))
            costs.append(float(sum((item["projected_cost_usd"] for item in rows), Decimal(0)) / Decimal(len(rows))))
            fallbacks.append(sum(item["fallback_used"] for item in rows) / len(rows))
        output[key] = {
            "acceptable_rate_95_percentile_interval": np.percentile(rates, [2.5, 97.5]).tolist(),
            "average_projected_cost_usd_95_percentile_interval": np.percentile(costs, [2.5, 97.5]).tolist(),
            "fallback_rate_95_percentile_interval": np.percentile(fallbacks, [2.5, 97.5]).tolist(),
        }
    return {
        "unit": "independent request group",
        "seed": seed,
        "replicates": replicates,
        "interval": "deterministic percentile bootstrap, descriptive",
        "thresholds": output,
    }


def cross_fitted_platt(records):
    calibrated = []
    audits = []
    for fold in sorted({int(item["fold"]) for item in records}):
        train = [item for item in records if int(item["fold"]) != fold and item["label_status"] == "valid"]
        test = [item for item in records if int(item["fold"]) == fold]
        train_tasks = {str(item["task_id"]) for item in train}
        test_tasks = {str(item["task_id"]) for item in test}
        if train_tasks & test_tasks:
            raise ValueError("calibration fold leakage")
        probabilities = np.clip(
            np.asarray([float(item["predicted_probability"]) for item in train]),
            1e-12, 1 - 1e-12,
        )
        logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
        target = np.asarray([int(bool(item["target"])) for item in train], dtype=int)
        calibrator = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        calibrator.fit(logits, target)
        test_probability = np.clip(
            np.asarray([float(item["predicted_probability"]) for item in test]),
            1e-12, 1 - 1e-12,
        )
        test_logits = np.log(test_probability / (1 - test_probability)).reshape(-1, 1)
        values = calibrator.predict_proba(test_logits)[:, 1]
        for item, probability in zip(test, values):
            calibrated.append({**item, "predicted_probability": float(probability)})
        audits.append({
            "fold": fold,
            "train_request_groups": len(train_tasks),
            "evaluation_request_groups": len(test_tasks),
            "group_overlap": 0,
            "train_valid_rows": len(train),
            "evaluation_rows": len(test),
        })
    calibrated.sort(key=lambda item: (str(item["task_id"]), str(item["candidate_id"])))
    return tuple(calibrated), audits


def frontier(threshold_summaries):
    non_dominated = []
    for key, summary in threshold_summaries.items():
        cost = Decimal(summary["average_projected_cost_per_request_usd"])
        quality = summary["observed_acceptable_rate_among_valid"]
        coverage = summary["valid_selected_labels"]
        dominated = any(
            other_key != key
            and other["valid_selected_labels"] == coverage
            and Decimal(other["average_projected_cost_per_request_usd"]) <= cost
            and other["observed_acceptable_rate_among_valid"] >= quality
            and (
                Decimal(other["average_projected_cost_per_request_usd"]) < cost
                or other["observed_acceptable_rate_among_valid"] > quality
            )
            for other_key, other in threshold_summaries.items()
        )
        if not dominated:
            non_dominated.append(key)
    return {
        "coverage_comparison_rule": "dominance comparisons require equal valid-selected-label counts",
        "non_dominated_thresholds": non_dominated,
        "automatic_winner_selected": False,
    }


def threshold_stability(selections_by_threshold, summaries):
    keys = list(selections_by_threshold)
    comparisons = []
    for left, right in zip(keys, keys[1:]):
        left_index = {item["task_id"]: item for item in selections_by_threshold[left]}
        right_index = {item["task_id"]: item for item in selections_by_threshold[right]}
        changed = sum(
            left_index[task]["candidate_id"] != right_index[task]["candidate_id"]
            for task in left_index
        )
        acceptance_delta = (
            summaries[right]["observed_acceptable_rate_among_valid"]
            - summaries[left]["observed_acceptable_rate_among_valid"]
        )
        fallback_delta = summaries[right]["fallback_rate"] - summaries[left]["fallback_rate"]
        left_cost = float(summaries[left]["average_projected_cost_per_request_usd"])
        right_cost = float(summaries[right]["average_projected_cost_per_request_usd"])
        relative_cost_change = (right_cost - left_cost) / left_cost if left_cost else 0.0
        brittle = (
            changed / 56 >= 0.20
            or abs(acceptance_delta) >= 0.10
            or abs(fallback_delta) >= 0.15
            or abs(relative_cost_change) >= 0.25
        )
        comparisons.append({
            "from_threshold": left,
            "to_threshold": right,
            "selected_candidate_changes": changed,
            "selected_candidate_change_rate": changed / 56,
            "acceptable_rate_change": acceptance_delta,
            "fallback_rate_change": fallback_delta,
            "relative_average_cost_change": relative_cost_change,
            "flagged_brittle": brittle,
        })
    return {
        "descriptive_brittleness_rule": "flag >=20% selection changes, >=10pp acceptance change, >=15pp fallback change, or >=25% relative cost change",
        "adjacent_fine_thresholds": comparisons,
        "brittle_intervals": [
            f"{item['from_threshold']}->{item['to_threshold']}"
            for item in comparisons if item["flagged_brittle"]
        ],
    }


def run_calibration_analysis(
    root: Path = ROOT,
    run_id: UUID = RUN_ID,
    *,
    report_path: Path | None = DEFAULT_REPORT_PATH,
):
    dataset, records, folds, provenance = load_grouped_oof_evidence(root, run_id)
    metrics = predictive_metrics(records)
    expected = {
        "log_loss": 0.407419898387,
        "brier_score": 0.129855392134,
        "roc_auc": 0.888299260548,
        "average_precision": 0.900570343522,
    }
    if any(abs(metrics[name] - value) > 1e-12 for name, value in expected.items()):
        raise ValueError("approved no-provider-pin predictive metrics did not reproduce")
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    ranking = _rank_variant(list(records), row_index)
    if (
        abs(ranking["top_1_acceptable_rate"] - 0.7962962962962963) > 1e-12
        or abs(ranking["pairwise"]["strict_accuracy"] - 0.9126984126984127) > 1e-12
    ):
        raise ValueError("approved no-provider-pin ranking metrics did not reproduce")

    width = reliability_table(records, bins=10, strategy="equal_width")
    quantile = reliability_table(records, bins=10, strategy="equal_frequency")
    candidate_diagnostics, category_diagnostics = subgroup_diagnostics(records, dataset)
    baselines, strongest_cost, strongest_audit = baseline_analysis(root, run_id, dataset, folds)

    all_selections = {}
    fine_summaries = {}
    for threshold in FINE_THRESHOLDS:
        key = f"{threshold:.2f}"
        selections = route_at_threshold(dataset, records, threshold)
        all_selections[key] = selections
        fine_summaries[key] = summarize_selections(selections, reference_cost=strongest_cost)
    coarse_summaries = {f"{threshold:.2f}": fine_summaries[f"{threshold:.2f}"] for threshold in THRESHOLDS}
    coarse_selections = {key: all_selections[key] for key in coarse_summaries}

    calibrated, calibration_audit = cross_fitted_platt(records)
    calibrated_metrics = predictive_metrics(calibrated)
    calibrated_width = reliability_table(calibrated, bins=10, strategy="equal_width")
    calibrated_quantile = reliability_table(calibrated, bins=10, strategy="equal_frequency")
    calibration_routing = {}
    selection_changes = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        selected = route_at_threshold(dataset, calibrated, threshold)
        calibration_routing[key] = summarize_selections(selected, reference_cost=strongest_cost)
        raw_index = {item["task_id"]: item["candidate_id"] for item in coarse_selections[key]}
        selection_changes[key] = sum(
            raw_index[item["task_id"]] != item["candidate_id"] for item in selected
        )

    slope_intercept = calibration_slope_intercept(records)
    uncertainty = grouped_bootstrap(coarse_selections)
    stability = threshold_stability(all_selections, fine_summaries)
    frontier_result = frontier(fine_summaries)
    conclusion = "MORE_DATA_REQUIRED_BEFORE_POLICY_FREEZE"
    if conclusion not in POLICY_CONCLUSIONS:
        raise ValueError("invalid policy conclusion")
    report = {
        "phase": "8G",
        "analysis_scope": "grouped OOF calibration and policy analysis; no production default selected",
        "provenance": provenance,
        "baseline_reproduction": {
            "status": "PASS",
            "predictive_metrics": metrics,
            "top_1_acceptable_rate": ranking["top_1_acceptable_rate"],
            "pairwise_ranking_accuracy": ranking["pairwise"]["strict_accuracy"],
        },
        "calibration": {
            "global_metrics": metrics,
            "slope_intercept": {
                **slope_intercept,
                "method": "descriptive logistic recalibration on OOF candidate rows; rows within each request are correlated",
            },
            "equal_width_10_bins": width,
            "equal_frequency_10_bins": quantile,
            "interpretation": "ranking discrimination does not establish probability calibration",
        },
        "candidate_diagnostics": candidate_diagnostics,
        "category_diagnostics": category_diagnostics,
        "routing": {
            "policy": "canonical Phase 8A CostAwareRoutingPolicy",
            "projected_cost_source": "frozen pre-generation routing export",
            "coarse_threshold_grid": list(THRESHOLDS),
            "fine_analysis_only_grid": list(FINE_THRESHOLDS),
            "coarse": coarse_summaries,
            "fine": fine_summaries,
            "threshold_semantics": "candidate qualification by predicted probability; not a guarantee of selected-answer quality",
        },
        "grouped_uncertainty": uncertainty,
        "baselines": {
            "policies": baselines,
            "matched_fold_local_strongest_audit": strongest_audit,
            "oracle_warning": "retrospective analysis only; not deployable",
        },
        "frontier": frontier_result,
        "threshold_stability": stability,
        "post_hoc_calibration": {
            "status": "EXPLORATORY_ONLY",
            "raw_metrics": metrics,
            "cross_fitted_platt_metrics": calibrated_metrics,
            "raw_equal_width_ece": width["ece"],
            "cross_fitted_platt_equal_width_ece": calibrated_width["ece"],
            "raw_equal_frequency_ece": quantile["ece"],
            "cross_fitted_platt_equal_frequency_ece": calibrated_quantile["ece"],
            "outer_group_audit": calibration_audit,
            "routing_on_same_evaluation_predictions": calibration_routing,
            "selected_candidate_changes_vs_raw": selection_changes,
            "isotonic": "SKIPPED: 56 independent request groups are insufficient for a flexible monotonic calibrator with nested grouped evaluation",
            "production_predictor_replaced": False,
            "feasibility_conclusion": "A two-parameter Platt check is methodologically possible but exploratory; effective sample size is too small to justify installing calibration.",
        },
        "quality_modes": {
            "implemented": False,
            "conclusion": "NOT_ENOUGH_EVIDENCE_YET",
            "reason": "56 independent requests and brittle threshold intervals do not support freezing mode boundaries",
        },
        "policy_decision": {
            "Q1_raw_probabilities_semantically_calibrated": "Globally promising, but not adequate for literal probability guarantees given maximum-bin gaps, subgroup variation, and grouped uncertainty.",
            "Q2_one_global_threshold_supported": "No; candidate/category diagnostics and only eight requests per category do not support a universal semantic threshold.",
            "Q3_post_hoc_calibration_justified": "Not for production; cross-fitted Platt is exploratory and isotonic is underpowered.",
            "Q4_stable_cost_quality_region": "Descriptive regions exist, but bootstrap uncertainty and adjacent-threshold sensitivity prevent a durable freeze.",
            "Q5_production_default_now": "No.",
            "Q6_api_caller_threshold": "Yes; retain the explicit caller threshold with careful documentation.",
            "Q7_coarse_quality_modes": "No; mode boundaries are not justified yet.",
            "conclusion": conclusion,
            "rationale": "Only 56 independent request groups support useful ranking and descriptive routing analysis, not a calibrated global policy freeze.",
        },
        "limitations": [
            "Only 56 independent request groups are available.",
            "Candidate rows within a request are correlated.",
            "Each category contains eight independent requests.",
            "Eight candidate labels are missing and are never imputed negative.",
            "OOF evidence is Foundation V3-specific and does not establish external generalization.",
            "Bootstrap intervals are deterministic descriptive intervals, not asymptotic guarantees.",
        ],
    }
    normalized = _normalized(report)
    if report_path is not None:
        _atomic_write(report_path, json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped-OOF Phase 8G calibration analysis")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", type=UUID, default=RUN_ID)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = run_calibration_analysis(args.root, args.run_id, report_path=args.report)
    print(json.dumps({
        "report": str(args.report),
        "policy_conclusion": report["policy_decision"]["conclusion"],
        "oof_rows": report["provenance"]["prediction_rows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
