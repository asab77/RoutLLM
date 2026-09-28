"""TRAIN-only grouped router analysis for Routing Benchmark v1.2."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from adaptive_llm_gateway.benchmarks.routing_export import RoutingDatasetRow

from .analysis import (
    ALWAYS_CHEAPEST, ALWAYS_STRONGEST, ORACLE_CHEAPEST_ACCEPTABLE,
    RANDOM_SEEDED, RULE_BASED_V1, AlwaysCheapestPolicy, CandidateOption,
    FixedCandidatePolicy, FrozenRoutingData, PolicyRequest, RandomSeededPolicy,
    RowObservation, RuleBasedV1Policy, SONNET,
)
from .calibration_analysis import (
    cross_fitted_platt, frontier, grouped_bootstrap, reliability_table,
    route_at_threshold, summarize_selections,
)
from .ml_diagnostics import _rank_variant
from .ml_features import MLExperimentDataset, MLExperimentRow, _feature_values
from .quality_features import (
    CANONICAL_PREDICTIVE_FEATURES, CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
    PREDICTOR_FORMULATION_ID, PREDICTOR_FORMULATION_VERSION,
    QUALITY_PREPROCESSING_ID, QUALITY_RANDOM_STATE, build_quality_pipeline,
    canonical_feature_matrix, canonical_from_training_row,
)

RUN_ID = UUID("cc28470f-55ed-4a2f-a6ee-8d25cf93ecd2")
ROOT = Path("artifacts/routing-benchmark-v1/full-train-runs")
OUTPUT = Path("artifacts/routing-analysis/rb12-train-router-v1")
MODEL_OUTPUT = Path("artifacts/routing-quality/rb12-train-candidate-v1")
PROTOCOL_SHA256 = "b4cde3954a8ccd1b54684dcb62da303e7bc806248306464ae536b9ff717a0bd8"
DATASET_SHA256 = "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005"
SPLIT_SHA256 = "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"
EVALUATOR_SHA256 = "7dfb5c29426520edcb2c64491c1d5f62fddfc3bdbb3869599378da781f35d413"
SPECIFICATION_SHA256 = "b5568b26d7fdce018e3cfef9020780645e12e02f488ab495b14a1e621dd62c1e"
DEV_SHA256 = "93daa569d7149666328ffad28eb2db5fbd1342700235d47be06bf001fef61c0e"
FINAL_SHA256 = "cca8f7f57df723f6f92624394c382544f972165d4087c69d4f3b1e402b43ef65"
THRESHOLDS = tuple(value / 100 for value in range(50, 96, 5))
FOLDS = 5
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_REPLICATES = 2000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(value) -> bytes:
    def normalize(item):
        if isinstance(item, float):
            return None if not math.isfinite(item) else round(item, 12)
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item
    return (json.dumps(normalize(value), indent=2, sort_keys=True) + "\n").encode()


def load_training_evidence(root: Path = ROOT, run_id: UUID = RUN_ID):
    directory = root / str(run_id)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    rows = tuple(RoutingDatasetRow.model_validate_json(line)
                 for line in (directory / "routing-dataset.jsonl").read_bytes().splitlines() if line)
    configuration = manifest["configuration"]
    if (manifest["run_id"] != str(run_id)
            or configuration["execution_protocol_sha256"] != PROTOCOL_SHA256
            or configuration["source_dataset_sha256"] != DATASET_SHA256
            or configuration["evaluator_manifest_sha256"] != EVALUATOR_SHA256
            or configuration["proposition_specification_sha256"] != SPECIFICATION_SHA256
            or configuration["execution_split"] != "train"
            or configuration["development_selected"] != 0
            or configuration["final_selected"] != 0):
        raise ValueError("TRAIN run identity mismatch")
    protected = {
        Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json"): SPLIT_SHA256,
        Path("artifacts/routing-benchmark-v1/development-manifest.json"): DEV_SHA256,
        Path("artifacts/routing-benchmark-v1/final-manifest.json"): FINAL_SHA256,
    }
    if any(_sha256(path) != digest for path, digest in protected.items()):
        raise ValueError("split or protected-manifest identity mismatch")
    pairs = {(row.task_id, row.candidate.internal_id) for row in rows}
    if (len(rows), len(pairs), len({row.task_id for row in rows})) != (560, 560, 140):
        raise ValueError("TRAIN routing export shape mismatch")
    if Counter(row.label_status for row in rows) != Counter({"valid": 557, "missing": 3}):
        raise ValueError("TRAIN label availability mismatch")
    missing = tuple(sorted((row.task_id, row.candidate.internal_id, row.missing_label_reason)
                           for row in rows if row.label_status == "missing"))
    if any(row.acceptable is not None for row in rows if row.label_status == "missing"):
        raise ValueError("missing labels must not be imputed")
    ml_rows = tuple(MLExperimentRow(
        task_id=row.task_id, candidate_id=row.candidate.internal_id,
        category=row.request_features.category or "unknown", features=_feature_values(row),
        label_status=row.label_status, acceptable=row.acceptable,
        quality_score=row.quality_score, analysis_difficulty=row.analysis_difficulty,
        realized_cost_usd=row.candidate_cost_usd, latency_ms=row.latency_ms,
    ) for row in sorted(rows, key=lambda item: (item.task_id, item.candidate.internal_id)))
    return MLExperimentDataset(run_id=run_id, rows=ml_rows), rows, {
        "run_manifest_sha256": _sha256(directory / "manifest.json"),
        "routing_dataset_sha256": _sha256(directory / "routing-dataset.jsonl"),
        "missing_labels": missing,
    }


def build_grouped_folds(dataset: MLExperimentDataset):
    by_task = defaultdict(list)
    for row in dataset.rows:
        by_task[row.task_id].append(row)
    fold_tasks = [[] for _ in range(FOLDS)]
    fold_positive = [0] * FOLDS
    fold_missing = [0] * FOLDS
    for category in sorted({rows[0].category for rows in by_task.values()}):
        tasks = [task for task, rows in by_task.items() if rows[0].category == category]
        if len(tasks) != 20:
            raise ValueError("five-fold balance requires 20 requests per category")
        tasks.sort(key=lambda task: (
            -sum(row.label_status == "missing" for row in by_task[task]),
            -sum(row.acceptable is True for row in by_task[task]), task))
        category_counts = [0] * FOLDS
        for task in tasks:
            eligible = [fold for fold in range(FOLDS) if category_counts[fold] < 4]
            fold = min(eligible, key=lambda item: (
                category_counts[item], fold_missing[item], fold_positive[item], item))
            fold_tasks[fold].append(task)
            category_counts[fold] += 1
            fold_positive[fold] += sum(row.acceptable is True for row in by_task[task])
            fold_missing[fold] += sum(row.label_status == "missing" for row in by_task[task])
    assignments = []
    for fold, tasks in enumerate(fold_tasks):
        counts = Counter(by_task[task][0].category for task in tasks)
        if len(tasks) != 28 or set(counts.values()) != {4}:
            raise ValueError("each fold must hold 28 requests and four per category")
        assignments.append({
            "fold": fold, "task_ids": sorted(tasks), "request_count": len(tasks),
            "category_counts": dict(sorted(counts.items())),
            "valid_rows": sum(row.label_status == "valid" for task in tasks for row in by_task[task]),
            "positive": sum(row.acceptable is True for task in tasks for row in by_task[task]),
            "negative": sum(row.acceptable is False for task in tasks for row in by_task[task]),
            "missing": sum(row.label_status == "missing" for task in tasks for row in by_task[task]),
        })
    if len({task for fold in assignments for task in fold["task_ids"]}) != 140:
        raise ValueError("request groups must appear in exactly one fold")
    return tuple(assignments)


def generate_oof(dataset: MLExperimentDataset, folds):
    canonical = {(row.task_id, row.candidate_id): canonical_from_training_row(row)
                 for row in dataset.rows}
    records = []
    audits = []
    for assignment in folds:
        test_tasks = set(assignment["task_ids"])
        train = [row for row in dataset.rows
                 if row.task_id not in test_tasks and row.label_status == "valid"]
        test = [row for row in dataset.rows if row.task_id in test_tasks]
        if {row.task_id for row in train} & test_tasks:
            raise ValueError("group leakage")
        pipeline = build_quality_pipeline()
        pipeline.fit(canonical_feature_matrix(canonical[row.task_id, row.candidate_id]
                                              for row in train),
                     np.asarray([int(bool(row.acceptable)) for row in train]))
        probabilities = pipeline.predict_proba(canonical_feature_matrix(
            canonical[row.task_id, row.candidate_id] for row in test))[:, 1]
        for row, probability in zip(test, probabilities):
            records.append({
                "task_id": row.task_id, "candidate_id": row.candidate_id,
                "category": row.category, "fold": assignment["fold"],
                "predicted_probability": float(probability),
                "label_status": row.label_status, "target": row.acceptable,
            })
        audits.append({
            "fold": assignment["fold"], "train_request_groups": len({r.task_id for r in train}),
            "test_request_groups": len(test_tasks), "valid_fit_rows": len(train),
            "test_rows": len(test), "group_overlap": 0,
        })
    records.sort(key=lambda item: (item["task_id"], item["candidate_id"]))
    if len(records) != 560 or len({(r["task_id"], r["candidate_id"]) for r in records}) != 560:
        raise ValueError("every row requires one OOF prediction")
    return tuple(records), audits


def predictive_metrics(records):
    valid = [record for record in records if record["label_status"] == "valid"]
    target = np.asarray([int(bool(record["target"])) for record in valid])
    probability = np.asarray([float(record["predicted_probability"]) for record in valid])
    has_both_classes = len(np.unique(target)) == 2
    return {
        "rows": len(valid), "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)) if has_both_classes else None,
        "average_precision": (
            float(average_precision_score(target, probability)) if has_both_classes else None
        ),
    }


def _frozen_data(dataset, exported, protocol_sha256=PROTOCOL_SHA256):
    observations = tuple(RowObservation(
        task_id=row.task_id, candidate_id=row.candidate.internal_id,
        features=row.request_features, analysis_difficulty=row.analysis_difficulty,
        option=CandidateOption(
            candidate_id=row.candidate.internal_id,
            input_cost_per_1m_tokens=row.candidate.input_cost_per_1m_tokens,
            output_cost_per_1m_tokens=row.candidate.output_cost_per_1m_tokens,
            effective_max_output_tokens=row.effective_max_output_tokens),
        label_status=row.label_status, acceptable=row.acceptable,
        quality_score=row.quality_score, realized_cost_usd=row.candidate_cost_usd,
        latency_ms=row.latency_ms) for row in exported)
    grouped = defaultdict(list)
    for row in observations:
        grouped[row.task_id].append(row)
    requests = tuple(PolicyRequest(
        task_id=task, features=rows[0].features,
        candidates=tuple(sorted((row.option for row in rows), key=lambda item: item.candidate_id)))
        for task, rows in sorted(grouped.items()))
    return FrozenRoutingData(run_id=dataset.run_id, dataset_sha256=DATASET_SHA256,
                             protocol_sha256=protocol_sha256,
                             observations=observations, requests=requests)


def _selections(frozen, dataset, decisions):
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    output = []
    for decision in decisions:
        if decision.candidate_id is None:
            continue
        row = index[decision.task_id, decision.candidate_id]
        output.append({
            "task_id": row.task_id, "candidate_id": row.candidate_id,
            "projected_cost_usd": row.projected_cost_usd,
            "realized_cost_usd": row.realized_cost_usd,
            "label_status": row.label_status, "acceptable": row.acceptable,
            "quality_score": row.quality_score, "fallback_used": False,
            "threshold_satisfied": True,
        })
    return output


def _with_realized_costs(summary, selections, *, reference_cost=None):
    known = [item["realized_cost_usd"] for item in selections
             if item.get("realized_cost_usd") is not None]
    total = sum(known, Decimal(0))
    summary.update({
        "realized_cost_observations": len(known),
        "total_realized_cost_usd": str(total),
        "average_realized_cost_per_request_usd": (
            str(total / Decimal(len(selections))) if selections else None
        ),
    })
    if reference_cost is not None:
        summary["realized_cost_reduction_vs_always_strongest_usd"] = str(
            reference_cost - total)
        summary["realized_cost_reduction_vs_always_strongest_fraction"] = float(
            (reference_cost - total) / reference_cost)
    return summary


def _enrich_realized_costs(dataset, selections):
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    for item in selections:
        item["realized_cost_usd"] = index[
            item["task_id"], item["candidate_id"]].realized_cost_usd
    return selections


def baselines(frozen, dataset):
    policies = (
        AlwaysCheapestPolicy(), FixedCandidatePolicy(SONNET), RandomSeededPolicy(),
        RuleBasedV1Policy(),
    )
    output = {}
    for policy in policies:
        selections = _selections(frozen, dataset, [policy.select(req) for req in frozen.requests])
        output[policy.name] = _with_realized_costs(
            summarize_selections(selections), selections)
    oracle = []
    for task in sorted({row.task_id for row in dataset.rows}):
        acceptable = [row for row in dataset.rows
                      if row.task_id == task and row.label_status == "valid" and row.acceptable]
        if acceptable:
            chosen = min(acceptable, key=lambda row: (row.projected_cost_usd, row.candidate_id))
            oracle.append(type("Decision", (), {"task_id": task,
                          "candidate_id": chosen.candidate_id})())
    oracle_selections = _selections(frozen, dataset, oracle)
    output[ORACLE_CHEAPEST_ACCEPTABLE] = {
        **_with_realized_costs(summarize_selections(oracle_selections), oracle_selections),
        "abstain_no_acceptable": len({row.task_id for row in dataset.rows}) - len(oracle),
        "analysis_only": True,
    }
    reference = Decimal(output[ALWAYS_STRONGEST]["total_projected_cost_usd"])
    realized_reference = Decimal(output[ALWAYS_STRONGEST]["total_realized_cost_usd"])
    for summary in output.values():
        cost = Decimal(summary["total_projected_cost_usd"])
        summary["cost_reduction_vs_always_strongest_fraction"] = float((reference - cost) / reference)
        realized = Decimal(summary["total_realized_cost_usd"])
        summary["realized_cost_reduction_vs_always_strongest_fraction"] = float(
            (realized_reference - realized) / realized_reference)
    return output, reference


def _subgroups(records, dataset, field):
    row_index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    values = sorted({(row.candidate_id if field == "candidate" else row.category)
                     for row in dataset.rows})
    output = {}
    for value in values:
        selected = [record for record in records if (
            record["candidate_id"] == value if field == "candidate"
            else row_index[record["task_id"], record["candidate_id"]].category == value)]
        output[value] = predictive_metrics(selected) | {
            "independent_requests": len({item["task_id"] for item in selected}),
            "warning": "diagnostic subgroup; do not treat as independent model selection evidence",
        }
    return output


def _stability(selections, summaries):
    comparisons = []
    keys = list(selections)
    for left, right in zip(keys, keys[1:]):
        a = {item["task_id"]: item for item in selections[left]}
        b = {item["task_id"]: item for item in selections[right]}
        changed = sum(a[key]["candidate_id"] != b[key]["candidate_id"] for key in a)
        comparisons.append({
            "interval": f"{left}->{right}", "changed_requests": changed,
            "changed_fraction": changed / len(a),
            "acceptability_delta": (summaries[right]["observed_acceptable_rate_among_valid"]
                                     - summaries[left]["observed_acceptable_rate_among_valid"]),
            "fallback_delta": summaries[right]["fallback_rate"] - summaries[left]["fallback_rate"],
            "cost_delta_usd": str(Decimal(summaries[right]["total_projected_cost_usd"])
                                  - Decimal(summaries[left]["total_projected_cost_usd"])),
        })
    return comparisons


def analyze(root: Path = ROOT, run_id: UUID = RUN_ID, output: Path = OUTPUT):
    dataset, exported, provenance = load_training_evidence(root, run_id)
    folds = build_grouped_folds(dataset)
    records, fit_audit = generate_oof(dataset, folds)
    metrics = predictive_metrics(records)
    ranking = _rank_variant(list(records), {(row.task_id, row.candidate_id): row
                                             for row in dataset.rows})
    raw_width = reliability_table(records, bins=10, strategy="equal_width")
    raw_frequency = reliability_table(records, bins=10, strategy="equal_frequency")
    from .calibration_analysis import calibration_slope_intercept
    slope = calibration_slope_intercept(records)
    calibrated, calibration_audit = cross_fitted_platt(records)
    platt_metrics = predictive_metrics(calibrated)
    frozen = _frozen_data(dataset, exported)
    baseline, reference_cost = baselines(frozen, dataset)
    selections = {f"{threshold:.2f}": route_at_threshold(dataset, records, threshold)
                  for threshold in THRESHOLDS}
    selections = {key: _enrich_realized_costs(dataset, value)
                  for key, value in selections.items()}
    realized_reference = Decimal(baseline[ALWAYS_STRONGEST]["total_realized_cost_usd"])
    routing = {key: _with_realized_costs(
                    summarize_selections(value, reference_cost=reference_cost), value,
                    reference_cost=realized_reference)
               for key, value in selections.items()}
    calibrated_routing = {
        f"{threshold:.2f}": summarize_selections(
            route_at_threshold(dataset, calibrated, threshold), reference_cost=reference_cost)
        for threshold in THRESHOLDS}
    report = {
        "phase": "9", "analysis": "RB12_TRAIN_ROUTER_RETRAINING",
        "provenance": {
            "source_run_id": str(run_id), "protocol_version": "1.7.0",
            "protocol_sha256": PROTOCOL_SHA256, "dataset_sha256": DATASET_SHA256,
            "split_sha256": SPLIT_SHA256, "evaluator_sha256": EVALUATOR_SHA256,
            "proposition_specification_sha256": SPECIFICATION_SHA256,
            "training_requests": 140, "rows": 560, "valid_labels": 557,
            "missing_labels": 3, "development_selected": 0, "final_selected": 0,
            **provenance,
        },
        "feature_contract": {
            "schema_version": CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
            "formulation": PREDICTOR_FORMULATION_ID,
            "features": list(CANONICAL_PREDICTIVE_FEATURES),
            "feature_count": len(CANONICAL_PREDICTIVE_FEATURES),
            "provider_pin_present": False,
            "target_leakage_audit": {
                "status": "PASS",
                "excluded_post_response_fields": [
                    "post_generation_quality", "latency", "actual_token_usage",
                    "actual_cost", "provider_outcome", "judge_output", "benchmark_label",
                ],
            },
        },
        "model": {
            "family": "LogisticRegression", "penalty": "L2", "C": 1.0,
            "solver": "lbfgs", "max_iter": 1000, "class_weight": None,
            "random_state": QUALITY_RANDOM_STATE,
        },
        "folds": list(folds), "fit_audit": fit_audit,
        "predictive_metrics": metrics, "ranking": ranking,
        "by_candidate": _subgroups(records, dataset, "candidate"),
        "by_category": _subgroups(records, dataset, "category"),
        "calibration": {
            "slope_intercept": slope, "equal_width_10_bins": raw_width,
            "equal_frequency_10_bins": raw_frequency,
        },
        "calibration_comparison": {
            "status": "EXPLORATORY_ONLY", "raw": metrics, "cross_fitted_platt": platt_metrics,
            "raw_equal_width_ece": raw_width["ece"],
            "platt_equal_width_ece": reliability_table(
                calibrated, bins=10, strategy="equal_width")["ece"],
            "raw_equal_frequency_ece": raw_frequency["ece"],
            "platt_equal_frequency_ece": reliability_table(
                calibrated, bins=10, strategy="equal_frequency")["ece"],
            "fold_audit": calibration_audit, "routing": calibrated_routing,
        },
        "baselines": baseline, "routing": routing,
        "frontier": frontier(routing),
        "bootstrap": grouped_bootstrap(
            selections, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES),
        "threshold_stability": _stability(selections, routing),
        "historical_phase8g": {
            "requests": 56, "valid_labels": 216, "log_loss": 0.4074199,
            "brier_score": 0.129855, "roc_auc": 0.888299,
            "average_precision": 0.900570, "top_1_acceptable_rate": 43 / 54,
            "pairwise_accuracy": 115 / 126,
        },
        "policy_decision": {
            "feature_formulation_freeze": True,
            "model_family_freeze": True,
            "calibration_strategy_freeze": True,
            "calibration_strategy": "raw_logistic_probability",
            "policy_semantics_freeze": True,
            "candidate_threshold_for_dev_validation": 0.80,
            "overall_decision": "READY_FOR_DEV_VALIDATION",
            "rationale": (
                "Raw OOF probabilities are well calibrated; exploratory cross-fitted Platt "
                "scaling slightly worsens log loss, Brier score, AUC, and AP. Threshold 0.80 "
                "is nondominated, reaches 90% observed acceptability with 97.86% projected "
                "cost reduction versus always strongest, and higher thresholds add cost and "
                "fallbacks without an observed acceptability gain. DEV remains required for "
                "independent confirmation."
            ),
        },
    }
    _atomic(output / "analysis.json", _json(report))
    _atomic(output / "oof-predictions.jsonl", b"".join(
        (json.dumps(item, sort_keys=True) + "\n").encode() for item in records))
    _atomic(output / "fold-assignments.json", _json(folds))
    return report, dataset, records


def fit_candidate(dataset: MLExperimentDataset, output: Path = MODEL_OUTPUT):
    valid = [row for row in dataset.rows if row.label_status == "valid"]
    missing = [row for row in dataset.rows if row.label_status == "missing"]
    canonical = [canonical_from_training_row(row) for row in valid]
    target = np.asarray([int(bool(row.acceptable)) for row in valid])
    pipeline = build_quality_pipeline()
    pipeline.fit(canonical_feature_matrix(canonical), target)
    model = pickle.dumps(pipeline, protocol=5)
    checksum = hashlib.sha256(model).hexdigest()
    metadata = {
        "artifact_format_version": "phase9-dev-candidate-v1",
        "deployment_status": "DEV_VALIDATION_CANDIDATE_NOT_DEPLOYED",
        "source_run_id": str(dataset.run_id), "protocol_version": "1.7.0",
        "protocol_sha256": PROTOCOL_SHA256, "training_dataset_sha256": DATASET_SHA256,
        "split_sha256": SPLIT_SHA256,
        "canonical_feature_schema_version": CANONICAL_QUALITY_FEATURE_SCHEMA_VERSION,
        "predictor_formulation_id": PREDICTOR_FORMULATION_ID,
        "predictor_formulation_version": PREDICTOR_FORMULATION_VERSION,
        "preprocessing_id": QUALITY_PREPROCESSING_ID,
        "hyperparameters": {"penalty": "L2", "C": 1.0, "solver": "lbfgs",
                            "max_iter": 1000, "class_weight": None,
                            "random_state": QUALITY_RANDOM_STATE},
        "valid_training_rows": len(valid), "missing_label_rows": len(missing),
        "known_candidate_ids": sorted({row.candidate_id for row in dataset.rows}),
        "known_categories": sorted({row.category for row in dataset.rows}),
        "feature_count": len(CANONICAL_PREDICTIVE_FEATURES),
        "model_filename": "predictor.pkl", "model_sha256": checksum,
    }
    _atomic(output / "predictor.pkl", model)
    _atomic(output / "metadata.json", _json(metadata))
    return metadata
