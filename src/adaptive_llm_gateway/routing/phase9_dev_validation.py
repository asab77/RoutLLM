"""Frozen TRAIN-predictor validation on the protected Routing Benchmark DEV split."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import numpy as np

from adaptive_llm_gateway.benchmarks.routing_export import RoutingDatasetRow

from .calibration_analysis import (
    calibration_slope_intercept, reliability_table, route_at_threshold,
    summarize_selections,
)
from .ml_diagnostics import _rank_variant
from .ml_features import MLExperimentDataset, MLExperimentRow, _feature_values
from .phase9_retraining import (
    DATASET_SHA256, DEV_SHA256, EVALUATOR_SHA256, FINAL_SHA256,
    PROTOCOL_SHA256, SPECIFICATION_SHA256, SPLIT_SHA256, _enrich_realized_costs,
    _frozen_data, _json, _with_realized_costs, baselines, predictive_metrics,
)
from .quality_features import canonical_feature_matrix, canonical_from_training_row

RUN_ID = UUID("96384f59-a988-4a8d-8f41-dabf9faebf3e")
ROOT = Path("artifacts/routing-benchmark-v1/dev-runs")
PREDICTOR = Path("artifacts/routing-quality/rb12-train-candidate-v1/predictor.pkl")
PREDICTOR_SHA256 = "502db83a54c4072c9741a8e4c406498ad88ec97e03bce1ddaf3e0a1b0001c0aa"
THRESHOLD = 0.80


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dev_evidence(root: Path = ROOT, run_id: UUID = RUN_ID):
    directory = root / str(run_id)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    summary = json.loads((directory / "evaluation-summary.json").read_bytes())
    rows = tuple(RoutingDatasetRow.model_validate_json(line)
                 for line in (directory / "routing-dataset.jsonl").read_bytes().splitlines()
                 if line)
    configuration = manifest["configuration"]
    if (manifest["run_id"] != str(run_id)
            or configuration["execution_protocol_sha256"] != PROTOCOL_SHA256
            or configuration["source_dataset_sha256"] != DATASET_SHA256
            or configuration["execution_split"] != "development"
            or configuration["final_evaluation_explicitly_unlocked"] is not False):
        raise ValueError("DEV run identity mismatch")
    if len(manifest["selected_task_ids"]) != 42:
        raise ValueError("DEV run must contain exactly 42 requests")
    if (len(rows), len({(row.task_id, row.candidate.internal_id) for row in rows}),
            len({row.task_id for row in rows})) != (168, 168, 42):
        raise ValueError("DEV evidence shape mismatch")
    statuses = Counter(row.label_status for row in rows)
    if statuses != Counter({"valid": 140, "missing": 28}):
        raise ValueError("DEV label availability mismatch")
    if any(row.acceptable is not None for row in rows if row.label_status == "missing"):
        raise ValueError("DEV missing labels must not be imputed")
    ml_rows = tuple(MLExperimentRow(
        task_id=row.task_id, candidate_id=row.candidate.internal_id,
        category=row.request_features.category or "unknown", features=_feature_values(row),
        label_status=row.label_status, acceptable=row.acceptable,
        quality_score=row.quality_score, analysis_difficulty=row.analysis_difficulty,
        realized_cost_usd=row.candidate_cost_usd, latency_ms=row.latency_ms,
    ) for row in sorted(rows, key=lambda item: (item.task_id, item.candidate.internal_id)))
    return MLExperimentDataset(run_id=run_id, rows=ml_rows), rows, manifest, summary


def _predict(dataset: MLExperimentDataset, predictor_path: Path = PREDICTOR):
    if _sha256(predictor_path) != PREDICTOR_SHA256:
        raise ValueError("frozen predictor checksum mismatch")
    pipeline = pickle.loads(predictor_path.read_bytes())
    canonical = [canonical_from_training_row(row) for row in dataset.rows]
    probabilities = pipeline.predict_proba(canonical_feature_matrix(canonical))[:, 1]
    return tuple({
        "task_id": row.task_id, "candidate_id": row.candidate_id,
        "category": row.category, "predicted_probability": float(probability),
        "label_status": row.label_status, "target": row.acceptable,
    } for row, probability in zip(dataset.rows, probabilities, strict=True))


def _candidate_diagnostics(records):
    return {candidate: predictive_metrics([
        item for item in records if item["candidate_id"] == candidate
    ]) for candidate in sorted({item["candidate_id"] for item in records})}


def _category_routing(dataset, selections):
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    output = {}
    for category in sorted({row.category for row in dataset.rows}):
        rows = [item for item in selections
                if index[item["task_id"], item["candidate_id"]].category == category]
        valid = [item for item in rows if item["label_status"] == "valid"]
        output[category] = {
            "request_count": len(rows), "valid_selected_labels": len(valid),
            "acceptable": sum(item["acceptable"] is True for item in valid),
            "acceptability": (sum(item["acceptable"] is True for item in valid) / len(valid)
                              if valid else None),
            "mean_quality": (float(np.mean([item["quality_score"] for item in valid]))
                             if valid else None),
            "selected_model_distribution": dict(sorted(Counter(
                item["candidate_id"] for item in rows).items())),
            "fallback_count": sum(bool(item.get("fallback_used")) for item in rows),
        }
    return output


def analyze(root: Path = ROOT, run_id: UUID = RUN_ID, predictor_path: Path = PREDICTOR):
    dataset, exported, manifest, evaluation_summary = load_dev_evidence(root, run_id)
    records = _predict(dataset, predictor_path)
    metrics = predictive_metrics(records)
    ranking = _rank_variant(list(records), {(row.task_id, row.candidate_id): row
                                             for row in dataset.rows})
    calibration = calibration_slope_intercept(records)
    reliability = reliability_table(records, bins=10, strategy="equal_frequency")
    frozen = _frozen_data(dataset, exported)
    baseline, projected_reference = baselines(frozen, dataset)
    selections = _enrich_realized_costs(
        dataset, route_at_threshold(dataset, records, THRESHOLD))
    realized_reference = Decimal(baseline["ALWAYS_STRONGEST"]["total_realized_cost_usd"])
    routing = _with_realized_costs(
        summarize_selections(selections, reference_cost=projected_reference), selections,
        reference_cost=realized_reference)
    missing = [row for row in dataset.rows if row.label_status == "missing"]
    report = {
        "phase": "9", "analysis": "FROZEN_ROUTER_DEV_VALIDATION",
        "decision_scope": "VALIDATION_ONLY_NO_TUNING",
        "source": {
            "run_id": str(run_id), "protocol_version": "1.7.0",
            "protocol_sha256": PROTOCOL_SHA256, "dataset_sha256": DATASET_SHA256,
            "split_sha256": SPLIT_SHA256, "evaluator_sha256": EVALUATOR_SHA256,
            "proposition_specification_sha256": SPECIFICATION_SHA256,
            "predictor_sha256": _sha256(predictor_path), "threshold": THRESHOLD,
            "requests": 42, "candidate_attempts": 168,
            "provider_successes": evaluation_summary["overall"]["successful_responses"],
            "provider_failures": 168 - evaluation_summary["overall"]["successful_responses"],
            "valid_labels": metrics["rows"], "missing_labels": len(missing),
            "final_evaluation_explicitly_unlocked": False,
        },
        "predictive_metrics": metrics, "ranking": ranking,
        "calibration": {"slope_intercept": calibration,
                        "equal_frequency_10_bins": reliability},
        "routing_threshold_0_80": routing,
        "baselines": baseline,
        "category_routing": _category_routing(dataset, selections),
        "candidate_diagnostics": _candidate_diagnostics(records),
        "missing_label_concentration": {
            "by_candidate": dict(sorted(Counter(row.candidate_id for row in missing).items())),
            "by_category": dict(sorted(Counter(row.category for row in missing).items())),
            "by_reason": dict(sorted(Counter(
                next(item.missing_label_reason for item in exported
                     if item.task_id == row.task_id
                     and item.candidate.internal_id == row.candidate_id)
                for row in missing).items())),
        },
        "cost": {
            "candidate_inference_usd": evaluation_summary["candidate_inference_cost_usd"],
            "judge_evaluation_usd": evaluation_summary["judge_evaluation_cost_usd"],
            "total_experiment_usd": str(
                Decimal(evaluation_summary["candidate_inference_cost_usd"])
                + Decimal(evaluation_summary["judge_evaluation_cost_usd"])),
            "judge_calls": evaluation_summary["judge_calls"],
        },
        "protected_hashes": {"development": DEV_SHA256, "final": FINAL_SHA256},
        "decision": {
            "recommendation": "FROZEN_ROUTER_DEV_VALIDATION_FAILED",
            "threshold_changed": False,
            "predictor_retrained": False,
            "concrete_failures": [
                "Frozen threshold routing acceptability fell to 33/41 valid selections (80.49%).",
                "Reasoning selected-response acceptability was 1/6 (16.67%).",
                "Twenty-seven of Gemini's 42 labels were missing because the frozen execution contract produced 26 output-budget exhaustions and one empty response.",
                "OOF-to-DEV calibration and ordering degraded materially: slope 0.6181, ECE 0.0952, and pairwise accuracy 62.86%.",
            ],
        },
    }
    output = root / str(run_id) / "frozen-router-dev-validation.json"
    output.write_bytes(_json(report))
    predictions = root / str(run_id) / "frozen-router-dev-predictions.jsonl"
    predictions.write_bytes(b"".join(
        (json.dumps(item, sort_keys=True) + "\n").encode() for item in records))
    return report
