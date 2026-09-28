"""TRAIN-only bounded structural-feature ablation for Phase 9."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import load_training_dataset

from .bounded_features import BOUNDED_FEATURE_NAMES, extract_bounded_structural_features
from .calibration_analysis import (
    calibration_slope_intercept,
    reliability_table,
    route_at_threshold,
    summarize_selections,
)
from .ml_diagnostics import _rank_variant
from .phase9_retraining import (
    DATASET_SHA256,
    EVALUATOR_SHA256,
    PROTOCOL_SHA256,
    RUN_ID,
    SPECIFICATION_SHA256,
    SPLIT_SHA256,
    _enrich_realized_costs,
    _json,
    _with_realized_costs,
    build_grouped_folds,
    generate_oof,
    load_training_evidence,
    predictive_metrics,
)
from .quality_features import (
    CANONICAL_BOOLEAN_FEATURES,
    CANONICAL_CATEGORICAL_FEATURES,
    CANONICAL_NUMERIC_FEATURES,
    CANONICAL_PREDICTIVE_FEATURES,
    QUALITY_RANDOM_STATE,
    canonical_feature_matrix,
    canonical_from_training_row,
)

DATASET_PATH = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
SPLIT_PATH = Path("benchmarks/protocols/routing-benchmark-v1/split-manifest.json")
OUTPUT = Path("artifacts/routing-analysis/rb12-bounded-feature-ablation-v1")
DECISION = "CURRENT_ROUTER_RETAINED"


def _as_float(values: np.ndarray) -> np.ndarray:
    return values.astype(float)


def _augmented_pipeline() -> Pipeline:
    categorical_end = len(CANONICAL_CATEGORICAL_FEATURES)
    numeric_end = (
        categorical_end + len(CANONICAL_NUMERIC_FEATURES) + len(BOUNDED_FEATURE_NAMES)
    )
    feature_count = numeric_end + len(CANONICAL_BOOLEAN_FEATURES)
    return Pipeline((
        ("preprocess", ColumnTransformer((
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(range(categorical_end))),
            ("numeric", StandardScaler(), list(range(categorical_end, numeric_end))),
            ("boolean", FunctionTransformer(_as_float, feature_names_out="one-to-one"),
             list(range(numeric_end, feature_count))),
        ), remainder="drop")),
        ("classifier", LogisticRegression(
            l1_ratio=0.0,
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
            random_state=QUALITY_RANDOM_STATE,
        )),
    ))


def _augmented_matrix(rows, canonical, structures) -> np.ndarray:
    base = canonical_feature_matrix(canonical[row.task_id, row.candidate_id] for row in rows)
    insert_at = len(CANONICAL_CATEGORICAL_FEATURES) + len(CANONICAL_NUMERIC_FEATURES)
    added = np.asarray([
        [getattr(structures[row.task_id], name) for name in BOUNDED_FEATURE_NAMES]
        for row in rows
    ], dtype=object)
    return np.concatenate((base[:, :insert_at], added, base[:, insert_at:]), axis=1)


def _generate_augmented_oof(dataset, folds, structures):
    canonical = {
        (row.task_id, row.candidate_id): canonical_from_training_row(row)
        for row in dataset.rows
    }
    records = []
    for assignment in folds:
        test_tasks = set(assignment["task_ids"])
        train = [row for row in dataset.rows
                 if row.task_id not in test_tasks and row.label_status == "valid"]
        test = [row for row in dataset.rows if row.task_id in test_tasks]
        pipeline = _augmented_pipeline()
        pipeline.fit(
            _augmented_matrix(train, canonical, structures),
            np.asarray([int(bool(row.acceptable)) for row in train]),
        )
        probabilities = pipeline.predict_proba(
            _augmented_matrix(test, canonical, structures)
        )[:, 1]
        for row, probability in zip(test, probabilities, strict=True):
            records.append({
                "task_id": row.task_id,
                "candidate_id": row.candidate_id,
                "category": row.category,
                "fold": assignment["fold"],
                "predicted_probability": float(probability),
                "label_status": row.label_status,
                "target": row.acceptable,
            })
    return tuple(sorted(records, key=lambda item: (item["task_id"], item["candidate_id"])))


def _request_representation(dataset):
    names = (
        "category", "prompt_characters", "approximate_input_tokens",
        "requested_max_output_tokens", "constraint_indicator_count",
        "reasoning_indicator_count", "contains_code", "requests_structured_output",
    )
    representation = {}
    for row in dataset.rows:
        representation.setdefault(row.task_id, tuple(row.features[name] for name in names))
    return representation


def _representation_audit(dataset, structures):
    representation = _request_representation(dataset)
    candidates = sorted({row.candidate_id for row in dataset.rows})
    outcome = {
        task_id: tuple(next(row.acceptable for row in dataset.rows
                            if row.task_id == task_id and row.candidate_id == candidate)
                       for candidate in candidates)
        for task_id in representation
    }
    groups = defaultdict(list)
    for task_id, values in representation.items():
        groups[values].append(task_id)
    duplicates = [tasks for tasks in groups.values() if len(tasks) > 1]
    reasoning_ids = sorted(
        task_id for task_id, values in representation.items() if values[0] == "reasoning"
    )
    near = []
    for offset, left in enumerate(reasoning_ids):
        for right in reasoning_ids[offset + 1:]:
            a, b = representation[left], representation[right]
            if abs(a[1] - b[1]) <= 15 and abs(a[2] - b[2]) <= 4 and a[3:] == b[3:]:
                near.append({
                    "left": left,
                    "right": right,
                    "different_candidate_outcomes": outcome[left] != outcome[right],
                })
    distributions = {}
    for name in BOUNDED_FEATURE_NAMES:
        values = [getattr(item, name) for item in structures.values()]
        distributions[name] = {
            "nonzero_requests": sum(value > 0 for value in values),
            "distribution": dict(sorted(Counter(values).items())),
        }
    return {
        "exact_duplicate_groups": len(duplicates),
        "requests_in_exact_duplicate_groups": sum(map(len, duplicates)),
        "exact_duplicate_groups_with_different_outcomes": sum(
            len({outcome[task] for task in tasks}) > 1 for tasks in duplicates
        ),
        "reasoning_exact_duplicate_groups": 0,
        "reasoning_near_duplicate_pair_count": len(near),
        "reasoning_near_duplicate_pairs_with_different_outcomes": sum(
            item["different_candidate_outcomes"] for item in near
        ),
        "reasoning_near_duplicate_pairs": near,
        "experimental_feature_distributions": distributions,
    }


def _summary(records, dataset):
    index = {(row.task_id, row.candidate_id): row for row in dataset.rows}
    ranking = _rank_variant(list(records), index)
    calibration = calibration_slope_intercept(records)
    selections = _enrich_realized_costs(
        dataset, route_at_threshold(dataset, records, 0.80)
    )
    routing = _with_realized_costs(summarize_selections(selections), selections)
    reasoning = [record for record in records if record["category"] == "reasoning"]
    reasoning_ranking = _rank_variant(reasoning, index)
    return {
        "predictive": predictive_metrics(records),
        "pairwise_accuracy": ranking["pairwise"]["strict_accuracy"],
        "top_1_acceptable_rate": ranking["top_1_acceptable_rate"],
        "calibration": {
            **calibration,
            "ece_equal_frequency_10": reliability_table(
                records, bins=10, strategy="equal_frequency"
            )["ece"],
            "ece_equal_width_10": reliability_table(
                records, bins=10, strategy="equal_width"
            )["ece"],
        },
        "routing_threshold_0_80": routing,
        "reasoning": {
            "predictive": predictive_metrics(reasoning),
            "pairwise_accuracy": reasoning_ranking["pairwise"]["strict_accuracy"],
            "top_1_acceptable_rate": reasoning_ranking["top_1_acceptable_rate"],
        },
    }


def analyze(output: Path = OUTPUT, *, persist: bool = True):
    dataset, _, provenance = load_training_evidence()
    training = load_training_dataset(DATASET_PATH, SPLIT_PATH)
    task_by_id = {task.task_id: task for task in training.tasks}
    if set(task_by_id) != {row.task_id for row in dataset.rows}:
        raise ValueError("bounded experiment must use exactly the TRAIN request set")
    structures = {
        task_id: extract_bounded_structural_features(task)
        for task_id, task in task_by_id.items()
    }
    folds = build_grouped_folds(dataset)
    baseline, _ = generate_oof(dataset, folds)
    augmented = _generate_augmented_oof(dataset, folds, structures)
    valid = [row for row in dataset.rows if row.label_status == "valid"]
    canonical = {(row.task_id, row.candidate_id): canonical_from_training_row(row)
                 for row in dataset.rows}
    final_pipeline = _augmented_pipeline()
    final_pipeline.fit(
        _augmented_matrix(valid, canonical, structures),
        np.asarray([int(bool(row.acceptable)) for row in valid]),
    )
    encoded = final_pipeline.named_steps["preprocess"].transform(
        _augmented_matrix(valid, canonical, structures)
    ).shape[1]
    report = {
        "phase": "9",
        "analysis": "FINAL_BOUNDED_ROUTER_IMPROVEMENT",
        "decision": DECISION,
        "source": {
            "run_id": str(RUN_ID),
            "training_requests": 140,
            "valid_training_rows": 557,
            "missing_training_rows": 3,
            "development_labels_used": 0,
            "final_labels_used": 0,
            "protocol_sha256": PROTOCOL_SHA256,
            "dataset_sha256": DATASET_SHA256,
            "split_sha256": SPLIT_SHA256,
            "evaluator_sha256": EVALUATOR_SHA256,
            "specification_sha256": SPECIFICATION_SHA256,
            **provenance,
        },
        "features": {
            "proposed": [*BOUNDED_FEATURE_NAMES, "enumeration_marker_count"],
            "tested": list(BOUNDED_FEATURE_NAMES),
            "rejected_before_fit": {
                "enumeration_marker_count": "zero variance across all 140 TRAIN requests",
            },
            "production_contract_changed": False,
            "old_canonical_feature_count": len(CANONICAL_PREDICTIVE_FEATURES),
            "experimental_canonical_feature_count": (
                len(CANONICAL_PREDICTIVE_FEATURES) + len(BOUNDED_FEATURE_NAMES)
            ),
            "old_encoded_dimension": 54,
            "experimental_encoded_dimension": encoded,
        },
        "representation_audit": _representation_audit(dataset, structures),
        "baseline": _summary(baseline, dataset),
        "augmented": _summary(augmented, dataset),
        "acceptance": {
            "accepted": False,
            "reason": (
                "The augmented representation worsened overall log loss, Brier score, "
                "AUC, AP, routing acceptability, quality, fallback rate, and cost; "
                "pairwise ordering and top-1 acceptability did not improve."
            ),
            "predictor_artifact_created": False,
            "feature_contract_version_created": False,
        },
    }
    if persist:
        output.mkdir(parents=True, exist_ok=True)
        (output / "analysis.json").write_bytes(_json(report))
        (output / "augmented-oof-predictions.jsonl").write_bytes(b"".join(
            (json.dumps(item, sort_keys=True) + "\n").encode() for item in augmented
        ))
    return report


if __name__ == "__main__":
    analyze()
