"""Phase 12.2B.1 offline category-detector experiment.

This module consumes only the guarded TRAIN/DEV prompt projection and local
pre-generation metadata. It has no provider or generation integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from adaptive_llm_gateway.benchmarks.routing_benchmark_v1 import (
    DEFAULT_ROOT,
    export_category_detector_prompts,
)
from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.providers.gateway_config import CANDIDATE_MODELS
from adaptive_llm_gateway.routing.features import ProductionRequestFeatureExtractor
from adaptive_llm_gateway.routing.predictor import (
    PRODUCTION_PREDICTOR_SHA256,
    PRODUCTION_QUALITY_THRESHOLD,
    SklearnQualityPredictor,
)
from adaptive_llm_gateway.routing.service import RoutingDecisionService

CATEGORIES = (
    "classification",
    "coding",
    "extraction",
    "structured_json",
    "qa",
    "reasoning",
    "summarization",
)
RANDOM_STATE = 20260929
CV_SPLITS = 5
RULES_VERSION = "phase12.2b.1-high-precision-v2"
NORMALIZATION = "unicode-preserving whitespace collapse plus casefold"
WORD_TFIDF = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
}
CHAR_TFIDF = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 1,
    "sublinear_tf": True,
    "max_features": 30_000,
}
MODEL_CONFIGS = (
    {"name": "logreg-c1", "C": 1.0},
    {"name": "logreg-c4", "C": 4.0},
)
ABSTENTION_POLICIES = (
    {"name": "top_0.50", "minimum_top_score": 0.50, "minimum_margin": 0.0},
    {"name": "margin_0.20", "minimum_top_score": 0.0, "minimum_margin": 0.20},
    {"name": "top_0.50_margin_0.15", "minimum_top_score": 0.50, "minimum_margin": 0.15},
)
BOUNDARY_CORRECTION_VERSION = "phase12.2b.1b-extraction-json-v1"
_TRANSFORMATION_OPERATION = re.compile(
    r"\b(?:map|mapping|sort|sorted|group|count|counting|deduplicat\w*|"
    r"duplicates? removed|join|joining|filter|aggregate|reorder|"
    r"first-occurrence|only when|otherwise null)\b",
    re.I,
)
_DIRECT_RECOVERY_SOURCE = re.compile(
    r"\b(?:contact|logs?|table|rows?|entries|section|draft|final|approved|"
    r"target customer|approver|requester)\b",
    re.I,
)
BENCHMARK_MAX_OUTPUT_TOKENS = {
    "classification": {"easy": 32, "medium": 32, "hard": 32},
    "coding": {"easy": 128, "medium": 192, "hard": 256},
    "extraction": {"easy": 64, "medium": 128, "hard": 192},
    "structured_json": {"easy": 64, "medium": 128, "hard": 192},
    "qa": {"easy": 64, "medium": 128, "hard": 160},
    "reasoning": {"easy": 160, "medium": 160, "hard": 160},
    "summarization": {"easy": 96, "medium": 128, "hard": 192},
}

_WHITESPACE = re.compile(r"\s+")
_CLASSIFICATION = re.compile(
    r"\b(?:classify|assign)\b.{0,100}\b(?:label|category|class)\b|"
    r"\b(?:label|category)\b.{0,80}\b(?:choose|select|return)\b",
    re.I | re.S,
)
_CODING = re.compile(
    r"\b(?:write|implement|define|debug|fix|modify|validate)\b.{0,100}"
    r"\b(?:python|function|code|class|executable)\b|"
    r"\breturn only the function definition\b",
    re.I | re.S,
)
_EXTRACTION = re.compile(
    r"\bextract\b.{0,120}\b(?:field|fields|value|values|record|records|from)\b|"
    r"\brecover\b.{0,100}\b(?:field|fields|fact|facts)\b",
    re.I | re.S,
)
_STRUCTURED_JSON = re.compile(
    r"\b(?:construct|transform|filter|aggregate|validate|convert)\b"
    r".{0,120}\bjson\b|\bjson\b.{0,100}\b(?:object|array|schema)\b",
    re.I | re.S,
)
_SUMMARIZATION = re.compile(
    r"\bsummarize\b.{0,100}\b(?:source|following|report|passage|text|document)\b|"
    r"\b(?:source|report|passage|document)\b.{0,100}\bsummary\b",
    re.I | re.S,
)
_REASONING = re.compile(
    r"\b(?:solve|calculate|compute)\b.{0,160}\b(?:return|answer|result)\b|"
    r"\b(?:earliest common|constraint|allocation|ordering|schedule)\b.{0,160}"
    r"\b(?:return|which|what)\b",
    re.I | re.S,
)


def normalize_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


def rule_predict(prompt: str) -> str | None:
    """Apply narrow deliverable-first rules; return None to abstain."""
    text = normalize_text(prompt)
    if _CLASSIFICATION.search(text):
        return "classification"
    if _CODING.search(text):
        return "coding"
    if _EXTRACTION.search(text):
        return "extraction"
    if _STRUCTURED_JSON.search(text):
        return "structured_json"
    if _SUMMARIZATION.search(text):
        return "summarization"
    if _REASONING.search(text):
        return "reasoning"
    return None


def correct_extraction_json_boundary(prompt: str, prediction: str) -> str:
    """Correct only the extraction/structured-JSON boundary using operation semantics."""
    if prediction not in {"extraction", "structured_json"}:
        return prediction
    text = normalize_text(prompt)
    if _TRANSFORMATION_OPERATION.search(text):
        return "structured_json"
    if _DIRECT_RECOVERY_SOURCE.search(text):
        return "extraction"
    return prediction


def _revised_predictions(
    records: Sequence[dict[str, Any]], predictions: Sequence[str],
) -> list[str]:
    hybrid, _ = _hybrid_predictions(records, predictions)
    return [
        correct_extraction_json_boundary(record["prompt"], prediction)
        for record, prediction in zip(records, hybrid, strict=True)
    ]


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    features = FeatureUnion((
        ("word", TfidfVectorizer(**WORD_TFIDF)),
        ("char", TfidfVectorizer(**CHAR_TFIDF)),
    ))
    model = LogisticRegression(
        C=float(config["C"]),
        max_iter=3_000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    return Pipeline((("features", features), ("model", model)))


def _manifest_records(split: str, artifact_root: Path) -> dict[str, dict[str, Any]]:
    name = "train-manifest.json" if split == "train" else "development-manifest.json"
    payload = json.loads((artifact_root / name).read_text(encoding="utf-8"))
    records = payload.get("tasks", [])
    by_id = {record["task_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate task ID in {split} manifest")
    return by_id


def _projected_records(split: str, artifact_root: Path) -> list[dict[str, Any]]:
    prompts = export_category_detector_prompts(split, artifact_root=artifact_root)
    manifest = _manifest_records(split, artifact_root)
    records = []
    for prompt in prompts:
        metadata = manifest[prompt["task_id"]]
        records.append({
            **prompt,
            "normalized_text": normalize_text(prompt["prompt"]),
            "task_family_id": metadata["task_family_id"],
            "difficulty": metadata["difficulty"],
        })
    return records


def _aligned_probabilities(pipeline: Pipeline, texts: Sequence[str]) -> np.ndarray:
    raw = pipeline.predict_proba(texts)
    classes = tuple(str(value) for value in pipeline.named_steps["model"].classes_)
    indexes = [classes.index(category) for category in CATEGORIES]
    return raw[:, indexes]


def cross_validated_probabilities(
    records: Sequence[dict[str, Any]], config: dict[str, Any],
) -> np.ndarray:
    texts = np.asarray([record["normalized_text"] for record in records], dtype=object)
    labels = np.asarray([record["category"] for record in records], dtype=object)
    groups = np.asarray([record["task_family_id"] for record in records], dtype=object)
    probabilities = np.zeros((len(records), len(CATEGORIES)), dtype=float)
    assigned = np.zeros(len(records), dtype=bool)
    splitter = StratifiedGroupKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for train_indexes, test_indexes in splitter.split(texts, labels, groups):
        pipeline = build_pipeline(config)
        pipeline.fit(texts[train_indexes].tolist(), labels[train_indexes].tolist())
        probabilities[test_indexes] = _aligned_probabilities(
            pipeline, texts[test_indexes].tolist()
        )
        assigned[test_indexes] = True
    if not assigned.all():
        raise RuntimeError("cross-validation did not assign every training record")
    return probabilities


def _predictions(probabilities: np.ndarray) -> list[str]:
    return [CATEGORIES[index] for index in probabilities.argmax(axis=1)]


def _hybrid_predictions(
    records: Sequence[dict[str, Any]], learned: Sequence[str],
) -> tuple[list[str], list[str | None]]:
    rules = [rule_predict(record["prompt"]) for record in records]
    hybrid = [rule or prediction for rule, prediction in zip(rules, learned, strict=True)]
    return hybrid, rules


def _metrics(labels: Sequence[str], predictions: Sequence[str]) -> dict[str, Any]:
    report = classification_report(
        labels,
        predictions,
        labels=list(CATEGORIES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "total": len(labels),
        "correct": int(sum(left == right for left, right in zip(labels, predictions, strict=True))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(CATEGORIES), average="macro")),
        "per_category": {
            category: {
                "precision": float(report[category]["precision"]),
                "recall": float(report[category]["recall"]),
                "f1": float(report[category]["f1-score"]),
                "support": int(report[category]["support"]),
            }
            for category in CATEGORIES
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(CATEGORIES)
        ).astype(int).tolist(),
        "confusion_matrix_labels": list(CATEGORIES),
    }


def _rule_metrics(records: Sequence[dict[str, Any]], rules: Sequence[str | None]) -> dict[str, Any]:
    covered = [(record, prediction) for record, prediction in zip(records, rules, strict=True)
               if prediction is not None]
    correct = sum(record["category"] == prediction for record, prediction in covered)
    per_true_category = {}
    for category in CATEGORIES:
        category_rows = [pair for pair in covered if pair[0]["category"] == category]
        per_true_category[category] = {
            "covered": len(category_rows),
            "correct": sum(row["category"] == prediction for row, prediction in category_rows),
            "total": sum(record["category"] == category for record in records),
        }
    per_predicted_category = {}
    for category in CATEGORIES:
        predicted = [pair for pair in covered if pair[1] == category]
        per_predicted_category[category] = {
            "predicted": len(predicted),
            "correct": sum(row["category"] == category for row, _ in predicted),
            "precision": (
                sum(row["category"] == category for row, _ in predicted) / len(predicted)
                if predicted else None
            ),
        }
    return {
        "coverage_count": len(covered),
        "coverage": len(covered) / len(records),
        "correct_covered": correct,
        "precision_when_fired": correct / len(covered) if covered else None,
        "unresolved_count": len(records) - len(covered),
        "unresolved_rate": 1 - len(covered) / len(records),
        "per_true_category": per_true_category,
        "per_predicted_category": per_predicted_category,
    }


def _score_summary(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(probabilities, axis=1)
    top = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    return top, margin


def _abstention_metrics(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[str],
    rules: Sequence[str | None],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    labels = [record["category"] for record in records]
    top, margin = _score_summary(probabilities)
    results = []
    for policy in ABSTENTION_POLICIES:
        accepted = np.asarray([
            rule is not None or (
                top[index] >= policy["minimum_top_score"]
                and margin[index] >= policy["minimum_margin"]
            )
            for index, rule in enumerate(rules)
        ])
        correct = np.asarray([
            label == prediction
            for label, prediction in zip(labels, predictions, strict=True)
        ])
        category_effects = {}
        for category in CATEGORIES:
            category_mask = np.asarray([label == category for label in labels])
            category_accepted = accepted & category_mask
            category_effects[category] = {
                "accepted": int(category_accepted.sum()),
                "total": int(category_mask.sum()),
                "correct_accepted": int((correct & category_accepted).sum()),
            }
        accepted_count = int(accepted.sum())
        correct_accepted = int((correct & accepted).sum())
        results.append({
            **policy,
            "accepted": accepted_count,
            "coverage": accepted_count / len(records),
            "abstained": len(records) - accepted_count,
            "abstention_rate": 1 - accepted_count / len(records),
            "correct_accepted": correct_accepted,
            "errors_accepted": accepted_count - correct_accepted,
            "accuracy_accepted": correct_accepted / accepted_count if accepted_count else None,
            "category_effects": category_effects,
        })
    return results


def _identity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    identity_rows = [{key: record[key] for key in ("task_id", "category", "prompt")}
                     for record in records]
    serialized = json.dumps(identity_rows, sort_keys=True, ensure_ascii=False).encode()
    return {
        "count": len(records),
        "unique_task_ids": len({record["task_id"] for record in records}),
        "per_category": dict(sorted(Counter(record["category"] for record in records).items())),
        "prompt_projection_sha256": hashlib.sha256(serialized).hexdigest(),
        "task_ids": [record["task_id"] for record in records],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_train_stage(
    *, artifact_root: Path = DEFAULT_ROOT, output_root: Path,
) -> dict[str, Any]:
    records = _projected_records("train", artifact_root)
    labels = [record["category"] for record in records]
    candidate_results = []
    candidate_probabilities = {}
    for config in MODEL_CONFIGS:
        probabilities = cross_validated_probabilities(records, config)
        learned = _predictions(probabilities)
        hybrid, rules = _hybrid_predictions(records, learned)
        candidate_probabilities[config["name"]] = probabilities
        candidate_results.append({
            "configuration": config,
            "learned": _metrics(labels, learned),
            "hybrid": _metrics(labels, hybrid),
        })
    selected = max(
        candidate_results,
        key=lambda result: (
            result["learned"]["macro_f1"], result["learned"]["accuracy"],
            -float(result["configuration"]["C"]),
        ),
    )
    selected_name = selected["configuration"]["name"]
    selected_probabilities = candidate_probabilities[selected_name]
    learned = _predictions(selected_probabilities)
    hybrid, rules = _hybrid_predictions(records, learned)
    primary_detector = max(
        ("learned", "hybrid"),
        key=lambda name: (
            _metrics(labels, learned if name == "learned" else hybrid)["macro_f1"],
            _metrics(labels, learned if name == "learned" else hybrid)["accuracy"],
            name == "learned",
        ),
    )
    train_results = {
        "stage": "train_cv",
        "identity": _identity(records),
        "cv": {
            "type": "StratifiedGroupKFold",
            "splits": CV_SPLITS,
            "group": "task_family_id",
            "random_state": RANDOM_STATE,
        },
        "rules": _rule_metrics(records, rules),
        "candidate_models": candidate_results,
        "selected_configuration": selected["configuration"],
        "learned": _metrics(labels, learned),
        "hybrid": _metrics(labels, hybrid),
        "abstention": _abstention_metrics(records, hybrid, rules, selected_probabilities),
        "primary_detector": primary_detector,
    }
    frozen = {
        "schema_version": "phase12.2b.1-freeze-v1",
        "rules_version": RULES_VERSION,
        "rule_precedence": [
            "classification", "coding", "extraction", "structured_json",
            "summarization", "reasoning", "abstain",
        ],
        "normalization": NORMALIZATION,
        "word_tfidf": WORD_TFIDF,
        "character_tfidf": CHAR_TFIDF,
        "logistic_regression": {
            **selected["configuration"],
            "solver": "lbfgs",
            "max_iter": 3000,
            "random_state": RANDOM_STATE,
        },
        "training_identity": train_results["identity"],
        "cv": train_results["cv"],
        "hybrid_behavior": "rule prediction when present; otherwise logistic-regression prediction",
        "abstention_behavior": "rule predictions accepted; learned predictions use top-score/margin policy",
        "abstention_policies": ABSTENTION_POLICIES,
        "primary_detector": primary_detector,
    }
    _write_json(output_root / "train-results.json", train_results)
    _write_json(output_root / "frozen-configuration.json", frozen)
    return train_results


def _validate_freeze(frozen: dict[str, Any], records: Sequence[dict[str, Any]]) -> None:
    expected = {
        "rules_version": RULES_VERSION,
        "normalization": NORMALIZATION,
        "word_tfidf": json.loads(json.dumps(WORD_TFIDF)),
        "character_tfidf": json.loads(json.dumps(CHAR_TFIDF)),
        "abstention_policies": json.loads(json.dumps(ABSTENTION_POLICIES)),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"frozen configuration mismatch: {key}")
    if frozen.get("training_identity") != _identity(records):
        raise ValueError("frozen training identity mismatch")


def _fit_frozen_model(records: Sequence[dict[str, Any]], frozen: dict[str, Any]) -> Pipeline:
    pipeline = build_pipeline(frozen["logistic_regression"])
    pipeline.fit(
        [record["normalized_text"] for record in records],
        [record["category"] for record in records],
    )
    return pipeline


def _route_impact(
    records: Sequence[dict[str, Any]], detected: Sequence[str], *, artifact_root: Path,
    predictor_root: Path,
) -> dict[str, Any]:
    manifest = _manifest_records("dev", artifact_root)
    predictor = SklearnQualityPredictor.from_trusted_artifact(predictor_root)
    service = RoutingDecisionService(predictor)
    extractor = ProductionRequestFeatureExtractor()
    comparisons = []
    for record, predicted_category in zip(records, detected, strict=True):
        difficulty = manifest[record["task_id"]]["difficulty"]
        max_tokens = BENCHMARK_MAX_OUTPUT_TOKENS[record["category"]][difficulty]
        request = InferenceRequest(
            prompt=record["prompt"], max_output_tokens=max_tokens, temperature=0
        )
        structured = record["category"] in {"extraction", "structured_json"}
        true_features = extractor.extract(
            request,
            category_hint=record["category"],
            structured_output_required=structured,
        )
        detected_features = extractor.extract(
            request,
            category_hint=predicted_category,
            structured_output_required=structured,
        )
        true_decision = service.route_features(
            true_features, CANDIDATE_MODELS, PRODUCTION_QUALITY_THRESHOLD
        )
        detected_decision = service.route_features(
            detected_features, CANDIDATE_MODELS, PRODUCTION_QUALITY_THRESHOLD
        )
        comparisons.append({
            "task_id": record["task_id"],
            "true_category": record["category"],
            "detected_category": predicted_category,
            "category_correct": record["category"] == predicted_category,
            "true_selected_model": true_decision.selected_model_id,
            "detected_selected_model": detected_decision.selected_model_id,
            "selected_model_changed": (
                true_decision.selected_model_id != detected_decision.selected_model_id
            ),
            "true_fallback": true_decision.fallback_used,
            "detected_fallback": detected_decision.fallback_used,
            "fallback_changed": true_decision.fallback_used != detected_decision.fallback_used,
            "true_projected_cost_usd": str(true_decision.selected_projected_cost_usd),
            "detected_projected_cost_usd": str(detected_decision.selected_projected_cost_usd),
            "projected_cost_delta_usd": str(
                detected_decision.selected_projected_cost_usd
                - true_decision.selected_projected_cost_usd
            ),
        })
    changed_errors = [row for row in comparisons
                      if not row["category_correct"] and row["selected_model_changed"]]
    confusion_changes = Counter(
        f"{row['true_category']}->{row['detected_category']}" for row in changed_errors
    )
    true_cost = sum(Decimal(row["true_projected_cost_usd"]) for row in comparisons)
    detected_cost = sum(Decimal(row["detected_projected_cost_usd"]) for row in comparisons)
    return {
        "requests": len(comparisons),
        "correct_detected_category": sum(row["category_correct"] for row in comparisons),
        "incorrect_detected_category": sum(not row["category_correct"] for row in comparisons),
        "same_selected_model": sum(not row["selected_model_changed"] for row in comparisons),
        "changed_selected_model": sum(row["selected_model_changed"] for row in comparisons),
        "same_fallback_state": sum(not row["fallback_changed"] for row in comparisons),
        "changed_fallback_state": sum(row["fallback_changed"] for row in comparisons),
        "true_total_projected_cost_usd": str(true_cost),
        "detected_total_projected_cost_usd": str(detected_cost),
        "projected_cost_delta_usd": str(detected_cost - true_cost),
        "route_changing_confusions": dict(sorted(confusion_changes.items())),
        "category_errors_that_changed_model": changed_errors,
        "comparisons": comparisons,
        "historical_acceptability_comparison": (
            "not performed; provider outcomes and validation labels are forbidden inputs"
        ),
    }


def run_dev_stage(
    *, artifact_root: Path = DEFAULT_ROOT, output_root: Path,
    predictor_root: Path = Path("deploy/router"),
) -> dict[str, Any]:
    frozen = json.loads((output_root / "frozen-configuration.json").read_text(encoding="utf-8"))
    train_records = _projected_records("train", artifact_root)
    _validate_freeze(frozen, train_records)
    pipeline = _fit_frozen_model(train_records, frozen)

    dev_records = _projected_records("dev", artifact_root)
    labels = [record["category"] for record in dev_records]
    probabilities = _aligned_probabilities(
        pipeline, [record["normalized_text"] for record in dev_records]
    )
    learned = _predictions(probabilities)
    hybrid, rules = _hybrid_predictions(dev_records, learned)
    primary = learned if frozen["primary_detector"] == "learned" else hybrid
    report = {
        "stage": "single_frozen_dev_evaluation",
        "identity": _identity(dev_records),
        "frozen_configuration": frozen,
        "rules": _rule_metrics(dev_records, rules),
        "learned": _metrics(labels, learned),
        "hybrid": _metrics(labels, hybrid),
        "primary_detector": frozen["primary_detector"],
        "abstention": _abstention_metrics(dev_records, hybrid, rules, probabilities),
        "routing_impact": _route_impact(
            dev_records, primary, artifact_root=artifact_root, predictor_root=predictor_root
        ),
        "safety": {
            "provider_calls": 0,
            "aws_calls": 0,
            "semantic_judge_calls": 0,
            "external_downloads": 0,
            "final_access": 0,
            "quality_threshold": PRODUCTION_QUALITY_THRESHOLD,
            "predictor_sha256": hashlib.sha256(
                (predictor_root / "predictor.pkl").read_bytes()
            ).hexdigest(),
            "expected_predictor_sha256": PRODUCTION_PREDICTOR_SHA256,
        },
    }
    _write_json(output_root / "dev-results.json", report)
    return report


def _boundary_correction_metrics(
    records: Sequence[dict[str, Any]], before: Sequence[str], after: Sequence[str],
) -> dict[str, Any]:
    changes = []
    fired = 0
    for record, previous, revised in zip(records, before, after, strict=True):
        if previous in {"extraction", "structured_json"} and (
            _TRANSFORMATION_OPERATION.search(record["normalized_text"])
            or _DIRECT_RECOVERY_SOURCE.search(record["normalized_text"])
        ):
            fired += 1
        if previous != revised:
            changes.append({
                "task_id": record["task_id"],
                "true_category": record["category"],
                "before": previous,
                "after": revised,
                "before_correct": previous == record["category"],
                "after_correct": revised == record["category"],
            })
    return {
        "version": BOUNDARY_CORRECTION_VERSION,
        "eligible_pair": ["extraction", "structured_json"],
        "precedence": ["transformation_operation", "direct_recovery_source", "unchanged"],
        "signal_coverage_count": fired,
        "prediction_changes": len(changes),
        "corrected_errors": sum(
            not item["before_correct"] and item["after_correct"] for item in changes
        ),
        "introduced_errors": sum(
            item["before_correct"] and not item["after_correct"] for item in changes
        ),
        "changes": changes,
    }


def run_correction_train_stage(
    *, artifact_root: Path = DEFAULT_ROOT,
    previous_root: Path = Path("artifacts/category-detector/phase12.2b.1"),
    output_root: Path,
) -> dict[str, Any]:
    previous_freeze = json.loads(
        (previous_root / "frozen-configuration.json").read_text(encoding="utf-8")
    )
    records = _projected_records("train", artifact_root)
    _validate_freeze(previous_freeze, records)
    config = previous_freeze["logistic_regression"]
    probabilities = cross_validated_probabilities(records, config)
    before = _predictions(probabilities)
    after = _revised_predictions(records, before)
    labels = [record["category"] for record in records]
    results = {
        "stage": "train_only_bounded_correction",
        "identity": _identity(records),
        "base_configuration": previous_freeze,
        "before": _metrics(labels, before),
        "after": _metrics(labels, after),
        "existing_rules": _rule_metrics(
            records, [rule_predict(record["prompt"]) for record in records]
        ),
        "boundary_correction": _boundary_correction_metrics(records, before, after),
    }
    frozen = {
        "schema_version": "phase12.2b.1b-freeze-v1",
        "base_configuration": previous_freeze,
        "boundary_correction_version": BOUNDARY_CORRECTION_VERSION,
        "eligible_predictions": ["extraction", "structured_json"],
        "precedence": ["transformation_operation", "direct_recovery_source", "unchanged"],
        "transformation_operation_pattern": _TRANSFORMATION_OPERATION.pattern,
        "direct_recovery_source_pattern": _DIRECT_RECOVERY_SOURCE.pattern,
        "training_identity": results["identity"],
        "train_before": results["before"],
        "train_after": results["after"],
        "correction_metrics": results["boundary_correction"],
        "primary_detector": "learned_plus_boundary_correction",
    }
    _write_json(output_root / "train-correction-results.json", results)
    _write_json(output_root / "frozen-correction.json", frozen)
    return results


def _validate_correction_freeze(
    frozen: dict[str, Any], records: Sequence[dict[str, Any]],
) -> None:
    expected = {
        "boundary_correction_version": BOUNDARY_CORRECTION_VERSION,
        "eligible_predictions": ["extraction", "structured_json"],
        "precedence": ["transformation_operation", "direct_recovery_source", "unchanged"],
        "transformation_operation_pattern": _TRANSFORMATION_OPERATION.pattern,
        "direct_recovery_source_pattern": _DIRECT_RECOVERY_SOURCE.pattern,
        "training_identity": _identity(records),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"frozen correction mismatch: {key}")
    _validate_freeze(frozen["base_configuration"], records)


def run_correction_dev_regression(
    *, artifact_root: Path = DEFAULT_ROOT,
    previous_root: Path = Path("artifacts/category-detector/phase12.2b.1"),
    output_root: Path,
    predictor_root: Path = Path("deploy/router"),
) -> dict[str, Any]:
    frozen = json.loads((output_root / "frozen-correction.json").read_text(encoding="utf-8"))
    train_records = _projected_records("train", artifact_root)
    _validate_correction_freeze(frozen, train_records)
    pipeline = _fit_frozen_model(train_records, frozen["base_configuration"])

    dev_records = _projected_records("dev", artifact_root)
    labels = [record["category"] for record in dev_records]
    probabilities = _aligned_probabilities(
        pipeline, [record["normalized_text"] for record in dev_records]
    )
    before = _predictions(probabilities)
    previous_dev = json.loads((previous_root / "dev-results.json").read_text(encoding="utf-8"))
    previous_predictions = {
        row["task_id"]: row["detected_category"]
        for row in previous_dev["routing_impact"]["comparisons"]
    }
    if any(
        previous_predictions.get(record["task_id"]) != prediction
        for record, prediction in zip(dev_records, before, strict=True)
    ):
        raise ValueError("recomputed DEV baseline does not match the prior frozen evaluation")
    after = _revised_predictions(dev_records, before)
    routing = _route_impact(
        dev_records, after, artifact_root=artifact_root, predictor_root=predictor_root
    )
    report = {
        "stage": "REUSED DEV REGRESSION CHECK",
        "fresh_holdout": False,
        "identity": _identity(dev_records),
        "frozen_correction": frozen,
        "before": _metrics(labels, before),
        "after": _metrics(labels, after),
        "boundary_correction": _boundary_correction_metrics(dev_records, before, after),
        "routing_impact": routing,
        "previous_routing_impact": {
            key: previous_dev["routing_impact"][key]
            for key in (
                "same_selected_model", "changed_selected_model",
                "same_fallback_state", "changed_fallback_state",
                "projected_cost_delta_usd",
            )
        },
        "safety": {
            "provider_calls": 0,
            "aws_calls": 0,
            "semantic_judge_calls": 0,
            "external_downloads": 0,
            "final_access": 0,
            "quality_threshold": PRODUCTION_QUALITY_THRESHOLD,
            "predictor_sha256": hashlib.sha256(
                (predictor_root / "predictor.pkl").read_bytes()
            ).hexdigest(),
            "expected_predictor_sha256": PRODUCTION_PREDICTOR_SHA256,
        },
    }
    _write_json(output_root / "reused-dev-regression.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("train", "dev", "correction-train", "correction-dev")
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/category-detector/phase12.2b.1"),
    )
    args = parser.parse_args()
    if args.stage == "train":
        result = run_train_stage(output_root=args.output_root)
    elif args.stage == "dev":
        result = run_dev_stage(output_root=args.output_root)
    elif args.stage == "correction-train":
        result = run_correction_train_stage(output_root=args.output_root)
    else:
        result = run_correction_dev_regression(output_root=args.output_root)
    print(json.dumps({
        "stage": result["stage"],
        "identity": result["identity"],
        "primary_detector": result.get("primary_detector"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
