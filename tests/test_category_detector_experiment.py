import json
import socket

import numpy as np

from adaptive_llm_gateway.experiments import category_detector as experiment


def test_normalization_is_small_and_deterministic():
    assert experiment.normalize_text("  Hello\n\tWORLD  ") == "hello world"


def test_rules_use_deliverable_precedence_and_can_abstain():
    assert experiment.rule_predict(
        "Classify this Python request into one declared label."
    ) == "classification"
    assert experiment.rule_predict(
        "Write a Python function that returns a JSON object."
    ) == "coding"
    assert experiment.rule_predict(
        "Extract the requested fields from this JSON source and return JSON."
    ) == "extraction"
    assert experiment.rule_predict(
        "Transform the supplied records and output a JSON object."
    ) == "structured_json"
    assert experiment.rule_predict(
        "Summarize the following source document in two sentences."
    ) == "summarization"
    assert experiment.rule_predict(
        "Calculate the schedule allocation and return the result."
    ) == "reasoning"
    assert experiment.rule_predict("What city is named in the context?") is None


def test_narrow_boundary_correction_uses_operation_semantics():
    assert experiment.correct_extraction_json_boundary(
        "Return JSON for these rows, sorted by score.", "extraction"
    ) == "structured_json"
    assert experiment.correct_extraction_json_boundary(
        "Return JSON with name and email. Contact: Mira Chen.", "structured_json"
    ) == "extraction"
    assert experiment.correct_extraction_json_boundary(
        "Return JSON for these rows, sorted by score.", "qa"
    ) == "qa"
    assert experiment.correct_extraction_json_boundary(
        "Return a JSON object with these supplied values.", "structured_json"
    ) == "structured_json"


def test_train_only_boundary_correction_improves_target_without_other_regressions(tmp_path):
    previous_root = tmp_path / "previous"
    experiment.run_train_stage(output_root=previous_root)
    result = experiment.run_correction_train_stage(
        previous_root=previous_root,
        output_root=tmp_path / "correction",
    )

    assert result["before"]["correct"] == 113
    assert result["after"]["correct"] == 129
    assert result["boundary_correction"]["introduced_errors"] == 0
    assert result["boundary_correction"]["corrected_errors"] == 16
    for category in set(experiment.CATEGORIES) - {"extraction", "structured_json"}:
        assert result["after"]["per_category"][category] == result["before"]["per_category"][category]


def test_pipeline_predictions_are_deterministic():
    texts = [
        "choose the declared label", "select the category label",
        "write python function", "implement executable code",
        "extract specified fields", "recover fields from source",
        "produce json object", "transform json array",
        "answer this direct question", "what is the supplied answer",
        "solve the allocation", "calculate the schedule",
        "summarize supplied source", "summarize this report",
    ]
    labels = [category for category in experiment.CATEGORIES for _ in range(2)]
    first = experiment.build_pipeline(experiment.MODEL_CONFIGS[0]).fit(texts, labels)
    second = experiment.build_pipeline(experiment.MODEL_CONFIGS[0]).fit(texts, labels)
    assert np.array_equal(first.predict(texts), second.predict(texts))
    assert np.array_equal(first.predict_proba(texts), second.predict_proba(texts))


def test_train_stage_is_offline_grouped_and_reproducible(tmp_path, monkeypatch):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(socket, "socket", network_forbidden)

    first = experiment.run_train_stage(output_root=tmp_path / "first")
    second = experiment.run_train_stage(output_root=tmp_path / "second")

    assert first == second
    assert first["identity"]["count"] == 140
    assert first["identity"]["unique_task_ids"] == 140
    assert first["cv"] == {
        "type": "StratifiedGroupKFold",
        "splits": 5,
        "group": "task_family_id",
        "random_state": experiment.RANDOM_STATE,
    }
    frozen = json.loads(
        (tmp_path / "first" / "frozen-configuration.json").read_text(encoding="utf-8")
    )
    assert frozen["training_identity"] == first["identity"]
    assert frozen["primary_detector"] in {"learned", "hybrid"}
    assert len(frozen["abstention_policies"]) == 3
