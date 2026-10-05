import asyncio
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from adaptive_llm_gateway.evaluation.final_harness import (
    CANDIDATES,
    PRIMARY_BASELINE,
    CandidateObservation,
    FinalPaths,
    FinalResults,
    FreezeVerification,
    FrozenExpectations,
    RouterSelection,
    _begin_semantic_judge_once,
    _canonical_digest,
    _prepare_semantic_evaluation,
    _semantic_evaluate,
    aggregate_strategies,
    build_final_results,
    main,
    render_markdown,
    recover_results_publication,
    replay_router,
    validate_frozen_sandbox,
    validate_final_evaluation,
    verify_execution_provenance,
    verify_final_freeze,
    write_results,
)
from adaptive_llm_gateway.evaluation.final_ledgers import (
    CandidateAttemptLedger,
    SemanticAttemptLedger,
)
from adaptive_llm_gateway.evaluation.models import EvaluationResult
from adaptive_llm_gateway.evaluation.sandbox import DockerPythonSandbox
from adaptive_llm_gateway.benchmarks.features import (
    build_request_feature_binding,
    extract_request_features,
    verify_request_feature_binding,
)
from adaptive_llm_gateway.benchmarks.models import BenchmarkResult, BenchmarkTask
from adaptive_llm_gateway.models import (
    ModelCapabilities,
    ModelConfig,
    ReasoningBehavior,
    ReasoningEffort,
)


class FixedProbabilityStep(ClassifierMixin, BaseEstimator):
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, values):
        probabilities = np.resize(self.probabilities, len(values))
        return np.column_stack((1 - probabilities, probabilities))

    def fit(self, values, target=None):
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model(model_id, input_price, output_price):
    return ModelConfig(
        model_id=model_id, provider="fake", provider_model_name=f"fake/{model_id}",
        input_cost_per_1m_tokens=input_price,
        output_cost_per_1m_tokens=output_price,
        context_window=10000,
        capabilities=ModelCapabilities(
            supports_temperature=True, supports_structured_output=True,
            reasoning=ReasoningBehavior.PROVIDER_DEFAULT),
        reasoning_effort=ReasoningEffort.NONE,
    )


def models():
    return tuple(model(candidate, str(index + 1), str(index + 1))
                 for index, candidate in enumerate(CANDIDATES))


def features(category="qa"):
    return {
        "category": category,
        "prompt_characters": 20,
        "system_prompt_characters": 0,
        "approximate_input_tokens": 5,
        "contains_code": False,
        "requests_structured_output": False,
        "max_output_tokens": 10,
        "constraint_indicator_count": 1,
        "reasoning_indicator_count": 0,
    }


def observation(task, candidate, *, category="qa", acceptable=True,
                provider_success=True, evaluation_status="evaluated", cost="0.01",
                judge_failure=False):
    return CandidateObservation(
        task_id=task, category=category, model_id=candidate,
        request_features=features(category), provider_success=provider_success,
        evaluation_status=evaluation_status,
        acceptable=(acceptable if evaluation_status == "evaluated" else None),
        realized_cost_usd=(Decimal(cost) if cost is not None else None),
        judge_failure=judge_failure,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def synthetic_provenance(_authorization, *, repository_root):
    return "b" * 40, {}, _canonical_digest({})


def freeze_fixture(tmp_path, *, authorized=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates = CANDIDATES
    files = {name: tmp_path / f"{name}.json" for name in (
        "dataset", "split", "proposition", "evaluator", "protocol",
        "final_manifest", "pricing_readiness", "authorization", "confirmation")}
    implementation = tmp_path / "implementation.py"
    implementation.write_text("FROZEN = True\n")
    routing_implementation = tmp_path / "routing_implementation.py"
    routing_implementation.write_text("FROZEN = True\n")
    predictor = tmp_path / "predictor"
    predictor.mkdir()
    (predictor / "predictor.pkl").write_bytes(b"synthetic-predictor")
    write_json(predictor / "metadata.json", {
        "approved_quality_threshold": 0.8,
        "known_candidate_ids": sorted(candidates),
        "predictor_sha256": digest(predictor / "predictor.pkl"),
    })
    policy = tmp_path / "policy.json"
    write_json(policy, {
        "policy_id": "routellm-frozen-final-policy", "policy_version": "1.0.0",
        "quality_threshold": "0.80", "candidate_portfolio": candidates,
        "predictor": {"artifact": "artifacts/routing-quality/rb12-train-candidate-v1",
                      "sha256": digest(predictor / "predictor.pkl")},
        "selection": {"ordered_keys": ["projected_cost_usd_ascending",
                                          "candidate_id_ascending"]},
        "fallback": {"ordered_keys": ["predicted_acceptability_descending",
                                         "projected_cost_usd_ascending",
                                         "candidate_id_ascending"]},
        "implementation_files": {
            str(routing_implementation): digest(routing_implementation)},
        "serving_cost": {"counterfactual_matrix_cost_excluded": True,
                         "evaluation_cost_excluded": True,
                         "selected_candidate_realized_inference_cost_only": True},
        "primary_fixed_baseline": PRIMARY_BASELINE,
    })
    for name in ("dataset", "split", "proposition", "final_manifest"):
        write_json(files[name], {"synthetic": name})
    write_json(files["protocol"], {"version": "1.7.0"})
    write_json(files["evaluator"], {"version": "1.3.0", "implementation_files": {
        str(implementation): digest(implementation)}})
    write_json(files["confirmation"], {
        "status": "confirmed", "calls_attempted": 5, "complete_responses": 5,
        "other_failures": 0, "output_budget_exhaustions": 0,
    })
    write_json(files["pricing_readiness"], {
        "status": "READY_FOR_PAID_EXECUTION", "protocol_version": "1.7.0",
        "protocol_sha256": digest(files["protocol"]),
        "models": [{"model_id": item} for item in (*candidates, "judge-gpt-6-astra")],
    })
    expected = FrozenExpectations(
        dataset_sha256=digest(files["dataset"]), split_sha256=digest(files["split"]),
        proposition_sha256=digest(files["proposition"]),
        evaluator_sha256=digest(files["evaluator"]),
        protocol_sha256=digest(files["protocol"]),
        final_manifest_sha256=digest(files["final_manifest"]),
        predictor_sha256=digest(predictor / "predictor.pkl"),
        policy_sha256=digest(policy), candidates=candidates,
    )
    experiment = {
        "dataset_sha256": expected.dataset_sha256,
        "evaluator_manifest_sha256": expected.evaluator_sha256,
        "final_manifest_sha256": expected.final_manifest_sha256,
        "policy_sha256": expected.policy_sha256,
        "predictor_sha256": expected.predictor_sha256,
        "proposition_specification_sha256": expected.proposition_sha256,
        "protocol_sha256": expected.protocol_sha256,
        "split_manifest_sha256": expected.split_sha256,
    }
    write_json(files["authorization"], {
        "version": "1.0.0", "experiment": experiment,
        "authorization": {
            "status": ("AUTHORIZED_FOR_FINAL_EXECUTION" if authorized
                       else "AWAITING_INDEPENDENT_REVIEW"),
            "final_paid_execution_authorized": authorized,
            "authorized_by": "synthetic-reviewer" if authorized else None,
            "authorized_on": datetime.now(timezone.utc).date().isoformat()
            if authorized else None,
            "pricing_reverified_by": "synthetic-reviewer" if authorized else None,
            "pricing_reverified_on": datetime.now(timezone.utc).date().isoformat()
            if authorized else None,
        },
        "call_limits": {"candidate_calls": 168, "candidate_retries": 0,
                        "semantic_judge_calls": 24, "semantic_judge_retries": 0},
        "cost_budget_usd": {
            "conservative_maximum": "1.55262531",
            "expected": "0.80157315",
            "repository_estimated_maximum": "1.41147755",
        },
        "evidence": {
            "pricing_readiness_sha256": digest(files["pricing_readiness"]),
            "gemini_native_minimal_confirmation_status_sha256": digest(files["confirmation"]),
        },
    })
    paths = FinalPaths(
        dataset=files["dataset"], split=files["split"],
        proposition=files["proposition"], evaluator=files["evaluator"],
        protocol=files["protocol"], final_manifest=files["final_manifest"],
        predictor_directory=predictor, policy=policy,
        pricing_readiness=files["pricing_readiness"],
        authorization=files["authorization"], confirmation_status=files["confirmation"],
    )
    fake_models = models()
    loader = lambda path: ({"version": "1.7.0"}, fake_models, digest(path))
    return paths, expected, loader


def test_freeze_verifier_accepts_exact_synthetic_state(tmp_path):
    paths, expected, loader = freeze_fixture(tmp_path)
    result = verify_final_freeze(
        paths=paths, expectations=expected, results_root=tmp_path / "runs",
        explicit_authorization=True, protocol_loader=loader,
        provenance_verifier=synthetic_provenance)
    assert result.status == "READY" and result.provider_calls_made == 0


def test_preflight_without_independent_authorization_is_blocked_but_zero_call(tmp_path):
    paths, expected, loader = freeze_fixture(tmp_path, authorized=False)
    result = verify_final_freeze(
        paths=paths, expectations=expected, results_root=tmp_path / "runs",
        require_authorization=False, protocol_loader=loader)
    assert result.status == "BLOCKED"
    assert result.authorization_status == "AWAITING_INDEPENDENT_REVIEW"
    assert result.provider_calls_made == 0


@pytest.mark.parametrize("field", ["dataset", "policy", "predictor", "protocol"])
def test_freeze_verifier_rejects_identity_mismatch(tmp_path, field):
    paths, expected, loader = freeze_fixture(tmp_path)
    target = (paths.predictor_directory / "predictor.pkl"
              if field == "predictor" else getattr(paths, field))
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "runs", explicit_authorization=True,
            protocol_loader=loader, provenance_verifier=synthetic_provenance)


@pytest.mark.parametrize("mismatch", ["threshold", "portfolio"])
def test_freeze_verifier_rejects_threshold_and_portfolio_mismatch(tmp_path, mismatch):
    paths, expected, loader = freeze_fixture(tmp_path)
    value = json.loads(paths.predictor_directory.joinpath("metadata.json").read_text())
    if mismatch == "threshold":
        value["approved_quality_threshold"] = 0.7
    else:
        value["known_candidate_ids"] = value["known_candidate_ids"][:-1]
    write_json(paths.predictor_directory / "metadata.json", value)
    with pytest.raises(ValueError, match="predictor approval metadata mismatch"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "runs", explicit_authorization=True,
            protocol_loader=loader)


def test_stale_readiness_and_missing_authorization_are_rejected(tmp_path):
    paths, expected, loader = freeze_fixture(tmp_path, authorized=False)
    with pytest.raises(ValueError, match="lacks same-day authorization"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "runs", explicit_authorization=True,
            protocol_loader=loader)


def test_evaluator_implementation_hash_mismatch_is_rejected(tmp_path):
    paths, expected, loader = freeze_fixture(tmp_path)
    implementation = Path(next(iter(
        json.loads(paths.evaluator.read_text())["implementation_files"])))
    implementation.write_text("FROZEN = False\n")
    with pytest.raises(ValueError, match="evaluator implementation hash mismatch"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "runs", explicit_authorization=True,
            protocol_loader=loader)
    paths, expected, loader = freeze_fixture(tmp_path / "stale")
    value = json.loads(paths.pricing_readiness.read_text())
    value["status"] = "REQUIRES_REVERIFICATION"
    write_json(paths.pricing_readiness, value)
    with pytest.raises(ValueError, match="pricing readiness is stale"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "stale-runs", explicit_authorization=True,
            protocol_loader=loader)


def test_duplicate_completed_execution_is_rejected(tmp_path):
    paths, expected, loader = freeze_fixture(tmp_path)
    ready = verify_final_freeze(paths=paths, expectations=expected,
        results_root=tmp_path / "runs", explicit_authorization=True,
        protocol_loader=loader, provenance_verifier=synthetic_provenance)
    run_id = UUID(int=1)
    run = tmp_path / "runs" / str(run_id)
    write_json(run / "status.json", {"status": "completed"})
    write_json(run / "manifest.json", {"configuration": {
        "final_experiment_identity": ready.experiment_identity}})
    with pytest.raises(ValueError, match="already exists"):
        verify_final_freeze(paths=paths, expectations=expected,
            results_root=tmp_path / "runs", explicit_authorization=True,
            protocol_loader=loader, provenance_verifier=synthetic_provenance)
    continued = verify_final_freeze(paths=paths, expectations=expected,
        results_root=tmp_path / "runs", explicit_authorization=True,
        allowed_run_id=run_id, protocol_loader=loader,
        provenance_verifier=synthetic_provenance)
    assert continued.status == "READY"


def test_semantic_judge_dry_run_makes_zero_calls(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["final_harness", "semantic-judge",
        "--root", "unused", "--run-id", "00000000-0000-0000-0000-000000000001",
        "--dry-run"])
    main()
    assert json.loads(capsys.readouterr().out)["provider_calls_made"] == 0


def test_semantic_judge_batch_marker_is_idempotent_because_ledger_guards_calls(tmp_path):
    run_id = UUID(int=2)
    (tmp_path / str(run_id) / "evaluations").mkdir(parents=True)
    _begin_semantic_judge_once(tmp_path, run_id)
    marker = json.loads(
        (tmp_path / str(run_id) / "semantic-judge-attempt.json").read_text())
    assert marker["status"] == "started" and marker["maximum_calls"] == 24
    _begin_semantic_judge_once(tmp_path, run_id)
    assert json.loads(
        (tmp_path / str(run_id) / "semantic-judge-attempt.json").read_text()
    )["status"] == "started"


def test_replay_rejects_older_generic_evaluation_configuration(tmp_path):
    run_id = UUID(int=3)
    directory = tmp_path / str(run_id)
    write_json(directory / "semantic-judge-attempt.json", {
        "status": "completed", "judge_calls": 0})
    metric = {
        "evaluated_tasks": 168, "fully_evaluated_tasks": 0,
        "incomplete_evaluations": 168, "successful_responses": 168,
        "acceptable_responses": 0, "mean_quality_score": None,
        "acceptable_rate": None, "total_estimated_cost_usd": "0",
        "average_cost_per_task_usd": "0",
        "cost_per_acceptable_response_usd": None, "average_latency_ms": 0,
    }
    write_json(directory / "evaluation-summary.json", {
        "run_id": str(run_id), "dataset_sha256": "synthetic",
        "evaluation_version": "test", "overall": metric,
        "by_model": [], "by_model_category": [], "comparisons": [],
        "judge_calls": 0,
        "evaluation_configuration": {
            "functional_sandbox": None, "semantic_judge": None,
            "hybrid_semantic_judge": None,
        },
    })
    with pytest.raises(ValueError, match="does not match Evaluator 1.3"):
        validate_final_evaluation(tmp_path, run_id)


def test_replay_selects_cheapest_qualifying_and_fallback_without_provider_calls(tmp_path):
    pipeline = Pipeline((("probability", FixedProbabilityStep([0.9, 0.85, 0.7, 0.6])),))
    predictor = tmp_path / "predictor.pkl"
    predictor.write_bytes(pickle.dumps(pipeline))
    rows = tuple(observation(task, candidate) for task in ("a", "b") for candidate in CANDIDATES)
    selected = replay_router(rows, predictor_path=predictor,
        predictor_sha256=digest(predictor), models=models())
    assert len(selected) == 2
    assert all(item.selected_model_id == CANDIDATES[0] and not item.fallback_used
               for item in selected)

    fallback_pipeline = Pipeline((("probability", FixedProbabilityStep([0.2, 0.4, 0.7, 0.7])),))
    predictor.write_bytes(pickle.dumps(fallback_pipeline))
    selected = replay_router(rows, predictor_path=predictor,
        predictor_sha256=digest(predictor), models=models())
    assert all(item.fallback_used and item.selected_model_id == CANDIDATES[2]
               for item in selected)
    with pytest.raises(ValueError, match="exactly 0.80"):
        replay_router(rows, predictor_path=predictor,
            predictor_sha256=digest(predictor), models=models(),
            threshold=Decimal("0.79"))

    tie_pipeline = Pipeline((
        ("probability", FixedProbabilityStep([0.9, 0.9, 0.9, 0.9])),))
    predictor.write_bytes(pickle.dumps(tie_pipeline))
    equal_price_models = tuple(model(candidate, "1", "1") for candidate in CANDIDATES)
    selected = replay_router(rows, predictor_path=predictor,
        predictor_sha256=digest(predictor), models=equal_price_models)
    assert all(item.selected_model_id == min(CANDIDATES) for item in selected)


def test_aggregation_uses_requests_for_y_and_excludes_judge_cost():
    rows = []
    selections = []
    for index in range(3):
        task = f"task-{index}"
        for candidate_index, candidate in enumerate(CANDIDATES):
            rows.append(observation(task, candidate, acceptable=index == 0,
                provider_success=index != 1,
                evaluation_status="partial" if index == 2 else "evaluated",
                cost=str(Decimal("0.01") * (candidate_index + 1)),
                judge_failure=index == 2))
        picked = rows[-4]
        selections.append(RouterSelection(
            task_id=task, category="qa", selected_model_id=picked.model_id,
            predicted_acceptability=0.9, projected_cost_usd=Decimal("0.01"),
            threshold_satisfied=True, fallback_used=False,
            provider_success=picked.provider_success,
            evaluation_status=picked.evaluation_status,
            acceptable=picked.acceptable, realized_cost_usd=picked.realized_cost_usd,
            judge_failure=picked.judge_failure))
    strategies = aggregate_strategies(tuple(rows), tuple(selections))
    router = strategies[0]
    assert router.independent_requests == 3 and router.acceptable == 1
    assert router.task_acceptability == pytest.approx(1 / 3)
    assert router.provider_failures == router.unresolved_evaluations == router.judge_failures == 1
    assert router.total_realized_inference_cost_usd == Decimal("0.03")
    assert next(item for item in strategies if item.analysis_only).strategy_id.startswith("oracle:")


def complete_fixture(*, missing_cost=False):
    categories = ("classification", "coding", "extraction", "structured_json",
                  "qa", "reasoning", "summarization")
    rows, selections = [], []
    for category in categories:
        for number in range(6):
            task = f"{category}-{number}"
            task_rows = []
            for index, candidate in enumerate(CANDIDATES):
                item = observation(task, candidate, category=category, acceptable=True,
                    cost=(None if missing_cost and candidate == CANDIDATES[0]
                          else str(Decimal(index + 1) / 100)))
                rows.append(item)
                task_rows.append(item)
            picked = task_rows[0]
            selections.append(RouterSelection(
                task_id=task, category=category, selected_model_id=picked.model_id,
                predicted_acceptability=0.9, projected_cost_usd=Decimal("0.01"),
                threshold_satisfied=True, fallback_used=False,
                provider_success=True, evaluation_status="evaluated", acceptable=True,
                realized_cost_usd=picked.realized_cost_usd, judge_failure=False))
    return tuple(rows), tuple(selections)


def verification():
    return FreezeVerification(status="READY", experiment_identity="e" * 64,
        identities={"synthetic": "f" * 64}, policy_sha256="a" * 64,
        authorization_status="AUTHORIZED_FOR_FINAL_EXECUTION",
        explicit_authorization=True, duplicate_completed_run=False)


def test_canonical_results_x_y_n_and_markdown_report():
    rows, selections = complete_fixture()
    result = build_final_results(
        verification=verification(), observations=rows, selections=selections,
        judge_call_count=24, git_commit="b" * 40,
        generated_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert result.final_request_count == result.primary_metrics.independent_requests == 42
    assert result.candidate_call_count == 168
    assert result.primary_baseline == PRIMARY_BASELINE
    assert result.primary_metrics.task_acceptability_percent == 100
    assert result.primary_metrics.cost_reduction_percent == 75
    assert result.primary_metrics.resume_statement is not None
    assert result.cost_accounting.candidate_matrix_collection_cost_usd == Decimal("4.20")
    assert result.cost_accounting.semantic_judge_cost_usd == 0
    assert all(value == 6 for value in result.category_counts.values())
    report = render_markdown(result)
    assert "Oracle (analysis only)" in report and "offline held-out" in report


def test_incomplete_cost_suppresses_x_and_resume_claim():
    rows, selections = complete_fixture(missing_cost=True)
    result = build_final_results(
        verification=verification(), observations=rows, selections=selections,
        judge_call_count=24, git_commit="b" * 40)
    assert not result.cost_complete
    assert result.primary_metrics.cost_reduction_percent is None
    assert result.primary_metrics.resume_statement is None


def test_judge_call_budget_is_enforced_before_result_creation():
    rows, selections = complete_fixture()
    with pytest.raises(ValueError, match="exceeds the frozen maximum"):
        build_final_results(
            verification=verification(), observations=rows, selections=selections,
            judge_call_count=25, git_commit="b" * 40)


def _candidate_bindings():
    return {f"task-{index}": {"binding_sha256": hashlib.sha256(
        f"task-{index}".encode()).hexdigest()} for index in range(42)}


def _failed_candidate(run_id, task_id, model_id, request_id):
    return BenchmarkResult(
        run_id=run_id, request_id=request_id, task_id=task_id,
        model_id=model_id, success=False, error_category="provider_failure",
        latency_ms=1,
    )


def test_candidate_ledger_resumes_after_57_without_repeating_completed_calls(tmp_path):
    path = tmp_path / "candidate-attempts.json"
    run_one, run_two = UUID(int=10), UUID(int=11)
    ledger = CandidateAttemptLedger(path, experiment_identity="e" * 64)
    bindings = _candidate_bindings()
    ledger.initialize(run_id=run_one, task_bindings=bindings, candidate_ids=CANDIDATES)
    pairs = [(task, candidate) for task in bindings for candidate in CANDIDATES]
    for index, (task, candidate) in enumerate(pairs[:57]):
        request_id = f"request-{index}"
        ledger.start(task, candidate, run_one, request_id)
        ledger.finish(_failed_candidate(run_one, task, candidate, request_id))

    restarted = CandidateAttemptLedger(path, experiment_identity="e" * 64)
    restarted.initialize(run_id=run_two, task_bindings=bindings, candidate_ids=CANDIDATES)
    assert restarted.counts() == {
        "pending": 111, "started": 0, "completed": 0, "failed": 57}
    for task, candidate in pairs[:57]:
        assert restarted.reusable(task, candidate, run_two) is not None
    assert restarted.reusable(*pairs[57], run_two) is None


def test_candidate_ledger_started_attempt_fails_closed_and_second_run_cannot_bypass(tmp_path):
    path = tmp_path / "candidate-attempts.json"
    bindings = _candidate_bindings()
    ledger = CandidateAttemptLedger(path, experiment_identity="e" * 64)
    ledger.initialize(run_id=UUID(int=12), task_bindings=bindings, candidate_ids=CANDIDATES)
    ledger.start("task-0", CANDIDATES[0], UUID(int=12), "ambiguous-request")
    with pytest.raises(ValueError, match="ambiguous started candidate"):
        CandidateAttemptLedger(path, experiment_identity="e" * 64).initialize(
            run_id=UUID(int=13), task_bindings=bindings, candidate_ids=CANDIDATES)
    assert ledger.counts()["started"] == 1


def _semantic_result(run_id, task_id, model_id, number):
    return EvaluationResult(
        benchmark_result_id=UUID(int=1000 + number), run_id=run_id,
        task_id=task_id, model_id=model_id, evaluator_name="synthetic",
        evaluator_version="1", quality_score=1, acceptable_threshold=0.8,
        acceptable=True, reason="synthetic", evaluation_call_made=True,
    )


def test_semantic_ledger_resumes_after_10_and_never_exceeds_24(tmp_path):
    path = tmp_path / "semantic-attempts.json"
    run_id = UUID(int=14)
    pairs = tuple((f"task-{index // 4}", CANDIDATES[index % 4]) for index in range(24))
    ledger = SemanticAttemptLedger(
        path, experiment_identity="e" * 64, evaluator_identity="v1",
        proposition_identity="p1", judge_identity="j1")
    ledger.initialize(run_id=run_id, pairs=pairs)
    for index, pair in enumerate(pairs[:10]):
        ledger.start(*pair)
        ledger.finish(_semantic_result(run_id, *pair, index))
    restarted = SemanticAttemptLedger(
        path, experiment_identity="e" * 64, evaluator_identity="v1",
        proposition_identity="p1", judge_identity="j1")
    restarted.initialize(run_id=run_id, pairs=pairs)
    assert restarted.counts()["completed"] == 10
    assert restarted.reusable(*pairs[0]) is not None
    for index, pair in enumerate(pairs[10:], start=10):
        restarted.start(*pair)
        restarted.finish(_semantic_result(run_id, *pair, index))
    assert restarted.counts()["completed"] == 24
    with pytest.raises(ValueError, match="exceeds the frozen call budget"):
        SemanticAttemptLedger(
            tmp_path / "too-many.json", experiment_identity="e" * 64,
            evaluator_identity="v1", proposition_identity="p1",
            judge_identity="j1").initialize(
                run_id=run_id, pairs=pairs + (("extra", CANDIDATES[0]),))


def test_semantic_ledger_ambiguous_started_attempt_fails_closed(tmp_path):
    path = tmp_path / "semantic-attempts.json"
    run_id = UUID(int=15)
    pairs = (("task", CANDIDATES[0]),)
    ledger = SemanticAttemptLedger(
        path, experiment_identity="e" * 64, evaluator_identity="v1",
        proposition_identity="p1", judge_identity="j1")
    ledger.initialize(run_id=run_id, pairs=pairs)
    ledger.start(*pairs[0])
    with pytest.raises(ValueError, match="ambiguous started semantic"):
        SemanticAttemptLedger(
            path, experiment_identity="e" * 64, evaluator_identity="v1",
            proposition_identity="p1", judge_identity="j1").initialize(
                run_id=run_id, pairs=pairs)


def test_semantic_credentials_are_checked_before_attempt_ledger_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "adaptive_llm_gateway.evaluation.final_harness.GatewaySettings.from_environment",
        lambda: SimpleNamespace(api_key=None))
    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        asyncio.run(_prepare_semantic_evaluation(
            tmp_path, UUID(int=16), FinalPaths(), verification(),
            DockerPythonSandbox().image))
    assert not (tmp_path / ".final-ledgers").exists()


@pytest.mark.parametrize("sandbox", [
    DockerPythonSandbox(image="python:latest"),
    DockerPythonSandbox(timeout_seconds=4),
    DockerPythonSandbox(memory="128m"),
    DockerPythonSandbox(cpus=1),
    DockerPythonSandbox(pids_limit=64),
])
def test_final_sandbox_rejects_image_and_resource_changes(sandbox):
    with pytest.raises(ValueError, match="exact frozen configuration"):
        validate_frozen_sandbox(sandbox)


def _initialize_git_repository(path, files):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=path, check=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "synthetic"], cwd=path, check=True)


def _authorized_source_identity(path, sources):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True).stdout.strip()
    hashes = {source: digest(path / source) for source in sources}
    return {"implementation": {"git_commit": commit, "source_hashes": hashes,
                                "source_set_sha256": _canonical_digest(hashes)}}


def test_provenance_rejects_dirty_and_untracked_outcome_critical_sources(tmp_path):
    sources = ("critical.py",)
    _initialize_git_repository(tmp_path, {"critical.py": "VALUE = 1\n"})
    authorization = _authorized_source_identity(tmp_path, sources)
    verify_execution_provenance(
        authorization, repository_root=tmp_path, sources=sources)
    (tmp_path / "critical.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        verify_execution_provenance(
            authorization, repository_root=tmp_path, sources=sources)
    subprocess.run(["git", "checkout", "--", "critical.py"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="clean Git worktree"):
        verify_execution_provenance(
            authorization, repository_root=tmp_path,
            sources=("critical.py", "untracked.py"))


def test_source_change_after_authorization_fails_closed(tmp_path):
    sources = ("critical.py",)
    _initialize_git_repository(tmp_path, {"critical.py": "VALUE = 1\n"})
    authorization = _authorized_source_identity(tmp_path, sources)
    (tmp_path / "critical.py").write_text("VALUE = 2\n")
    subprocess.run(["git", "add", "critical.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "changed"], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="differs from authorized"):
        verify_execution_provenance(
            authorization, repository_root=tmp_path, sources=sources)


def test_authorization_only_descendant_commit_preserves_reviewed_source_binding(tmp_path):
    sources = ("critical.py",)
    _initialize_git_repository(tmp_path, {"critical.py": "VALUE = 1\n"})
    authorization = _authorized_source_identity(tmp_path, sources)
    reviewed_commit = authorization["implementation"]["git_commit"]
    (tmp_path / "authorization.json").write_text('{"authorized": true}\n')
    subprocess.run(["git", "add", "authorization.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "authorize"], cwd=tmp_path, check=True)
    execution_commit, _, _ = verify_execution_provenance(
        authorization, repository_root=tmp_path, sources=sources)
    assert execution_commit != reviewed_commit


def test_task_and_feature_binding_detects_task_and_snapshot_tampering():
    task = BenchmarkTask(task_id="synthetic", category="qa", prompt="Original prompt")
    snapshot = extract_request_features(task)
    binding = build_request_feature_binding(task, snapshot)
    verify_request_feature_binding(task, snapshot, binding)
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_request_feature_binding(
            task.model_copy(update={"prompt": "Altered prompt"}), snapshot, binding)
    altered = snapshot.model_copy(update={"prompt_characters": snapshot.prompt_characters + 1})
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_request_feature_binding(task, altered, binding)


def test_replay_rejects_wrong_feature_extractor_identity(tmp_path):
    pipeline = Pipeline((("probability", FixedProbabilityStep([0.9] * 4)),))
    predictor = tmp_path / "predictor.pkl"
    predictor.write_bytes(pickle.dumps(pipeline))
    rows = tuple(observation("task", candidate) for candidate in CANDIDATES)
    bindings = {"task": {"task_id": "task", "feature_extractor_sha256": "wrong",
                         "feature_snapshot_sha256": _canonical_digest(features())}}
    with pytest.raises(ValueError, match="binding mismatch"):
        replay_router(rows, predictor_path=predictor,
            predictor_sha256=digest(predictor), models=models(),
            feature_bindings=bindings)


def test_report_publication_recovers_after_json_write_without_provider_work(tmp_path, monkeypatch):
    rows, selections = complete_fixture()
    result = build_final_results(
        verification=verification(), observations=rows, selections=selections,
        judge_call_count=24, git_commit="b" * 40,
        generated_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    from adaptive_llm_gateway.evaluation import final_harness
    monkeypatch.setattr(final_harness, "VercelGatewayProvider",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("publication must not construct a provider")))
    original = final_harness._atomic_write
    writes = 0

    def crash_on_report(path, content):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("synthetic publication crash")
        original(path, content)

    monkeypatch.setattr(final_harness, "_atomic_write", crash_on_report)
    with pytest.raises(RuntimeError, match="publication crash"):
        write_results(result, tmp_path)
    assert (tmp_path / "final-results.json").is_file()
    assert not (tmp_path / "final-report.md").exists()
    monkeypatch.setattr(final_harness, "_atomic_write", original)
    assert all(path.is_file() for path in recover_results_publication(tmp_path))
    reconstructed = build_final_results(
        verification=verification(), observations=rows, selections=selections,
        judge_call_count=24, git_commit="b" * 40,
        generated_at=datetime(2031, 1, 1, tzinfo=timezone.utc))
    assert all(path.is_file() for path in write_results(reconstructed, tmp_path))
    assert FinalResults.model_validate_json(
        (tmp_path / "final-results.json").read_bytes()).generated_at.year == 2030


def test_dry_run_does_not_construct_provider(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider construction is forbidden in dry-run")
    monkeypatch.setattr(
        "adaptive_llm_gateway.evaluation.final_harness.VercelGatewayProvider", forbidden)
    monkeypatch.setattr(sys, "argv", ["final_harness", "semantic-judge",
        "--root", "unused", "--run-id", "00000000-0000-0000-0000-000000000001",
        "--dry-run"])
    main()
    assert json.loads(capsys.readouterr().out)["provider_calls_made"] == 0


def test_full_synthetic_final_benchmark_cli_path_uses_only_fake_provider(
        tmp_path, monkeypatch, capsys):
    from adaptive_llm_gateway.application.service import InferenceService
    from adaptive_llm_gateway.benchmarks import __main__ as benchmark_cli
    from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset
    from adaptive_llm_gateway.providers.fake import FakeProvider
    from adaptive_llm_gateway.providers.resolver import ProviderResolver
    from adaptive_llm_gateway.registry import ModelRegistry

    dataset_path = tmp_path / "synthetic-dataset.json"
    dataset = BenchmarkDataset(
        name="synthetic-final-cli", version="1.0.0",
        tasks=tuple(BenchmarkTask(
            task_id=f"task-{index}", category="qa", prompt=f"Synthetic prompt {index}")
            for index in range(42)))
    dataset_path.write_text(dataset.model_dump_json(indent=2))
    protocol_path = tmp_path / "protocol.json"
    split_path = tmp_path / "split.json"
    write_json(protocol_path, {"synthetic": True})
    write_json(split_path, {"synthetic": True})
    output = tmp_path / "runs"
    fake_models = models()

    registry = ModelRegistry()
    for item in fake_models:
        registry.register(item)
    resolver = ProviderResolver()
    resolver.register("fake", FakeProvider)
    service = InferenceService(registry, resolver)
    dataset_sha = digest(dataset_path)
    identities = {
        "dataset_sha256": dataset_sha,
        "split_manifest_sha256": digest(split_path),
        "protocol_sha256": digest(protocol_path),
        "predictor_sha256": "p" * 64,
    }
    verified = FreezeVerification(
        status="READY", experiment_identity="e" * 64, identities=identities,
        policy_sha256="q" * 64,
        authorization_status="AUTHORIZED_FOR_FINAL_EXECUTION",
        explicit_authorization=True, duplicate_completed_run=False,
        git_commit="b" * 40)

    monkeypatch.setattr(benchmark_cli, "create_development_service", lambda: service)
    monkeypatch.setattr(benchmark_cli, "configure_gateway", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_cli, "load_execution_protocol", lambda _path: ({
        "protocol": "routellm-routing-benchmark-v1.2", "version": "1.7.0",
        "dataset_sha256": dataset_sha, "candidate_task_max_output_tokens": {},
    }, fake_models, digest(protocol_path)))
    monkeypatch.setattr(benchmark_cli, "load_pricing_readiness",
                        lambda *_args, **_kwargs: SimpleNamespace(version="synthetic"))
    monkeypatch.setattr(
        "adaptive_llm_gateway.evaluation.final_harness.verify_final_freeze",
        lambda **_kwargs: verified)
    readiness = Path("benchmarks/protocols/routing-benchmark-v1.2/execution-readiness-1.7.json")
    monkeypatch.setattr(sys, "argv", ["benchmarks", "--dataset", str(dataset_path),
        "--models", *CANDIDATES, "--limit", "42", "--output", str(output),
        "--protocol", str(protocol_path), "--pricing-readiness", str(readiness),
        "--split", "final", "--split-manifest", str(split_path),
        "--allow-final-evaluation", "--authorize-final",
        "--final-results-root", str(output)])
    benchmark_cli.main()
    run_directories = [path for path in output.iterdir()
                       if path.is_dir() and path.name != ".final-ledgers"]
    assert len(run_directories) == 1
    assert len(tuple((run_directories[0] / "results").glob("*.json"))) == 168
    ledger = CandidateAttemptLedger(
        output / ".final-ledgers" / verified.experiment_identity
        / "candidate-attempts.json", experiment_identity=verified.experiment_identity)
    assert ledger.counts()["completed"] == 168
    assert "Benchmark run:" in capsys.readouterr().out


def test_semantic_workflow_end_to_end_persists_each_call_and_reuses_it(tmp_path):
    from adaptive_llm_gateway.benchmarks.models import BenchmarkDataset, BenchmarkRun
    from adaptive_llm_gateway.benchmarks.summarization_spec import (
        PropositionSpecification, RationalWeight, SemanticProposition,
        SourceEvidence, SummarizationTaskSpecification,
    )
    from adaptive_llm_gateway.evaluation.hybrid_judge import FakeHybridJudge
    from adaptive_llm_gateway.evaluation.service import EvaluationService
    from adaptive_llm_gateway.models import InferenceResponse

    task = BenchmarkTask(
        task_id="synthetic-summary", category="summarization",
        prompt="Synthetic source fact.", acceptable_threshold=0.8,
        evaluation_metadata={
            "semantic_requirements": ["Synthetic source fact."],
            "semantic_specification_id": "synthetic-spec",
            "deterministic_constraints": {"max_words": 10},
        })
    candidate = model("synthetic-candidate", "1", "1")
    run = BenchmarkRun(
        dataset=BenchmarkDataset(name="synthetic", version="1", tasks=(task,)),
        dataset_sha256="synthetic", selected_task_ids=(task.task_id,),
        models=(candidate,), configuration={})
    result = BenchmarkResult(
        run_id=run.run_id, request_id="synthetic-request", task_id=task.task_id,
        model_id=candidate.model_id, success=True, latency_ms=1,
        response=InferenceResponse(
            text="Synthetic source fact.", model_id=candidate.model_id,
            provider="fake", input_tokens=3, output_tokens=3, latency_ms=1,
            estimated_cost_usd=0))
    directory = tmp_path / str(run.run_id)
    (directory / "results").mkdir(parents=True)
    write_json(directory / "status.json", {"status": "completed"})
    (directory / "manifest.json").write_text(run.model_dump_json())
    (directory / "results" / f"{result.result_id}.json").write_text(
        result.model_dump_json())
    specification = PropositionSpecification(
        source_benchmarks=("synthetic",),
        tasks=(SummarizationTaskSpecification(
            specification_id="synthetic-spec", canonical_task_id=task.task_id,
            applies_to=(task.task_id,), source_text=task.prompt,
            legacy_requirements=("Synthetic source fact.",),
            propositions=(SemanticProposition(
                proposition_id="synthetic-proposition", legacy_requirement_index=1,
                description="The fact is stated.", requirement_type="atomic",
                weight=RationalWeight(numerator=1, denominator=1),
                source_evidence=SourceEvidence(quote="Synthetic source fact.")),)),))
    ledger = SemanticAttemptLedger(
        tmp_path / ".final-ledgers" / "semantic.json",
        experiment_identity="e" * 64, evaluator_identity="v1",
        proposition_identity="p1", judge_identity="j1")
    ledger.initialize(run_id=run.run_id, pairs=((task.task_id, candidate.model_id),))
    judge = FakeHybridJudge()
    service = EvaluationService(
        tmp_path, hybrid_semantic_judge=judge,
        proposition_specification=specification)
    first = asyncio.run(_semantic_evaluate(service, ledger, run.run_id))
    second = asyncio.run(_semantic_evaluate(service, ledger, run.run_id))
    assert first.judge_calls == second.judge_calls == 1
    assert len(judge.requests) == 1
    assert ledger.counts()["completed"] == 1
    assert len(tuple((directory / "evaluations").glob("*.json"))) == 1
