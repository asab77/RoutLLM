import json
import hashlib
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from adaptive_llm_gateway.benchmarks.models import (
    BenchmarkDataset, BenchmarkResult, BenchmarkRun, BenchmarkTask, load_dataset)
from adaptive_llm_gateway.benchmarks.summarization_spec import (
    EntailmentVerdict, MaterialErrorCategory, MaterialErrorFinding,
    PropositionSpecification, SourceEvidence)
from adaptive_llm_gateway.evaluation.hybrid_judge import (
    HYBRID_SEMANTIC_EVALUATOR_VERSION, FakeHybridJudge, HybridJudgeRequest,
    HybridSemanticRubric, PropositionJudgment, VercelHybridSemanticJudge,
    validate_rubric)
from adaptive_llm_gateway.evaluation.hybrid_replay import prepare_historical_replay
from adaptive_llm_gateway.evaluation.hybrid_validation import (
    HybridValidationRunner, load_validation_cases)
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.models import InferenceResponse, ModelConfig
from adaptive_llm_gateway.errors import ProviderFailureError

SPEC_PATH = Path("benchmarks/specifications/summarization-propositions-v1.0.0.json")
DATASET_PATH = Path("benchmarks/datasets/routing-benchmark-v1.2.json")
VALIDATION_INPUT = Path("benchmarks/validation/summarization-proposition-validation-v1.0.0.json")
VALIDATION_LABELS = Path("benchmarks/validation/summarization-proposition-validation-v1.0.0-labels.json")


def specification():
    return PropositionSpecification.model_validate_json(SPEC_PATH.read_bytes())


def task_spec(specification_id="rb12-summarization-medium-011"):
    return next(item for item in specification().tasks
                if item.specification_id == specification_id)


def request(specification_id="rb12-summarization-medium-011"):
    return HybridJudgeRequest.from_specification(task_spec(specification_id), "Candidate summary.")


def rubric_for(request, verdicts=None, findings=()):
    verdicts = verdicts or {}
    return HybridSemanticRubric(proposition_judgments=tuple(PropositionJudgment(
        proposition_id=item.proposition_id,
        verdict=verdicts.get(item.proposition_id, EntailmentVerdict.ENTAILED),
        source_evidence_reference=item.source_evidence,
        equivalence_or_derivation_used=None,
        reason="Narrow proposition verdict.") for item in request.required_propositions),
        material_error_findings=findings)


class RecordingProvider:
    def __init__(self, text="", error=None):
        self.text, self.error, self.requests = text, error, []

    async def generate(self, inference_request):
        self.requests.append(inference_request)
        if self.error:
            raise self.error
        return InferenceResponse(text=self.text, model_id="astra", provider="vercel",
            input_tokens=20, output_tokens=10, latency_ms=12,
            estimated_cost_usd=Decimal("0.0007"))


def write_run(root, task_id="summarization-medium-011", candidate="Candidate summary."):
    source = next(item for item in load_dataset(DATASET_PATH).tasks if item.task_id == task_id)
    model = ModelConfig(model_id="candidate", provider="fake", provider_model_name="fake",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="1", context_window=1000)
    run = BenchmarkRun(dataset=BenchmarkDataset(name="routing-fixture", version="1.2.0", tasks=(source,)),
        dataset_sha256="fixture", selected_task_ids=(task_id,), models=(model,), configuration={})
    result = BenchmarkResult(run_id=run.run_id, request_id=str(uuid4()), task_id=task_id,
        model_id=model.model_id, success=True, latency_ms=1,
        response=InferenceResponse(text=candidate, model_id=model.model_id, provider="fake",
            input_tokens=1, output_tokens=1, latency_ms=1, estimated_cost_usd=0))
    directory = root / str(run.run_id)
    (directory / "results").mkdir(parents=True)
    (directory / "manifest.json").write_text(run.model_dump_json())
    (directory / "status.json").write_text('{"status":"completed"}')
    (directory / "results" / f"{result.result_id}.json").write_text(result.model_dump_json())
    return run


def write_foundation_run(root, candidate="- Candidate summary."):
    task_id = "summarization-hard-03"
    source = next(item for item in load_dataset(
        Path("benchmarks/datasets/foundation-v2.json")).tasks if item.task_id == task_id)
    model = ModelConfig(model_id="candidate", provider="fake", provider_model_name="fake",
        input_cost_per_1m_tokens="1", output_cost_per_1m_tokens="1", context_window=1000)
    run = BenchmarkRun(dataset=BenchmarkDataset(name="foundation-fixture", version="2.0.0", tasks=(source,)),
        dataset_sha256="fixture", selected_task_ids=(task_id,), models=(model,), configuration={})
    result = BenchmarkResult(run_id=run.run_id, request_id=str(uuid4()), task_id=task_id,
        model_id=model.model_id, success=True, latency_ms=1,
        response=InferenceResponse(text=candidate, model_id=model.model_id, provider="fake",
            input_tokens=1, output_tokens=1, latency_ms=1, estimated_cost_usd=0))
    directory = root / str(run.run_id)
    (directory / "results").mkdir(parents=True)
    (directory / "manifest.json").write_text(run.model_dump_json())
    (directory / "status.json").write_text('{"status":"completed"}')
    (directory / "results" / f"{result.result_id}.json").write_text(result.model_dump_json())
    return run


def test_blind_request_and_output_schema_exclude_holistic_scores_and_identity():
    req = request()
    assert set(req.model_dump()) == {"source_material", "candidate_summary",
        "required_propositions", "material_error_contract"}
    payload = req.model_dump_json().casefold()
    for forbidden in ("model_id", "provider", "price", "difficulty",
                      "expected_verdict", "routing_outcome", "candidate_ranking"):
        assert forbidden not in payload
    assert set(HybridSemanticRubric.model_fields) == {
        "proposition_judgments", "material_error_findings"}
    assert not ({"coverage", "quality_score", "acceptable", "pass"}
                & set(HybridSemanticRubric.model_fields))


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate"])
def test_expected_proposition_ids_are_exactly_once(mutation):
    req = request()
    judgments = list(rubric_for(req).proposition_judgments)
    if mutation == "missing":
        judgments.pop()
    elif mutation == "unknown":
        judgments[-1] = judgments[-1].model_copy(update={"proposition_id": "unknown"})
    else:
        judgments[-1] = judgments[0].model_copy()
    if mutation == "duplicate":
        with pytest.raises(ValueError):
            HybridSemanticRubric(proposition_judgments=judgments)
    else:
        assert validate_rubric(req, HybridSemanticRubric(
            proposition_judgments=judgments)) == "proposition_id_mismatch"


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "category"), [
    ("not json", "malformed_json"),
    ('{"proposition_judgments":[],"material_error_findings":[]}', "invalid_schema"),
    ('{"proposition_judgments":[{"proposition_id":"unknown","verdict":"ENTAILED",'
     '"source_evidence_reference":"source","equivalence_or_derivation_used":null,'
     '"reason":"direct"}],"material_error_findings":[]}', "proposition_id_mismatch"),
])
async def test_provider_adapter_rejects_malformed_or_incomplete_judgments(text, category):
    provider = RecordingProvider(text)
    outcome = await VercelHybridSemanticJudge(
        provider, provider_name="vercel", model_id="astra").judge(request())
    assert outcome.status == "infrastructure_failure"
    assert outcome.error_category == category and outcome.rubric is None
    assert len(provider.requests) == 1 and outcome.usage is not None


@pytest.mark.asyncio
async def test_provider_failure_is_distinct_and_makes_one_attempt():
    provider = RecordingProvider(error=ProviderFailureError("upstream unavailable"))
    outcome = await VercelHybridSemanticJudge(
        provider, provider_name="vercel", model_id="astra").judge(request())
    assert outcome.status == "infrastructure_failure"
    assert outcome.error_category == "provider_error" and outcome.usage is None
    assert len(provider.requests) == 1


def test_allowed_disallowed_and_conditional_derivations_are_enforced():
    spec = specification()
    allowed_task = next(item for item in spec.tasks
        if any(rule.policy == "ALLOWED" for prop in item.propositions for rule in prop.derivations))
    allowed_req = HybridJudgeRequest.from_specification(allowed_task, "candidate")
    allowed_rubric = rubric_for(allowed_req)
    judgments = list(allowed_rubric.proposition_judgments)
    prop = next(prop for prop in allowed_task.propositions if prop.derivations)
    rule = next(rule for rule in prop.derivations if rule.policy == "ALLOWED")
    index = next(i for i, item in enumerate(judgments) if item.proposition_id == prop.proposition_id)
    judgments[index] = judgments[index].model_copy(update={
        "equivalence_or_derivation_used": rule.derivation_id})
    assert validate_rubric(allowed_req, HybridSemanticRubric(
        proposition_judgments=judgments)) is None

    disallowed_task = next(item for item in spec.tasks
        if any(rule.policy == "NOT_ALLOWED" for prop in item.propositions for rule in prop.derivations))
    disallowed_req = HybridJudgeRequest.from_specification(disallowed_task, "candidate")
    judgments = list(rubric_for(disallowed_req).proposition_judgments)
    prop = next(prop for prop in disallowed_task.propositions
                if any(rule.policy == "NOT_ALLOWED" for rule in prop.derivations))
    rule = next(rule for rule in prop.derivations if rule.policy == "NOT_ALLOWED")
    index = next(i for i, item in enumerate(judgments) if item.proposition_id == prop.proposition_id)
    judgments[index] = judgments[index].model_copy(update={
        "equivalence_or_derivation_used": rule.derivation_id})
    assert validate_rubric(disallowed_req, HybridSemanticRubric(
        proposition_judgments=judgments)) == "invalid_rule_use"

    conditional_task = next(item for item in spec.tasks
        if any(rule.policy == "CONDITIONAL" for prop in item.propositions for rule in prop.derivations))
    conditional_req = HybridJudgeRequest.from_specification(conditional_task, "candidate")
    prop = next(prop for prop in conditional_task.propositions
                if any(rule.policy == "CONDITIONAL" for rule in prop.derivations))
    rule = next(rule for rule in prop.derivations if rule.policy == "CONDITIONAL")
    judgments = list(rubric_for(conditional_req).proposition_judgments)
    for index, judgment in enumerate(judgments):
        if judgment.proposition_id in rule.requires_proposition_ids:
            judgments[index] = judgment.model_copy(update={"verdict": "NOT_ENTAILED"})
        if judgment.proposition_id == prop.proposition_id:
            judgments[index] = judgment.model_copy(update={
                "equivalence_or_derivation_used": rule.derivation_id})
    assert validate_rubric(conditional_req, HybridSemanticRubric(
        proposition_judgments=judgments)) == "invalid_rule_use"


@pytest.mark.asyncio
@pytest.mark.parametrize(("entailed_count", "acceptable", "failure_type"), [
    (5, True, "COMPLETE_SEMANTIC_RESULT"),
    (4, False, "COVERAGE_FAILURE"),
    (3, False, "COVERAGE_FAILURE"),
])
async def test_exact_weighted_boundary_and_partial_coverage(
        tmp_path, entailed_count, acceptable, failure_type):
    req = request()
    verdicts = {item.proposition_id: ("ENTAILED" if index < entailed_count else "NOT_ENTAILED")
                for index, item in enumerate(req.required_propositions)}
    run = write_run(tmp_path)
    summary = await EvaluationService(tmp_path, hybrid_semantic_judge=FakeHybridJudge(
        rubric_for(req, verdicts)), proposition_specification=specification()).evaluate(run.run_id)
    evaluation = json.loads(next((tmp_path / str(run.run_id) / "evaluations").glob("*.json")).read_text())
    assert bool(summary.overall.acceptable_responses) is acceptable
    assert evaluation["details"]["failure_type"] == failure_type
    if entailed_count == 4:
        assert evaluation["details"]["coverage_fraction"] == {"numerator": 2, "denominator": 3}


def test_exact_four_fifths_boundary_uses_rational_arithmetic():
    from adaptive_llm_gateway.benchmarks.summarization_spec import weighted_coverage
    task = task_spec("foundation-shared-summarization-hard-03")
    verdicts = {item.proposition_id: (EntailmentVerdict.NOT_ENTAILED if index == 0
                else EntailmentVerdict.ENTAILED)
                for index, item in enumerate(task.propositions)}
    coverage = weighted_coverage(task, verdicts)
    assert (coverage.coverage_numerator, coverage.coverage_denominator) == (4, 5)
    assert coverage.coverage_numerator * 5 == coverage.coverage_denominator * 4


@pytest.mark.asyncio
async def test_foundation_historical_threshold_of_one_is_preserved(tmp_path):
    req = request("foundation-shared-summarization-hard-03")
    verdicts = {item.proposition_id: ("NOT_ENTAILED" if index == 0 else "ENTAILED")
                for index, item in enumerate(req.required_propositions)}
    run = write_foundation_run(tmp_path)
    summary = await EvaluationService(tmp_path, hybrid_semantic_judge=FakeHybridJudge(
        rubric_for(req, verdicts)), proposition_specification=specification()).evaluate(run.run_id)
    assert summary.overall.acceptable_responses == 0
    evaluation = json.loads(next((tmp_path / str(run.run_id) / "evaluations").glob("*.json")).read_text())
    assert evaluation["acceptable_threshold"] == 1


@pytest.mark.asyncio
async def test_material_veto_ambiguity_and_deterministic_precedence(tmp_path):
    req = request()
    finding = MaterialErrorFinding(candidate_claim="Invented cause",
        source_evidence=SourceEvidence(quote="Source has no cause."),
        category=MaterialErrorCategory.UNSUPPORTED_CAUSATION,
        materiality="material", reason="The candidate invents a causal relation.")
    with pytest.raises(ValueError):
        HybridSemanticRubric(proposition_judgments=rubric_for(req).proposition_judgments,
                             material_error_findings=(finding, finding))
    run = write_run(tmp_path / "material")
    await EvaluationService(tmp_path / "material", hybrid_semantic_judge=FakeHybridJudge(
        rubric_for(req, findings=(finding,))), proposition_specification=specification()).evaluate(run.run_id)
    result = json.loads(next((tmp_path / "material" / str(run.run_id) / "evaluations").glob("*.json")).read_text())
    assert result["details"]["failure_type"] == "MATERIAL_SEMANTIC_FAILURE"

    ambiguous = {req.required_propositions[0].proposition_id: "AMBIGUOUS"}
    run = write_run(tmp_path / "ambiguous")
    summary = await EvaluationService(tmp_path / "ambiguous", hybrid_semantic_judge=FakeHybridJudge(
        rubric_for(req, ambiguous)), proposition_specification=specification()).evaluate(run.run_id)
    assert summary.overall.incomplete_evaluations == 1

    borderline = finding.model_copy(update={"materiality": "borderline"})
    run = write_run(tmp_path / "interpretive")
    summary = await EvaluationService(tmp_path / "interpretive",
        hybrid_semantic_judge=FakeHybridJudge(rubric_for(req, findings=(borderline,))),
        proposition_specification=specification()).evaluate(run.run_id)
    assert summary.overall.incomplete_evaluations == 1

    run = write_run(tmp_path / "deterministic", candidate="word " * 100)
    await EvaluationService(tmp_path / "deterministic", hybrid_semantic_judge=FakeHybridJudge(
        rubric_for(req, findings=(finding,))), proposition_specification=specification()).evaluate(run.run_id)
    result = json.loads(next((tmp_path / "deterministic" / str(run.run_id) / "evaluations").glob("*.json")).read_text())
    assert result["details"]["failure_type"] == "DETERMINISTIC_TASK_FAILURE"


@pytest.mark.local_evidence
def test_historical_replay_prepares_69_blind_requests_without_execution():
    replay = prepare_historical_replay(specification())
    assert len(replay) == 69
    assert sum(len(item.blind_judge_request.required_propositions) for item in replay) == 254
    for item in replay:
        blind = item.blind_judge_request.model_dump_json().casefold()
        assert item.model_id.casefold() not in blind


@pytest.mark.asyncio
async def test_all_24_frozen_validation_cases_run_locally_and_remain_blind():
    inputs, labels = load_validation_cases(VALIDATION_INPUT, VALIDATION_LABELS)

    class LabelFixtureJudge(FakeHybridJudge):
        async def judge(self, req):
            expected = labels[req.required_propositions[0].proposition_id].expected_verdict
            self.rubric = rubric_for(req, {req.required_propositions[0].proposition_id: expected})
            return await super().judge(req)

    judge = LabelFixtureJudge()
    report = await HybridValidationRunner(judge).run(inputs, labels)
    assert report.total_cases == 24 and report.exact_agreement == 1
    assert report.structured_output_validity == 1
    assert report.malformed_rate == report.provider_failure_rate == 0
    assert set(report.accuracy_by_label) == {"ENTAILED", "NOT_ENTAILED", "AMBIGUOUS"}
    for req in judge.requests:
        serialized = req.model_dump_json().casefold()
        assert "expected_verdict" not in serialized


def test_live_validation_requires_explicit_paid_authorization():
    class NonFake:
        async def judge(self, request):  # pragma: no cover - authorization blocks first
            raise AssertionError("must not be called")
    inputs, labels = load_validation_cases(VALIDATION_INPUT, VALIDATION_LABELS)
    with pytest.raises(PermissionError):
        import asyncio
        asyncio.run(HybridValidationRunner(NonFake()).run(inputs, labels))


def test_proposed_manifest_identities_and_protocol_are_deterministic():
    root = Path("benchmarks/protocols/routing-benchmark-v1.2")
    identities = json.loads((root / "identities.json").read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert identities["evaluator_manifest_sha256"] == digest(root / "evaluator-manifest.json")
    assert identities["protocol_sha256"] == digest(root / "protocol.json")
    assert identities["cost_estimate_sha256"] == digest(root / "cost-estimate.json")
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["version"] == "1.4.0"
    assert protocol["status"] == "PROPOSED_AWAITING_LIVE_VALIDATION"
    assert protocol["paid_execution_authorized"] is False
    assert protocol["dataset_sha256"] == "1a9cadc2d557eacb4a7dce450100cdb495a2659c2862845ce9894cf0a7791005"
    assert protocol["split_manifest_sha256"] == "98c639be29da4e11fdf48073d74e83805f16fdfb72b0102b5f5543213ab4960b"


def test_cost_plan_counts_one_judge_call_per_summary():
    plan = json.loads(Path(
        "benchmarks/protocols/routing-benchmark-v1.2/cost-estimate.json").read_text())
    assert plan["validation"]["calls"] == 24
    assert plan["corrected_pilot"]["calls"] == 12
    assert plan["full_benchmark"]["calls"] == 128
