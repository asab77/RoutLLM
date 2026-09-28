from pathlib import Path
from fractions import Fraction
from uuid import UUID

from adaptive_llm_gateway.errors import EvaluationArtifactError

from .aggregation import aggregate
from .evaluators import evaluator_for
from .judge import SEMANTIC_JUDGE_VERSION, SemanticJudge, SemanticJudgeRequest
from .hybrid_judge import (
    HYBRID_SEMANTIC_EVALUATOR_VERSION,
    HybridJudgeRequest,
    HybridSemanticJudge,
)
from .models import EvaluationResult, EvaluationSummary
from .normalization import strip_code_fence
from .repository import FileEvaluationRepository
from .sandbox import FunctionalEvaluationRequest, FunctionalSandbox, FunctionalTestCase


class EvaluationService:
    def __init__(self, root: Path = Path("benchmark-results"), *,
                 functional_sandbox: FunctionalSandbox | None = None,
                 semantic_judge: SemanticJudge | None = None,
                 hybrid_semantic_judge: HybridSemanticJudge | None = None,
                 proposition_specification=None) -> None:
        self.repository = FileEvaluationRepository(root)
        self.functional_sandbox = functional_sandbox
        self.semantic_judge = semantic_judge
        self.hybrid_semantic_judge = hybrid_semantic_judge
        self.proposition_specification = proposition_specification

    @property
    def configuration(self) -> dict:
        configuration = {"functional_sandbox": None, "semantic_judge": None,
                         "hybrid_semantic_judge": None}
        if self.functional_sandbox is not None:
            configuration["functional_sandbox"] = getattr(
                self.functional_sandbox, "configuration", {"type": type(self.functional_sandbox).__name__})
        if self.semantic_judge is not None:
            configuration["semantic_judge"] = getattr(
                self.semantic_judge, "configuration", {"type": type(self.semantic_judge).__name__})
        if self.hybrid_semantic_judge is not None:
            configuration["hybrid_semantic_judge"] = getattr(
                self.hybrid_semantic_judge, "configuration",
                {"type": type(self.hybrid_semantic_judge).__name__})
        return configuration

    async def evaluate(self, run_id: UUID) -> EvaluationSummary:
        run, results = await self.repository.load_run(run_id)
        task_by_id = {task.task_id: task for task in run.dataset.tasks}
        evaluations: list[EvaluationResult] = []
        for result in results:
            try:
                task = task_by_id[result.task_id]
            except KeyError:
                raise EvaluationArtifactError(
                    f"Result references unknown task {result.task_id!r}") from None
            try:
                evaluation = evaluator_for(task).evaluate(task, result)
                if (evaluation.evaluation_status == "requires_functional_execution"
                        and self.functional_sandbox is not None and result.response is not None):
                    evaluation = await self._evaluate_functionally(task, result, evaluation)
                if (evaluation.evaluation_status == "requires_semantic_judge"
                        and result.response is not None):
                    if self.hybrid_semantic_judge is not None:
                        evaluation = await self._evaluate_hybrid_semantically(
                            task, result, evaluation)
                    elif self.semantic_judge is not None:
                        evaluation = await self._evaluate_semantically(task, result, evaluation)
                evaluations.append(evaluation)
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationArtifactError(
                    f"Task {task.task_id!r} has invalid evaluation metadata") from exc
        try:
            summary = aggregate(run, results, evaluations,
                                evaluation_configuration=self.configuration)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationArtifactError("Benchmark evaluations could not be aggregated") from exc
        await self.repository.save(run_id, evaluations, summary)
        return summary

    async def summary(self, run_id: UUID) -> EvaluationSummary:
        return await self.repository.load_summary(run_id)

    async def _evaluate_functionally(self, task, result, static: EvaluationResult) -> EvaluationResult:
        metadata = task.evaluation_metadata
        tests = metadata.get("functional_tests")
        if not isinstance(tests, list):
            raise ValueError("Functional tests must be a list")
        request = FunctionalEvaluationRequest(
            source=strip_code_fence(result.response.text),
            function_name=metadata["function_name"], parameter_count=metadata["parameters"],
            tests=tuple(FunctionalTestCase.model_validate(case) for case in tests),
            preserve_inputs="mutat" in task.prompt.casefold(),
        )
        outcome = await self.functional_sandbox.evaluate(request)
        details = {"static_validation": static.model_dump(mode="json"),
                   "functional_execution": outcome.model_dump(mode="json")}
        if outcome.execution_status == "infrastructure_failure":
            return static.model_copy(update={
                "reason": "Functional evaluation infrastructure failed; candidate quality was not scored.",
                "details": details,
            })
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name="docker_python_functional",
            evaluator_version=outcome.evaluator_version,
            quality_score=outcome.quality_score, acceptable_threshold=1,
            acceptable=outcome.acceptable,
            reason=(f"Passed {outcome.tests_passed} of {outcome.total_tests} hidden functional tests."
                    if outcome.execution_status == "passed" else
                    f"Functional evaluation failed: {outcome.error_category}."),
            component_scores={"functional_tests": outcome.quality_score or 0},
            details=details,
        )

    async def _evaluate_semantically(self, task, result,
                                     deterministic: EvaluationResult) -> EvaluationResult:
        metadata = task.evaluation_metadata
        request = SemanticJudgeRequest(
            source_text=task.prompt, candidate_summary=result.response.text,
            semantic_requirements=tuple(metadata["semantic_requirements"]),
            output_constraints=metadata.get("deterministic_constraints", {}),
        )
        outcome = await self.semantic_judge.judge(request)
        usage = outcome.usage
        usage_fields = {
            "evaluation_call_made": True,
            "evaluation_input_tokens": usage.input_tokens if usage else 0,
            "evaluation_output_tokens": usage.output_tokens if usage else 0,
            "evaluation_latency_ms": usage.latency_ms if usage else 0,
            "evaluation_cost_usd": usage.estimated_cost_usd if usage else 0,
        }
        details = {"deterministic": deterministic.model_dump(mode="json"),
                   "semantic_judge": outcome.model_dump(mode="json")}
        if outcome.status == "infrastructure_failure" or outcome.rubric is None:
            return deterministic.model_copy(update={
                "reason": "Semantic judge infrastructure failed; candidate quality was not scored.",
                "details": details, **usage_fields,
            })
        rubric = outcome.rubric
        threshold = task.acceptable_threshold if task.acceptable_threshold is not None else 0.8
        deterministic_passed = all(score == 1 for score in deterministic.component_scores.values())
        quality = (min(rubric.required_fact_coverage, rubric.instruction_compliance)
                   if deterministic_passed and not rubric.material_errors else 0.0)
        components = {**{f"deterministic_{key}": value
                         for key, value in deterministic.component_scores.items()},
                      "required_fact_coverage": rubric.required_fact_coverage,
                      "instruction_compliance": rubric.instruction_compliance,
                      "no_unsupported_claims": float(not rubric.unsupported_claims),
                      "no_contradictions": float(not rubric.contradictions),
                      "no_causal_claim_errors": float(not rubric.causal_claim_errors),
                      "no_attribution_errors": float(not rubric.attribution_errors),
                      "no_quantity_or_time_errors": float(not rubric.quantity_or_time_errors),
                      "no_certainty_distortions": float(not rubric.certainty_distortions)}
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name="semantic_summary_judge",
            evaluator_version=SEMANTIC_JUDGE_VERSION,
            quality_score=quality, acceptable_threshold=threshold,
            acceptable=quality >= threshold,
            reason=rubric.reason, component_scores=components, details=details,
            **usage_fields,
        )

    async def _evaluate_hybrid_semantically(self, task, result,
                                            deterministic: EvaluationResult) -> EvaluationResult:
        if self.proposition_specification is None:
            raise ValueError("Hybrid semantic evaluation requires a proposition specification")
        specification_id = task.evaluation_metadata.get("semantic_specification_id")
        specifications = {item.specification_id: item
                          for item in self.proposition_specification.tasks}
        if isinstance(specification_id, str) and specification_id in specifications:
            task_specification = specifications[specification_id]
        else:
            historical_matches = [item for item in self.proposition_specification.tasks
                                  if item.canonical_task_id == task.task_id]
            if len(historical_matches) != 1:
                raise ValueError(
                    "Task does not reference a known semantic proposition specification")
            task_specification = historical_matches[0]
        request = HybridJudgeRequest.from_specification(
            task_specification, result.response.text).model_copy(update={
                "material_error_contract": self.proposition_specification.material_error_contract})
        outcome = await self.hybrid_semantic_judge.judge(request)
        usage = outcome.usage
        usage_fields = {
            "evaluation_call_made": True,
            "evaluation_input_tokens": usage.input_tokens if usage else 0,
            "evaluation_output_tokens": usage.output_tokens if usage else 0,
            "evaluation_latency_ms": usage.latency_ms if usage else 0,
            "evaluation_cost_usd": usage.estimated_cost_usd if usage else 0,
        }
        details = {"failure_type": None,
                   "deterministic": deterministic.model_dump(mode="json"),
                   "semantic_judge": outcome.model_dump(mode="json")}
        if outcome.status == "infrastructure_failure" or outcome.rubric is None:
            failure_type = ("PROVIDER_FAILURE" if outcome.error_category in {
                "provider_error", "timeout"} else "MALFORMED_JUDGMENT")
            details["failure_type"] = failure_type
            return deterministic.model_copy(update={
                "evaluator_name": "hybrid_proposition_semantic_summary",
                "evaluator_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
                "evaluation_status": "partial", "quality_score": None,
                "acceptable": None,
                "reason": f"{failure_type}: semantic quality was not scored.",
                "details": details, **usage_fields,
            })

        from adaptive_llm_gateway.benchmarks.summarization_spec import (
            EntailmentVerdict, weighted_coverage)

        rubric = outcome.rubric
        verdicts = {item.proposition_id: item.verdict
                    for item in rubric.proposition_judgments}
        coverage = weighted_coverage(task_specification, verdicts)
        borderline = tuple(item for item in rubric.material_error_findings
                           if item.materiality == "borderline")
        if coverage.status == "incomplete" or borderline:
            details["failure_type"] = "AMBIGUOUS_SEMANTIC_RESULT"
            details["ambiguous_proposition_ids"] = list(coverage.ambiguous_proposition_ids)
            details["borderline_material_findings"] = [
                item.model_dump(mode="json") for item in borderline]
            return deterministic.model_copy(update={
                "evaluator_name": "hybrid_proposition_semantic_summary",
                "evaluator_version": HYBRID_SEMANTIC_EVALUATOR_VERSION,
                "evaluation_status": "partial", "quality_score": None,
                "acceptable": None,
                "reason": "AMBIGUOUS_SEMANTIC_RESULT: human review is required.",
                "details": details, **usage_fields,
            })

        assert coverage.coverage_numerator is not None
        assert coverage.coverage_denominator is not None
        coverage_score = coverage.coverage_numerator / coverage.coverage_denominator
        deterministic_passed = all(
            score == 1 for score in deterministic.component_scores.values())
        material_findings = tuple(item for item in rubric.material_error_findings
                                  if item.materiality == "material")
        threshold = task.acceptable_threshold if task.acceptable_threshold is not None else 0.8
        if not deterministic_passed:
            quality, failure_type = 0.0, "DETERMINISTIC_TASK_FAILURE"
        elif material_findings:
            quality, failure_type = 0.0, "MATERIAL_SEMANTIC_FAILURE"
        elif Fraction(coverage.coverage_numerator, coverage.coverage_denominator) < Fraction(str(threshold)):
            quality, failure_type = coverage_score, "COVERAGE_FAILURE"
        else:
            quality, failure_type = coverage_score, "COMPLETE_SEMANTIC_RESULT"
        details["failure_type"] = failure_type
        details["coverage_fraction"] = {
            "numerator": coverage.coverage_numerator,
            "denominator": coverage.coverage_denominator,
        }
        details["material_error_veto_count"] = len(material_findings)
        components = {
            **{f"deterministic_{key}": value
               for key, value in deterministic.component_scores.items()},
            "weighted_proposition_coverage": coverage_score,
            "no_material_semantic_errors": float(not material_findings),
        }
        return EvaluationResult(
            benchmark_result_id=result.result_id, run_id=result.run_id,
            task_id=task.task_id, model_id=result.model_id,
            evaluator_name="hybrid_proposition_semantic_summary",
            evaluator_version=HYBRID_SEMANTIC_EVALUATOR_VERSION,
            quality_score=quality, acceptable_threshold=threshold,
            acceptable=quality >= threshold,
            reason=f"{failure_type}: weighted semantic coverage is "
                   f"{coverage.coverage_numerator}/{coverage.coverage_denominator}.",
            component_scores=components, details=details, **usage_fields,
        )
