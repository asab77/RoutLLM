"""Frozen proposition validation runner. Live use requires explicit authorization."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from pydantic import Field

from adaptive_llm_gateway.benchmarks.summarization_spec import (
    EntailmentVerdict,
    MaterialErrorContract,
    PropositionJudgeInput,
    ValidationExpectation,
)
from adaptive_llm_gateway.models.schemas import DomainModel

from .hybrid_judge import (
    FakeHybridJudge,
    HybridJudgeRequest,
    HybridSemanticJudge,
    JudgeProposition,
)

LIVE_VALIDATION_AUTHORIZATION = "AUTHORIZE_24_CASE_PAID_VALIDATION"


class ValidationCaseResult(DomainModel):
    case_id: str
    expected_verdict: EntailmentVerdict
    actual_verdict: EntailmentVerdict | None = None
    phenomenon: str
    structured_output_valid: bool
    error_category: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    cost_usd: Decimal = Field(default=Decimal(0), ge=0)


class ValidationReport(DomainModel):
    total_cases: int
    exact_agreement: float = Field(ge=0, le=1)
    accuracy_by_label: dict[str, float]
    accuracy_by_phenomenon: dict[str, float]
    structured_output_validity: float = Field(ge=0, le=1)
    malformed_rate: float = Field(ge=0, le=1)
    provider_failure_rate: float = Field(ge=0, le=1)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    total_cost_usd: Decimal = Field(ge=0)
    results: tuple[ValidationCaseResult, ...]


def load_validation_cases(input_path: Path, label_path: Path) -> tuple[
        tuple[PropositionJudgeInput, ...], dict[str, ValidationExpectation]]:
    inputs_raw = json.loads(input_path.read_text())
    labels_raw = json.loads(label_path.read_text())
    inputs = tuple(PropositionJudgeInput.model_validate(item)
                   for item in inputs_raw["judge_inputs"])
    labels = {key: ValidationExpectation.model_validate(value)
              for key, value in labels_raw["expected_labels"].items()}
    if {item.case_id for item in inputs} != set(labels):
        raise ValueError("validation inputs and labels must have identical case IDs")
    return inputs, labels


class HybridValidationRunner:
    def __init__(self, judge: HybridSemanticJudge) -> None:
        self.judge = judge

    async def run(self, inputs: tuple[PropositionJudgeInput, ...],
                  labels: dict[str, ValidationExpectation], *,
                  authorization: str | None = None) -> ValidationReport:
        if not isinstance(self.judge, FakeHybridJudge) and authorization != LIVE_VALIDATION_AUTHORIZATION:
            raise PermissionError("live validation requires explicit human paid authorization")
        if {item.case_id for item in inputs} != set(labels):
            raise ValueError("validation inputs and labels must remain separate and aligned")
        results: list[ValidationCaseResult] = []
        for item in inputs:
            request = HybridJudgeRequest(
                source_material=item.source_evidence,
                candidate_summary=item.candidate_text,
                required_propositions=(JudgeProposition(
                    proposition_id=item.case_id,
                    description=item.required_proposition,
                    source_evidence=item.source_evidence),),
                material_error_contract=MaterialErrorContract(),
            )
            outcome = await self.judge.judge(request)
            usage = outcome.usage
            actual = (outcome.rubric.proposition_judgments[0].verdict
                      if outcome.status == "evaluated" and outcome.rubric else None)
            expected = labels[item.case_id]
            results.append(ValidationCaseResult(
                case_id=item.case_id, expected_verdict=expected.expected_verdict,
                actual_verdict=actual, phenomenon=expected.phenomenon,
                structured_output_valid=outcome.status == "evaluated",
                error_category=outcome.error_category,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                latency_ms=usage.latency_ms if usage else 0,
                cost_usd=usage.estimated_cost_usd if usage else 0,
            ))
        return self._report(results)

    @staticmethod
    def _report(results: list[ValidationCaseResult]) -> ValidationReport:
        total = len(results)
        correct = [item.actual_verdict == item.expected_verdict for item in results]
        label_totals = Counter(item.expected_verdict.value for item in results)
        label_correct = Counter(item.expected_verdict.value for item in results
                                if item.actual_verdict == item.expected_verdict)
        phenomenon_totals = Counter(item.phenomenon for item in results)
        phenomenon_correct = Counter(item.phenomenon for item in results
                                     if item.actual_verdict == item.expected_verdict)
        malformed = sum(item.error_category in {
            "malformed_json", "invalid_schema", "proposition_id_mismatch", "invalid_rule_use"
        } for item in results)
        provider = sum(item.error_category in {"provider_error", "timeout"} for item in results)
        return ValidationReport(
            total_cases=total, exact_agreement=sum(correct) / total,
            accuracy_by_label={key: label_correct[key] / value
                               for key, value in sorted(label_totals.items())},
            accuracy_by_phenomenon={key: phenomenon_correct[key] / value
                                    for key, value in sorted(phenomenon_totals.items())},
            structured_output_validity=sum(item.structured_output_valid for item in results) / total,
            malformed_rate=malformed / total, provider_failure_rate=provider / total,
            total_input_tokens=sum(item.input_tokens for item in results),
            total_output_tokens=sum(item.output_tokens for item in results),
            total_latency_ms=sum(item.latency_ms for item in results),
            total_cost_usd=sum((item.cost_usd for item in results), Decimal(0)),
            results=tuple(results),
        )
