"""Offline cost planning for the frozen proposition judge contract."""
from __future__ import annotations

from decimal import Decimal

from adaptive_llm_gateway.benchmarks.summarization_spec import (
    MaterialErrorContract,
    PropositionJudgeInput,
    PropositionSpecification,
)

from .hybrid_judge import (
    HYBRID_JUDGE_SYSTEM_PROMPT,
    HybridJudgeRequest,
    JudgeProposition,
    VercelHybridSemanticJudge,
)
from .hybrid_replay import HistoricalReplayItem

ASTRA_INPUT_USD_PER_MILLION = Decimal("10")
ASTRA_OUTPUT_USD_PER_MILLION = Decimal("50")
EXPECTED_OUTPUT_BASE_TOKENS = 96
EXPECTED_OUTPUT_TOKENS_PER_PROPOSITION = 56
WORST_CASE_OUTPUT_TOKENS = 768


def estimated_tokens(text: str) -> int:
    """Conservative deterministic approximation used only for preflight budgets."""
    return (len(text.encode("utf-8")) + 3) // 4


def expected_output_tokens(proposition_count: int) -> int:
    return EXPECTED_OUTPUT_BASE_TOKENS + EXPECTED_OUTPUT_TOKENS_PER_PROPOSITION * proposition_count


def cost(input_tokens: int, output_tokens: int) -> Decimal:
    return ((Decimal(input_tokens) * ASTRA_INPUT_USD_PER_MILLION
             + Decimal(output_tokens) * ASTRA_OUTPUT_USD_PER_MILLION)
            / Decimal(1_000_000))


def request_input_tokens(request: HybridJudgeRequest) -> int:
    return estimated_tokens(
        HYBRID_JUDGE_SYSTEM_PROMPT + "\n" + VercelHybridSemanticJudge.prompt(request))


def estimate_requests(requests: tuple[HybridJudgeRequest, ...]) -> dict:
    input_tokens = sum(request_input_tokens(item) for item in requests)
    expected_output = sum(expected_output_tokens(len(item.required_propositions))
                          for item in requests)
    worst_output = WORST_CASE_OUTPUT_TOKENS * len(requests)
    return {
        "calls": len(requests),
        "estimated_input_tokens": input_tokens,
        "expected_output_tokens": expected_output,
        "worst_case_output_tokens": worst_output,
        "expected_cost_usd": str(cost(input_tokens, expected_output)),
        "worst_case_cost_usd": str(cost(input_tokens, worst_output)),
    }


def validation_requests(inputs: tuple[PropositionJudgeInput, ...]) -> tuple[HybridJudgeRequest, ...]:
    return tuple(HybridJudgeRequest(
        source_material=item.source_evidence,
        candidate_summary=item.candidate_text,
        required_propositions=(JudgeProposition(
            proposition_id=item.case_id, description=item.required_proposition,
            source_evidence=item.source_evidence),),
        material_error_contract=MaterialErrorContract(),
    ) for item in inputs)


def routing_full_requests(specification: PropositionSpecification, *,
                          candidate_count: int = 4) -> tuple[HybridJudgeRequest, ...]:
    requests = []
    for task in specification.tasks:
        if not task.specification_id.startswith("rb12-"):
            continue
        # Expected candidate length is represented without model identity. The fixed
        # placeholder is deliberately conservative for the short summaries in this suite.
        candidate = "summary " * 40
        request = HybridJudgeRequest.from_specification(task, candidate).model_copy(update={
            "material_error_contract": specification.material_error_contract})
        requests.extend(request for _ in range(candidate_count))
    return tuple(requests)


def build_cost_plan(validation_inputs: tuple[PropositionJudgeInput, ...],
                    pilot_replay: tuple[HistoricalReplayItem, ...],
                    specification: PropositionSpecification) -> dict:
    validation = estimate_requests(validation_requests(validation_inputs))
    pilot_requests = tuple(item.blind_judge_request for item in pilot_replay
                           if "1aa850f6" in item.run_id)
    pilot = estimate_requests(pilot_requests)
    full = estimate_requests(routing_full_requests(specification))
    pilot_candidates = Decimal("0.03213190")
    full_candidates = Decimal("0.34830600")
    contingency = Decimal("0.10")
    pilot_base = pilot_candidates + Decimal(pilot["expected_cost_usd"])
    full_base = full_candidates + Decimal(full["expected_cost_usd"])
    return {
        "version": "1.3.0",
        "pricing_status": "FROZEN_REPOSITORY_ASSUMPTION_REQUIRES_LIVE_REVERIFICATION",
        "pricing_usd_per_million_tokens": {
            "astra_input": str(ASTRA_INPUT_USD_PER_MILLION),
            "astra_output": str(ASTRA_OUTPUT_USD_PER_MILLION),
        },
        "method": {
            "input_token_approximation": (
                "ceil((system prompt + exact request prompt) UTF-8 bytes / 4)"),
            "expected_output_tokens": "96 + 56 * proposition_count",
            "worst_case_output_tokens_per_call": WORST_CASE_OUTPUT_TOKENS,
            "full_candidate_placeholder": "40 words per stored task source",
            "candidate_costs": "unchanged frozen routing cost estimate v1.1.0",
            "total_contingency_fraction": str(contingency),
        },
        "validation": validation,
        "corrected_pilot": pilot | {
            "candidate_cost_usd": str(pilot_candidates),
            "expected_total_cost_usd": str(pilot_base * (Decimal(1) + contingency)),
        },
        "full_benchmark": full | {
            "candidate_cost_usd": str(full_candidates),
            "expected_total_cost_usd": str(full_base * (Decimal(1) + contingency)),
        },
    }
