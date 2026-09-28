"""Offline diagnostics for benchmark generation-allowance failures."""

from typing import Any

from adaptive_llm_gateway.models import InferenceResponse, TerminationReason


def is_output_budget_exhaustion(details: dict[str, Any]) -> bool:
    """Recognize an empty completion whose combined token budget was exhausted."""
    limit = details.get("output_token_limit")
    output = details.get("output_tokens")
    reasoning = details.get("reasoning_tokens")
    return (
        details.get("http_status") == 200
        and details.get("content_empty") is True
        and (details.get("termination_reason") == "length"
             or details.get("finish_reason") == "length")
        and type(limit) is int and limit > 0
        and type(output) is int and output >= max(0, limit - 4)
        and type(reasoning) is int and reasoning == output
    )


def output_budget_diagnostic(details: dict[str, Any]) -> dict[str, Any]:
    """Return safe token accounting without provider text or reasoning content."""
    exhausted = is_output_budget_exhaustion(details)
    return {
        "classification": "OUTPUT_BUDGET_EXHAUSTION" if exhausted else "OTHER_FAILURE",
        "requested_budget": details.get("output_token_limit"),
        "effective_budget": details.get("output_token_limit"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "total_output_tokens": details.get("output_tokens"),
        "visible_tokens": 0 if details.get("content_empty") is True else None,
        "finish_reason": details.get("finish_reason"),
        "normalized_outcome": "gateway_empty_response" if exhausted else None,
    }


def visible_response_appears_truncated(text: str) -> bool:
    """Conservative fallback for providers that omit termination metadata."""
    value = text.rstrip()
    if not value:
        return True
    if value.count("```") % 2:
        return True
    if value.endswith(("len", "cl", "14:0")):
        return True
    if value[-1] in "([{,:=+-*/\\":
        return True
    if "\n" in value and not value.endswith((".", "!", "?", "]", ")", "}", "'", '"')):
        return True
    return False


def response_output_budget_exhaustion(
        response: InferenceResponse, output_token_limit: int) -> bool:
    """Prefer authoritative termination; fall back only near the allowance."""
    if response.termination_reason is TerminationReason.LENGTH:
        return True
    if response.termination_reason is not TerminationReason.UNKNOWN:
        return False
    return (
        response.output_tokens >= max(1, output_token_limit - 8)
        and visible_response_appears_truncated(response.text)
    )


__all__ = [
    "is_output_budget_exhaustion", "output_budget_diagnostic",
    "response_output_budget_exhaustion", "visible_response_appears_truncated",
]
