"""Pure deterministic checks; response content is never logged or retained."""
import json
from decimal import Decimal
from time import perf_counter

from adaptive_llm_gateway.models import InferenceResponse, TerminationReason
from .contracts import (
    MAX_FAILURES, MAX_INPUT_CHARACTERS, MAX_JSON_DEPTH, JsonType,
    ValidationContext, ValidationFailure, ValidationFailureCode as Code,
    ValidationResult, ValidationStatus as Status,
)


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonstandard constant")


def _too_deep(text: str) -> bool:
    # Scan before parsing, ignoring brackets in strings and escaped quotes.
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                return True
        elif char in "]}":
            depth -= 1
    return False


def _matches(value: object, expected: JsonType) -> bool:
    return {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "number": type(value) is Decimal,
        "integer": type(value) is Decimal and value == value.to_integral_value(),
        "boolean": type(value) is bool,
        "null": value is None,
    }[expected]


class DeterministicResponseValidator:
    def validate(self, response: InferenceResponse, context: ValidationContext) -> ValidationResult:
        started = perf_counter()
        if context.contract is None:
            return ValidationResult(status=Status.NOT_RUN)
        try:
            codes = self._check(response, context)
            status = Status.FAILED if codes else Status.PASSED
        except Exception:
            # Never return exception strings, text, field names, or label values.
            codes, status = [Code.VALIDATOR_ERROR], Status.ERROR
        return ValidationResult(
            status=status,
            failures=tuple(
                ValidationFailure(
                    code=code,
                    recoverable_by_escalation=(code is not Code.VALIDATOR_ERROR),
                )
                for code in codes[:MAX_FAILURES]
            ),
            duration_ms=(perf_counter() - started) * 1000,
        )

    def _check(self, response: InferenceResponse, context: ValidationContext) -> list[Code]:
        contract = context.contract
        assert contract is not None
        codes = []
        if response.termination_reason == TerminationReason.LENGTH:
            codes.append(Code.OUTPUT_TRUNCATED)
        if len(response.text) > MAX_INPUT_CHARACTERS:
            return codes + [Code.INPUT_SIZE_LIMIT]
        text = response.text.strip()
        if not text:
            return codes + [Code.EMPTY_RESPONSE]
        if contract.min_characters is not None and len(text) < contract.min_characters:
            codes.append(Code.MINIMUM_CHARACTERS)
        if contract.format == "label" and text not in contract.allowed_labels:
            codes.append(Code.LABEL_NOT_ALLOWED)
        if contract.format != "json":
            return codes
        if _too_deep(text):
            return codes + [Code.JSON_DEPTH_LIMIT]
        try:
            value = json.loads(response.text, object_pairs_hook=_pairs, parse_constant=_constant,
                               parse_int=Decimal, parse_float=Decimal)
        except (ValueError, ArithmeticError, RecursionError):
            return codes + [Code.INVALID_JSON]
        root_type = contract.root_type or ("object" if contract.required_fields or contract.field_types else None)
        if root_type is not None and not _matches(value, root_type):
            return codes + [Code.ROOT_TYPE]
        if isinstance(value, dict):
            for key in contract.required_fields:
                if key not in value:
                    codes.append(Code.MISSING_FIELD)
            for key, expected in contract.field_types.items():
                # Optional typed fields may be absent; required_fields owns presence.
                if key in value and not _matches(value[key], expected):
                    codes.append(Code.FIELD_TYPE)
        return codes
