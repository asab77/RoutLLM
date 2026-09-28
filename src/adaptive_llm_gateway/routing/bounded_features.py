"""Small deterministic request-structure signals for the Phase 9 bounded ablation.

These features are experimental and are not part of the deployed canonical
quality-feature contract. They consume request text only and are available
before candidate selection or provider execution.
"""

from __future__ import annotations

import re

from pydantic import Field

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.models.schemas import DomainModel

from .features import request_text

BOUNDED_FEATURE_NAMES = (
    "numeric_quantity_count",
    "conditional_operator_count",
    "symbolic_math_operator_count",
    "comparison_structure_count",
)

_NUMERIC_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?(?![A-Za-z0-9_])"
)
_CONDITIONAL_OPERATOR = re.compile(
    r"\b(?:if|unless|otherwise|except|when|provided\s+that|either|neither)\b",
    re.I,
)
_SYMBOLIC_MATH_OPERATOR = re.compile(r"<=|>=|==|!=|[=<>+*/%]")
_COMPARISON_STRUCTURE = re.compile(
    r"\b(?:at\s+least|at\s+most|more\s+than|less\s+than|fewer\s+than|"
    r"greater\s+than|above|below|exceed(?:s|ed|ing)?|highest|lowest|maximum|"
    r"minimum|before|after)\b",
    re.I,
)


class BoundedStructuralFeatures(DomainModel):
    numeric_quantity_count: int = Field(ge=0, strict=True)
    conditional_operator_count: int = Field(ge=0, strict=True)
    symbolic_math_operator_count: int = Field(ge=0, strict=True)
    comparison_structure_count: int = Field(ge=0, strict=True)


def extract_bounded_structural_features(
    request: InferenceRequest,
) -> BoundedStructuralFeatures:
    """Extract four provider-independent pre-generation counts."""
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be an InferenceRequest")
    text = request_text(request)
    return BoundedStructuralFeatures(
        numeric_quantity_count=len(_NUMERIC_QUANTITY.findall(text)),
        conditional_operator_count=len(_CONDITIONAL_OPERATOR.findall(text)),
        symbolic_math_operator_count=len(_SYMBOLIC_MATH_OPERATOR.findall(text)),
        comparison_structure_count=len(_COMPARISON_STRUCTURE.findall(text)),
    )


__all__ = [
    "BOUNDED_FEATURE_NAMES",
    "BoundedStructuralFeatures",
    "extract_bounded_structural_features",
]
