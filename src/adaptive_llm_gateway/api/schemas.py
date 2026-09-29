from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from adaptive_llm_gateway.evaluation.models import EvaluationSummary
from adaptive_llm_gateway.models import (
    InferenceRequest,
    InferenceResponse,
    TerminationReason,
)
from adaptive_llm_gateway.models.schemas import Identifier
from adaptive_llm_gateway.routing.features import RoutingCategory
from adaptive_llm_gateway.routing.policy import (
    RoutingDecisionReason,
    validate_quality_threshold,
)
from adaptive_llm_gateway.telemetry.contracts import TelemetrySummary
from adaptive_llm_gateway.validation import ValidationContract


class MetricsSummary(TelemetrySummary):
    """Public aggregate telemetry response; no ORM objects."""


class BenchmarkEvaluationSummary(EvaluationSummary):
    """Read-only aggregate evaluation; raw local paths are never exposed."""


class InferencePayload(InferenceRequest):
    """Reuse all domain validation while adding explicit model selection."""

    model_id: Identifier

    def to_domain(self) -> InferenceRequest:
        return InferenceRequest(**self.model_dump(exclude={"model_id"}))


class InferenceResult(InferenceResponse):
    termination_reason: TerminationReason = Field(
        default=TerminationReason.UNKNOWN, exclude=True)
    provider_termination_reason: str | None = Field(default=None, exclude=True)
    request_id: str


class AdaptiveInferencePayload(InferenceRequest):
    """Public adaptive intent; the gateway owns artifact and candidate configuration."""

    validation: ValidationContract | None = Field(
        default=None,
        description="Optional deterministic output checks with bounded escalation.",
    )
    category: RoutingCategory
    quality_threshold: float | None = None

    @field_validator("quality_threshold", mode="before")
    @classmethod
    def apply_routing_threshold_contract(cls, value: object) -> float | None:
        if value is None:
            return value
        return validate_quality_threshold(value)

    def to_domain(self) -> InferenceRequest:
        return InferenceRequest(
            **self.model_dump(exclude={"category", "quality_threshold", "validation"})
        )


class PublicRoutingMetadata(BaseModel):
    selected_model_id: str
    threshold_satisfied: bool
    fallback_used: bool
    reason: RoutingDecisionReason


class PublicExecutionMetadata(BaseModel):
    attempts: int = Field(gt=0, le=3)
    escalated: bool
    validation_outcome: Literal["passed"]
    total_estimated_cost_usd: Decimal | None
    total_latency_ms: float = Field(ge=0, allow_inf_nan=False)


class AdaptiveInferenceResult(InferenceResponse):
    termination_reason: TerminationReason = Field(
        default=TerminationReason.UNKNOWN, exclude=True)
    provider_termination_reason: str | None = Field(default=None, exclude=True)
    request_id: str
    routing: PublicRoutingMetadata
    execution: PublicExecutionMetadata | None = None


class PublicModel(BaseModel):
    """Intentionally omit pricing and provider-internal model names."""

    model_id: str
    provider: str
    context_window: int


class ModelList(BaseModel):
    models: list[PublicModel]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str
