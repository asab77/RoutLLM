from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from adaptive_llm_gateway.models.schemas import DomainModel
from adaptive_llm_gateway.validation import ValidationResult, ValidationStatus, ValidationFailureCode
from adaptive_llm_gateway.validation.contracts import MAX_FAILURES, VALIDATOR_VERSION


@dataclass(frozen=True, kw_only=True)
class TelemetryEvent:
    request_id: str
    model_id: str
    provider: str
    success: bool
    latency_ms: float
    max_output_tokens: int
    temperature: float
    prompt_characters: int
    system_prompt_characters: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    error_category: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TelemetrySummary(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_estimated_cost_usd: Decimal = Decimal(0)
    average_latency_ms: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class InferenceTelemetryRepository(Protocol):
    async def record(self, event: TelemetryEvent) -> None: ...

    async def summary(self) -> TelemetrySummary: ...


class ValidationTelemetry(DomainModel):
    """Privacy-safe projection for future attempt storage; not a new DB event.

    Never attach a ValidationContext, contract, response, or exception here.
    """

    status: ValidationStatus
    failure_codes: tuple[ValidationFailureCode, ...] = Field(default=(), max_length=MAX_FAILURES)
    validator_version: Literal["deterministic-v1"] = VALIDATOR_VERSION
    duration_ms: float = Field(ge=0, allow_inf_nan=False)

    @classmethod
    def from_result(cls, result: ValidationResult) -> "ValidationTelemetry":
        return cls(status=result.status, failure_codes=tuple(f.code for f in result.failures),
                   validator_version=result.validator_version, duration_ms=result.duration_ms)


class AdaptiveTerminalOutcome(StrEnum):
    RETURNED = "returned"
    VALIDATION_FAILED = "validation_failed"
    PROVIDER_FAILURE = "provider_failure"
    DEADLINE_EXCEEDED = "deadline_exceeded"


class AdaptiveExecutionTelemetry(DomainModel):
    """One bounded, privacy-safe summary for a validation-enabled request."""

    id: UUID = Field(default_factory=uuid4)
    request_id: str = Field(min_length=1, max_length=128)
    initial_routed_model_id: str = Field(min_length=1)
    returned_model_id: str | None = Field(default=None, min_length=1)
    attempt_count: int = Field(gt=0, le=3, strict=True)
    escalated: bool
    validation_outcome: ValidationStatus
    terminal_outcome: AdaptiveTerminalOutcome
    cumulative_known_cost_usd: Decimal = Field(default=Decimal(0), ge=0)
    cost_complete: bool
    cumulative_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    validator_version: str | None = Field(default=None, max_length=64)
    validation_duration_ms: float = Field(ge=0, allow_inf_nan=False)
    failure_codes: tuple[ValidationFailureCode, ...] = Field(
        default=(), max_length=MAX_FAILURES
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveExecutionTelemetryRepository(Protocol):
    async def record_adaptive_execution(
        self, event: AdaptiveExecutionTelemetry
    ) -> None: ...
