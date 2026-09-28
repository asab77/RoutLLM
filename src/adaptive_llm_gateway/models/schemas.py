"""Validated, immutable configuration and inference data."""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
TokenCount = Annotated[int, Field(ge=0, strict=True)]
PositiveTokenCount = Annotated[int, Field(gt=0, strict=True)]


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReasoningBehavior(StrEnum):
    """How a registered model handles reasoning when no override is requested."""

    UNSUPPORTED = "unsupported"
    PROVIDER_DEFAULT = "provider_default"
    ADAPTIVE = "adaptive"


class ReasoningEffort(StrEnum):
    """Explicit provider-agnostic reasoning effort; omission uses the provider default."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningControlMechanism(StrEnum):
    """Wire mechanism used to apply the effective reasoning effort."""

    GATEWAY_SHARED = "gateway_shared"
    GOOGLE_PROVIDER_NATIVE = "google_provider_native"


class OutputTokenAccounting(StrEnum):
    """What an upstream max-output limit counts for a model path."""

    VISIBLE_ONLY = "visible_only"
    REASONING_AND_VISIBLE = "reasoning_and_visible"


class TerminationReason(StrEnum):
    """Provider-independent completion termination categories."""

    COMPLETE = "complete"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"
    UNKNOWN = "unknown"


class ModelCapabilities(DomainModel):
    """Small, explicit set of generation capabilities used by request builders."""

    supports_temperature: bool = Field(default=True, strict=True)
    supports_structured_output: bool = Field(default=False, strict=True)
    reasoning: ReasoningBehavior = ReasoningBehavior.UNSUPPORTED
    output_token_accounting: OutputTokenAccounting = OutputTokenAccounting.VISIBLE_ONLY


class CategoryOutputTokenAllowance(DomainModel):
    """Optional pre-generation output allowance for one canonical category."""

    category: Identifier
    max_output_tokens: PositiveTokenCount


class OutputTokenPolicy(DomainModel):
    """Bounded provider allowance policy independent of model identity."""

    category_overrides: tuple[CategoryOutputTokenAllowance, ...] = ()
    reasoning_headroom_tokens: int = Field(default=0, ge=0, le=1024, strict=True)
    expected_reasoning_tokens: int = Field(default=0, ge=0, le=1024, strict=True)

    @model_validator(mode="after")
    def category_overrides_are_unique(self) -> "OutputTokenPolicy":
        categories = [item.category for item in self.category_overrides]
        if len(categories) != len(set(categories)):
            raise ValueError("output-token category overrides must be unique")
        if self.expected_reasoning_tokens > self.reasoning_headroom_tokens:
            raise ValueError("expected reasoning tokens cannot exceed bounded headroom")
        return self


class ModelConfig(DomainModel):
    """Public model metadata. Prices are USD per one million tokens."""

    model_id: Identifier
    provider: Identifier
    provider_model_name: Identifier
    input_cost_per_1m_tokens: Money
    output_cost_per_1m_tokens: Money
    context_window: PositiveTokenCount
    enabled: bool = Field(default=True, strict=True)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_control: ReasoningControlMechanism = ReasoningControlMechanism.GATEWAY_SHARED
    output_token_policy: OutputTokenPolicy = Field(default_factory=OutputTokenPolicy)

    @model_validator(mode="after")
    def reject_unsupported_reasoning_effort(self) -> "ModelConfig":
        if (self.reasoning_effort is not None
                and self.capabilities.reasoning is ReasoningBehavior.UNSUPPORTED):
            raise ValueError("explicit reasoning effort requires reasoning support")
        if self.reasoning_control is ReasoningControlMechanism.GOOGLE_PROVIDER_NATIVE:
            if not self.provider_model_name.startswith("google/"):
                raise ValueError("Google provider-native reasoning requires a Google model")
            if self.reasoning_effort is None or self.reasoning_effort is ReasoningEffort.NONE:
                raise ValueError("Google provider-native reasoning requires an effective effort")
        if (self.capabilities.output_token_accounting
                is OutputTokenAccounting.VISIBLE_ONLY
                and self.output_token_policy.reasoning_headroom_tokens):
            raise ValueError("visible-only output accounting cannot reserve reasoning headroom")
        return self

    def provider_output_allowance(self, visible_output_tokens: int, *, category: str) -> int:
        """Translate a canonical visible requirement into a bounded upstream allowance."""
        if type(visible_output_tokens) is not int or visible_output_tokens <= 0:
            raise ValueError("visible output requirement must be a positive integer")
        matching = tuple(
            item.max_output_tokens for item in self.output_token_policy.category_overrides
            if item.category == category
        )
        if len(matching) > 1:
            raise ValueError("output-token policy contains duplicate category overrides")
        visible_requirement = matching[0] if matching else visible_output_tokens
        if (self.capabilities.output_token_accounting
                is OutputTokenAccounting.REASONING_AND_VISIBLE):
            return visible_requirement + self.output_token_policy.reasoning_headroom_tokens
        return visible_requirement


class InferenceRequest(DomainModel):
    """Single-turn input; adapters map this schema to provider requests."""

    prompt: str = Field(min_length=1)
    system_prompt: str | None = None
    max_output_tokens: PositiveTokenCount = 256
    temperature: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)

    @field_validator("prompt", "system_prompt")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("prompt text must contain non-whitespace characters")
        return value


class InferenceResponse(DomainModel):
    """Normalized generation result. Cost is an estimate in USD."""

    text: str
    model_id: Identifier
    provider: Identifier
    input_tokens: TokenCount
    output_tokens: TokenCount
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    estimated_cost_usd: Money
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    provider_termination_reason: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}$")
