"""Bounded output contracts; passing checks does not establish semantic correctness."""
from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from adaptive_llm_gateway.models import InferenceResponse
from adaptive_llm_gateway.models.schemas import DomainModel

MAX_INPUT_CHARACTERS = 262_144
MAX_JSON_DEPTH = 32
MAX_CONSTRAINTS = 64
MAX_FAILURES = 16
VALIDATOR_VERSION = "deterministic-v1"
Name = Annotated[str, StringConstraints(min_length=1, max_length=128)]
JsonType = Literal["object", "array", "string", "number", "integer", "boolean", "null"]


class ValidationContract(DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    format: Literal["text", "json", "label"]
    min_characters: int | None = Field(default=None, strict=True, ge=1, le=MAX_INPUT_CHARACTERS)
    root_type: JsonType | None = None
    required_fields: tuple[Name, ...] = Field(default=(), max_length=MAX_CONSTRAINTS)
    field_types: dict[Name, JsonType] = Field(default_factory=dict, max_length=MAX_CONSTRAINTS)
    allowed_labels: tuple[Name, ...] = Field(default=(), max_length=MAX_CONSTRAINTS)

    @model_validator(mode="after")
    def compatible_constraints(self) -> "ValidationContract":
        if self.format != "json" and (self.root_type is not None or self.required_fields or self.field_types):
            raise ValueError("JSON constraints require JSON format")
        if (self.required_fields or self.field_types) and self.root_type not in (None, "object"):
            raise ValueError("field constraints require an object root")
        if self.format == "label":
            if not self.allowed_labels:
                raise ValueError("label format requires allowed labels")
            if any(label != label.strip() or not label.strip() for label in self.allowed_labels):
                raise ValueError("labels must not contain surrounding whitespace")
            if self.min_characters is not None and any(len(label) < self.min_characters for label in self.allowed_labels):
                raise ValueError("minimum length excludes an allowed label")
        elif self.allowed_labels:
            raise ValueError("allowed labels require label format")
        if len(set(self.required_fields)) != len(self.required_fields) or len(set(self.allowed_labels)) != len(self.allowed_labels):
            raise ValueError("duplicate constraints are not allowed")
        return self


class ValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    ERROR = "error"


class ValidationFailureCode(StrEnum):
    EMPTY_RESPONSE = "empty_response"
    MINIMUM_CHARACTERS = "minimum_characters"
    INVALID_JSON = "invalid_json"
    ROOT_TYPE = "root_type"
    MISSING_FIELD = "missing_field"
    FIELD_TYPE = "field_type"
    LABEL_NOT_ALLOWED = "label_not_allowed"
    OUTPUT_TRUNCATED = "output_truncated"
    INPUT_SIZE_LIMIT = "input_size_limit"
    JSON_DEPTH_LIMIT = "json_depth_limit"
    VALIDATOR_ERROR = "validator_error"


class ValidationFailure(DomainModel):
    code: ValidationFailureCode
    recoverable_by_escalation: bool = True


class ValidationResult(DomainModel):
    status: ValidationStatus
    failures: tuple[ValidationFailure, ...] = Field(default=(), max_length=MAX_FAILURES)
    validator_version: Literal["deterministic-v1"] = VALIDATOR_VERSION
    duration_ms: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_status(self) -> "ValidationResult":
        if (self.status in (ValidationStatus.FAILED, ValidationStatus.ERROR)) != bool(self.failures):
            raise ValueError("validation status and failures disagree")
        if self.status == ValidationStatus.ERROR and any(f.code != ValidationFailureCode.VALIDATOR_ERROR for f in self.failures):
            raise ValueError("validation errors require a controlled internal error")
        if self.status != ValidationStatus.ERROR and any(f.code == ValidationFailureCode.VALIDATOR_ERROR for f in self.failures):
            raise ValueError("internal errors require error status")
        return self


class ValidationContext(DomainModel):
    contract: ValidationContract | None = None


class ResponseValidator(Protocol):
    def validate(self, response: InferenceResponse, context: ValidationContext) -> ValidationResult: ...
