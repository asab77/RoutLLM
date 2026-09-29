"""Adaptive routing composition that reuses the explicit inference service."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from adaptive_llm_gateway.errors import (
    CompletionRejectedError,
    InferenceDeadlineExceededError,
    ModelDisabledError,
    ProviderUnavailableError,
    ResponseValidationError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier, Money
from adaptive_llm_gateway.request_deadline import RequestDeadline
from adaptive_llm_gateway.routing.features import RoutingCategory
from adaptive_llm_gateway.routing.policy import RoutingDecision
from adaptive_llm_gateway.routing.service import RoutingDecisionService, RoutingResult
from adaptive_llm_gateway.telemetry.contracts import (
    AdaptiveExecutionTelemetry,
    AdaptiveTerminalOutcome,
    ValidationTelemetry,
)
from adaptive_llm_gateway.validation import (
    DeterministicResponseValidator,
    ResponseValidator,
    ValidationContext,
    ValidationContract,
    ValidationResult,
    ValidationStatus,
)

from .service import InferenceService


class AdaptiveAttempt(DomainModel):
    """Privacy-safe in-memory accounting for one executed candidate."""

    attempt_number: int = Field(gt=0, strict=True)
    model_id: Identifier
    validation: ValidationTelemetry
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    estimated_cost_usd: Money | None = None


class AdaptiveExecution(DomainModel):
    attempts: tuple[AdaptiveAttempt, ...]
    escalated: bool
    validation_outcome: ValidationStatus
    total_estimated_cost_usd: Money | None
    total_latency_ms: float = Field(ge=0, allow_inf_nan=False)


class AdaptiveInferenceResult(DomainModel):
    """Generated response plus the unchanged pre-generation routing decision."""

    response: InferenceResponse
    routing_decision: RoutingDecision
    execution: AdaptiveExecution | None = None


class AdaptiveInferenceService:
    """Route once and optionally validate up to three unique candidates."""

    def __init__(
        self,
        inference_service: InferenceService,
        routing_service: RoutingDecisionService,
        *,
        validator: ResponseValidator | None = None,
        max_validation_attempts: int = 3,
    ) -> None:
        if type(max_validation_attempts) is not int or not 1 <= max_validation_attempts <= 3:
            raise ValueError("max validation attempts must be an integer from 1 to 3")
        self._inference_service = inference_service
        self._routing_service = routing_service
        self._validator = validator if validator is not None else DeterministicResponseValidator()
        self._max_validation_attempts = max_validation_attempts
        self.predictor_metadata = None

    @classmethod
    def from_trusted_artifact(
        cls,
        inference_service: InferenceService,
        artifact_directory: str | Path,
        *,
        candidate_model_ids: Sequence[Identifier | str] | None = None,
    ) -> AdaptiveInferenceService:
        """Explicitly load an application-owned artifact without building it."""
        # Keep sklearn and pickle loading outside ordinary explicit API imports.
        from adaptive_llm_gateway.routing.predictor import SklearnQualityPredictor

        predictor = SklearnQualityPredictor.from_trusted_artifact(
            Path(artifact_directory)
        )
        instance = cls(inference_service, RoutingDecisionService(predictor))
        instance.predictor_metadata = predictor.metadata
        if candidate_model_ids is not None:
            candidates = instance._resolve_candidates(candidate_model_ids)
            supported = set(predictor.metadata.known_candidate_ids)
            unsupported = sorted(
                candidate.model_id for candidate in candidates
                if candidate.model_id not in supported
            )
            if unsupported:
                raise UnsupportedPredictorCandidateError(
                    "configured candidate is not supported by the predictor artifact"
                )
        return instance

    async def generate(
        self,
        request: InferenceRequest,
        *,
        category: RoutingCategory | str | None,
        quality_threshold: float | Decimal,
        candidate_model_ids: Sequence[Identifier | str],
        validation: ValidationContract | None = None,
        structured_output_required: bool = False,
        request_id: str | None = None,
        deadline: RequestDeadline | None = None,
    ) -> AdaptiveInferenceResult:
        request_deadline = deadline or self._inference_service.new_deadline()
        if request_deadline.expired():
            raise InferenceDeadlineExceededError("inference_deadline_exceeded")
        candidates = self._resolve_candidates(candidate_model_ids)
        if validation is None:
            # Preserve the legacy single-attempt path exactly when validation is absent.
            decision = self._routing_service.route(
                request,
                candidates,
                quality_threshold,
                category_hint=category,
                structured_output_required=structured_output_required,
            )
            self._inference_service.observability.routing_decision(
                selected_model=decision.selected_model_id,
                reason=decision.reason.value,
            )
            correlation_id = request_id or str(uuid4())
            try:
                response = await self._inference_service.generate(
                    decision.selected_model_id,
                    request,
                    request_id=correlation_id,
                    deadline=request_deadline,
                    request_mode="adaptive",
                )
            except Exception as exc:
                terminal = (
                    AdaptiveTerminalOutcome.DEADLINE_EXCEEDED
                    if isinstance(exc, InferenceDeadlineExceededError)
                    else AdaptiveTerminalOutcome.PROVIDER_FAILURE
                )
                self._inference_service.observability.adaptive_request(
                    terminal_outcome=terminal.value,
                    escalated=False,
                    recovered=False,
                    attempts=1,
                    request_id=correlation_id,
                    validation_outcome=ValidationStatus.NOT_RUN.value,
                )
                raise
            self._inference_service.observability.adaptive_request(
                terminal_outcome=AdaptiveTerminalOutcome.RETURNED.value,
                escalated=False,
                recovered=False,
                attempts=1,
                request_id=correlation_id,
                validation_outcome=ValidationStatus.NOT_RUN.value,
            )
            return AdaptiveInferenceResult(
                response=response,
                routing_decision=decision,
            )

        routing = self._routing_service.route_with_snapshot(
            request,
            candidates,
            quality_threshold,
            category_hint=category,
            structured_output_required=structured_output_required,
        )
        self._inference_service.observability.routing_decision(
            selected_model=routing.decision.selected_model_id,
            reason=routing.decision.reason.value,
        )
        return await self._generate_validated(
            request,
            validation,
            routing,
            request_id=request_id,
            deadline=request_deadline,
        )

    async def _generate_validated(
        self,
        request: InferenceRequest,
        validation: ValidationContract,
        routing: RoutingResult,
        *,
        request_id: str | None,
        deadline: RequestDeadline,
    ) -> AdaptiveInferenceResult:
        decision = routing.decision
        execution_id = uuid4()
        correlation_id = request_id or str(uuid4())
        eligible = sorted(
            (
                candidate for candidate in routing.candidates
                if candidate.qualifies and candidate.model_id != decision.selected_model_id
            ),
            key=lambda candidate: (candidate.projected_cost_usd, candidate.model_id),
        )
        model_ids = [decision.selected_model_id]
        if not decision.fallback_used:
            model_ids.extend(candidate.model_id for candidate in eligible)
        model_ids = model_ids[: self._max_validation_attempts]

        attempts: list[AdaptiveAttempt] = []
        for attempt_number, model_id in enumerate(model_ids, start=1):
            if deadline.expired():
                raise InferenceDeadlineExceededError("inference_deadline_exceeded")
            try:
                response = await self._inference_service.generate(
                    model_id,
                    request,
                    request_id=correlation_id,
                    deadline=deadline,
                    request_mode="adaptive",
                )
            except CompletionRejectedError as exc:
                response = exc.completion
            except Exception as exc:
                terminal = (
                    AdaptiveTerminalOutcome.DEADLINE_EXCEEDED
                    if isinstance(exc, InferenceDeadlineExceededError)
                    else AdaptiveTerminalOutcome.PROVIDER_FAILURE
                )
                await self._record_execution(
                    AdaptiveExecutionTelemetry(
                        id=execution_id,
                        request_id=correlation_id,
                        initial_routed_model_id=decision.selected_model_id,
                        returned_model_id=None,
                        attempt_count=attempt_number,
                        escalated=attempt_number > 1,
                        validation_outcome=(
                            attempts[-1].validation.status
                            if attempts else ValidationStatus.NOT_RUN
                        ),
                        terminal_outcome=terminal,
                        cumulative_known_cost_usd=self._known_cost(attempts),
                        cost_complete=False,
                        cumulative_latency_ms=sum(
                            item.latency_ms or 0 for item in attempts
                        ),
                        validator_version=(
                            attempts[-1].validation.validator_version
                            if attempts else None
                        ),
                        validation_duration_ms=sum(
                            item.validation.duration_ms for item in attempts
                        ),
                    ),
                    deadline,
                )
                self._inference_service.observability.adaptive_request(
                    terminal_outcome=terminal.value,
                    escalated=attempt_number > 1,
                    recovered=False,
                    attempts=attempt_number,
                    request_id=correlation_id,
                    validation_outcome=(
                        attempts[-1].validation.status.value
                        if attempts else ValidationStatus.NOT_RUN.value
                    ),
                )
                raise

            if deadline.expired():
                raise InferenceDeadlineExceededError("inference_deadline_exceeded")
            result = self.validate_response(response, validation)
            if deadline.expired():
                raise InferenceDeadlineExceededError("inference_deadline_exceeded")
            for failure in result.failures:
                self._inference_service.observability.validation_failure(
                    model=model_id,
                    reason=failure.code.value,
                    attempt=attempt_number,
                )
            attempts.append(AdaptiveAttempt(
                attempt_number=attempt_number,
                model_id=model_id,
                validation=ValidationTelemetry.from_result(result),
                latency_ms=response.latency_ms,
                estimated_cost_usd=response.estimated_cost_usd,
            ))
            if result.status is ValidationStatus.PASSED:
                execution = self._execution(attempts, result.status)
                await self._record_execution(AdaptiveExecutionTelemetry(
                    id=execution_id,
                    request_id=correlation_id,
                    initial_routed_model_id=decision.selected_model_id,
                    returned_model_id=response.model_id,
                    attempt_count=len(attempts),
                    escalated=len(attempts) > 1,
                    validation_outcome=result.status,
                    terminal_outcome=AdaptiveTerminalOutcome.RETURNED,
                    cumulative_known_cost_usd=self._known_cost(attempts),
                    cost_complete=(execution.total_estimated_cost_usd is not None),
                    cumulative_latency_ms=execution.total_latency_ms,
                    validator_version=result.validator_version,
                    validation_duration_ms=sum(
                        item.validation.duration_ms for item in attempts
                    ),
                ), deadline)
                recovered = (
                    len(attempts) > 1
                    and any(
                        item.validation.status is ValidationStatus.FAILED
                        for item in attempts[:-1]
                    )
                )
                self._inference_service.observability.adaptive_request(
                    terminal_outcome=AdaptiveTerminalOutcome.RETURNED.value,
                    escalated=len(attempts) > 1,
                    recovered=recovered,
                    attempts=len(attempts),
                    request_id=correlation_id,
                    validation_outcome=result.status.value,
                )
                return AdaptiveInferenceResult(
                    response=response,
                    routing_decision=decision,
                    execution=execution,
                )

            recoverable = (
                result.status is ValidationStatus.FAILED
                and bool(result.failures)
                and all(item.recoverable_by_escalation for item in result.failures)
            )
            if not recoverable or attempt_number == len(model_ids):
                execution = self._execution(attempts, result.status)
                await self._record_execution(AdaptiveExecutionTelemetry(
                    id=execution_id,
                    request_id=correlation_id,
                    initial_routed_model_id=decision.selected_model_id,
                    returned_model_id=None,
                    attempt_count=len(attempts),
                    escalated=len(attempts) > 1,
                    validation_outcome=result.status,
                    terminal_outcome=AdaptiveTerminalOutcome.VALIDATION_FAILED,
                    cumulative_known_cost_usd=self._known_cost(attempts),
                    cost_complete=(execution.total_estimated_cost_usd is not None),
                    cumulative_latency_ms=execution.total_latency_ms,
                    validator_version=result.validator_version,
                    validation_duration_ms=sum(
                        item.validation.duration_ms for item in attempts
                    ),
                    failure_codes=tuple(item.code for item in result.failures),
                ), deadline)
                self._inference_service.observability.adaptive_request(
                    terminal_outcome=AdaptiveTerminalOutcome.VALIDATION_FAILED.value,
                    escalated=len(attempts) > 1,
                    recovered=False,
                    attempts=len(attempts),
                    request_id=correlation_id,
                    validation_outcome=result.status.value,
                )
                raise ResponseValidationError("response_validation_failed")

        raise AssertionError("validated execution requires an initial candidate")

    @staticmethod
    def _total_cost(attempts: Sequence[AdaptiveAttempt]) -> Decimal | None:
        costs = [item.estimated_cost_usd for item in attempts]
        if any(cost is None for cost in costs):
            return None
        return sum(costs, start=Decimal(0))

    @staticmethod
    def _known_cost(attempts: Sequence[AdaptiveAttempt]) -> Decimal:
        return sum(
            (item.estimated_cost_usd for item in attempts
             if item.estimated_cost_usd is not None),
            start=Decimal(0),
        )

    def _execution(
        self,
        attempts: Sequence[AdaptiveAttempt],
        outcome: ValidationStatus,
    ) -> AdaptiveExecution:
        return AdaptiveExecution(
            attempts=tuple(attempts),
            escalated=len(attempts) > 1,
            validation_outcome=outcome,
            total_estimated_cost_usd=self._total_cost(attempts),
            total_latency_ms=sum(item.latency_ms or 0 for item in attempts),
        )

    async def _record_execution(
        self,
        event: AdaptiveExecutionTelemetry,
        deadline: RequestDeadline,
    ) -> None:
        repository = self._inference_service.telemetry
        record = getattr(repository, "record_adaptive_execution", None)
        if record is None:
            self._inference_service.observability.telemetry_write(
                record_type="adaptive",
                outcome="skipped_unconfigured",
                duration_seconds=None,
            )
            return
        timeout = deadline.constrain_timeout(
            self._inference_service.telemetry_timeout
        )
        if timeout is None:
            self._inference_service.observability.telemetry_write(
                record_type="adaptive",
                outcome="skipped_deadline",
                duration_seconds=None,
            )
            return
        started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(timeout):
                await record(event)
        except Exception:  # noqa: BLE001 - telemetry must not fail paid inference
            duration = asyncio.get_running_loop().time() - started
            self._inference_service.observability.telemetry_write(
                record_type="adaptive",
                outcome="failure",
                duration_seconds=duration,
            )
            self._inference_service.observability.events.emit(
                "telemetry_write_failed",
                level=logging.WARNING,
                request_id=event.request_id,
                record_type="adaptive",
                outcome="failure",
                latency_ms=duration * 1000,
            )
        else:
            self._inference_service.observability.telemetry_write(
                record_type="adaptive",
                outcome="success",
                duration_seconds=asyncio.get_running_loop().time() - started,
            )

    def validate_response(
        self, response: InferenceResponse, validation: ValidationContract | None = None,
    ) -> ValidationResult:
        """Apply the injected provider-independent validator."""
        return self._validator.validate(response, ValidationContext(contract=validation))

    def _resolve_candidates(
        self,
        candidate_model_ids: Sequence[Identifier | str],
    ) -> tuple[ModelConfig, ...]:
        candidates: list[ModelConfig] = []
        for model_id in candidate_model_ids:
            model = self._inference_service.registry.get(str(model_id))
            if not model.enabled:
                raise ModelDisabledError(f"Model is disabled: {model.model_id!r}")
            if not self._inference_service.resolver.supports(model.provider):
                raise ProviderUnavailableError(
                    "No provider adapter is registered for an adaptive candidate"
                )
            candidates.append(model)
        return tuple(candidates)


__all__ = [
    "AdaptiveAttempt",
    "AdaptiveExecution",
    "AdaptiveInferenceResult",
    "AdaptiveInferenceService",
]
