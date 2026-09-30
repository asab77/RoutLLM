"""Frontend-oriented orchestration over the existing inference services."""

from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from enum import StrEnum
import logging
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import ConfigDict

from adaptive_llm_gateway.errors import (
    AdaptiveRoutingUnavailableError,
    InferenceDeadlineExceededError,
    ProviderFailureError,
    ProviderUnavailableError,
    ResponseValidationError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse
from adaptive_llm_gateway.models.schemas import DomainModel, Identifier
from adaptive_llm_gateway.registry import ModelNotFoundError
from adaptive_llm_gateway.request_deadline import RequestDeadline
from adaptive_llm_gateway.routing.features import RoutingCategory
from adaptive_llm_gateway.routing.policy import RoutingDecision
from adaptive_llm_gateway.telemetry.contracts import (
    ChatActivityOutcome,
    ChatActivityRecord,
    ChatExecutionMode as ActivityExecutionMode,
)
from adaptive_llm_gateway.validation import ValidationContract

from .service import InferenceService

if TYPE_CHECKING:
    from .adaptive import AdaptiveExecution
    from .adaptive_config import AdaptiveRuntime

DEFAULT_MODEL_ID_ENV = "ROUTELLM_DEFAULT_MODEL_ID"


class ChatRoutingMode(StrEnum):
    AUTO = "auto"
    MANUAL = "manual"


class ChatExecutionMode(StrEnum):
    DIRECT = "direct"
    ADAPTIVE = "adaptive"


class CategorySource(StrEnum):
    MANUAL = "manual"


class ChatConfiguration(DomainModel):
    """Trusted process configuration for the direct AUTO path."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    default_model_id: Identifier

    @classmethod
    def from_environment(cls) -> "ChatConfiguration":
        model_id = os.environ.get(DEFAULT_MODEL_ID_ENV, "").strip()
        if not model_id:
            raise ValueError(f"{DEFAULT_MODEL_ID_ENV} is required")
        return cls(default_model_id=model_id)


@dataclass(frozen=True)
class ChatOrchestrationResult:
    response: InferenceResponse
    execution_mode: ChatExecutionMode
    category: RoutingCategory | None = None
    category_source: CategorySource | None = None
    routing_decision: RoutingDecision | None = None
    execution: "AdaptiveExecution | None" = None


@dataclass(frozen=True)
class ChatOrchestrationService:
    """Choose direct or adaptive execution without duplicating either path."""

    inference_service: InferenceService
    default_model_id: str
    adaptive_runtime: "AdaptiveRuntime | None" = None

    async def generate(
        self,
        request: InferenceRequest,
        *,
        routing_mode: ChatRoutingMode,
        category: RoutingCategory | None,
        validation: ValidationContract | None,
        request_id: str,
        deadline: RequestDeadline,
    ) -> ChatOrchestrationResult:
        if routing_mode is ChatRoutingMode.AUTO:
            try:
                response = await self.inference_service.generate(
                    self.default_model_id,
                    request,
                    request_id=request_id,
                    deadline=deadline,
                )
            except (ProviderFailureError, ProviderUnavailableError,
                    InferenceDeadlineExceededError) as exc:
                model = self.inference_service.registry.get(self.default_model_id)
                await self._record_activity(ChatActivityRecord(
                    request_id=request_id,
                    execution_mode=ActivityExecutionMode.DIRECT,
                    initial_model_id=self.default_model_id,
                    provider=model.provider,
                    outcome=self._failure_outcome(exc),
                    error_category=self._failure_outcome(exc).value,
                    cost_complete=False,
                ), deadline)
                raise
            await self._record_activity(ChatActivityRecord(
                request_id=request_id,
                execution_mode=ActivityExecutionMode.DIRECT,
                final_model_id=response.model_id,
                provider=response.provider,
                attempt_count=1,
                escalated=False,
                outcome=ChatActivityOutcome.RETURNED,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                estimated_cost_usd=response.estimated_cost_usd,
                cost_complete=True,
            ), deadline)
            return ChatOrchestrationResult(
                response=response,
                execution_mode=ChatExecutionMode.DIRECT,
            )

        runtime = self.adaptive_runtime
        if runtime is None:
            raise AdaptiveRoutingUnavailableError("adaptive routing is not configured")
        try:
            result = await runtime.service.generate(
                request,
                **({"validation": validation} if validation is not None else {}),
                category=category,
                quality_threshold=runtime.approved_quality_threshold,
                candidate_model_ids=runtime.candidate_model_ids,
                request_id=request_id,
                deadline=deadline,
            )
        except (ProviderFailureError, ProviderUnavailableError,
                InferenceDeadlineExceededError, ResponseValidationError) as exc:
            await self._record_activity(ChatActivityRecord(
                request_id=request_id,
                execution_mode=ActivityExecutionMode.ADAPTIVE,
                category=category.value if category is not None else None,
                category_source="manual",
                outcome=self._failure_outcome(exc),
                error_category=self._failure_outcome(exc).value,
                cost_complete=False,
            ), deadline)
            raise
        decision = result.routing_decision
        execution = result.execution
        total_cost = (
            execution.total_estimated_cost_usd if execution is not None
            else result.response.estimated_cost_usd
        )
        total_latency = (
            execution.total_latency_ms if execution is not None
            else result.response.latency_ms
        )
        await self._record_activity(ChatActivityRecord(
            request_id=request_id,
            execution_mode=ActivityExecutionMode.ADAPTIVE,
            category=category.value if category is not None else None,
            category_source="manual",
            initial_model_id=decision.selected_model_id,
            final_model_id=result.response.model_id,
            provider=result.response.provider,
            routing_threshold_satisfied=decision.threshold_satisfied,
            routing_fallback_used=decision.fallback_used,
            attempt_count=len(execution.attempts) if execution is not None else 1,
            escalated=execution.escalated if execution is not None else False,
            validation_outcome=(execution.validation_outcome.value if execution is not None else None),
            outcome=ChatActivityOutcome.RETURNED,
            input_tokens=result.response.input_tokens,
            output_tokens=result.response.output_tokens,
            latency_ms=total_latency,
            estimated_cost_usd=total_cost,
            cost_complete=total_cost is not None,
        ), deadline)
        return ChatOrchestrationResult(
            response=result.response,
            execution_mode=ChatExecutionMode.ADAPTIVE,
            category=category,
            category_source=CategorySource.MANUAL,
            routing_decision=result.routing_decision,
            execution=result.execution,
        )

    @staticmethod
    def _failure_outcome(exc: Exception) -> ChatActivityOutcome:
        if isinstance(exc, ResponseValidationError):
            return ChatActivityOutcome.VALIDATION_FAILED
        if isinstance(exc, InferenceDeadlineExceededError):
            return ChatActivityOutcome.DEADLINE_EXCEEDED
        return ChatActivityOutcome.PROVIDER_FAILURE

    async def _record_activity(
        self, event: ChatActivityRecord, deadline: RequestDeadline
    ) -> None:
        repository = self.inference_service.telemetry
        record = getattr(repository, "record_chat_activity", None)
        if record is None:
            return
        timeout = deadline.constrain_timeout(self.inference_service.telemetry_timeout)
        if timeout is None:
            return
        started = perf_counter()
        try:
            async with asyncio.timeout(timeout):
                await record(event)
        except Exception:
            duration = perf_counter() - started
            self.inference_service.observability.telemetry_write(
                record_type="chat_activity",
                outcome="failure",
                duration_seconds=duration,
            )
            self.inference_service.observability.events.emit(
                "telemetry_write_failed",
                level=logging.WARNING,
                request_id=event.request_id,
                record_type="chat_activity",
                outcome="failure",
                latency_ms=duration * 1000,
            )
        else:
            self.inference_service.observability.telemetry_write(
                record_type="chat_activity",
                outcome="success",
                duration_seconds=perf_counter() - started,
            )


def build_chat_orchestration_service(
    inference_service: InferenceService,
    config: ChatConfiguration,
    adaptive_runtime: "AdaptiveRuntime | None",
) -> ChatOrchestrationService:
    """Validate the trusted default against the configured model registry."""
    try:
        model = inference_service.registry.get(config.default_model_id)
    except ModelNotFoundError:
        raise ValueError("configured default model is not registered") from None
    if not model.enabled:
        raise ValueError("configured default model is disabled")
    return ChatOrchestrationService(
        inference_service=inference_service,
        default_model_id=model.model_id,
        adaptive_runtime=adaptive_runtime,
    )


__all__ = [
    "DEFAULT_MODEL_ID_ENV",
    "CategorySource",
    "ChatConfiguration",
    "ChatExecutionMode",
    "ChatOrchestrationResult",
    "ChatOrchestrationService",
    "ChatRoutingMode",
    "build_chat_orchestration_service",
]
