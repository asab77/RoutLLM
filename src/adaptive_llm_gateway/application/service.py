import asyncio
import logging
from time import perf_counter
from uuid import uuid4

from adaptive_llm_gateway.telemetry.contracts import InferenceTelemetryRepository, TelemetryEvent
from adaptive_llm_gateway.errors import (
    CompletionRejectedError,
    ContextLimitError,
    GatewayError,
    InferenceDeadlineExceededError,
    ModelDisabledError,
    ProviderFailureError,
    ProviderUnavailableError,
)
from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse, ModelConfig
from adaptive_llm_gateway.observability import Observability
from adaptive_llm_gateway.providers.resolver import ProviderResolver
from adaptive_llm_gateway.request_deadline import RequestDeadline
from adaptive_llm_gateway.registry import ModelRegistry

class InferenceService:
    def __init__(self, registry: ModelRegistry, resolver: ProviderResolver,
                 telemetry: InferenceTelemetryRepository | None = None,
                 telemetry_timeout: float = 2.0,
                 inference_deadline_seconds: float = 60.0,
                 observability: Observability | None = None) -> None:
        if inference_deadline_seconds <= 0:
            raise ValueError("inference deadline must be positive")
        self.registry = registry
        self.resolver = resolver
        self.telemetry = telemetry
        self.telemetry_timeout = telemetry_timeout
        self.inference_deadline_seconds = inference_deadline_seconds
        self.observability = observability or Observability()

    def new_deadline(self) -> RequestDeadline:
        return RequestDeadline(self.inference_deadline_seconds)

    def list_models(self) -> list[ModelConfig]:
        """Only advertise enabled models with a registered adapter."""
        return [
            model for model in self.registry.list_enabled_models()
            if self.resolver.supports(model.provider)
        ]

    async def generate(self, model_id: str, request: InferenceRequest,
                       *, request_id: str | None = None,
                       deadline: RequestDeadline | None = None,
                       request_mode: str = "explicit") -> InferenceResponse:
        if request_mode not in {"explicit", "adaptive"}:
            raise ValueError("request mode must be explicit or adaptive")
        request_deadline = deadline or self.new_deadline()
        if request_deadline.expired():
            raise InferenceDeadlineExceededError("inference_deadline_exceeded")
        # Unknown IDs and HTTP validation errors are not model inference attempts.
        model = self.registry.get(model_id)
        correlation_id = request_id if request_id is not None else str(uuid4())
        started = perf_counter()
        try:
            result = await self._generate(
                model,
                request,
                request_deadline,
                request_id=correlation_id,
                request_mode=request_mode,
            )
        except (ContextLimitError, InferenceDeadlineExceededError, ModelDisabledError,
                ProviderUnavailableError, ProviderFailureError) as exc:
            categories = {ContextLimitError: "context_limit_exceeded", ModelDisabledError: "model_disabled",
                          ProviderUnavailableError: "provider_unavailable", ProviderFailureError: "provider_failure",
                          InferenceDeadlineExceededError: "inference_deadline_exceeded"}
            category = next(value for kind, value in categories.items() if isinstance(exc, kind))
            if isinstance(exc, GatewayError):
                category = exc.category.value
            completion = exc.completion if isinstance(exc, CompletionRejectedError) else None
            await self._record(TelemetryEvent(
                request_id=correlation_id, model_id=model.model_id, provider=model.provider,
                success=False, error_category=category,
                latency_ms=(completion.latency_ms if completion is not None
                            else (perf_counter() - started) * 1000),
                input_tokens=(completion.input_tokens if completion is not None else None),
                output_tokens=(completion.output_tokens if completion is not None else None),
                estimated_cost_usd=(completion.estimated_cost_usd
                                    if completion is not None else None),
                max_output_tokens=request.max_output_tokens, temperature=request.temperature,
                prompt_characters=len(request.prompt), system_prompt_characters=len(request.system_prompt or ""),
            ), request_deadline)
            raise
        await self._record(TelemetryEvent(
            request_id=correlation_id, model_id=result.model_id, provider=result.provider,
            success=True, input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            latency_ms=result.latency_ms, estimated_cost_usd=result.estimated_cost_usd,
            max_output_tokens=request.max_output_tokens, temperature=request.temperature,
            prompt_characters=len(request.prompt), system_prompt_characters=len(request.system_prompt or ""),
        ), request_deadline)
        return result

    async def _record(self, event: TelemetryEvent, deadline: RequestDeadline) -> None:
        if self.telemetry is None:
            self.observability.telemetry_write(
                record_type="inference",
                outcome="skipped_unconfigured",
                duration_seconds=None,
            )
            return
        timeout = deadline.constrain_timeout(self.telemetry_timeout)
        if timeout is None:
            self.observability.telemetry_write(
                record_type="inference",
                outcome="skipped_deadline",
                duration_seconds=None,
            )
            return
        started = perf_counter()
        try:
            async with asyncio.timeout(timeout):
                await self.telemetry.record(event)
        except Exception:
            duration = perf_counter() - started
            self.observability.telemetry_write(
                record_type="inference",
                outcome="failure",
                duration_seconds=duration,
            )
            self.observability.events.emit(
                "telemetry_write_failed",
                level=logging.WARNING,
                request_id=event.request_id,
                record_type="inference",
                outcome="failure",
                latency_ms=duration * 1000,
            )
        else:
            self.observability.telemetry_write(
                record_type="inference",
                outcome="success",
                duration_seconds=perf_counter() - started,
            )

    async def _generate(
        self,
        model: ModelConfig,
        request: InferenceRequest,
        deadline: RequestDeadline,
        *,
        request_id: str,
        request_mode: str,
    ) -> InferenceResponse:
        if not model.enabled:
            raise ModelDisabledError(f"Model is disabled: {model.model_id!r}")
        try:
            provider = self.resolver.resolve(model)
        except (ProviderUnavailableError, ProviderFailureError):
            raise
        except Exception as exc:
            raise ProviderFailureError("Provider could not complete inference") from exc

        started = perf_counter()
        try:
            timeout = deadline.remaining_seconds()
            if timeout <= 0:
                raise InferenceDeadlineExceededError("inference_deadline_exceeded")
            try:
                async with asyncio.timeout(timeout):
                    generate_with_timeout = getattr(
                        provider, "generate_with_timeout", None
                    )
                    if generate_with_timeout is None:
                        response = await provider.generate(request)
                    else:
                        response = await generate_with_timeout(
                            request, timeout_seconds=timeout
                        )
            except TimeoutError:
                if deadline.expired():
                    raise InferenceDeadlineExceededError(
                        "inference_deadline_exceeded"
                    ) from None
                raise
            if not isinstance(response, InferenceResponse):
                raise TypeError("Provider returned an invalid response")
            if response.model_id != model.model_id or response.provider != model.provider:
                raise ValueError("Provider returned inconsistent model metadata")
        except asyncio.CancelledError:
            self.observability.provider_attempt(
                provider=model.provider,
                model=model.model_id,
                outcome="cancelled",
                error_category="cancelled",
                duration_seconds=perf_counter() - started,
                request_mode=request_mode,
                estimated_cost_usd=None,
                request_id=request_id,
            )
            raise
        except (ContextLimitError, InferenceDeadlineExceededError, ModelDisabledError,
                ProviderUnavailableError, ProviderFailureError) as exc:
            if isinstance(exc, GatewayError):
                category = exc.category.value
            elif isinstance(exc, InferenceDeadlineExceededError):
                category = "inference_deadline_exceeded"
            elif isinstance(exc, ContextLimitError):
                category = "context_limit_exceeded"
            elif isinstance(exc, ModelDisabledError):
                category = "model_disabled"
            elif isinstance(exc, ProviderUnavailableError):
                category = "provider_unavailable"
            else:
                category = "provider_failure"
            completion = exc.completion if isinstance(exc, CompletionRejectedError) else None
            self.observability.provider_attempt(
                provider=model.provider,
                model=model.model_id,
                outcome="rejected" if completion is not None else "error",
                error_category=category,
                duration_seconds=perf_counter() - started,
                request_mode=request_mode,
                estimated_cost_usd=(
                    completion.estimated_cost_usd if completion is not None else None
                ),
                request_id=request_id,
            )
            raise
        except Exception as exc:
            self.observability.provider_attempt(
                provider=model.provider,
                model=model.model_id,
                outcome="error",
                error_category="provider_failure",
                duration_seconds=perf_counter() - started,
                request_mode=request_mode,
                estimated_cost_usd=None,
                request_id=request_id,
            )
            raise ProviderFailureError("Provider could not complete inference") from exc
        self.observability.provider_attempt(
            provider=model.provider,
            model=model.model_id,
            outcome="success",
            error_category="none",
            duration_seconds=perf_counter() - started,
            request_mode=request_mode,
            estimated_cost_usd=response.estimated_cost_usd,
            request_id=request_id,
        )
        return response
