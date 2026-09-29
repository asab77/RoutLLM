"""ASGI entry point: uvicorn adaptive_llm_gateway.api.app:app."""

import asyncio
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.routing import Match

from adaptive_llm_gateway.api.limits import RequestLimitSettings
from adaptive_llm_gateway.application.adaptive_config import (
    AdaptiveRoutingConfig,
    AdaptiveRuntime,
    build_adaptive_runtime,
)
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import (
    AdaptiveRoutingUnavailableError,
    ContextLimitError,
    EvaluationArtifactError,
    EvaluationNotFoundError,
    GatewayError,
    GatewayErrorCategory,
    InferenceDeadlineExceededError,
    InvalidQualityThresholdError,
    MissingRoutingCategoryError,
    ModelDisabledError,
    NoEligibleCandidatesError,
    PredictorArtifactError,
    ProviderFailureError,
    ProviderUnavailableError,
    RateLimitExceededError,
    RateLimitUnavailableError,
    RequestLimitExceededError,
    ResponseValidationError,
    UnsupportedPredictorCandidateError,
)
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.observability import Observability, ObservabilitySettings
from adaptive_llm_gateway.rate_limit import (
    DisabledRateLimiter,
    InferenceRateLimiter,
    RedisRateLimiter,
)
from adaptive_llm_gateway.registry import ModelNotFoundError
from adaptive_llm_gateway.runtime import application_service, rate_limiter_service
from adaptive_llm_gateway.telemetry.query import TelemetryUnavailableError

from .routes import router
from .schemas import ErrorDetail, ErrorResponse

_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ERRORS = {
    InferenceDeadlineExceededError: (504, "inference_deadline_exceeded", "The inference deadline was exceeded."),
    RateLimitExceededError: (429, "rate_limit_exceeded", "The inference rate limit was exceeded."),
    RateLimitUnavailableError: (503, "rate_limit_unavailable", "Inference protection is currently unavailable."),
    ResponseValidationError: (502, "response_validation_failed", "No response passed the configured validation checks."),
    AdaptiveRoutingUnavailableError: (503, "adaptive_routing_unavailable", "Adaptive inference is not configured."),
    PredictorArtifactError: (503, "adaptive_routing_unavailable", "Adaptive inference is unavailable."),
    UnsupportedPredictorCandidateError: (503, "adaptive_routing_unavailable", "Adaptive inference is unavailable."),
    NoEligibleCandidatesError: (503, "adaptive_routing_unavailable", "Adaptive inference is unavailable."),
    InvalidQualityThresholdError: (422, "invalid_adaptive_request", "The adaptive routing request is invalid."),
    RequestLimitExceededError: (422, "invalid_request", "The request exceeds the configured production limits."),
    MissingRoutingCategoryError: (422, "invalid_adaptive_request", "The adaptive routing request is invalid."),
    EvaluationNotFoundError: (404, "evaluation_not_found", "The benchmark evaluation was not found."),
    EvaluationArtifactError: (422, "evaluation_artifact_invalid", "The benchmark evaluation artifact is invalid."),
    TelemetryUnavailableError: (503, "telemetry_unavailable", "Telemetry is currently unavailable."),
    ModelNotFoundError: (404, "model_not_found", "The requested model was not found."),
    ModelDisabledError: (403, "model_disabled", "The requested model is disabled."),
    ContextLimitError: (422, "context_limit_exceeded", "Input plus requested output exceeds the context window."),
    ProviderUnavailableError: (503, "provider_unavailable", "No adapter is available for this model."),
    ProviderFailureError: (502, "provider_failure", "The provider could not complete inference."),
}


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message),
                         request_id=request.state.request_id)
    return JSONResponse(status_code=status, content=body.model_dump(),
                        headers={"X-Request-ID": request.state.request_id})


def _matched_route(routes: list, scope: dict) -> str:
    """Resolve a controlled route template, including FastAPI included routers."""
    for candidate in routes:
        original_router = getattr(candidate, "original_router", None)
        if original_router is not None:
            nested = _matched_route(original_router.routes, scope)
            if nested != "unmatched":
                return nested
            continue
        match, _ = candidate.matches(scope)
        if match is Match.FULL:
            return getattr(candidate, "path", "unmatched")
    return "unmatched"


def create_app(
    service: InferenceService | None = None,
    adaptive_runtime: AdaptiveRuntime | None = None,
    rate_limiter: InferenceRateLimiter | None = None,
    observability: Observability | None = None,
    request_limits: RequestLimitSettings | None = None,
) -> FastAPI:
    configured_observability = observability or Observability(
        ObservabilitySettings.from_environment()
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is not None:
            application.state.initialized = True
            try:
                yield
            finally:
                application.state.initialized = False
        else:
            try:
                service_context = application_service(configured_observability)
            except TypeError:
                # Preserve the existing zero-argument injectable test boundary.
                service_context = application_service()
            async with (
                service_context as configured_service,
                rate_limiter_service(configured_observability) as configured_rate_limiter,
            ):
                configured_service.observability = configured_observability
                if isinstance(configured_rate_limiter, RedisRateLimiter):
                    configured_rate_limiter.observability = configured_observability
                application.state.inference_service = configured_service
                application.state.rate_limiter = configured_rate_limiter
                config = AdaptiveRoutingConfig.from_environment()
                application.state.adaptive_required = config.enabled
                application.state.adaptive_runtime = build_adaptive_runtime(
                    configured_service, config
                )
                application.state.initialized = True
                try:
                    yield
                finally:
                    application.state.initialized = False

    app = FastAPI(
        title="Adaptive LLM Gateway",
        version="0.5.5",
        description="Explicit inference plus optional configured adaptive routing through one provider-execution path.",
        lifespan=lifespan,
    )
    app.state.inference_service = service if service is not None else create_development_service()
    app.state.inference_service.observability = configured_observability
    app.state.adaptive_runtime = adaptive_runtime
    app.state.adaptive_required = adaptive_runtime is not None
    app.state.rate_limiter = rate_limiter or DisabledRateLimiter()
    app.state.observability = configured_observability
    app.state.request_limits = request_limits or RequestLimitSettings.from_environment()
    app.state.initialized = False
    app.state.evaluation_service = EvaluationService(
        Path(os.environ.get("BENCHMARK_RESULTS_DIR", "benchmark-results")))

    @app.middleware("http")
    async def correlation_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        route = _matched_route(request.app.router.routes, request.scope)
        method = request.method.upper()
        started = perf_counter()
        configured_observability.http_started(route)
        supplied_ids = request.headers.getlist("x-request-id")
        request.state.request_id = str(uuid4())
        status_code = 500
        status_class = "5xx"
        outcome = "server_error"
        cancelled = False
        try:
            if supplied_ids:
                if len(supplied_ids) != 1 or not _REQUEST_ID.fullmatch(supplied_ids[0]):
                    response = error_response(
                        request,
                        400,
                        "invalid_request_id",
                        "X-Request-ID must contain 1-128 ASCII letters, digits, dots, underscores or hyphens, starting with a letter or digit.",
                    )
                else:
                    request.state.request_id = supplied_ids[0]
                    response = await call_next(request)
            else:
                response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            status_code = response.status_code
            status_class = f"{status_code // 100}xx"
            outcome = (
                "success" if status_code < 400
                else "client_error" if status_code < 500
                else "server_error"
            )
            return response
        except asyncio.CancelledError:
            cancelled = True
            status_code = 499
            status_class = "cancelled"
            outcome = "cancelled"
            raise
        finally:
            duration = perf_counter() - started
            configured_observability.http_finished(
                route=route,
                method=method,
                status_class=status_class,
                outcome=outcome,
                duration_seconds=duration,
            )
            configured_observability.events.emit(
                "request_cancelled" if cancelled else "http_request_completed",
                request_id=request.state.request_id,
                route=route,
                method=method,
                status_code=status_code,
                outcome=outcome,
                latency_ms=duration * 1000,
            )

    async def application_error(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, GatewayError):
            status = {GatewayErrorCategory.NOT_CONFIGURED: 503, GatewayErrorCategory.RATE_LIMIT: 429,
                      GatewayErrorCategory.TIMEOUT: 504, GatewayErrorCategory.CONTEXT_LIMIT: 422,
                      GatewayErrorCategory.INVALID_REQUEST: 422}.get(exc.category, 502)
            return error_response(request, status, exc.category.value, "The gateway request could not be completed.")
        # Resolve subclasses as well as the explicitly registered error types.
        for error_type, (status, code, message) in _ERRORS.items():
            if isinstance(exc, error_type):
                response = error_response(request, status, code, message)
                if isinstance(exc, RateLimitExceededError):
                    response.headers["Retry-After"] = str(exc.retry_after_seconds)
                return response
        return error_response(request, 500, "internal_error", "An internal error occurred.")

    for error_type in _ERRORS:
        app.add_exception_handler(error_type, application_error)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Avoid echoing prompts, supplied input, or Pydantic exception context.
        return error_response(request, 422, "invalid_request", "Request body does not match the inference schema.")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        response = error_response(request, exc.status_code, "http_error", "The HTTP request could not be completed.")
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        return error_response(request, 500, "internal_error", "An internal error occurred.")

    app.include_router(router)
    return app


app = create_app()
