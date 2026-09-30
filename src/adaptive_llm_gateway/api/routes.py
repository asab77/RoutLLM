import hmac
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import JSONResponse, Response

from adaptive_llm_gateway.api.limits import RequestLimitSettings
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.application.chat import ChatOrchestrationService, ChatRoutingMode
from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.errors import InvalidQualityThresholdError
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.observability import Observability
from adaptive_llm_gateway.rate_limit import InferenceKind, InferenceRateLimiter
from adaptive_llm_gateway.telemetry.query import ChatActivityQueryService, TelemetryQueryService

from .dependencies import (
    get_adaptive_runtime,
    get_chat_service,
    get_evaluation_service,
    get_observability,
    get_rate_limiter,
    get_request_limits,
    get_service,
)
from .schemas import (
    AdaptiveInferencePayload,
    AdaptiveInferenceResult,
    ActivityItem,
    ActivityPage,
    BenchmarkEvaluationSummary,
    ChatPayload,
    ChatResult,
    ErrorResponse,
    HealthResponse,
    InferencePayload,
    InferenceResult,
    MetricsSummary,
    ModelList,
    PublicExecutionMetadata,
    PublicModel,
    PublicRoutingMetadata,
    ReadyResponse,
)

router = APIRouter()
Service = Annotated[InferenceService, Depends(get_service)]
Evaluation = Annotated[EvaluationService, Depends(get_evaluation_service)]
Adaptive = Annotated[AdaptiveRuntime, Depends(get_adaptive_runtime)]
Chat = Annotated[ChatOrchestrationService, Depends(get_chat_service)]
RateLimiter = Annotated[InferenceRateLimiter, Depends(get_rate_limiter)]
Metrics = Annotated[Observability, Depends(get_observability)]
RequestLimits = Annotated[RequestLimitSettings, Depends(get_request_limits)]


def _effective_client(request: Request) -> str:
    """Use only the peer identity supplied by ASGI/trusted proxy handling."""
    return request.client.host if request.client is not None else "unknown"


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["health"],
    responses={503: {"model": ReadyResponse}},
)
async def ready(
    request: Request,
    service: Service,
    rate_limiter: RateLimiter,
    metrics: Metrics,
) -> ReadyResponse | JSONResponse:
    initialized = bool(request.app.state.initialized)
    try:
        providers_available = bool(service.list_models())
    except Exception:  # noqa: BLE001 - readiness must fail closed for any dependency error
        providers_available = False
    adaptive_available = (
        not request.app.state.adaptive_required
        or request.app.state.adaptive_runtime is not None
    )
    try:
        redis_available = await rate_limiter.ready()
    except Exception:  # noqa: BLE001 - readiness must fail closed for any dependency error
        redis_available = False
    if initialized and providers_available and adaptive_available and redis_available:
        return ReadyResponse(status="ready")
    dependency = (
        "initialization" if not initialized
        else "provider_registry" if not providers_available
        else "adaptive_routing" if not adaptive_available
        else "redis"
    )
    metrics.events.emit(
        "dependency_degraded",
        level=logging.WARNING,
        dependency=dependency,
        outcome="not_ready",
    )
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request, metrics: Metrics) -> Response:
    settings = metrics.settings
    if not settings.metrics_enabled:
        return Response(status_code=404)
    expected = settings.metrics_bearer_token
    if expected is None:
        return Response(status_code=404)
    authorization = request.headers.get("authorization", "")
    candidate = authorization[7:] if authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(candidate, expected.get_secret_value()):
        return Response(
            status_code=401,
            content="Unauthorized",
            media_type="text/plain",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.get("/v1/models", response_model=ModelList, tags=["models"])
async def list_models(service: Service) -> ModelList:
    return ModelList(models=[
        PublicModel(model_id=model.model_id, provider=model.provider,
                    context_window=model.context_window)
        for model in service.list_models()
    ])


@router.post(
    "/v1/inference", response_model=InferenceResult, tags=["inference"],
    responses={status: {"model": ErrorResponse} for status in (400, 403, 404, 422, 429, 502, 503, 504)},
    openapi_extra={"parameters": [{
        "name": "X-Request-ID", "in": "header", "required": False,
        "description": "Optional correlation label; generated UUID4 when omitted.",
        "schema": {"type": "string", "maxLength": 128,
                   "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
    }]},
)
async def inference(
    payload: InferencePayload,
    request: Request,
    service: Service,
    rate_limiter: RateLimiter,
    request_limits: RequestLimits,
) -> InferenceResult:
    request_limits.validate_request(payload)
    deadline = service.new_deadline()
    await rate_limiter.admit(
        _effective_client(request), InferenceKind.EXPLICIT, deadline
    )
    result = await service.generate(
        payload.model_id,
        payload.to_domain(),
        request_id=request.state.request_id,
        deadline=deadline,
    )
    return InferenceResult(**result.model_dump(), request_id=request.state.request_id)


@router.post(
    "/v1/inference/adaptive",
    response_model=AdaptiveInferenceResult,
    response_model_exclude_none=True,
    tags=["inference"],
    responses={status: {"model": ErrorResponse} for status in (400, 403, 404, 422, 429, 502, 503, 504)},
)
async def adaptive_inference(
    payload: AdaptiveInferencePayload,
    request: Request,
    runtime: Adaptive,
    rate_limiter: RateLimiter,
    service: Service,
    request_limits: RequestLimits,
) -> AdaptiveInferenceResult:
    request_limits.validate_request(payload)
    approved_threshold = runtime.approved_quality_threshold
    if (
        payload.quality_threshold is not None
        and payload.quality_threshold != approved_threshold
    ):
        raise InvalidQualityThresholdError("unapproved production threshold")
    deadline = service.new_deadline()
    await rate_limiter.admit(
        _effective_client(request), InferenceKind.ADAPTIVE, deadline
    )
    result = await runtime.service.generate(
        payload.to_domain(),
        **({"validation": payload.validation} if payload.validation is not None else {}),
        category=payload.category,
        quality_threshold=approved_threshold,
        candidate_model_ids=runtime.candidate_model_ids,
        request_id=request.state.request_id,
        deadline=deadline,
    )
    decision = result.routing_decision
    execution = result.execution
    return AdaptiveInferenceResult(
        **result.response.model_dump(),
        request_id=request.state.request_id,
        routing=PublicRoutingMetadata(
            selected_model_id=decision.selected_model_id,
            threshold_satisfied=decision.threshold_satisfied,
            fallback_used=decision.fallback_used,
            reason=decision.reason,
        ),
        execution=(
            PublicExecutionMetadata(
                attempts=len(execution.attempts),
                escalated=execution.escalated,
                validation_outcome="passed",
                total_estimated_cost_usd=execution.total_estimated_cost_usd,
                total_latency_ms=execution.total_latency_ms,
            )
            if execution is not None else None
        ),
    )


@router.post(
    "/v1/chat",
    response_model=ChatResult,
    response_model_exclude_none=True,
    tags=["inference"],
    responses={status: {"model": ErrorResponse} for status in (400, 403, 404, 422, 429, 502, 503, 504)},
)
async def chat_inference(
    payload: ChatPayload,
    request: Request,
    chat: Chat,
    rate_limiter: RateLimiter,
    service: Service,
    request_limits: RequestLimits,
) -> ChatResult:
    request_limits.validate_request(payload)
    deadline = service.new_deadline()
    kind = (
        InferenceKind.EXPLICIT
        if payload.routing_mode is ChatRoutingMode.AUTO
        else InferenceKind.ADAPTIVE
    )
    await rate_limiter.admit(_effective_client(request), kind, deadline)
    result = await chat.generate(
        payload.to_domain(),
        routing_mode=payload.routing_mode,
        category=payload.category,
        validation=payload.validation,
        request_id=request.state.request_id,
        deadline=deadline,
    )
    decision = result.routing_decision
    execution = result.execution
    return ChatResult(
        **result.response.model_dump(),
        request_id=request.state.request_id,
        execution_mode=result.execution_mode,
        category=result.category,
        category_source=result.category_source,
        routing=(
            PublicRoutingMetadata(
                selected_model_id=decision.selected_model_id,
                threshold_satisfied=decision.threshold_satisfied,
                fallback_used=decision.fallback_used,
                reason=decision.reason,
            )
            if decision is not None else None
        ),
        execution=(
            PublicExecutionMetadata(
                attempts=len(execution.attempts),
                escalated=execution.escalated,
                validation_outcome="passed",
                total_estimated_cost_usd=execution.total_estimated_cost_usd,
                total_latency_ms=execution.total_latency_ms,
            )
            if execution is not None else None
        ),
    )


@router.get("/v1/metrics/summary", response_model=MetricsSummary, tags=["metrics"],
            responses={503: {"model": ErrorResponse}})
async def metrics_summary(service: Service) -> MetricsSummary:
    summary = await TelemetryQueryService(service.telemetry, service.telemetry_timeout).summary()
    return MetricsSummary(**summary.model_dump())


@router.get(
    "/v1/activity",
    response_model=ActivityPage,
    tags=["activity"],
    responses={status: {"model": ErrorResponse} for status in (400, 422, 503)},
)
async def activity(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    cursor: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
) -> ActivityPage:
    page = await ChatActivityQueryService(
        service.telemetry, service.telemetry_timeout
    ).page(limit=limit, cursor=cursor)
    return ActivityPage(
        items=[ActivityItem(**item.model_dump()) for item in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/v1/benchmarks/{run_id}/summary", response_model=BenchmarkEvaluationSummary,
            tags=["benchmarks"], responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def benchmark_summary(run_id: UUID, evaluation: Evaluation) -> BenchmarkEvaluationSummary:
    summary = await evaluation.summary(run_id)
    return BenchmarkEvaluationSummary(**summary.model_dump())
