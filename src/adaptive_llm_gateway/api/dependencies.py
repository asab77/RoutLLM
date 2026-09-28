from fastapi import Request

from adaptive_llm_gateway.application.service import InferenceService
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.errors import AdaptiveRoutingUnavailableError
from adaptive_llm_gateway.evaluation.service import EvaluationService
from adaptive_llm_gateway.rate_limit import InferenceRateLimiter
from adaptive_llm_gateway.observability import Observability


async def get_service(request: Request) -> InferenceService:
    """App-local dependency; replace through create_app or dependency_overrides."""
    return request.app.state.inference_service


async def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation_service


async def get_adaptive_runtime(request: Request) -> AdaptiveRuntime:
    runtime = request.app.state.adaptive_runtime
    if runtime is None:
        raise AdaptiveRoutingUnavailableError(
            "adaptive routing is not configured"
        )
    return runtime


async def get_rate_limiter(request: Request) -> InferenceRateLimiter:
    return request.app.state.rate_limiter


async def get_observability(request: Request) -> Observability:
    return request.app.state.observability
