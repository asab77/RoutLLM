from abc import ABC, abstractmethod

from adaptive_llm_gateway.models import InferenceRequest, InferenceResponse


class LLMProvider(ABC):
    """Adapter bound to one model; concrete adapters own configuration.

    Implementations translate requests and normalize usage, timing, and cost.
    A future caller selects a configured adapter before invoking generation.
    """

    @abstractmethod
    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Generate a provider-independent result for one request."""
        raise NotImplementedError

    async def generate_with_timeout(
        self,
        request: InferenceRequest,
        *,
        timeout_seconds: float,
    ) -> InferenceResponse:
        """Invoke a provider with a caller budget when the adapter supports it.

        The application layer still enforces the same timeout around this call.
        This default preserves compatibility for providers without HTTP-specific
        timeout controls.
        """
        return await self.generate(request)
