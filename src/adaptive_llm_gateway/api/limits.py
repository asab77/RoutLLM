"""Trusted HTTP admission bounds for the public production API."""

import os

from pydantic import BaseModel, ConfigDict, Field

from adaptive_llm_gateway.errors import RequestLimitExceededError
from adaptive_llm_gateway.models import InferenceRequest


class RequestLimitSettings(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    # Broad development defaults preserve existing context-limit behavior. The
    # production Compose contract sets the documented, conservative demo values.
    max_prompt_characters: int = Field(default=100_000, gt=0, le=100_000)
    max_output_tokens: int = Field(default=8_192, gt=0, le=8_192)

    @classmethod
    def from_environment(cls) -> "RequestLimitSettings":
        return cls(
            max_prompt_characters=os.environ.get(
                "ROUTELLM_MAX_PROMPT_CHARACTERS", "100000"
            ),
            max_output_tokens=os.environ.get(
                "ROUTELLM_MAX_OUTPUT_TOKENS", "8192"
            ),
        )

    def validate_request(self, request: InferenceRequest) -> None:
        total_characters = len(request.prompt) + len(request.system_prompt or "")
        if total_characters > self.max_prompt_characters:
            raise RequestLimitExceededError("prompt_limit_exceeded")
        if request.max_output_tokens > self.max_output_tokens:
            raise RequestLimitExceededError("output_limit_exceeded")


__all__ = ["RequestLimitSettings"]
