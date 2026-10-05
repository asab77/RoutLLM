import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from adaptive_llm_gateway.models import InferenceRequest
from adaptive_llm_gateway.models.schemas import DomainModel
from adaptive_llm_gateway.routing.features import (
    approximate_input_tokens,
    constraint_indicator_count,
    reasoning_indicator_count,
    request_text,
    text_contains_code,
)


class RequestFeatures(DomainModel):
    category: str | None = None
    prompt_characters: int = Field(ge=0)
    system_prompt_characters: int = Field(ge=0)
    approximate_input_tokens: int = Field(ge=0)
    contains_code: bool
    requests_structured_output: bool
    max_output_tokens: int = Field(gt=0)
    constraint_indicator_count: int = Field(ge=0)
    reasoning_indicator_count: int = Field(ge=0)


def extract_request_features(request: InferenceRequest) -> RequestFeatures:
    """Derive only information available before model selection."""
    text = request_text(request)
    output_type = getattr(request, "expected_output_type", "text")
    return RequestFeatures(
        category=getattr(request, "category", None),
        prompt_characters=len(request.prompt),
        system_prompt_characters=len(request.system_prompt or ""),
        approximate_input_tokens=approximate_input_tokens(request),
        contains_code=text_contains_code(text),
        requests_structured_output=output_type in {"json", "code"},
        max_output_tokens=request.max_output_tokens,
        constraint_indicator_count=constraint_indicator_count(text),
        reasoning_indicator_count=reasoning_indicator_count(text),
    )


FEATURE_BINDING_SCHEMA_VERSION = "1.0.0"


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def feature_extractor_sha256() -> str:
    """Identify the exact implementation that produced persisted features."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_request_feature_binding(request: InferenceRequest,
                                  snapshot: RequestFeatures | dict[str, Any]) -> dict[str, str]:
    """Bind a pre-routing feature snapshot to its exact request and extractor."""
    task_id = getattr(request, "task_id", None)
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Feature binding requires a stable task ID")
    request_payload = request.model_dump(mode="json")
    snapshot_payload = (snapshot.model_dump(mode="json")
                        if isinstance(snapshot, RequestFeatures) else snapshot)
    binding = {
        "schema_version": FEATURE_BINDING_SCHEMA_VERSION,
        "task_id": task_id,
        "task_content_sha256": _canonical_digest(request_payload),
        "feature_extractor_sha256": feature_extractor_sha256(),
        "feature_snapshot_sha256": _canonical_digest(snapshot_payload),
    }
    binding["binding_sha256"] = _canonical_digest(binding)
    return binding


def verify_request_feature_binding(request: InferenceRequest,
                                   snapshot: RequestFeatures | dict[str, Any],
                                   binding: dict[str, Any]) -> None:
    expected = build_request_feature_binding(request, snapshot)
    if binding != expected:
        raise ValueError(f"request-feature integrity mismatch for {expected['task_id']}")
