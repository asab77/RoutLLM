from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from adaptive_llm_gateway.api.app import create_app
from adaptive_llm_gateway.application.adaptive import (
    AdaptiveAttempt,
    AdaptiveExecution,
    AdaptiveInferenceResult,
)
from adaptive_llm_gateway.application.adaptive_config import AdaptiveRuntime
from adaptive_llm_gateway.bootstrap import create_development_service
from adaptive_llm_gateway.errors import ResponseValidationError
from adaptive_llm_gateway.models import InferenceResponse
from adaptive_llm_gateway.routing.policy import RoutingDecision, RoutingDecisionReason
from adaptive_llm_gateway.telemetry.contracts import (
    ChatActivityOutcome,
    ChatActivityRecord,
    ChatExecutionMode,
    ValidationTelemetry,
)
from adaptive_llm_gateway.validation import ValidationStatus


class MemoryRepository:
    def __init__(self):
        self.activity: list[ChatActivityRecord] = []
        self.attempts = []
        self.last_limit = None

    async def record(self, event):
        self.attempts.append(event)

    async def record_chat_activity(self, event):
        if any(item.request_id == event.request_id for item in self.activity):
            raise RuntimeError("duplicate request")
        self.activity.append(event)

    async def list_chat_activity(self, *, limit, before):
        self.last_limit = limit
        rows = sorted(self.activity, key=lambda item: (item.created_at, item.id), reverse=True)
        if before is not None:
            rows = [item for item in rows if (item.created_at, item.id) < before]
        return rows[:limit], len(rows) > limit


class BrokenActivityRepository(MemoryRepository):
    async def record_chat_activity(self, event):
        raise RuntimeError("private database failure")


def normalized_response(model_id="fake-small", cost="0.00000135", latency=5):
    return InferenceResponse(
        text="private response text",
        model_id=model_id,
        provider="fake",
        input_tokens=3,
        output_tokens=2,
        latency_ms=latency,
        estimated_cost_usd=cost,
    )


def routing(selected="fake-small", fallback=False):
    return RoutingDecision(
        selected_model_id=selected,
        selected_predicted_acceptability=0.7 if fallback else 0.9,
        selected_projected_cost_usd="0.000001",
        quality_threshold=0.8,
        threshold_satisfied=not fallback,
        fallback_used=fallback,
        eligible_candidate_count=2,
        qualifying_candidate_count=0 if fallback else 1,
        reason=(RoutingDecisionReason.NO_MODEL_MET_THRESHOLD_FALLBACK if fallback
                else RoutingDecisionReason.QUALITY_THRESHOLD_MET),
    )


class StubAdaptive:
    def __init__(self, result=None, failure=None):
        self.result = result
        self.failure = failure

    async def generate(self, request, **kwargs):
        if self.failure is not None:
            raise self.failure
        return self.result


def app_with(repository, adaptive=None):
    service = create_development_service()
    service.telemetry = repository
    runtime = AdaptiveRuntime(adaptive, ("fake-small", "fake-large")) if adaptive else None
    return create_app(service, runtime, default_model_id="fake-small")


def test_successful_direct_chat_creates_one_private_request_level_row():
    repository = MemoryRepository()
    with TestClient(app_with(repository)) as client:
        response = client.post(
            "/v1/chat",
            json={"prompt": "private prompt", "routing_mode": "auto"},
            headers={"X-Request-ID": "activity-direct-1"},
        )
    assert response.status_code == 200
    assert len(repository.activity) == 1
    row = repository.activity[0]
    assert row.request_id == "activity-direct-1"
    assert row.execution_mode is ChatExecutionMode.DIRECT
    assert row.category is row.category_source is None
    assert row.initial_model_id is None
    assert row.final_model_id == "fake-small"
    assert row.attempt_count == 1 and row.escalated is False
    assert row.outcome is ChatActivityOutcome.RETURNED
    assert row.cost_complete is True
    serialized = row.model_dump_json()
    assert "private prompt" not in serialized
    assert "private response text" not in serialized
    assert len(repository.attempts) == 1


def test_successful_adaptive_single_attempt_preserves_category_and_fallback():
    repository = MemoryRepository()
    adaptive = StubAdaptive(AdaptiveInferenceResult(
        response=normalized_response(), routing_decision=routing(fallback=True)
    ))
    with TestClient(app_with(repository, adaptive)) as client:
        response = client.post("/v1/chat", json={
            "prompt": "private", "routing_mode": "manual", "category": "coding",
        })
    assert response.status_code == 200
    row = repository.activity[0]
    assert row.execution_mode is ChatExecutionMode.ADAPTIVE
    assert (row.category, row.category_source) == ("coding", "manual")
    assert row.initial_model_id == row.final_model_id == "fake-small"
    assert row.routing_threshold_satisfied is False
    assert row.routing_fallback_used is True
    assert row.attempt_count == 1 and row.escalated is False
    assert row.validation_outcome is None


def test_adaptive_escalation_uses_final_model_and_cumulative_cost_latency():
    repository = MemoryRepository()
    attempts = (
        AdaptiveAttempt(
            attempt_number=1, model_id="fake-small",
            validation=ValidationTelemetry(status="failed", failure_codes=("invalid_json",), duration_ms=1),
            latency_ms=4, estimated_cost_usd="0.001",
        ),
        AdaptiveAttempt(
            attempt_number=2, model_id="fake-large",
            validation=ValidationTelemetry(status="passed", duration_ms=1),
            latency_ms=8, estimated_cost_usd="0.004",
        ),
    )
    adaptive = StubAdaptive(AdaptiveInferenceResult(
        response=normalized_response("fake-large", cost="0.004", latency=8),
        routing_decision=routing("fake-small"),
        execution=AdaptiveExecution(
            attempts=attempts, escalated=True, validation_outcome=ValidationStatus.PASSED,
            total_estimated_cost_usd=Decimal("0.005"), total_latency_ms=12,
        ),
    ))
    with TestClient(app_with(repository, adaptive)) as client:
        response = client.post("/v1/chat", json={
            "prompt": "private", "routing_mode": "manual", "category": "structured_json",
            "validation": {"format": "json"},
        })
    assert response.status_code == 200
    row = repository.activity[0]
    assert (row.initial_model_id, row.final_model_id) == ("fake-small", "fake-large")
    assert row.attempt_count == 2 and row.escalated is True
    assert row.validation_outcome == "passed"
    assert row.estimated_cost_usd == Decimal("0.005")
    assert row.latency_ms == 12 and row.cost_complete is True


def test_defined_adaptive_failure_is_safe_and_invalid_payload_creates_no_row():
    repository = MemoryRepository()
    adaptive = StubAdaptive(failure=ResponseValidationError("private raw failure"))
    with TestClient(app_with(repository, adaptive)) as client:
        failed = client.post("/v1/chat", json={
            "prompt": "private", "routing_mode": "manual", "category": "qa",
        })
        invalid = client.post("/v1/chat", json={
            "prompt": "private", "routing_mode": "manual",
        })
    assert failed.status_code == 502 and invalid.status_code == 422
    assert len(repository.activity) == 1
    row = repository.activity[0]
    assert row.outcome is ChatActivityOutcome.VALIDATION_FAILED
    assert row.error_category == "validation_failed"
    assert row.final_model_id is row.provider is row.attempt_count is None
    assert "private raw failure" not in row.model_dump_json()


def test_activity_write_failure_and_duplicate_do_not_fail_inference():
    broken = BrokenActivityRepository()
    with TestClient(app_with(broken)) as client:
        response = client.post("/v1/chat", json={"prompt": "x", "routing_mode": "auto"})
    assert response.status_code == 200

    repository = MemoryRepository()
    with TestClient(app_with(repository)) as client:
        first = client.post("/v1/chat", json={"prompt": "x", "routing_mode": "auto"}, headers={"X-Request-ID": "same-id"})
        second = client.post("/v1/chat", json={"prompt": "y", "routing_mode": "auto"}, headers={"X-Request-ID": "same-id"})
    assert first.status_code == second.status_code == 200
    assert len(repository.activity) == 1


def activity_record(index: int, when: datetime) -> ChatActivityRecord:
    return ChatActivityRecord(
        request_id=f"request-{index}", execution_mode="direct",
        final_model_id="fake-small", provider="fake", attempt_count=1,
        escalated=False, outcome="returned", input_tokens=1, output_tokens=1,
        latency_ms=1, estimated_cost_usd="0.000001", cost_complete=True,
        created_at=when,
    )


def test_activity_api_is_bounded_newest_first_and_cursor_paginated():
    repository = MemoryRepository()
    now = datetime.now(timezone.utc)
    repository.activity = [activity_record(index, now - timedelta(seconds=index // 2)) for index in range(5)]
    with TestClient(app_with(repository)) as client:
        first = client.get("/v1/activity?limit=2")
        second = client.get("/v1/activity", params={"limit": 2, "cursor": first.json()["next_cursor"]})
        third = client.get("/v1/activity", params={"limit": 2, "cursor": second.json()["next_cursor"]})
    assert first.status_code == second.status_code == third.status_code == 200
    ids = [item["request_id"] for page in (first, second, third) for item in page.json()["items"]]
    assert len(ids) == len(set(ids)) == 5
    assert first.json()["next_cursor"] and second.json()["next_cursor"]
    assert third.json().get("next_cursor") is None
    assert set(first.json()["items"][0]).isdisjoint({"prompt", "system_prompt", "text", "response"})


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "cursor=not-a-cursor"])
def test_activity_api_rejects_invalid_bounds_and_cursor(query):
    with TestClient(app_with(MemoryRepository())) as client:
        response = client.get(f"/v1/activity?{query}")
    assert response.status_code == 422


def test_activity_api_empty_and_database_unavailable_are_distinct():
    repository = MemoryRepository()
    with TestClient(app_with(repository)) as client:
        empty = client.get("/v1/activity")
    assert empty.status_code == 200 and empty.json() == {"items": [], "next_cursor": None}
    assert repository.last_limit == 25

    class BrokenRead(MemoryRepository):
        async def list_chat_activity(self, **kwargs):
            raise RuntimeError("private database URL")

    with TestClient(app_with(BrokenRead())) as client:
        failed = client.get("/v1/activity")
    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "telemetry_unavailable"
    assert "private" not in failed.text
