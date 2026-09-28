"""Low-cardinality Prometheus metrics and allowlisted structured events."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


_SAFE_LOG_FIELDS = frozenset({
    "request_id",
    "route",
    "method",
    "status_code",
    "model",
    "provider",
    "attempt_number",
    "routing_fallback",
    "escalated",
    "recovered",
    "validation_outcome",
    "provider_error_category",
    "latency_ms",
    "estimated_cost_usd",
    "terminal_outcome",
    "scope",
    "record_type",
    "outcome",
    "dependency",
})


class ObservabilitySettings(BaseModel):
    """Trusted configuration for the narrowly scoped scrape endpoint."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    metrics_enabled: bool = False
    metrics_bearer_token: SecretStr | None = Field(default=None, min_length=32)

    @model_validator(mode="after")
    def require_token_when_enabled(self) -> "ObservabilitySettings":
        if self.metrics_enabled and self.metrics_bearer_token is None:
            raise ValueError(
                "METRICS_BEARER_TOKEN is required when METRICS_ENABLED is true"
            )
        return self

    @classmethod
    def from_environment(cls) -> "ObservabilitySettings":
        return cls(
            metrics_enabled=os.environ.get("METRICS_ENABLED", "false"),
            metrics_bearer_token=os.environ.get("METRICS_BEARER_TOKEN") or None,
        )


class JsonEventLogger:
    """Emit JSON objects containing only explicitly approved fields."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("routellm.events")

    def emit(self, event: str, *, level: int = logging.INFO, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": logging.getLevelName(level).lower(),
            "event": event,
        }
        for key, value in fields.items():
            if key not in _SAFE_LOG_FIELDS or value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, Decimal):
                value = str(value)
            payload[key] = value
        try:
            self._logger.log(
                level,
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
            )
        except Exception:
            # Observability must never change application behavior.
            return


class Observability:
    """Per-application metrics registry with safe, non-throwing record methods."""

    def __init__(
        self,
        settings: ObservabilitySettings | None = None,
        *,
        registry: CollectorRegistry | None = None,
        event_logger: JsonEventLogger | None = None,
    ) -> None:
        self.settings = settings or ObservabilitySettings()
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.events = event_logger or JsonEventLogger()

        self.http_requests = Counter(
            "routellm_http_requests_total",
            "Completed HTTP requests.",
            ("route", "method", "status_class", "outcome"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "routellm_http_request_duration_seconds",
            "End-to-end HTTP request duration.",
            ("route", "method", "status_class"),
            registry=self.registry,
        )
        self.http_in_progress = Gauge(
            "routellm_http_requests_in_progress",
            "HTTP requests currently executing.",
            ("route",),
            registry=self.registry,
        )
        self.provider_attempts = Counter(
            "routellm_provider_attempts_total",
            "Provider attempts by controlled provider/model and outcome.",
            ("provider", "model", "outcome", "error_category"),
            registry=self.registry,
        )
        self.provider_duration = Histogram(
            "routellm_provider_attempt_duration_seconds",
            "Provider attempt wall-clock duration.",
            ("provider", "model", "outcome"),
            registry=self.registry,
        )
        self.routing_decisions = Counter(
            "routellm_routing_decisions_total",
            "Adaptive routing decisions.",
            ("selected_model", "reason"),
            registry=self.registry,
        )
        self.validation_failures = Counter(
            "routellm_validation_failures_total",
            "Controlled deterministic validation failures.",
            ("model", "reason", "attempt"),
            registry=self.registry,
        )
        self.adaptive_requests = Counter(
            "routellm_adaptive_requests_total",
            "Adaptive requests classified at one terminal point.",
            ("terminal_outcome", "escalated", "recovered"),
            registry=self.registry,
        )
        self.adaptive_attempts = Histogram(
            "routellm_adaptive_attempts",
            "Provider attempt count per terminal adaptive request.",
            ("terminal_outcome",),
            buckets=(1, 2, 3),
            registry=self.registry,
        )
        self.estimated_cost = Counter(
            "routellm_estimated_cost_usd_total",
            "Known estimated cost of provider completions.",
            ("request_mode", "model"),
            registry=self.registry,
        )
        self.telemetry_writes = Counter(
            "routellm_telemetry_writes_total",
            "Durable telemetry write outcomes.",
            ("record_type", "outcome"),
            registry=self.registry,
        )
        self.telemetry_duration = Histogram(
            "routellm_telemetry_write_duration_seconds",
            "Durable telemetry write duration.",
            ("record_type", "outcome"),
            registry=self.registry,
        )
        self.rate_limit_rejections = Counter(
            "routellm_rate_limit_rejections_total",
            "Atomic Redis admission rejections.",
            ("scope", "route"),
            registry=self.registry,
        )

    @staticmethod
    def _safe(operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception:
            return

    def http_started(self, route: str) -> None:
        self._safe(lambda: self.http_in_progress.labels(route=route).inc())

    def http_finished(
        self,
        *,
        route: str,
        method: str,
        status_class: str,
        outcome: str,
        duration_seconds: float,
    ) -> None:
        self._safe(lambda: self.http_requests.labels(
                route=route,
                method=method,
                status_class=status_class,
                outcome=outcome,
            ).inc())
        self._safe(lambda: self.http_duration.labels(
                route=route, method=method, status_class=status_class
            ).observe(duration_seconds))
        self._safe(lambda: self.http_in_progress.labels(route=route).dec())

    def provider_attempt(
        self,
        *,
        provider: str,
        model: str,
        outcome: str,
        error_category: str,
        duration_seconds: float,
        request_mode: str,
        estimated_cost_usd: Decimal | None,
        request_id: str,
    ) -> None:
        self._safe(lambda: self.provider_attempts.labels(
            provider=provider,
            model=model,
            outcome=outcome,
            error_category=error_category,
        ).inc())
        self._safe(lambda: self.provider_duration.labels(
                provider=provider, model=model, outcome=outcome
            ).observe(duration_seconds))
        if estimated_cost_usd is not None:
            self._safe(lambda: self.estimated_cost.labels(
                    request_mode=request_mode, model=model
                ).inc(float(estimated_cost_usd)))
        self.events.emit(
            "provider_attempt_completed",
            request_id=request_id,
            provider=provider,
            model=model,
            outcome=outcome,
            provider_error_category=error_category,
            latency_ms=duration_seconds * 1000,
            estimated_cost_usd=estimated_cost_usd,
        )

    def routing_decision(self, *, selected_model: str, reason: str) -> None:
        self._safe(lambda: self.routing_decisions.labels(
            selected_model=selected_model, reason=reason
        ).inc())

    def validation_failure(self, *, model: str, reason: str, attempt: int) -> None:
        self._safe(lambda: self.validation_failures.labels(
            model=model, reason=reason, attempt=str(attempt)
        ).inc())

    def adaptive_request(
        self,
        *,
        terminal_outcome: str,
        escalated: bool,
        recovered: bool,
        attempts: int,
        request_id: str,
        validation_outcome: str,
    ) -> None:
        labels = {
            "terminal_outcome": terminal_outcome,
            "escalated": str(escalated).lower(),
            "recovered": str(recovered).lower(),
        }
        self._safe(lambda: self.adaptive_requests.labels(**labels).inc())
        self._safe(lambda: self.adaptive_attempts.labels(
                terminal_outcome=terminal_outcome
            ).observe(attempts))
        self.events.emit(
            "adaptive_execution_completed",
            request_id=request_id,
            attempt_number=attempts,
            escalated=escalated,
            recovered=recovered,
            validation_outcome=validation_outcome,
            terminal_outcome=terminal_outcome,
        )

    def telemetry_write(
        self, *, record_type: str, outcome: str, duration_seconds: float | None
    ) -> None:
        self._safe(lambda: self.telemetry_writes.labels(
            record_type=record_type, outcome=outcome
        ).inc())
        if duration_seconds is not None:
            self._safe(lambda: self.telemetry_duration.labels(
                    record_type=record_type, outcome=outcome
                ).observe(duration_seconds))

    def rate_limit_rejection(self, *, scope: str, route: str) -> None:
        self._safe(lambda: self.rate_limit_rejections.labels(
            scope=scope, route=route
        ).inc())
        self.events.emit("rate_limit_rejected", scope=scope, route=route)

    def render(self) -> bytes:
        return generate_latest(self.registry)


__all__ = [
    "JsonEventLogger",
    "Observability",
    "ObservabilitySettings",
]
