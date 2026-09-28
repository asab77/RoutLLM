"""Monotonic whole-request inference deadline and trusted configuration."""

from collections.abc import Callable
import math
import os
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field


class InferenceDeadlineSettings(BaseModel):
    """Validated process configuration for one total inference budget."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    seconds: float = Field(default=60.0, gt=0, le=300, allow_inf_nan=False)

    @classmethod
    def from_environment(cls) -> "InferenceDeadlineSettings":
        return cls(seconds=os.environ.get("INFERENCE_DEADLINE_SECONDS", "60"))


class RequestDeadline:
    """One immutable end point measured using a monotonic clock."""

    def __init__(
        self,
        total_seconds: float,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(total_seconds, bool) or not isinstance(total_seconds, (int, float)):
            raise ValueError("inference deadline must be a finite positive number")
        seconds = float(total_seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("inference deadline must be a finite positive number")
        self._clock = clock
        self._expires_at = clock() + seconds

    def remaining_seconds(self) -> float:
        """Returns the remaining time in seconds, or zero if expired."""
        return max(0.0, self._expires_at - self._clock())

    def expired(self) -> bool:
        """Checks if the deadline has expired."""
        return self.remaining_seconds() <= 0

    def constrain_timeout(self, max_timeout: float) -> float | None:
        """Returns the smaller of the remaining time or the given max timeout."""
        remaining = self.remaining_seconds()
        if remaining <= 0:
            return None
        return min(remaining, max_timeout)


__all__ = ["InferenceDeadlineSettings", "RequestDeadline"]
