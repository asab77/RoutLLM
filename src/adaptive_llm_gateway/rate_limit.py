"""Redis-backed admission control for paid inference operations."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import math
import os
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from redis.asyncio import Redis
from redis.exceptions import RedisError

from adaptive_llm_gateway.errors import (
    InferenceDeadlineExceededError,
    RateLimitExceededError,
    RateLimitUnavailableError,
)
from adaptive_llm_gateway.observability import Observability
from adaptive_llm_gateway.request_deadline import RequestDeadline


# One script makes the client and global decision without partially consuming
# either fixed window. Windows begin on their first admitted request.
_ADMIT_SCRIPT = """
local weight = tonumber(ARGV[1])
local client_capacity = tonumber(ARGV[2])
local client_window_ms = tonumber(ARGV[3])
local global_capacity = tonumber(ARGV[4])
local global_window_ms = tonumber(ARGV[5])

local client_used = tonumber(redis.call('GET', KEYS[1]) or '0')
local global_used = tonumber(redis.call('GET', KEYS[2]) or '0')
local client_blocked = client_used + weight > client_capacity
local global_blocked = global_used + weight > global_capacity

if client_blocked or global_blocked then
  local retry_ms = 0
  local scope = 0
  if client_blocked then
    local ttl = redis.call('PTTL', KEYS[1])
    if ttl < 1 then ttl = client_window_ms end
    retry_ms = ttl
    scope = 1
  end
  if global_blocked then
    local ttl = redis.call('PTTL', KEYS[2])
    if ttl < 1 then ttl = global_window_ms end
    if ttl > retry_ms then retry_ms = ttl end
    if scope == 1 then scope = 3 else scope = 2 end
  end
  return {0, retry_ms, scope}
end

local new_client = redis.call('INCRBY', KEYS[1], weight)
if redis.call('PTTL', KEYS[1]) < 0 then
  redis.call('PEXPIRE', KEYS[1], client_window_ms)
end
local new_global = redis.call('INCRBY', KEYS[2], weight)
if redis.call('PTTL', KEYS[2]) < 0 then
  redis.call('PEXPIRE', KEYS[2], global_window_ms)
end
return {1, 0, client_capacity - new_client, global_capacity - new_global}
"""


class InferenceKind(StrEnum):
    EXPLICIT = "explicit"
    ADAPTIVE = "adaptive"


class RateLimitSettings(BaseModel):
    """Trusted admission configuration; defaults are development placeholders."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    required: bool = False
    redis_url: SecretStr | None = None
    hmac_secret: SecretStr | None = None
    redis_timeout_seconds: float = Field(default=0.25, gt=0, le=5, allow_inf_nan=False)
    client_capacity: int = Field(default=10, gt=0, le=1_000_000)
    client_window_seconds: int = Field(default=60, gt=0, le=86_400)
    global_capacity: int = Field(default=100, gt=0, le=10_000_000)
    global_window_seconds: int = Field(default=60, gt=0, le=86_400)
    explicit_weight: int = Field(default=1, gt=0, le=1_000_000)
    adaptive_weight: int = Field(default=2, gt=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_required_settings(self) -> "RateLimitSettings":
        if self.required:
            if self.redis_url is None or self.hmac_secret is None:
                raise ValueError(
                    "REDIS_URL and RATE_LIMIT_HMAC_SECRET are required when rate limiting is required"
                )
        if self.redis_url is not None:
            parsed = urlsplit(self.redis_url.get_secret_value())
            if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
                raise ValueError("REDIS_URL must be a redis:// or rediss:// URL with a host")
        if self.hmac_secret is not None:
            if len(self.hmac_secret.get_secret_value()) < 32:
                raise ValueError("RATE_LIMIT_HMAC_SECRET must contain at least 32 characters")
        if self.explicit_weight > min(self.client_capacity, self.global_capacity):
            raise ValueError("explicit weight must fit both configured capacities")
        if self.adaptive_weight > min(self.client_capacity, self.global_capacity):
            raise ValueError("adaptive weight must fit both configured capacities")
        return self

    @classmethod
    def from_environment(cls) -> "RateLimitSettings":
        return cls(
            required=os.environ.get("RATE_LIMIT_REQUIRED", "false"),
            redis_url=os.environ.get("REDIS_URL") or None,
            hmac_secret=os.environ.get("RATE_LIMIT_HMAC_SECRET") or None,
            redis_timeout_seconds=os.environ.get("REDIS_TIMEOUT_SECONDS", "0.25"),
            client_capacity=os.environ.get("RATE_LIMIT_CLIENT_CAPACITY", "10"),
            client_window_seconds=os.environ.get("RATE_LIMIT_CLIENT_WINDOW_SECONDS", "60"),
            global_capacity=os.environ.get("RATE_LIMIT_GLOBAL_CAPACITY", "100"),
            global_window_seconds=os.environ.get("RATE_LIMIT_GLOBAL_WINDOW_SECONDS", "60"),
            explicit_weight=os.environ.get("RATE_LIMIT_EXPLICIT_WEIGHT", "1"),
            adaptive_weight=os.environ.get("RATE_LIMIT_ADAPTIVE_WEIGHT", "2"),
        )


class InferenceRateLimiter(Protocol):
    async def admit(
        self, client_identity: str, kind: InferenceKind, deadline: RequestDeadline
    ) -> None: ...

    async def ready(self) -> bool: ...


def normalize_client_identity(identity: str) -> str:
    """Normalize the ASGI-provided peer identity without consulting headers."""
    value = identity.strip()
    if not value:
        return "unknown"
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        # ASGI test transports and Unix-socket deployments may expose a name.
        return value.casefold()


def client_identity_digest(secret: str, identity: str) -> str:
    normalized = normalize_client_identity(identity)
    return hmac.new(
        secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class DisabledRateLimiter:
    """Explicit development mode; never used when protection is required."""

    async def admit(
        self, client_identity: str, kind: InferenceKind, deadline: RequestDeadline
    ) -> None:
        if deadline.expired():
            raise InferenceDeadlineExceededError("inference_deadline_exceeded")

    async def ready(self) -> bool:
        return True


class RedisRateLimiter:
    """Atomic fixed-window client and deployment-wide admission control."""

    def __init__(
        self,
        client: Redis,
        settings: RateLimitSettings,
        observability: Observability | None = None,
    ) -> None:
        if not settings.required or settings.hmac_secret is None:
            raise ValueError("RedisRateLimiter requires enabled, complete settings")
        self._client = client
        self._settings = settings
        self._secret = settings.hmac_secret.get_secret_value()
        self.observability = observability or Observability()
        # Hash tag keeps both keys in one Redis Cluster slot.
        self._key_prefix = "routellm:rate:{admission}"

    async def admit(
        self, client_identity: str, kind: InferenceKind, deadline: RequestDeadline
    ) -> None:
        timeout = deadline.constrain_timeout(self._settings.redis_timeout_seconds)
        if timeout is None:
            raise InferenceDeadlineExceededError("inference_deadline_exceeded")
        weight = (
            self._settings.explicit_weight
            if kind is InferenceKind.EXPLICIT
            else self._settings.adaptive_weight
        )
        digest = client_identity_digest(self._secret, client_identity)
        keys = (
            f"{self._key_prefix}:client:{digest}",
            f"{self._key_prefix}:global",
        )
        try:
            async with asyncio.timeout(timeout):
                result = await self._client.eval(
                    _ADMIT_SCRIPT,
                    2,
                    *keys,
                    weight,
                    self._settings.client_capacity,
                    self._settings.client_window_seconds * 1000,
                    self._settings.global_capacity,
                    self._settings.global_window_seconds * 1000,
                )
        except TimeoutError:
            if deadline.expired():
                raise InferenceDeadlineExceededError(
                    "inference_deadline_exceeded"
                ) from None
            raise RateLimitUnavailableError("rate_limit_unavailable") from None
        except RedisError:
            raise RateLimitUnavailableError("rate_limit_unavailable") from None

        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RateLimitUnavailableError("rate_limit_unavailable")
        try:
            admitted = int(result[0])
            retry_ms = int(result[1])
        except (TypeError, ValueError, OverflowError):
            raise RateLimitUnavailableError("rate_limit_unavailable") from None
        if admitted not in {0, 1}:
            raise RateLimitUnavailableError("rate_limit_unavailable")
        if admitted == 0:
            try:
                scope_code = int(result[2])
            except (IndexError, TypeError, ValueError, OverflowError):
                raise RateLimitUnavailableError("rate_limit_unavailable") from None
            scope = {1: "client", 2: "global", 3: "both"}.get(scope_code)
            if scope is None:
                raise RateLimitUnavailableError("rate_limit_unavailable")
            retry_after = max(1, math.ceil(retry_ms / 1000))
            route = (
                "/v1/inference"
                if kind is InferenceKind.EXPLICIT
                else "/v1/inference/adaptive"
            )
            self.observability.rate_limit_rejection(scope=scope, route=route)
            raise RateLimitExceededError(
                retry_after_seconds=retry_after, scope=scope
            )

    async def ready(self) -> bool:
        try:
            async with asyncio.timeout(self._settings.redis_timeout_seconds):
                return bool(await self._client.ping())
        except (TimeoutError, RedisError):
            return False


def create_redis_client(settings: RateLimitSettings) -> Redis:
    if settings.redis_url is None:
        raise ValueError("REDIS_URL is required")
    return Redis.from_url(
        settings.redis_url.get_secret_value(),
        socket_connect_timeout=settings.redis_timeout_seconds,
        socket_timeout=settings.redis_timeout_seconds,
        retry_on_timeout=False,
    )


__all__ = [
    "DisabledRateLimiter",
    "InferenceKind",
    "InferenceRateLimiter",
    "RateLimitSettings",
    "RedisRateLimiter",
    "client_identity_digest",
    "create_redis_client",
    "normalize_client_identity",
]
