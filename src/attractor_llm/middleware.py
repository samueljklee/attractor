"""Middleware/interceptor chain for the Unified LLM Client.

Middleware intercepts LLM requests and responses, enabling cross-cutting
concerns like logging, caching, token counting, rate limiting, and
request transformation without modifying the core Client or adapters.

Middleware is composable: multiple middleware can be chained, with each
wrapping the next. The chain executes in order for requests (outer to
inner) and reverse order for responses (inner to outer).

Usage::

    from attractor_llm.middleware import (
        Middleware, LoggingMiddleware, TokenCountingMiddleware,
        CachingMiddleware, apply_middleware,
    )

    client = Client()
    client.register_adapter("anthropic", adapter)

    # Apply middleware chain
    client = apply_middleware(client, [
        LoggingMiddleware(),
        TokenCountingMiddleware(),
        CachingMiddleware(max_size=100),
    ])

    # Client works normally -- middleware intercepts transparently
    response = await client.complete(request)

Spec reference: unified-llm-spec S2.3.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

from attractor_llm.types import Request, Response, Usage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Middleware Protocol
# ------------------------------------------------------------------ #


class Middleware(Protocol):
    """Protocol for LLM request/response interceptors."""

    async def before_request(self, request: Request) -> Request:
        """Transform or inspect the request before it's sent.

        Return the (possibly modified) request. Raise to abort the call.
        """
        ...

    async def after_response(self, request: Request, response: Response) -> Response:
        """Transform or inspect the response after it's received.

        Return the (possibly modified) response.
        """
        ...


# ------------------------------------------------------------------ #
# Built-in middleware implementations
# ------------------------------------------------------------------ #


class LoggingMiddleware:
    """Logs all LLM requests and responses.

    Logs model, provider, token usage, latency, and finish reason
    at INFO level. Logs full message content at DEBUG level.
    """

    def __init__(self, logger_name: str = "attractor_llm") -> None:
        self._logger = logging.getLogger(logger_name)
        # Per-request timing keyed by id(request) -- concurrency safe
        self._timings: dict[int, float] = {}

    async def before_request(self, request: Request) -> Request:
        self._logger.info(
            "LLM request: model=%s provider=%s messages=%d tools=%d",
            request.model,
            request.provider or "auto",
            len(request.messages),
            len(request.tools or []),
        )
        self._timings[id(request)] = time.monotonic()
        return request

    async def after_response(self, request: Request, response: Response) -> Response:
        start = self._timings.pop(id(request), time.monotonic())
        duration = time.monotonic() - start
        self._logger.info(
            "LLM response: model=%s finish=%s tokens=%d duration=%.1fs",
            response.model,
            response.finish_reason,
            response.usage.total_tokens,
            duration,
        )
        return response


@dataclass
class TokenCountingMiddleware:
    """Tracks cumulative token usage across all LLM calls.

    Access totals via .total_usage, .call_count, .total_cost_estimate.
    """

    total_usage: Usage = field(default_factory=Usage)
    call_count: int = 0

    # Note: cost estimate uses approximate mid-range rates.
    # Per-model tracking would require storing model per call.

    async def before_request(self, request: Request) -> Request:
        return request

    async def after_response(self, request: Request, response: Response) -> Response:
        self.total_usage = self.total_usage + response.usage
        self.call_count += 1
        return response

    @property
    def total_cost_estimate(self) -> float:
        """Rough cost estimate in USD (very approximate)."""
        # Use a middle-ground estimate
        return (
            self.total_usage.input_tokens * 3.0 / 1_000_000
            + self.total_usage.output_tokens * 15.0 / 1_000_000
        )


class CachingMiddleware:
    """Caches LLM responses for identical requests.

    Cache key is derived from: model + messages + system + tools + temperature.
    Only caches non-tool-call responses (tool calls are non-deterministic).

    Uses LRU eviction with configurable max size.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._cache: OrderedDict[str, Response] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    async def before_request(self, request: Request) -> Request:
        # Store the cache key on the request for lookup in after_response
        request._cache_key = self._make_key(request)  # type: ignore[attr-defined]
        return request

    async def after_response(self, request: Request, response: Response) -> Response:
        key = getattr(request, "_cache_key", None)
        if key and response.finish_reason != "tool_calls":
            self._cache[key] = response
            self._cache.move_to_end(key)
            # Evict oldest if over limit
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        return response

    def lookup(self, request: Request) -> Response | None:
        """Check if a cached response exists for this request."""
        key = self._make_key(request)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def _make_key(self, request: Request) -> str:
        """Create a cache key from request content."""
        key_parts = [
            request.model,
            request.system or "",
            str(request.temperature),
            str(request.reasoning_effort or ""),
        ]
        for msg in request.messages:
            key_parts.append(f"{msg.role}:{msg.text or ''}")
        if request.tools:
            for tool in request.tools:
                key_parts.append(f"tool:{tool.name}")

        raw = "|".join(key_parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class RateLimitMiddleware:
    """Enforces client-side rate limiting.

    Tracks requests per time window and delays if the limit is hit.
    This is separate from provider-side rate limiting (429 errors) --
    it prevents hitting the provider's limit in the first place.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
    ) -> None:
        self._max_rpm = max_requests_per_minute
        self._window: list[float] = []

    async def before_request(self, request: Request) -> Request:
        import asyncio

        now = time.monotonic()
        # Remove entries older than 60 seconds
        self._window = [t for t in self._window if now - t < 60.0]

        if len(self._window) >= self._max_rpm:
            # Wait until the oldest entry expires
            wait_time = 60.0 - (now - self._window[0])
            if wait_time > 0:
                logger.info(
                    "Rate limit: waiting %.1fs (%d/%d requests in window)",
                    wait_time,
                    len(self._window),
                    self._max_rpm,
                )
                await asyncio.sleep(wait_time)

        self._window.append(time.monotonic())
        return request

    async def after_response(self, request: Request, response: Response) -> Response:
        return response


# ------------------------------------------------------------------ #
# Middleware application
# ------------------------------------------------------------------ #


class MiddlewareClient:
    """Wraps a Client with a middleware chain.

    Intercepts complete() calls with before_request/after_response hooks.
    Delegates all other operations to the wrapped client.
    """

    def __init__(self, client: Any, middleware: list[Any]) -> None:
        self._client = client
        self._middleware = middleware

    async def complete(self, request: Request) -> Response:
        """Send request through middleware chain, then to client."""
        # Before: outer to inner
        req = request
        for mw in self._middleware:
            req = await mw.before_request(req)

        # Check cache (if CachingMiddleware is present)
        for mw in self._middleware:
            if isinstance(mw, CachingMiddleware):
                cached = mw.lookup(req)
                if cached is not None:
                    # Still run after_response for accounting
                    resp = cached
                    for mw_after in reversed(self._middleware):
                        resp = await mw_after.after_response(req, resp)
                    return resp
                break

        # Actual LLM call
        response = await self._client.complete(req)

        # After: inner to outer
        resp = response
        for mw in reversed(self._middleware):
            resp = await mw.after_response(req, resp)

        return resp

    async def stream(self, request: Request) -> Any:
        """Stream passes through to the underlying client (no middleware)."""
        return await self._client.stream(request)

    def register_adapter(self, name: str, adapter: Any) -> None:
        self._client.register_adapter(name, adapter)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> MiddlewareClient:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.__aexit__(*args)


def apply_middleware(client: Any, middleware: list[Any]) -> MiddlewareClient:
    """Wrap a Client with a middleware chain.

    Returns a MiddlewareClient that intercepts complete() calls.
    """
    return MiddlewareClient(client, middleware)
