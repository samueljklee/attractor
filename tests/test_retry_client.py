"""Tests for retry_with_policy() and Client routing.

These were identified as the #1 and #2 coverage gaps by the swarm review:
- retry_with_policy: the core reliability engine, previously untested
- Client: the top-level SDK entry point, previously untested
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from attractor_llm.client import Client
from attractor_llm.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    SDKError,
    ServerError,
)
from attractor_llm.retry import RetryPolicy, retry_with_policy
from attractor_llm.types import (
    FinishReason,
    Message,
    Request,
    Response,
    StreamEvent,
    StreamEventKind,
    Usage,
)

# ================================================================== #
# retry_with_policy
# ================================================================== #


class TestRetryWithPolicy:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_policy(fn, RetryPolicy(max_retries=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failure(self):
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServerError("temporary", provider="test")
            return "ok"

        result = await retry_with_policy(fn, RetryPolicy(max_retries=3, initial_delay=0.01))
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_reraises_non_retryable_immediately(self):
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("bad key", provider="test")

        with pytest.raises(AuthenticationError):
            await retry_with_policy(fn, RetryPolicy(max_retries=3))

        assert call_count == 1  # no retries

    @pytest.mark.asyncio
    async def test_exhaustion_raises_last_error(self):
        async def fn() -> str:
            raise ServerError("still failing", provider="test")

        with pytest.raises(ServerError, match="still failing"):
            await retry_with_policy(fn, RetryPolicy(max_retries=2, initial_delay=0.01))

    @pytest.mark.asyncio
    async def test_honors_rate_limit_retry_after(self):
        """retry_after from RateLimitError should be honored."""
        delays: list[float] = []

        async def fn() -> str:
            raise RateLimitError("slow", retry_after=0.05, provider="test")

        async def on_retry(attempt: int, error: SDKError, delay: float) -> None:
            delays.append(delay)

        with pytest.raises(RateLimitError):
            await retry_with_policy(
                fn,
                RetryPolicy(max_retries=1, initial_delay=0.01),
                on_retry=on_retry,
            )

        # The delay should be at least the retry_after value (0.05)
        assert len(delays) == 1
        assert delays[0] >= 0.05

    @pytest.mark.asyncio
    async def test_on_retry_callback_invoked(self):
        attempts: list[int] = []
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServerError("fail", provider="test")
            return "ok"

        async def on_retry(attempt: int, error: SDKError, delay: float) -> None:
            attempts.append(attempt)

        result = await retry_with_policy(
            fn,
            RetryPolicy(max_retries=3, initial_delay=0.01),
            on_retry=on_retry,
        )
        assert result == "ok"
        assert attempts == [0, 1]  # called before retry 1 and 2

    @pytest.mark.asyncio
    async def test_negative_max_retries_raises(self):
        async def fn() -> str:
            return "ok"

        with pytest.raises(ValueError, match="max_retries"):
            await retry_with_policy(fn, RetryPolicy(max_retries=-1))

    @pytest.mark.asyncio
    async def test_zero_retries_no_retry(self):
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise ServerError("fail", provider="test")

        with pytest.raises(ServerError):
            await retry_with_policy(fn, RetryPolicy(max_retries=0))

        assert call_count == 1


# ================================================================== #
# Client routing
# ================================================================== #


class _MockAdapter:
    """Minimal mock adapter for testing Client routing."""

    def __init__(self, provider: str, response_text: str = "mock") -> None:
        self._provider = provider
        self._response_text = response_text

    @property
    def provider_name(self) -> str:
        return self._provider

    async def complete(self, request: Request) -> Response:
        return Response(
            id="mock-id",
            model=request.model,
            provider=self._provider,
            message=Message.assistant(self._response_text),
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=10, output_tokens=5),
        )

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            kind=StreamEventKind.START,
            model=request.model,
            provider=self._provider,
        )
        yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=self._response_text)
        yield StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP)

    async def close(self) -> None:
        pass


class TestClient:
    @pytest.mark.asyncio
    async def test_register_and_route_by_explicit_provider(self):
        client = Client()
        client.register_adapter("anthropic", _MockAdapter("anthropic", "hello"))

        resp = await client.complete(
            Request(model="claude-sonnet-4-5", provider="anthropic", messages=[Message.user("hi")])
        )
        assert resp.text == "hello"
        assert resp.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_route_by_catalog_lookup(self):
        """Model in catalog -> auto-resolves to correct provider."""
        client = Client()
        client.register_adapter("anthropic", _MockAdapter("anthropic", "from-catalog"))

        resp = await client.complete(
            Request(model="claude-sonnet-4-5", messages=[Message.user("hi")])
        )
        assert resp.text == "from-catalog"

    @pytest.mark.asyncio
    async def test_route_by_prefix_heuristic(self):
        """claude-* prefix -> anthropic, gpt-* -> openai, gemini-* -> gemini."""
        client = Client()
        client.register_adapter("anthropic", _MockAdapter("anthropic"))
        client.register_adapter("openai", _MockAdapter("openai"))
        client.register_adapter("gemini", _MockAdapter("gemini"))

        r1 = await client.complete(
            Request(model="claude-unknown-model", messages=[Message.user("x")])
        )
        assert r1.provider == "anthropic"

        r2 = await client.complete(Request(model="gpt-future", messages=[Message.user("x")]))
        assert r2.provider == "openai"

        r3 = await client.complete(Request(model="gemini-future", messages=[Message.user("x")]))
        assert r3.provider == "gemini"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        client = Client()

        with pytest.raises(InvalidRequestError, match="not registered"):
            await client.complete(
                Request(model="x", provider="nonexistent", messages=[Message.user("hi")])
            )

    @pytest.mark.asyncio
    async def test_unresolvable_model_raises(self):
        client = Client()
        # No adapters registered, unknown model name

        with pytest.raises(InvalidRequestError, match="Cannot resolve"):
            await client.complete(Request(model="unknown-model-xyz", messages=[Message.user("hi")]))

    @pytest.mark.asyncio
    async def test_context_manager(self):
        adapter = _MockAdapter("anthropic")
        client = Client()
        client.register_adapter("anthropic", adapter)

        async with client:
            resp = await client.complete(
                Request(model="claude-sonnet-4-5", messages=[Message.user("hi")])
            )
            assert resp.text == "mock"
        # After exit, close was called (no error)

    @pytest.mark.asyncio
    async def test_multiple_adapters(self):
        client = Client()
        client.register_adapter("anthropic", _MockAdapter("anthropic", "from-anthropic"))
        client.register_adapter("openai", _MockAdapter("openai", "from-openai"))

        r1 = await client.complete(Request(model="claude-sonnet-4-5", messages=[Message.user("x")]))
        assert r1.text == "from-anthropic"

        r2 = await client.complete(Request(model="gpt-5.2", messages=[Message.user("x")]))
        assert r2.text == "from-openai"
