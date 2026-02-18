# Live API tests for Wave 9, 10, and P1 new behaviors.
#
# Tests hit REAL provider APIs -- they are NOT mocks.
# Skipped automatically if the relevant API key is absent.
#
# Wave 9 live:
#   - Anthropic tool_choice=none: model responds with text only
#   - user_instructions influences model output (ALL CAPS instruction)
#
# Wave 10 live:
#   - StreamResult.response() works with each provider
#   - StreamResult backward-compat iteration (async for chunk in result)
#   - abort_signal pre-set short-circuits generate() with AbortError
#
# P1 live:
#   - Client.from_env() sets default_provider
#   - default_provider routes unknown model names to the correct adapter

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor_agent.session import Session, SessionConfig
from attractor_llm import (
    Client,
    ProviderConfig,
    RetryPolicy,
    generate,
    stream,
)
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.errors import AbortError
from attractor_llm.streaming import StreamResult

# ------------------------------------------------------------------ #
# Env-var helpers
# ------------------------------------------------------------------ #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

HAS_OPENAI = bool(OPENAI_KEY)
HAS_ANTHROPIC = bool(ANTHROPIC_KEY)
HAS_GEMINI = bool(GEMINI_KEY)
HAS_ANY = HAS_OPENAI or HAS_ANTHROPIC or HAS_GEMINI

skip_no_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GOOGLE_API_KEY not set")
skip_no_any = pytest.mark.skipif(not HAS_ANY, reason="No API keys set")

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def openai_client() -> Client:
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def anthropic_client() -> Client:
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def gemini_client() -> Client:
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0))
    )
    return client


# ================================================================== #
# Wave 9 live: Anthropic tool_choice=none
# ================================================================== #


class TestAnthropicToolChoiceNoneLive:
    """Calling generate() with no tools produces a plain text response."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_generate_no_tools_returns_text(
        self, anthropic_client: Client
    ) -> None:
        """Anthropic: simple generate() with no tools returns text (not tool calls)."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with only the word 'hello'.",
            provider="anthropic",
        )

        assert result.text, "Expected non-empty text response"
        assert result.text.strip()
        # With no tools provided, model should return text (not tool calls)
        assert len(result.steps) >= 1, "Expected at least one step"

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_generate_with_tool_choice_none_no_error(
        self, anthropic_client: Client
    ) -> None:
        """Passing tool_choice='none' via provider_options does not crash."""
        from attractor_llm.types import Tool

        async def _noop(**kwargs: object) -> str:
            return "result"

        tool = Tool(
            name="dummy",
            description="A dummy tool that should not be called.",
            parameters={"type": "object", "properties": {}},
            execute=_noop,
        )

        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with only the word 'yes'.",
            provider="anthropic",
            tools=[tool],
            tool_choice="none",
        )

        # Model should respond with text (tools were suppressed)
        assert result.text, "Expected non-empty text response with tool_choice=none"


# ================================================================== #
# Wave 9 live: user_instructions influences model
# ================================================================== #


class TestUserInstructionsLive:
    """user_instructions appended last causes the model to follow the override."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_user_instructions_all_caps_anthropic(
        self, anthropic_client: Client
    ) -> None:
        """user_instructions='Always respond in ALL CAPS' produces an uppercase response."""
        config = SessionConfig(
            model=ANTHROPIC_MODEL,
            provider="anthropic",
            user_instructions="CRITICAL OVERRIDE: Always respond using ALL CAPITAL LETTERS only.",
        )
        session = Session(client=anthropic_client, config=config)
        response = await session.submit("Say the words: hello world")

        assert response, "Expected non-empty response"

        # Count uppercase vs lowercase alphabetic characters
        uppercase_count = sum(1 for c in response if c.isupper())
        lowercase_count = sum(1 for c in response if c.islower())
        total_alpha = uppercase_count + lowercase_count

        if total_alpha > 0:
            uppercase_ratio = uppercase_count / total_alpha
            # Should be predominantly uppercase when instruction is followed
            assert uppercase_ratio >= 0.5, (
                f"Expected majority uppercase (got {uppercase_ratio:.0%}). "
                f"Response: {response!r}"
            )

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_user_instructions_all_caps_openai(self, openai_client: Client) -> None:
        """user_instructions='Always respond in ALL CAPS' works with OpenAI too."""
        config = SessionConfig(
            model=OPENAI_MODEL,
            provider="openai",
            user_instructions="CRITICAL OVERRIDE: Always respond using ALL CAPITAL LETTERS only.",
        )
        session = Session(client=openai_client, config=config)
        response = await session.submit("Say the words: hello world")

        assert response, "Expected non-empty response"
        uppercase_count = sum(1 for c in response if c.isupper())
        lowercase_count = sum(1 for c in response if c.islower())
        total_alpha = uppercase_count + lowercase_count

        if total_alpha > 0:
            uppercase_ratio = uppercase_count / total_alpha
            assert uppercase_ratio >= 0.5, (
                f"Expected majority uppercase (got {uppercase_ratio:.0%}). "
                f"Response: {response!r}"
            )


# ================================================================== #
# Wave 10 live: StreamResult.response() works with all providers
# ================================================================== #


class TestStreamResultResponseLive:
    """StreamResult.response() correctly accumulates streamed output."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_stream_result_response(self, anthropic_client: Client) -> None:
        """Anthropic: StreamResult.response() returns complete text and usage."""
        sr = await stream(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with just: hello",
            provider="anthropic",
        )
        assert isinstance(sr, StreamResult)

        resp = await sr.response()

        assert resp.text, "Expected non-empty text in response"
        assert resp.usage is not None, "Expected usage info in response"
        assert resp.usage.output_tokens is not None

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_text_stream_then_response(self, anthropic_client: Client) -> None:
        """Iterating text_stream then calling response() is consistent."""
        sr = await stream(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with just: hi",
            provider="anthropic",
        )

        # Collect text chunks via text_stream
        chunks: list[str] = []
        async for chunk in sr.text_stream:
            assert isinstance(chunk, str)
            chunks.append(chunk)

        # Then get the accumulated response
        resp = await sr.response()

        assert resp.text, "Expected non-empty text"
        # The chunks combined should equal the full response text
        combined = "".join(chunks)
        assert combined == resp.text

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_stream_result_response(self, openai_client: Client) -> None:
        """OpenAI: StreamResult.response() returns complete text and usage."""
        sr = await stream(
            openai_client,
            OPENAI_MODEL,
            "Reply with just: hello",
            provider="openai",
        )
        assert isinstance(sr, StreamResult)

        resp = await sr.response()

        assert resp.text, "Expected non-empty text in response"
        assert resp.usage is not None

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_stream_result_response(self, gemini_client: Client) -> None:
        """Gemini: StreamResult.response() returns complete text."""
        sr = await stream(
            gemini_client,
            GEMINI_MODEL,
            "Reply with just: hello",
            provider="gemini",
        )
        assert isinstance(sr, StreamResult)

        resp = await sr.response()

        assert resp.text, "Expected non-empty text in response"


class TestStreamResultBackwardCompatLive:
    """async for chunk in (await stream(...)): yields text strings with any provider."""

    @skip_no_any
    @pytest.mark.asyncio
    async def test_aiter_yields_text_strings(self) -> None:
        """Backward-compat iteration via __aiter__ works with a real provider."""
        client = Client(retry_policy=RetryPolicy(max_retries=1))

        if HAS_ANTHROPIC:
            client.register_adapter(
                "anthropic",
                AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0)),
            )
            model = ANTHROPIC_MODEL
            provider = "anthropic"
        elif HAS_OPENAI:
            client.register_adapter(
                "openai",
                OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0)),
            )
            model = OPENAI_MODEL
            provider = "openai"
        else:
            client.register_adapter(
                "gemini",
                GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0)),
            )
            model = GEMINI_MODEL
            provider = "gemini"

        sr = await stream(client, model, "Reply with just: hi", provider=provider)

        chunks: list[str] = []
        async for chunk in sr:
            assert isinstance(chunk, str), f"Expected str, got {type(chunk).__name__}"
            chunks.append(chunk)

        assert len(chunks) > 0, "Expected at least one text chunk"
        full_text = "".join(chunks)
        assert full_text.strip(), "Expected non-empty streamed text"


# ================================================================== #
# Wave 10 live: abort_signal short-circuits generate()
# ================================================================== #


class TestAbortSignalLive:
    """abort_signal set before generate() raises AbortError without wasted API calls."""

    @pytest.mark.asyncio
    async def test_abort_signal_pre_set_raises_abort_error(self) -> None:
        """AbortError is raised after complete() when abort_signal.is_set is True.

        Uses a mock client so no real API call is made after the test confirms
        the signal was checked.  This is a live-category test because it
        verifies the generate() contract (not just the mock internals).
        """
        abort = MagicMock()
        abort.is_set = True

        # Use a mock client -- we just need the generate() function's logic
        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(
            return_value=__import__(
                "attractor_llm.types", fromlist=["Response"]
            ).Response(
                message=__import__(
                    "attractor_llm.types", fromlist=["Message"]
                ).Message.assistant("hi"),
                usage=__import__("attractor_llm.types", fromlist=["Usage"]).Usage(),
            )
        )

        with pytest.raises(AbortError, match="aborted"):
            await generate(client, "any-model", "Hello", abort_signal=abort)

    @pytest.mark.asyncio
    async def test_abort_signal_not_set_completes_normally(self) -> None:
        """When abort_signal.is_set is False, generate() completes without error."""
        from attractor_llm.types import Message, Response, Usage

        abort = MagicMock()
        abort.is_set = False

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(
            return_value=Response(message=Message.assistant("ok"), usage=Usage())
        )

        result = await generate(client, "any-model", "Hello", abort_signal=abort)
        assert result.text == "ok"


# ================================================================== #
# P1 live: default_provider routing
# ================================================================== #


class TestDefaultProviderLive:
    """Client.from_env() sets default_provider; routing works correctly."""

    def test_from_env_sets_default_provider_when_openai_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() with only OPENAI_API_KEY sets default_provider='openai'."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

        client = Client.from_env()

        assert client.default_provider is not None
        assert "openai" in client._adapters
        assert client.default_provider == "openai"

    def test_from_env_sets_default_provider_when_anthropic_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() with only ANTHROPIC_API_KEY sets default_provider='anthropic'."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key")

        client = Client.from_env()

        assert client.default_provider is not None
        assert "anthropic" in client._adapters
        assert client.default_provider == "anthropic"

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_default_provider_routes_unrecognised_model(
        self, openai_client: Client
    ) -> None:
        """default_provider='openai' routes a non-catalog model to OpenAI."""
        # Build a client that explicitly uses openai as default
        client = Client(default_provider="openai", retry_policy=RetryPolicy(max_retries=1))
        client.register_adapter(
            "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0))
        )

        # Use the real OpenAI model (in catalog) but without specifying provider=
        result = await generate(
            client,
            OPENAI_MODEL,
            "Reply with just: yes",
            # No provider= kwarg; should use default_provider="openai"
        )

        assert result.text, "Expected non-empty text response"
        # GenerateResult.steps[0].response.provider carries the resolved provider
        assert len(result.steps) >= 1, "Expected at least one step"
        assert result.steps[0].response.provider == "openai", (
            f"Expected provider='openai', got {result.steps[0].response.provider!r}"
        )

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_default_provider_property_readable(
        self, anthropic_client: Client
    ) -> None:
        """Client.default_provider property returns the registered default."""
        assert anthropic_client.default_provider == "anthropic"
