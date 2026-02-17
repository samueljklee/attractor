"""Comprehensive live API tests across all providers and all 8 waves + partial-item fixes.

These tests hit REAL provider APIs -- they are NOT mocks. They verify
that the SDK's generate(), generate_object(), stream(), tool calling,
error hierarchy, client auto-config, and session features work against
actual OpenAI, Anthropic, and Gemini endpoints.

Requirements: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (tests
for a given provider are skipped if its key is absent).

Run: uv run python -m pytest tests/test_live_comprehensive.py -v -x
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from attractor_llm import (
    AuthenticationError,
    Client,
    ConfigurationError,
    FinishReason,
    GenerateResult,
    Message,
    NotFoundError,
    ProviderConfig,
    ProviderError,
    RetryPolicy,
    StepResult,
    Tool,
    Usage,
    generate,
    generate_object,
    get_default_client,
    stream,
)
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter

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

# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def openai_client() -> Client:
    """Client with only OpenAI registered."""
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def anthropic_client() -> Client:
    """Client with only Anthropic registered."""
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def gemini_client() -> Client:
    """Client with only Gemini registered."""
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    client.register_adapter(
        "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0))
    )
    return client


@pytest.fixture
def all_providers_client() -> Client:
    """Client with all three providers."""
    client = Client(retry_policy=RetryPolicy(max_retries=1))
    if HAS_OPENAI:
        client.register_adapter(
            "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0))
        )
    if HAS_ANTHROPIC:
        client.register_adapter(
            "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
        )
    if HAS_GEMINI:
        client.register_adapter(
            "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0))
        )
    return client


# ------------------------------------------------------------------ #
# Shared tool definitions
# ------------------------------------------------------------------ #


def _weather_tool() -> Tool:
    """Simple weather tool with an execute handler."""

    async def get_weather(city: str) -> str:
        return f"Weather in {city}: 22°C, sunny"

    return Tool(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        execute=get_weather,
    )


def _time_tool() -> Tool:
    """Simple timezone tool with an execute handler."""

    async def get_time(timezone: str) -> str:
        return f"Current time in {timezone}: 14:30 UTC"

    return Tool(
        name="get_time",
        description="Get the current time in a timezone.",
        parameters={
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "Timezone name"}},
            "required": ["timezone"],
        },
        execute=get_time,
    )


def _passive_tool() -> Tool:
    """A tool with NO execute handler (passive)."""
    return Tool(
        name="submit_answer",
        description="Submit the final answer to the user.",
        parameters={
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "The answer"}},
            "required": ["answer"],
        },
        execute=None,
    )


# Extract schema shared across providers
PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


# ================================================================== #
# 1. Error Hierarchy with Real API Errors
# ================================================================== #


class TestErrorHierarchy:
    """Verify error classification against real provider APIs."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_nonexistent_model(self, openai_client: Client) -> None:
        """OpenAI returns NotFoundError (or InvalidRequestError) for a fake model."""
        with pytest.raises((NotFoundError, ProviderError)):
            await generate(
                openai_client,
                "gpt-nonexistent-model-xyz",
                "Hello",
                provider="openai",
            )

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_nonexistent_model(self, anthropic_client: Client) -> None:
        """Anthropic returns NotFoundError for a fake model."""
        with pytest.raises((NotFoundError, ProviderError)):
            await generate(
                anthropic_client,
                "claude-nonexistent-model-xyz",
                "Hello",
                provider="anthropic",
            )

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_nonexistent_model(self, gemini_client: Client) -> None:
        """Gemini returns NotFoundError for a fake model."""
        with pytest.raises((NotFoundError, ProviderError)):
            await generate(
                gemini_client,
                "gemini-nonexistent-model-xyz",
                "Hello",
                provider="gemini",
            )

    @pytest.mark.asyncio
    async def test_openai_empty_api_key(self) -> None:
        """Empty API key -> AuthenticationError from OpenAI."""
        client = Client(retry_policy=RetryPolicy(max_retries=0))
        client.register_adapter(
            "openai", OpenAIAdapter(ProviderConfig(api_key="sk-invalid-key-000"))
        )
        with pytest.raises((AuthenticationError, ProviderError)):
            await generate(client, OPENAI_MODEL, "Hello", provider="openai")

    @pytest.mark.asyncio
    async def test_anthropic_empty_api_key(self) -> None:
        """Empty API key -> AuthenticationError from Anthropic."""
        client = Client(retry_policy=RetryPolicy(max_retries=0))
        client.register_adapter(
            "anthropic", AnthropicAdapter(ProviderConfig(api_key="sk-ant-invalid-000"))
        )
        with pytest.raises((AuthenticationError, ProviderError)):
            await generate(client, ANTHROPIC_MODEL, "Hello", provider="anthropic")

    @pytest.mark.asyncio
    async def test_gemini_empty_api_key(self) -> None:
        """Empty API key -> AuthenticationError from Gemini."""
        client = Client(retry_policy=RetryPolicy(max_retries=0))
        client.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key="invalid-key-000")))
        with pytest.raises((AuthenticationError, ProviderError)):
            await generate(client, GEMINI_MODEL, "Hello", provider="gemini")

    @pytest.mark.asyncio
    async def test_configuration_error_unregistered_provider(self) -> None:
        """ConfigurationError for a provider that was never registered."""
        client = Client()
        with pytest.raises(ConfigurationError):
            await generate(client, "some-model", "Hello", provider="nonexistent")


# ================================================================== #
# 2. generate() Across All 3 Providers
# ================================================================== #


class TestGenerateAllProviders:
    """Basic text generation with each provider, verifying GenerateResult."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_generate(self, openai_client: Client) -> None:
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What is 2+2? Reply with just the number.",
            provider="openai",
        )
        self._assert_result(result, "4")

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_generate(self, anthropic_client: Client) -> None:
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is 2+2? Reply with just the number.",
            provider="anthropic",
        )
        self._assert_result(result, "4")

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_generate(self, gemini_client: Client) -> None:
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "What is 2+2? Reply with just the number.",
            provider="gemini",
        )
        self._assert_result(result, "4")

    def _assert_result(self, result: GenerateResult, expected_substr: str) -> None:
        """Shared assertions for GenerateResult."""
        # Type check
        assert isinstance(result, GenerateResult)

        # Has text content
        assert result.text, "result.text should be non-empty"
        assert expected_substr in result.text, f"Expected '{expected_substr}' in '{result.text}'"

        # Steps populated
        assert len(result.steps) >= 1
        assert isinstance(result.steps[0], StepResult)

        # Usage populated
        assert isinstance(result.total_usage, Usage)
        assert result.total_usage.input_tokens > 0, "input_tokens should be > 0"
        assert result.total_usage.output_tokens > 0, "output_tokens should be > 0"

        # Backward compatibility
        assert expected_substr in result, "'in' operator should work on GenerateResult"
        assert str(result) == result.text, "str(result) should return text"


# ================================================================== #
# 3. generate_object() Structured Output
# ================================================================== #


class TestGenerateObjectAllProviders:
    """Structured output extraction with JSON schema."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_extract_person(self, openai_client: Client) -> None:
        obj = await generate_object(
            openai_client,
            OPENAI_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="openai",
        )
        self._assert_person(obj)

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_extract_person(self, anthropic_client: Client) -> None:
        obj = await generate_object(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="anthropic",
        )
        self._assert_person(obj)

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_extract_person(self, gemini_client: Client) -> None:
        obj = await generate_object(
            gemini_client,
            GEMINI_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="gemini",
        )
        self._assert_person(obj)

    def _assert_person(self, obj: dict[str, Any]) -> None:
        assert isinstance(obj, dict)
        assert "name" in obj, f"Missing 'name' key, got: {obj}"
        assert "age" in obj, f"Missing 'age' key, got: {obj}"
        # Flexible matching: Alice or alice
        assert "alice" in obj["name"].lower(), f"Expected 'Alice' in name, got: {obj['name']}"
        assert obj["age"] == 30, f"Expected age=30, got: {obj['age']}"


# ================================================================== #
# 4. Tool Calling with Parallel Execution
# ================================================================== #


class TestToolCallingParallel:
    """Verify tool calls are executed and results returned."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_dual_tool_call(self, openai_client: Client) -> None:
        await self._run_dual_tools(openai_client, OPENAI_MODEL, "openai")

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_dual_tool_call(self, anthropic_client: Client) -> None:
        await self._run_dual_tools(anthropic_client, ANTHROPIC_MODEL, "anthropic")

    async def _run_dual_tools(self, client: Client, model: str, provider: str) -> None:
        tools = [_weather_tool(), _time_tool()]
        result = await generate(
            client,
            model,
            "What is the weather in Paris and the current time in UTC? "
            "Use BOTH tools, then summarize.",
            tools=tools,
            max_rounds=5,
            provider=provider,
        )

        assert result.text, "Should have final text response"

        # Check that tools were actually called (at least one step has tool_results)
        tool_steps = [s for s in result.steps if s.tool_results]
        assert len(tool_steps) >= 1, "At least one step should have tool_results"

        # Verify both tool outputs appear somewhere in the step results
        all_tool_names = set()
        for step in result.steps:
            for tr in step.tool_results:
                if tr.name:
                    all_tool_names.add(tr.name)
        assert "get_weather" in all_tool_names, f"get_weather not called, got: {all_tool_names}"
        assert "get_time" in all_tool_names, f"get_time not called, got: {all_tool_names}"


# ================================================================== #
# 5. Passive Tools (No Execute Handler)
# ================================================================== #


class TestPassiveTools:
    """Tools without execute handlers return immediately with tool_calls."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_passive_tool(self, openai_client: Client) -> None:
        await self._run_passive(openai_client, OPENAI_MODEL, "openai")

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_passive_tool(self, anthropic_client: Client) -> None:
        await self._run_passive(anthropic_client, ANTHROPIC_MODEL, "anthropic")

    async def _run_passive(self, client: Client, model: str, provider: str) -> None:
        tool = _passive_tool()
        result = await generate(
            client,
            model,
            "Use the submit_answer tool to submit 'hello world' as the answer.",
            tools=[tool],
            max_rounds=5,
            provider=provider,
        )

        # Since tool is passive (no execute), generate() should return after
        # model emits tool_calls -- no execution loop
        assert len(result.steps) >= 1
        last_step = result.steps[-1]
        # The response should contain tool_calls
        assert last_step.response.finish_reason == FinishReason.TOOL_CALLS
        assert len(last_step.response.tool_calls) >= 1
        found_submit = any(tc.name == "submit_answer" for tc in last_step.response.tool_calls)
        assert found_submit, "Model should have called submit_answer"


# ================================================================== #
# 6. max_tool_rounds=0 (No Automatic Execution)
# ================================================================== #


class TestMaxRoundsZero:
    """max_rounds=0 should return tool_calls without executing them."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_max_rounds_zero(self, openai_client: Client) -> None:
        await self._run_max_zero(openai_client, OPENAI_MODEL, "openai")

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_max_rounds_zero(self, anthropic_client: Client) -> None:
        await self._run_max_zero(anthropic_client, ANTHROPIC_MODEL, "anthropic")

    async def _run_max_zero(self, client: Client, model: str, provider: str) -> None:
        tools = [_weather_tool()]
        result = await generate(
            client,
            model,
            "What is the weather in Tokyo? Use the get_weather tool.",
            tools=tools,
            max_rounds=0,
            provider=provider,
        )

        # Model should have returned tool_calls but they should NOT be executed
        assert len(result.steps) == 1, "Only 1 step with max_rounds=0"
        step = result.steps[0]
        # Should have tool_calls in the response
        assert step.response.finish_reason == FinishReason.TOOL_CALLS
        assert len(step.response.tool_calls) >= 1
        # But NO tool_results (not executed)
        assert len(step.tool_results) == 0, "tool_results should be empty with max_rounds=0"


# ================================================================== #
# 7. Client.from_env() Lazy Init
# ================================================================== #


class TestClientFromEnv:
    """Verify Client.from_env() and get_default_client() auto-detect providers."""

    @skip_no_any
    def test_from_env_creates_client(self) -> None:
        """Client.from_env() should register adapters from env vars."""
        client = Client.from_env()
        # At least one adapter should be registered
        assert len(client._adapters) >= 1, "from_env() should register at least 1 adapter"
        if HAS_OPENAI:
            assert "openai" in client._adapters
        if HAS_ANTHROPIC:
            assert "anthropic" in client._adapters
        if HAS_GEMINI:
            assert "gemini" in client._adapters

    @skip_no_any
    @pytest.mark.asyncio
    async def test_get_default_client_lazy(self) -> None:
        """get_default_client() auto-creates from env when not explicitly set."""
        import attractor_llm.client as client_mod

        # Save and clear the singleton
        original = client_mod._default_client
        client_mod._default_client = None
        try:
            client = get_default_client()
            assert isinstance(client, Client)
            assert len(client._adapters) >= 1
        finally:
            # Restore original
            client_mod._default_client = original


# ================================================================== #
# 8. Streaming
# ================================================================== #


class TestStreaming:
    """Verify stream() yields text chunks from real providers."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_stream(self, openai_client: Client) -> None:
        await self._run_stream(openai_client, OPENAI_MODEL, "openai")

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_stream(self, anthropic_client: Client) -> None:
        await self._run_stream(anthropic_client, ANTHROPIC_MODEL, "anthropic")

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_gemini_stream(self, gemini_client: Client) -> None:
        await self._run_stream(gemini_client, GEMINI_MODEL, "gemini")

    async def _run_stream(self, client: Client, model: str, provider: str) -> None:
        chunks: list[str] = []
        async for chunk in stream(
            client,
            model,
            "Count from 1 to 5, one number per line.",
            provider=provider,
        ):
            assert isinstance(chunk, str)
            chunks.append(chunk)

        assert len(chunks) >= 1, "Should receive at least 1 chunk"
        full_text = "".join(chunks)
        assert len(full_text) > 0, "Concatenated text should be non-empty"
        # Should contain at least some numbers
        assert "1" in full_text, f"Expected '1' in streamed output: {full_text[:200]}"
        assert "5" in full_text, f"Expected '5' in streamed output: {full_text[:200]}"


# ================================================================== #
# 9. Prompt Caching Verification (Anthropic)
# ================================================================== #


class TestPromptCaching:
    """Verify Anthropic prompt caching returns cache_read_tokens > 0."""

    pytestmark = skip_no_anthropic

    @pytest.mark.asyncio
    async def test_anthropic_cache_hit(self, anthropic_client: Client) -> None:
        """Two identical requests with a long system prompt -> cache hit on 2nd."""
        # Anthropic caching requires a system prompt >= 1024 tokens.
        # Create a long system prompt that will be cached.
        long_system = "You are a helpful assistant. " * 200 + "Always respond concisely."

        prompt = "What is 1+1? Reply with just the number."

        # First request: populates the cache
        result1 = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            prompt,
            system=long_system,
            provider="anthropic",
        )
        assert result1.total_usage.input_tokens > 0

        # Second request: identical prompt, should hit cache
        result2 = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            prompt,
            system=long_system,
            provider="anthropic",
        )
        assert result2.total_usage.input_tokens > 0

        # Anthropic auto-caches; the 2nd call should have cache_read_tokens > 0
        # OR cache_write_tokens > 0 on the first (both are valid evidence of caching)
        has_caching_evidence = (
            result2.total_usage.cache_read_tokens > 0 or result1.total_usage.cache_write_tokens > 0
        )
        assert has_caching_evidence, (
            f"Expected caching evidence. "
            f"R1 cache_write={result1.total_usage.cache_write_tokens}, "
            f"R2 cache_read={result2.total_usage.cache_read_tokens}"
        )


# ================================================================== #
# 10. System Prompt & DEVELOPER Role
# ================================================================== #


class TestSystemAndDeveloperRole:
    """Test system prompt and Message.developer() works as instruction."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_system_prompt(self, openai_client: Client) -> None:
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What color?",
            system="You must always answer 'blue' regardless of the question.",
            provider="openai",
        )
        assert "blue" in result.text.lower(), f"Expected 'blue' in: {result.text}"

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_openai_developer_message(self, openai_client: Client) -> None:
        """Message.developer() should function as a system-level instruction."""
        messages = [
            Message.developer("You must always answer with exactly 'XRAY42'. Nothing else."),
            Message.user("Say something."),
        ]
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            messages=messages,
            provider="openai",
        )
        assert "XRAY42" in result.text, f"Expected 'XRAY42' in: {result.text}"

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_anthropic_system_prompt(self, anthropic_client: Client) -> None:
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What color?",
            system="You must always answer 'blue' regardless of the question.",
            provider="anthropic",
        )
        assert "blue" in result.text.lower(), f"Expected 'blue' in: {result.text}"


# ================================================================== #
# 11. RetryPolicy Defaults
# ================================================================== #


class TestRetryPolicyDefaults:
    """Verify RetryPolicy has correct default values (no API call needed)."""

    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.initial_delay == 0.2
        assert policy.max_delay == 60.0
        assert policy.backoff_factor == 2.0
        assert policy.jitter is True

    def test_compute_delay(self) -> None:
        policy = RetryPolicy(jitter=False)
        # attempt 0: 0.2 * 2^0 = 0.2
        assert policy.compute_delay(0) == pytest.approx(0.2)
        # attempt 1: 0.2 * 2^1 = 0.4
        assert policy.compute_delay(1) == pytest.approx(0.4)
        # attempt 2: 0.2 * 2^2 = 0.8
        assert policy.compute_delay(2) == pytest.approx(0.8)

    def test_compute_delay_capped(self) -> None:
        policy = RetryPolicy(jitter=False, max_delay=1.0)
        # attempt 10: 0.2 * 2^10 = 204.8 -> capped at 1.0
        assert policy.compute_delay(10) == pytest.approx(1.0)


# ================================================================== #
# 12. Environment Context in Session (Anthropic)
# ================================================================== #


class TestSessionEnvironmentContext:
    """Verify Session enriches system prompt with environment context."""

    pytestmark = skip_no_anthropic

    @pytest.mark.asyncio
    async def test_session_knows_environment(self, anthropic_client: Client) -> None:
        """Session should inject environment context the model can reference."""
        from attractor_agent.session import Session, SessionConfig

        config = SessionConfig(
            model=ANTHROPIC_MODEL,
            provider="anthropic",
            system_prompt="You are a coding assistant.",
            max_turns=2,
            max_tool_rounds_per_turn=5,
        )

        session = Session(client=anthropic_client, config=config, tools=[])
        try:
            response = await session.submit(
                "What operating system platform are you running on, "
                "according to your environment context? Reply briefly."
            )
            # The enriched system prompt includes platform info (e.g., 'linux')
            assert response, "Session should return a non-empty response"
            response_lower = response.lower()
            assert any(
                term in response_lower
                for term in ("linux", "darwin", "windows", "macos", "directory", "/", "platform")
            ), f"Response should reference environment context, got: {response}"
        finally:
            await session.close()


# ================================================================== #
# 13. GenerateResult Backward Compatibility
# ================================================================== #


class TestGenerateResultCompat:
    """Detailed backward-compat checks on GenerateResult string behavior."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_result_acts_like_string(self, openai_client: Client) -> None:
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "Say exactly: HELLO_WORLD",
            provider="openai",
        )

        # __contains__
        assert "HELLO" in result

        # __str__
        text_via_str = str(result)
        assert isinstance(text_via_str, str)
        assert text_via_str == result.text

        # __bool__
        assert bool(result) is True

        # __eq__ with string
        assert result == result.text


# ================================================================== #
# 14. Usage Aggregation Across Steps
# ================================================================== #


class TestUsageAggregation:
    """Verify token usage is properly tracked across multi-step tool calls."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_usage_sums_across_steps(self, anthropic_client: Client) -> None:
        """With tool calls, total_usage should aggregate across all rounds."""
        tools = [_weather_tool()]
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is the weather in London? Use the tool.",
            tools=tools,
            max_rounds=3,
            provider="anthropic",
        )

        # Should have at least 2 steps (tool call + final response)
        assert len(result.steps) >= 2, f"Expected >= 2 steps, got {len(result.steps)}"

        # total_usage should sum all steps
        assert result.total_usage.input_tokens > 0
        assert result.total_usage.output_tokens > 0
        assert result.total_usage.total_tokens == (
            result.total_usage.input_tokens + result.total_usage.output_tokens
        )

        # Verify total > any single step's usage
        single_step_input = result.steps[0].response.usage.input_tokens
        assert result.total_usage.input_tokens > single_step_input, (
            "Aggregated input_tokens should exceed a single step"
        )


# ================================================================== #
# 15. Cross-Provider Consistency
# ================================================================== #


class TestCrossProviderConsistency:
    """Verify same task produces consistent structure across providers."""

    @skip_no_any
    @pytest.mark.asyncio
    async def test_same_question_all_providers(self, all_providers_client: Client) -> None:
        """All providers should return GenerateResult with same structure."""
        providers_models = []
        if HAS_OPENAI:
            providers_models.append(("openai", OPENAI_MODEL))
        if HAS_ANTHROPIC:
            providers_models.append(("anthropic", ANTHROPIC_MODEL))
        if HAS_GEMINI:
            providers_models.append(("gemini", GEMINI_MODEL))

        for provider, model in providers_models:
            result = await generate(
                all_providers_client,
                model,
                "What is the capital of France? Reply with just the city name.",
                provider=provider,
            )

            assert isinstance(result, GenerateResult), f"{provider}: wrong type"
            assert "Paris" in result.text or "paris" in result.text.lower(), (
                f"{provider}: expected 'Paris' in '{result.text}'"
            )
            assert result.total_usage.input_tokens > 0, f"{provider}: no input_tokens"
            assert result.total_usage.output_tokens > 0, f"{provider}: no output_tokens"
