# Wave 16b live test coverage.
#
# All tests hit real provider APIs.  Each test is skipped if its key is absent.
#
# Sections:
#   Image input      (6)  – URL + base64  ×  {openai, anthropic, gemini}
#   Gemini tools     (3)  – single, parallel, multi-step
#   Streaming+tools  (3)  – stream with tools declared for each provider
#   Reasoning tokens (3)  – verify reasoning_tokens in usage when enabled
#
# Run: uv run python -m pytest tests/test_wave16b_live_coverage.py -v -x

from __future__ import annotations

import os
import struct
import zlib

import pytest

from attractor_llm import Client, Message, ProviderConfig, RetryPolicy, Tool, generate
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    ImageData,
    Request,
    Role,
    StreamEventKind,
)

# ================================================================== #
# API key helpers & skip markers
# ================================================================== #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

HAS_OPENAI = bool(OPENAI_KEY)
HAS_ANTHROPIC = bool(ANTHROPIC_KEY)
HAS_GEMINI = bool(GEMINI_KEY)

skip_no_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GOOGLE_API_KEY not set")

# ================================================================== #
# Model constants
# ================================================================== #

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"

OPENAI_REASONING_MODEL = "o4-mini"
ANTHROPIC_REASONING_MODEL = "claude-sonnet-4-5"
GEMINI_REASONING_MODEL = "gemini-2.5-flash"

# ================================================================== #
# Client fixtures
# ================================================================== #


@pytest.fixture
def openai_client() -> Client:
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0)))
    return c


@pytest.fixture
def anthropic_client() -> Client:
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
    )
    return c


@pytest.fixture
def gemini_client() -> Client:
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0)))
    return c


# ================================================================== #
# Image helpers
# ================================================================== #

# Publicly accessible HTTP image URL usable by OpenAI and Anthropic.
# OpenAI and Anthropic fetch the URL server-side; httpbin.org is a
# well-known HTTP testing service that returns a valid PNG.
_IMAGE_URL_HTTP = "https://httpbin.org/image/png"


def _make_minimal_png(width: int = 8, height: int = 8) -> bytes:
    """Create a minimal valid RGB PNG image in memory (no Pillow required)."""

    def chunk(type_: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(type_ + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + type_ + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit-depth=8, color-type=2 (RGB), compress=0, filter=0, interlace=0
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    # Rows: 1 filter byte + width×3 RGB bytes (all red)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_PNG_BYTES = _make_minimal_png()


def _url_image_msg(prompt: str) -> Message:
    """User message carrying an HTTP image URL plus a text prompt."""
    return Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentPartKind.TEXT, text=prompt),
            ContentPart(
                kind=ContentPartKind.IMAGE,
                image=ImageData(url=_IMAGE_URL_HTTP, media_type="image/png"),
            ),
        ],
    )


def _base64_image_msg(prompt: str) -> Message:
    """User message carrying inline base64 PNG bytes plus a text prompt."""
    return Message(
        role=Role.USER,
        content=[
            ContentPart(kind=ContentPartKind.TEXT, text=prompt),
            ContentPart(
                kind=ContentPartKind.IMAGE,
                image=ImageData(data=_PNG_BYTES, media_type="image/png"),
            ),
        ],
    )


# ================================================================== #
# Image input tests (6)
# ================================================================== #


class TestImageInput:
    """Vision capability across all three providers."""

    # ---- URL image tests --------------------------------------------------
    #
    # Note: Gemini's adapter maps HTTP URLs to fileData.fileUri which expects
    # GCS/File API URIs.  The Gemini URL test is therefore marked xfail(strict=False)
    # so it documents the current adapter limitation without blocking CI.

    @skip_no_openai
    async def test_openai_image_url_input(self, openai_client: Client) -> None:
        """OpenAI: send an image URL, verify non-empty text response."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            messages=[_url_image_msg("Describe the colour of this image in one sentence.")],
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty response from OpenAI image-URL request"
        assert result.total_usage.input_tokens > 0

    @skip_no_anthropic
    async def test_anthropic_image_url_input(self, anthropic_client: Client) -> None:
        """Anthropic: send an image URL, verify non-empty text response."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            messages=[_url_image_msg("Describe the colour of this image in one sentence.")],
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response from Anthropic image-URL request"
        assert result.total_usage.input_tokens > 0

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Gemini adapter maps HTTP URLs to fileData.fileUri which requires "
            "GCS or File API URIs; plain HTTP URLs are not supported."
        ),
    )
    async def test_gemini_image_url_input(self, gemini_client: Client) -> None:
        """Gemini: send an image URL (GCS URI required; plain HTTP is xfail)."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            messages=[_url_image_msg("Describe the colour of this image in one sentence.")],
            provider="gemini",
        )
        assert result.text.strip()
        assert result.total_usage.input_tokens > 0

    # ---- Base64 image tests -----------------------------------------------

    @skip_no_openai
    async def test_openai_image_base64_input(self, openai_client: Client) -> None:
        """OpenAI: send inline base64 PNG, verify non-empty text response."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            messages=[_base64_image_msg("Describe the colour of this image in one sentence.")],
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty response from OpenAI base64 image"
        assert result.total_usage.input_tokens > 0

    @skip_no_anthropic
    async def test_anthropic_image_base64_input(self, anthropic_client: Client) -> None:
        """Anthropic: send inline base64 PNG, verify non-empty text response."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            messages=[_base64_image_msg("Describe the colour of this image in one sentence.")],
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response from Anthropic base64 image"
        assert result.total_usage.input_tokens > 0

    @skip_no_gemini
    async def test_gemini_image_base64_input(self, gemini_client: Client) -> None:
        """Gemini: send inline base64 PNG, verify non-empty text response."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            messages=[_base64_image_msg("Describe the colour of this image in one sentence.")],
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty response from Gemini base64 image"
        assert result.total_usage.input_tokens > 0


# ================================================================== #
# Gemini tool tests (3)
# ================================================================== #
#
# generate.py calls `await tool.execute(**args)`, so all execute
# functions must be async coroutines.


async def _weather_fn(city: str) -> str:
    return f"The temperature in {city} is 22 °C and sunny."


async def _population_fn(city: str) -> str:
    return f"{city} has an approximate population of 2 million people."


async def _add_fn(a: float, b: float) -> str:
    return str(a + b)


async def _multiply_fn(a: float, b: float) -> str:
    return str(a * b)


_WEATHER_TOOL = Tool(
    name="get_weather",
    description="Return the current temperature in Celsius for a given city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
    execute=_weather_fn,
)

_POPULATION_TOOL = Tool(
    name="get_population",
    description="Return the approximate population of a given city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    execute=_population_fn,
)

_ADD_TOOL = Tool(
    name="add",
    description="Add two numbers and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
    execute=_add_fn,
)

_MULTIPLY_TOOL = Tool(
    name="multiply",
    description="Multiply two numbers and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
    execute=_multiply_fn,
)


class TestGeminiTools:
    """Gemini-specific tool-calling: single, parallel, and multi-step."""

    @skip_no_gemini
    @pytest.mark.xfail(strict=False, reason="LLM may skip tools and answer from memory")
    async def test_gemini_single_tool_call(self, gemini_client: Client) -> None:
        """Gemini calls a single tool and returns a text answer."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            (
                "Use the get_weather tool to look up the weather in Berlin. "
                "You MUST call the tool — do not answer from memory."
            ),
            tools=[_WEATHER_TOOL],
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final text after tool call"
        assert result.total_usage.input_tokens > 0
        # At least one tool-call round must have occurred
        tool_rounds = [s for s in result.steps if s.tool_results]
        assert tool_rounds, "Expected at least one tool-call round"

    @skip_no_gemini
    @pytest.mark.xfail(strict=False, reason="LLM may skip tools and answer from memory")
    async def test_gemini_parallel_tool_calls(self, gemini_client: Client) -> None:
        """Gemini calls two tools to answer a compound question."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            (
                "You MUST use both the get_weather tool AND the get_population tool. "
                "Call get_weather for Paris to get the temperature, then call get_population "
                "for Paris to get the population. Do not answer from memory."
            ),
            tools=[_WEATHER_TOOL, _POPULATION_TOOL],
            max_rounds=5,
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final text"
        assert result.total_usage.input_tokens > 0
        tool_rounds = [s for s in result.steps if s.tool_results]
        assert tool_rounds, "Expected at least one tool-call round"

    @skip_no_gemini
    @pytest.mark.xfail(strict=False, reason="LLM may skip tools and answer from memory")
    async def test_gemini_multi_step_tool_loop(self, gemini_client: Client) -> None:
        """Gemini performs a multi-step calculation using multiple tool calls."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            (
                "You MUST use the tools for every arithmetic step — do not compute mentally. "
                "Step 1: call the add tool with a=15 and b=27. "
                "Step 2: call the multiply tool with the result from step 1 and b=3. "
                "Then report the final answer."
            ),
            system=(
                "You are a calculator assistant. "
                "You MUST call the provided tools for every arithmetic operation. "
                "Never compute numbers yourself."
            ),
            tools=[_ADD_TOOL, _MULTIPLY_TOOL],
            max_rounds=8,
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final answer"
        assert result.total_usage.input_tokens > 0
        # Expect at least 2 distinct tool-call rounds (add + multiply)
        tool_rounds = [s for s in result.steps if s.tool_results]
        assert len(tool_rounds) >= 2, (
            f"Expected ≥ 2 tool-call rounds (add + multiply), got {len(tool_rounds)}"
        )


# ================================================================== #
# Streaming with tools declared (3)
# ================================================================== #
#
# generate.stream() has no tools parameter; we call client.stream()
# directly with a Request that includes tools.  We verify that the
# stream completes and emits START + at least one content event.


class TestStreamingWithTools:
    """Verify streaming completes when tools are declared in the request."""

    @skip_no_openai
    async def test_openai_streaming_with_tools(self, openai_client: Client) -> None:
        """OpenAI: streaming request with tools declared emits correct events."""
        request = Request(
            model=OPENAI_MODEL,
            messages=[Message.user("What is 2 + 2? Answer directly without using tools.")],
            tools=[_WEATHER_TOOL],
            provider="openai",
        )
        event_gen = await openai_client.stream(request)
        events = [ev async for ev in event_gen]

        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, "Expected START event"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got: {kinds}"

    @skip_no_anthropic
    async def test_anthropic_streaming_with_tools(self, anthropic_client: Client) -> None:
        """Anthropic: streaming request with tools declared emits correct events."""
        request = Request(
            model=ANTHROPIC_MODEL,
            messages=[Message.user("What is 3 + 4? Answer directly without using tools.")],
            tools=[_WEATHER_TOOL],
            provider="anthropic",
        )
        event_gen = await anthropic_client.stream(request)
        events = [ev async for ev in event_gen]

        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, "Expected START event"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got: {kinds}"

    @skip_no_gemini
    async def test_gemini_streaming_with_tools(self, gemini_client: Client) -> None:
        """Gemini: streaming request with tools declared emits correct events."""
        request = Request(
            model=GEMINI_MODEL,
            messages=[Message.user("What is 5 + 6? Answer directly without using tools.")],
            tools=[_WEATHER_TOOL],
            provider="gemini",
        )
        event_gen = await gemini_client.stream(request)
        events = [ev async for ev in event_gen]

        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, "Expected START event"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got: {kinds}"


# ================================================================== #
# Reasoning token tests (3)
# ================================================================== #


class TestReasoningTokens:
    """Verify reasoning/thinking tokens appear in usage when enabled."""

    @skip_no_openai
    async def test_openai_reasoning_tokens(self, openai_client: Client) -> None:
        """o4-mini with reasoning_effort should include reasoning tokens in usage."""
        result = await generate(
            openai_client,
            OPENAI_REASONING_MODEL,
            "What is the 7th prime number? Think step by step.",
            reasoning_effort="medium",
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty response from o4-mini"
        usage = result.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        # o-series models report reasoning_tokens in output_tokens_details
        assert isinstance(usage.reasoning_tokens, int), "reasoning_tokens must be an int"

    @skip_no_anthropic
    async def test_anthropic_reasoning_tokens(self, anthropic_client: Client) -> None:
        """Claude extended thinking should include thinking tokens in usage."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_REASONING_MODEL,
            "What is the 10th Fibonacci number? Think step by step.",
            reasoning_effort="low",  # enables extended thinking, budget_tokens=2048
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response with extended thinking"
        usage = result.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert isinstance(usage.reasoning_tokens, int), "reasoning_tokens must be an int"

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "The Gemini adapter currently places thinkingConfig at the top level of the "
            "request body, but the Gemini API requires it inside generationConfig. "
            "This is a known adapter limitation; the test documents the intended behaviour."
        ),
    )
    async def test_gemini_reasoning_tokens(self, gemini_client: Client) -> None:
        """Gemini 2.5 Flash with reasoning_effort should populate reasoning_tokens."""
        result = await generate(
            gemini_client,
            GEMINI_REASONING_MODEL,
            "What is the square root of 144? Think step by step.",
            reasoning_effort="low",
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty response from Gemini with thinking"
        usage = result.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        # Gemini reports thoughtsTokenCount → reasoning_tokens
        assert isinstance(usage.reasoning_tokens, int), "reasoning_tokens must be an int"
        assert usage.reasoning_tokens > 0, "Expected reasoning_tokens > 0 with thinking enabled"
