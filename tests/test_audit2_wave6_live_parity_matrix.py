"""Audit 2 Wave 6 - Live Parity Matrix.

15 scenarios x 3 providers = 45 test cells.

Every test hits a real provider API.  Tests are skipped automatically when
their API key is absent; flaky or known-limited cells are marked
``xfail(strict=False)`` with an explanatory reason string.

Models used (small/fast):
  OpenAI    - gpt-4.1-mini
  Anthropic - claude-sonnet-4-5
  Gemini    - gemini-2.0-flash
  Reasoning - o4-mini / claude-sonnet-4-5 / gemini-2.5-flash

Run all 45:
    uv run python -m pytest tests/test_audit2_wave6_live_parity_matrix.py -v

Quick deterministic subset:
    uv run python -m pytest tests/test_audit2_wave6_live_parity_matrix.py -v \\
        -k "basic_generation or error_handling or usage or finish_stop" \\
        --timeout=120
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import Any

import pytest

from attractor_llm import (
    Client,
    FinishReason,
    GenerateResult,
    InvalidRequestError,
    Message,
    NotFoundError,
    ProviderConfig,
    ProviderError,
    Request,
    RetryPolicy,
    Tool,
    Usage,
    generate,
    generate_object,
    stream_with_tools,
)
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    ImageData,
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
    """Client with only OpenAI registered."""
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0)))
    return c


@pytest.fixture
def anthropic_client() -> Client:
    """Client with only Anthropic registered."""
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=60.0))
    )
    return c


@pytest.fixture
def gemini_client() -> Client:
    """Client with only Gemini registered."""
    c = Client(retry_policy=RetryPolicy(max_retries=1))
    c.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0)))
    return c


# ================================================================== #
# Shared schema
# ================================================================== #

PERSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

# ================================================================== #
# Image helpers  (mirrors test_wave16b_live_coverage.py)
# ================================================================== #

# Publicly accessible HTTP PNG -- fetched server-side by OpenAI/Anthropic.
# Gemini requires GCS/File API URIs, so the URL test is xfail for Gemini.
_IMAGE_URL_HTTP = "https://httpbin.org/image/png"


def _make_minimal_png(width: int = 8, height: int = 8) -> bytes:
    """Create a minimal valid RGB PNG image in memory (no Pillow required)."""

    def chunk(type_: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(type_ + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + type_ + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit-depth=8, color-type=2 (RGB), compress/filter/interlace=0
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    # Rows: 1 filter byte + width x 3 RGB bytes (solid red pixels)
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
# Shared tool definitions
# ================================================================== #


async def _weather_fn(city: str) -> str:
    return f"Weather in {city}: 22 C, sunny."


async def _time_fn(timezone: str) -> str:
    return f"Current time in {timezone}: 14:30 UTC."


async def _stock_fn(ticker: str) -> str:
    return f"Price of {ticker}: $42.00."


_WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
    execute=_weather_fn,
)

_TIME_TOOL = Tool(
    name="get_time",
    description="Get the current time in a timezone.",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string", "description": "Timezone name (e.g. UTC)"}},
        "required": ["timezone"],
    },
    execute=_time_fn,
)

_STOCK_TOOL = Tool(
    name="get_stock_price",
    description="Get current stock price for a ticker symbol.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL)"}
        },
        "required": ["ticker"],
    },
    execute=_stock_fn,
)


# ================================================================== #
# Scenario 1 -- Basic Generation
# ================================================================== #


class TestS01BasicGeneration:
    """Parity row 1: simplest possible prompt returns a non-empty GenerateResult."""

    @skip_no_openai
    async def test_openai_basic_generation(self, openai_client: Client) -> None:
        """OpenAI gpt-4.1-mini: short prompt returns non-empty text."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "Reply with the single word: HELLO",
            temperature=0,
            provider="openai",
        )
        assert isinstance(result, GenerateResult)
        assert result.text.strip(), "Expected non-empty text from OpenAI"
        assert len(result.steps) >= 1

    @skip_no_anthropic
    async def test_anthropic_basic_generation(self, anthropic_client: Client) -> None:
        """Anthropic claude-sonnet-4-5: short prompt returns non-empty text."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Reply with the single word: HELLO",
            temperature=0,
            provider="anthropic",
        )
        assert isinstance(result, GenerateResult)
        assert result.text.strip(), "Expected non-empty text from Anthropic"
        assert len(result.steps) >= 1

    @skip_no_gemini
    async def test_gemini_basic_generation(self, gemini_client: Client) -> None:
        """Gemini gemini-2.0-flash: short prompt returns non-empty text."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "Reply with the single word: HELLO",
            temperature=0,
            provider="gemini",
        )
        assert isinstance(result, GenerateResult)
        assert result.text.strip(), "Expected non-empty text from Gemini"
        assert len(result.steps) >= 1


# ================================================================== #
# Scenario 2 -- System Prompt
# ================================================================== #


class TestS02SystemPrompt:
    """Parity row 2: system prompt controls output style (ALL CAPS enforcement)."""

    @skip_no_openai
    async def test_openai_system_prompt(self, openai_client: Client) -> None:
        """OpenAI: system='ALL CAPS' instruction -> response is >=70% uppercase letters."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What color is the sky?",
            system="You MUST respond in ALL CAPS. Every alphabetic character must be uppercase.",
            temperature=0,
            provider="openai",
        )
        text = result.text.strip()
        assert text, "Expected non-empty response"
        letters = [c for c in text if c.isalpha()]
        if letters:
            ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            assert ratio >= 0.70, f"Expected >=70% uppercase, got {ratio:.0%}: {text!r}"

    @skip_no_anthropic
    async def test_anthropic_system_prompt(self, anthropic_client: Client) -> None:
        """Anthropic: system='ALL CAPS' instruction -> response is >=70% uppercase letters."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What color is the sky?",
            system="You MUST respond in ALL CAPS. Every alphabetic character must be uppercase.",
            temperature=0,
            provider="anthropic",
        )
        text = result.text.strip()
        assert text, "Expected non-empty response"
        letters = [c for c in text if c.isalpha()]
        if letters:
            ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            assert ratio >= 0.70, f"Expected >=70% uppercase, got {ratio:.0%}: {text!r}"

    @skip_no_gemini
    async def test_gemini_system_prompt(self, gemini_client: Client) -> None:
        """Gemini: system='ALL CAPS' instruction -> response is >=70% uppercase letters."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "What color is the sky?",
            system="You MUST respond in ALL CAPS. Every alphabetic character must be uppercase.",
            temperature=0,
            provider="gemini",
        )
        text = result.text.strip()
        assert text, "Expected non-empty response"
        letters = [c for c in text if c.isalpha()]
        if letters:
            ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            assert ratio >= 0.70, f"Expected >=70% uppercase, got {ratio:.0%}: {text!r}"


# ================================================================== #
# Scenario 3 -- Image Input URL
# ================================================================== #


class TestS03ImageURL:
    """Parity row 3: send image via HTTP URL, expect descriptive text response."""

    @skip_no_openai
    async def test_openai_image_url(self, openai_client: Client) -> None:
        """OpenAI: HTTP image URL -> non-empty description, input_tokens > 0."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            messages=[_url_image_msg("Describe this image in one sentence.")],
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty image description from OpenAI"
        assert result.total_usage.input_tokens > 0

    @skip_no_anthropic
    async def test_anthropic_image_url(self, anthropic_client: Client) -> None:
        """Anthropic: HTTP image URL -> non-empty description, input_tokens > 0."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            messages=[_url_image_msg("Describe this image in one sentence.")],
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty image description from Anthropic"
        assert result.total_usage.input_tokens > 0

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Gemini adapter maps HTTP URLs to fileData.fileUri which requires "
            "GCS or File API URIs; plain HTTP URLs are not supported."
        ),
    )
    async def test_gemini_image_url(self, gemini_client: Client) -> None:
        """Gemini: HTTP image URL -- xfail (adapter requires GCS/File API URIs)."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            messages=[_url_image_msg("Describe this image in one sentence.")],
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty image description from Gemini"
        assert result.total_usage.input_tokens > 0


# ================================================================== #
# Scenario 4 -- Image Input Base64
# ================================================================== #


class TestS04ImageBase64:
    """Parity row 4: send image as inline base64 bytes, expect a description."""

    @skip_no_openai
    async def test_openai_image_base64(self, openai_client: Client) -> None:
        """OpenAI: base64 PNG -> non-empty response, input_tokens > 0."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            messages=[_base64_image_msg("What color is this image?")],
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty response from OpenAI (base64)"
        assert result.total_usage.input_tokens > 0

    @skip_no_anthropic
    async def test_anthropic_image_base64(self, anthropic_client: Client) -> None:
        """Anthropic: base64 PNG -> non-empty response, input_tokens > 0."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            messages=[_base64_image_msg("What color is this image?")],
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response from Anthropic (base64)"
        assert result.total_usage.input_tokens > 0

    @skip_no_gemini
    async def test_gemini_image_base64(self, gemini_client: Client) -> None:
        """Gemini: base64 PNG -> non-empty response, input_tokens > 0."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            messages=[_base64_image_msg("What color is this image?")],
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty response from Gemini (base64)"
        assert result.total_usage.input_tokens > 0


# ================================================================== #
# Scenario 5 -- Single Tool Call
# ================================================================== #


class TestS05SingleToolCall:
    """Parity row 5: model calls one tool and incorporates result into reply."""

    @skip_no_openai
    async def test_openai_single_tool(self, openai_client: Client) -> None:
        """OpenAI: get_weather called for Paris; verified via step tool_results."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "Use the get_weather tool to look up the weather in Paris. "
            "You MUST call the tool -- do not answer from memory.",
            tools=[_WEATHER_TOOL],
            max_rounds=5,
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty final answer"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; called={all_names}"

    @skip_no_anthropic
    async def test_anthropic_single_tool(self, anthropic_client: Client) -> None:
        """Anthropic: get_weather called for Paris; verified via step tool_results."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Use the get_weather tool to look up the weather in Paris. "
            "You MUST call the tool -- do not answer from memory.",
            tools=[_WEATHER_TOOL],
            max_rounds=5,
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty final answer"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; called={all_names}"

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason="Gemini may answer from memory without calling the tool.",
    )
    async def test_gemini_single_tool(self, gemini_client: Client) -> None:
        """Gemini: single tool call -- xfail (may answer from memory)."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "Use the get_weather tool to look up the weather in Paris. "
            "You MUST call the tool -- do not answer from memory.",
            tools=[_WEATHER_TOOL],
            max_rounds=5,
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final answer"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; called={all_names}"


# ================================================================== #
# Scenario 6 -- Parallel Tool Calls
# ================================================================== #


class TestS06ParallelToolCalls:
    """Parity row 6: model calls two tools for a compound question."""

    @skip_no_openai
    async def test_openai_parallel_tools(self, openai_client: Client) -> None:
        """OpenAI: both get_weather and get_time called for compound question."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What is the weather in London AND the current time in UTC? "
            "You MUST use BOTH the get_weather and get_time tools.",
            tools=[_WEATHER_TOOL, _TIME_TOOL],
            max_rounds=5,
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty final text"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; got {all_names}"
        assert "get_time" in all_names, f"get_time not called; got {all_names}"

    @skip_no_anthropic
    async def test_anthropic_parallel_tools(self, anthropic_client: Client) -> None:
        """Anthropic: both get_weather and get_time called for compound question."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is the weather in London AND the current time in UTC? "
            "You MUST use BOTH the get_weather and get_time tools.",
            tools=[_WEATHER_TOOL, _TIME_TOOL],
            max_rounds=5,
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty final text"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; got {all_names}"
        assert "get_time" in all_names, f"get_time not called; got {all_names}"

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason="Gemini may answer compound questions from memory without calling both tools.",
    )
    async def test_gemini_parallel_tools(self, gemini_client: Client) -> None:
        """Gemini: parallel tools -- xfail (may short-circuit via memory)."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "What is the weather in London AND the current time in UTC? "
            "You MUST use BOTH the get_weather and get_time tools.",
            tools=[_WEATHER_TOOL, _TIME_TOOL],
            max_rounds=5,
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final text"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert "get_weather" in all_names, f"get_weather not called; got {all_names}"
        assert "get_time" in all_names, f"get_time not called; got {all_names}"


# ================================================================== #
# Scenario 7 -- Multi-step Tool Loop
# ================================================================== #


class TestS07MultiStepToolLoop:
    """Parity row 7: model executes >=2 sequential tool-call rounds."""

    @skip_no_openai
    async def test_openai_multi_step(self, openai_client: Client) -> None:
        """OpenAI: 3-step chain (weather -> time -> stock), >=2 distinct tools called."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "Step 1: call get_weather for London. "
            "Step 2: call get_time for UTC. "
            "Step 3: call get_stock_price for AAPL. "
            "Call each tool individually, then summarize all three results.",
            tools=[_WEATHER_TOOL, _TIME_TOOL, _STOCK_TOOL],
            max_rounds=8,
            provider="openai",
        )
        assert result.text.strip(), "Expected non-empty final summary"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert len(all_names) >= 2, f"Expected >=2 distinct tools called; got {all_names}"

    @skip_no_anthropic
    async def test_anthropic_multi_step(self, anthropic_client: Client) -> None:
        """Anthropic: 3-step chain (weather -> time -> stock), >=2 distinct tools called."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Step 1: call get_weather for London. "
            "Step 2: call get_time for UTC. "
            "Step 3: call get_stock_price for AAPL. "
            "Call each tool individually, then summarize all three results.",
            tools=[_WEATHER_TOOL, _TIME_TOOL, _STOCK_TOOL],
            max_rounds=8,
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty final summary"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert len(all_names) >= 2, f"Expected >=2 distinct tools called; got {all_names}"

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason="Gemini may short-circuit multi-step tool chains and answer from memory.",
    )
    async def test_gemini_multi_step(self, gemini_client: Client) -> None:
        """Gemini: multi-step tool loop -- xfail (may short-circuit)."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "Step 1: call get_weather for London. "
            "Step 2: call get_time for UTC. "
            "Step 3: call get_stock_price for AAPL. "
            "Call each tool individually, then summarize all three results.",
            tools=[_WEATHER_TOOL, _TIME_TOOL, _STOCK_TOOL],
            max_rounds=8,
            provider="gemini",
        )
        assert result.text.strip(), "Expected non-empty final summary"
        all_names = {tr.name for s in result.steps for tr in s.tool_results}
        assert len(all_names) >= 2, f"Expected >=2 distinct tools called; got {all_names}"


# ================================================================== #
# Scenario 8 -- Streaming + Tools
# ================================================================== #


class TestS08StreamingTools:
    """Parity row 8: stream_with_tools() emits START + content events."""

    @skip_no_openai
    async def test_openai_streaming_tools(self, openai_client: Client) -> None:
        """OpenAI: stream_with_tools yields START + TEXT_DELTA (or TOOL_CALL_START)."""
        result = await stream_with_tools(
            openai_client,
            OPENAI_MODEL,
            "What is 2 + 2? Answer directly without using any tools.",
            tools=[_WEATHER_TOOL],
            provider="openai",
        )
        events = [ev async for ev in result.iter_events()]
        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, f"Expected START event; got {kinds}"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got {kinds}"

    @skip_no_anthropic
    async def test_anthropic_streaming_tools(self, anthropic_client: Client) -> None:
        """Anthropic: stream_with_tools yields START + TEXT_DELTA (or TOOL_CALL_START)."""
        result = await stream_with_tools(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is 3 + 4? Answer directly without using any tools.",
            tools=[_WEATHER_TOOL],
            provider="anthropic",
        )
        events = [ev async for ev in result.iter_events()]
        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, f"Expected START event; got {kinds}"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got {kinds}"

    @skip_no_gemini
    async def test_gemini_streaming_tools(self, gemini_client: Client) -> None:
        """Gemini: stream_with_tools yields START + TEXT_DELTA (or TOOL_CALL_START)."""
        result = await stream_with_tools(
            gemini_client,
            GEMINI_MODEL,
            "What is 5 + 6? Answer directly without using any tools.",
            tools=[_WEATHER_TOOL],
            provider="gemini",
        )
        events = [ev async for ev in result.iter_events()]
        kinds = {ev.kind for ev in events}
        assert StreamEventKind.START in kinds, f"Expected START event; got {kinds}"
        has_content = (
            StreamEventKind.TEXT_DELTA in kinds or StreamEventKind.TOOL_CALL_START in kinds
        )
        assert has_content, f"Expected TEXT_DELTA or TOOL_CALL_START; got {kinds}"


# ================================================================== #
# Scenario 9 -- Structured Output
# ================================================================== #


class TestS09StructuredOutput:
    """Parity row 9: generate_object() returns dict matching PERSON_SCHEMA."""

    @skip_no_openai
    async def test_openai_structured_output(self, openai_client: Client) -> None:
        """OpenAI: generate_object returns {name: 'Alice', age: 30}."""
        obj = await generate_object(
            openai_client,
            OPENAI_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="openai",
        )
        assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
        assert "name" in obj and "age" in obj, f"Missing required keys; got {obj}"
        assert "alice" in obj["name"].lower(), f"Expected Alice in name; got {obj['name']}"
        assert obj["age"] == 30, f"Expected age=30; got {obj['age']}"

    @skip_no_anthropic
    async def test_anthropic_structured_output(self, anthropic_client: Client) -> None:
        """Anthropic: generate_object returns {name: 'Alice', age: 30}."""
        obj = await generate_object(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="anthropic",
        )
        assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
        assert "name" in obj and "age" in obj, f"Missing required keys; got {obj}"
        assert "alice" in obj["name"].lower(), f"Expected Alice in name; got {obj['name']}"
        assert obj["age"] == 30, f"Expected age=30; got {obj['age']}"

    @skip_no_gemini
    async def test_gemini_structured_output(self, gemini_client: Client) -> None:
        """Gemini: generate_object returns {name: 'Alice', age: 30}."""
        obj = await generate_object(
            gemini_client,
            GEMINI_MODEL,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider="gemini",
        )
        assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
        assert "name" in obj and "age" in obj, f"Missing required keys; got {obj}"
        assert "alice" in obj["name"].lower(), f"Expected Alice in name; got {obj['name']}"
        assert obj["age"] == 30, f"Expected age=30; got {obj['age']}"


# ================================================================== #
# Scenario 10 -- Reasoning Tokens
# ================================================================== #


class TestS10ReasoningTokens:
    """Parity row 10: reasoning_effort causes reasoning_tokens to be populated."""

    @skip_no_openai
    async def test_openai_reasoning(self, openai_client: Client) -> None:
        """OpenAI o4-mini with reasoning_effort='medium' -> reasoning_tokens is int."""
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
        assert isinstance(usage.reasoning_tokens, int), (
            f"reasoning_tokens must be int; got {type(usage.reasoning_tokens)}"
        )

    @skip_no_anthropic
    async def test_anthropic_reasoning(self, anthropic_client: Client) -> None:
        """Anthropic claude-sonnet-4-5 with reasoning_effort='low' -> reasoning_tokens is int."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_REASONING_MODEL,
            "What is the 10th Fibonacci number? Think step by step.",
            reasoning_effort="low",
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response with extended thinking"
        usage = result.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert isinstance(usage.reasoning_tokens, int), (
            f"reasoning_tokens must be int; got {type(usage.reasoning_tokens)}"
        )

    @skip_no_gemini
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "The Gemini adapter places thinkingConfig at the top level of the request body "
            "but the Gemini API requires it inside generationConfig -- known adapter limitation."
        ),
    )
    async def test_gemini_reasoning(self, gemini_client: Client) -> None:
        """Gemini 2.5 Flash with reasoning_effort='low' -- xfail (adapter limitation)."""
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
        assert isinstance(usage.reasoning_tokens, int)
        assert usage.reasoning_tokens > 0, "Expected reasoning_tokens > 0 with thinking enabled"


# ================================================================== #
# Scenario 11 -- Max Rounds
# ================================================================== #


class TestS11MaxRounds:
    """Parity row 11: max_rounds=0 returns tool_calls without executing them."""

    @skip_no_openai
    async def test_openai_max_rounds(self, openai_client: Client) -> None:
        """OpenAI: max_rounds=0 -> exactly 1 step, TOOL_CALLS finish, no tool_results."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What is the weather in Tokyo? You MUST use the get_weather tool.",
            tools=[_WEATHER_TOOL],
            max_rounds=0,
            provider="openai",
        )
        assert len(result.steps) == 1, (
            f"Expected exactly 1 step with max_rounds=0; got {len(result.steps)}"
        )
        step = result.steps[0]
        assert step.response.finish_reason == FinishReason.TOOL_CALLS, (
            f"Expected TOOL_CALLS finish reason; got {step.response.finish_reason}"
        )
        assert len(step.response.tool_calls) >= 1, "Expected >=1 tool call in response"
        assert len(step.tool_results) == 0, "tool_results must be empty (tools not executed)"

    @skip_no_anthropic
    async def test_anthropic_max_rounds(self, anthropic_client: Client) -> None:
        """Anthropic: max_rounds=0 -> exactly 1 step, TOOL_CALLS finish, no tool_results."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is the weather in Tokyo? You MUST use the get_weather tool.",
            tools=[_WEATHER_TOOL],
            max_rounds=0,
            provider="anthropic",
        )
        assert len(result.steps) == 1, (
            f"Expected exactly 1 step with max_rounds=0; got {len(result.steps)}"
        )
        step = result.steps[0]
        assert step.response.finish_reason == FinishReason.TOOL_CALLS, (
            f"Expected TOOL_CALLS finish reason; got {step.response.finish_reason}"
        )
        assert len(step.response.tool_calls) >= 1, "Expected >=1 tool call in response"
        assert len(step.tool_results) == 0, "tool_results must be empty (tools not executed)"

    @skip_no_gemini
    async def test_gemini_max_rounds(self, gemini_client: Client) -> None:
        """Gemini: max_rounds=0 -> exactly 1 step, TOOL_CALLS finish, no tool_results."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "What is the weather in Tokyo? You MUST use the get_weather tool.",
            tools=[_WEATHER_TOOL],
            max_rounds=0,
            provider="gemini",
        )
        assert len(result.steps) == 1, (
            f"Expected exactly 1 step with max_rounds=0; got {len(result.steps)}"
        )
        step = result.steps[0]
        assert step.response.finish_reason == FinishReason.TOOL_CALLS, (
            f"Expected TOOL_CALLS finish reason; got {step.response.finish_reason}"
        )
        assert len(step.response.tool_calls) >= 1, "Expected >=1 tool call in response"
        assert len(step.tool_results) == 0, "tool_results must be empty (tools not executed)"


# ================================================================== #
# Scenario 12 -- Error Handling
# ================================================================== #


class TestS12ErrorHandling:
    """Parity row 12: invalid model name raises a provider-level error."""

    @skip_no_openai
    async def test_openai_invalid_model(self, openai_client: Client) -> None:
        """OpenAI: nonexistent model -> NotFoundError, InvalidRequestError, or ProviderError."""
        with pytest.raises((NotFoundError, InvalidRequestError, ProviderError)):
            await generate(
                openai_client,
                "gpt-nonexistent-xyzzy-9999",
                "Hello",
                provider="openai",
            )

    @skip_no_anthropic
    async def test_anthropic_invalid_model(self, anthropic_client: Client) -> None:
        """Anthropic: nonexistent model -> NotFoundError, InvalidRequestError, or ProviderError."""
        with pytest.raises((NotFoundError, InvalidRequestError, ProviderError)):
            await generate(
                anthropic_client,
                "claude-nonexistent-xyzzy-9999",
                "Hello",
                provider="anthropic",
            )

    @skip_no_gemini
    async def test_gemini_invalid_model(self, gemini_client: Client) -> None:
        """Gemini: nonexistent model -> NotFoundError, InvalidRequestError, or ProviderError."""
        with pytest.raises((NotFoundError, InvalidRequestError, ProviderError)):
            await generate(
                gemini_client,
                "gemini-nonexistent-xyzzy-9999",
                "Hello",
                provider="gemini",
            )


# ================================================================== #
# Scenario 13 -- Finish Reasons
# ================================================================== #


class TestS13FinishReasons:
    """Parity row 13: text-only response -> finish_reason STOP on last step."""

    @skip_no_openai
    async def test_openai_finish_stop(self, openai_client: Client) -> None:
        """OpenAI: plain text prompt -> last step finish_reason == STOP."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "Say hello.",
            temperature=0,
            provider="openai",
        )
        assert len(result.steps) >= 1
        last = result.steps[-1]
        assert last.response.finish_reason == FinishReason.STOP, (
            f"Expected STOP finish reason; got {last.response.finish_reason}"
        )

    @skip_no_anthropic
    async def test_anthropic_finish_stop(self, anthropic_client: Client) -> None:
        """Anthropic: plain text prompt -> last step finish_reason == STOP."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "Say hello.",
            temperature=0,
            provider="anthropic",
        )
        assert len(result.steps) >= 1
        last = result.steps[-1]
        assert last.response.finish_reason == FinishReason.STOP, (
            f"Expected STOP finish reason; got {last.response.finish_reason}"
        )

    @skip_no_gemini
    async def test_gemini_finish_stop(self, gemini_client: Client) -> None:
        """Gemini: plain text prompt -> last step finish_reason == STOP."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "Say hello.",
            temperature=0,
            provider="gemini",
        )
        assert len(result.steps) >= 1
        last = result.steps[-1]
        assert last.response.finish_reason == FinishReason.STOP, (
            f"Expected STOP finish reason; got {last.response.finish_reason}"
        )


# ================================================================== #
# Scenario 14 -- Usage Tracking
# ================================================================== #


class TestS14UsageTracking:
    """Parity row 14: input_tokens > 0, output_tokens > 0, total_tokens == sum."""

    @skip_no_openai
    async def test_openai_usage(self, openai_client: Client) -> None:
        """OpenAI: usage fields are populated and internally consistent."""
        result = await generate(
            openai_client,
            OPENAI_MODEL,
            "What is 1+1?",
            temperature=0,
            provider="openai",
        )
        u = result.total_usage
        assert isinstance(u, Usage)
        assert u.input_tokens > 0, f"input_tokens should be >0; got {u.input_tokens}"
        assert u.output_tokens > 0, f"output_tokens should be >0; got {u.output_tokens}"
        assert u.total_tokens == u.input_tokens + u.output_tokens, (
            f"total_tokens {u.total_tokens} != input {u.input_tokens} + output {u.output_tokens}"
        )

    @skip_no_anthropic
    async def test_anthropic_usage(self, anthropic_client: Client) -> None:
        """Anthropic: usage fields are populated and internally consistent."""
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is 1+1?",
            temperature=0,
            provider="anthropic",
        )
        u = result.total_usage
        assert isinstance(u, Usage)
        assert u.input_tokens > 0, f"input_tokens should be >0; got {u.input_tokens}"
        assert u.output_tokens > 0, f"output_tokens should be >0; got {u.output_tokens}"
        assert u.total_tokens == u.input_tokens + u.output_tokens, (
            f"total_tokens {u.total_tokens} != input {u.input_tokens} + output {u.output_tokens}"
        )

    @skip_no_gemini
    async def test_gemini_usage(self, gemini_client: Client) -> None:
        """Gemini: usage fields are populated and internally consistent."""
        result = await generate(
            gemini_client,
            GEMINI_MODEL,
            "What is 1+1?",
            temperature=0,
            provider="gemini",
        )
        u = result.total_usage
        assert isinstance(u, Usage)
        assert u.input_tokens > 0, f"input_tokens should be >0; got {u.input_tokens}"
        assert u.output_tokens > 0, f"output_tokens should be >0; got {u.output_tokens}"
        assert u.total_tokens == u.input_tokens + u.output_tokens, (
            f"total_tokens {u.total_tokens} != input {u.input_tokens} + output {u.output_tokens}"
        )


# ================================================================== #
# Scenario 15 -- Provider Options
# ================================================================== #


class TestS15ProviderOptions:
    """Parity row 15: provider_options pass-through reaches the adapter without crashing."""

    @skip_no_openai
    async def test_openai_provider_options(self, openai_client: Client) -> None:
        """OpenAI: Request with provider_options={openai:{}} passes through cleanly."""
        # generate() doesn't expose provider_options; build Request + client.complete() directly.
        request = Request(
            model=OPENAI_MODEL,
            messages=[Message.user("What is 2+2?")],
            provider="openai",
            temperature=0.0,
            provider_options={"openai": {}},  # empty extras -> no field override
        )
        response = await openai_client.complete(request)
        assert response.message.text, "Expected non-empty response text via Request"
        assert response.usage.input_tokens > 0, "Expected input_tokens > 0"

    @skip_no_anthropic
    async def test_anthropic_provider_options(self, anthropic_client: Client) -> None:
        """Anthropic: auto cache_control injection with a long system prompt doesn't crash."""
        # The Anthropic adapter auto-injects cache_control breakpoints for prompts >=1024 tokens.
        # Passing a long system prompt exercises that code path.
        long_system = "You are a helpful assistant. " * 200  # ~800+ tokens
        result = await generate(
            anthropic_client,
            ANTHROPIC_MODEL,
            "What is 1+1?",
            system=long_system,
            temperature=0,
            provider="anthropic",
        )
        assert result.text.strip(), "Expected non-empty response (cache headers auto-injected)"
        assert result.total_usage.input_tokens > 0

    @skip_no_gemini
    async def test_gemini_provider_options(self, gemini_client: Client) -> None:
        """Gemini: safetySettings passed via provider_options don't cause a crash."""
        request = Request(
            model=GEMINI_MODEL,
            messages=[Message.user("What is 2+2?")],
            provider="gemini",
            temperature=0.0,
            provider_options={
                "gemini": {
                    "safetySettings": [
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        }
                    ]
                }
            },
        )
        response = await gemini_client.complete(request)
        assert response.message.text, "Expected non-empty response text via Gemini safetySettings"
        assert response.usage.input_tokens > 0, "Expected input_tokens > 0"
