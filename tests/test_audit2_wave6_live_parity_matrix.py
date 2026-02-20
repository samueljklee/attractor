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

import pytest

from attractor_llm import (
    Client,
    FinishReason,
    GenerateResult,
    InvalidRequestError,
    Message,
    NotFoundError,
    ProviderError,
    Request,
    Usage,
    generate,
    generate_object,
    stream_with_tools,
)
from attractor_llm.types import StreamEventKind
from tests.conftest import (
    ANTHROPIC_MODEL,
    ANTHROPIC_REASONING_MODEL,
    GEMINI_MODEL,
    GEMINI_REASONING_MODEL,
    OPENAI_MODEL,
    OPENAI_REASONING_MODEL,
    skip_no_anthropic,
    skip_no_gemini,
    skip_no_openai,
)
from tests.live_helpers import (
    PERSON_SCHEMA,
    STOCK_TOOL,
    TIME_TOOL,
    WEATHER_TOOL,
    assert_mostly_uppercase,
    base64_image_msg,
    url_image_msg,
)

# ================================================================== #
# Provider parametrization
# ================================================================== #
#
# Each pytest.param supplies (provider_name, provider_name, model).
# The first value feeds ``provider_client`` via indirect (fixture
# resolves it to a Client); the second is the plain provider string
# used in generate(... provider=); the third is the model name.

_OPENAI = pytest.param(
    "openai",
    "openai",
    OPENAI_MODEL,
    marks=[skip_no_openai],
    id="openai",
)
_ANTHROPIC = pytest.param(
    "anthropic",
    "anthropic",
    ANTHROPIC_MODEL,
    marks=[skip_no_anthropic],
    id="anthropic",
)
_GEMINI = pytest.param(
    "gemini",
    "gemini",
    GEMINI_MODEL,
    marks=[skip_no_gemini],
    id="gemini",
)

ALL_PROVIDERS = [_OPENAI, _ANTHROPIC, _GEMINI]

# Standard arg names for the 3-tuple parametrization.
_P = ("provider_client", "provider", "model")


def _providers(
    params: list[pytest.ParameterSet],
) -> pytest.MarkDecorator:
    """Parametrize over provider/model with indirect client resolution."""
    return pytest.mark.parametrize(
        _P,
        params,
        indirect=["provider_client"],
    )


def _gemini_xfail(reason: str) -> pytest.ParameterSet:
    """Gemini param with an additional non-strict xfail."""
    return pytest.param(
        "gemini",
        "gemini",
        GEMINI_MODEL,
        marks=[
            skip_no_gemini,
            pytest.mark.xfail(strict=False, reason=reason),
        ],
        id="gemini",
    )


@pytest.fixture
def provider_client(
    request: pytest.FixtureRequest,
    openai_client: Client,
    anthropic_client: Client,
    gemini_client: Client,
) -> Client:
    """Resolve the correct client for the parametrized provider name."""
    return {
        "openai": openai_client,
        "anthropic": anthropic_client,
        "gemini": gemini_client,
    }[request.param]


# ================================================================== #
# Helpers
# ================================================================== #


def _tool_names(result: GenerateResult) -> set[str]:
    """Extract tool names called across all steps."""
    return {tr.name for s in result.steps for tr in s.tool_results if tr.name}


# ================================================================== #
# Scenario 1 -- Basic Generation
# ================================================================== #


class TestS01BasicGeneration:
    """Parity row 1: simplest prompt returns a non-empty GenerateResult."""

    @_providers(ALL_PROVIDERS)
    async def test_basic_generation(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "Reply with the single word: HELLO",
            temperature=0,
            provider=provider,
        )
        assert isinstance(result, GenerateResult)
        assert result.text.strip(), f"Expected non-empty text from {provider}"
        assert len(result.steps) >= 1


# ================================================================== #
# Scenario 2 -- System Prompt
# ================================================================== #


class TestS02SystemPrompt:
    """Parity row 2: system prompt controls output style (ALL CAPS)."""

    @_providers(ALL_PROVIDERS)
    async def test_system_prompt(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "What color is the sky?",
            system=("You MUST respond in ALL CAPS. Every alphabetic character must be uppercase."),
            temperature=0,
            provider=provider,
        )
        text = result.text.strip()
        assert text, "Expected non-empty response"
        assert_mostly_uppercase(text)


# ================================================================== #
# Scenario 3 -- Image Input URL
# ================================================================== #


class TestS03ImageURL:
    """Parity row 3: send image via HTTP URL, expect descriptive text."""

    @_providers(
        [
            _OPENAI,
            _ANTHROPIC,
            _gemini_xfail(
                "Gemini adapter maps HTTP URLs to fileData.fileUri "
                "which requires GCS or File API URIs; plain HTTP "
                "URLs are not supported."
            ),
        ]
    )
    async def test_image_url(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            messages=[url_image_msg("Describe this image in one sentence.")],
            provider=provider,
        )
        assert result.text.strip(), f"Expected non-empty image description from {provider}"
        assert result.total_usage.input_tokens > 0


# ================================================================== #
# Scenario 4 -- Image Input Base64
# ================================================================== #


class TestS04ImageBase64:
    """Parity row 4: send image as inline base64 bytes."""

    @_providers(ALL_PROVIDERS)
    async def test_image_base64(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            messages=[base64_image_msg("What color is this image?")],
            provider=provider,
        )
        assert result.text.strip(), f"Expected non-empty response from {provider} (base64)"
        assert result.total_usage.input_tokens > 0


# ================================================================== #
# Scenario 5 -- Single Tool Call
# ================================================================== #


class TestS05SingleToolCall:
    """Parity row 5: model calls one tool and incorporates the result."""

    @_providers(
        [
            _OPENAI,
            _ANTHROPIC,
            _gemini_xfail("Gemini may answer from memory without calling the tool."),
        ]
    )
    async def test_single_tool(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "Use the get_weather tool to look up the weather in Paris. "
            "You MUST call the tool -- do not answer from memory.",
            tools=[WEATHER_TOOL],
            max_rounds=5,
            provider=provider,
        )
        assert result.text.strip(), "Expected non-empty final answer"
        names = _tool_names(result)
        assert "get_weather" in names, f"get_weather not called; called={names}"


# ================================================================== #
# Scenario 6 -- Parallel Tool Calls
# ================================================================== #


class TestS06ParallelToolCalls:
    """Parity row 6: model calls two tools for a compound question."""

    @_providers(
        [
            _OPENAI,
            _ANTHROPIC,
            _gemini_xfail(
                "Gemini may answer compound questions from memory without calling both tools."
            ),
        ]
    )
    async def test_parallel_tools(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "What is the weather in London AND the current time in UTC? "
            "You MUST use BOTH the get_weather and get_time tools.",
            tools=[WEATHER_TOOL, TIME_TOOL],
            max_rounds=5,
            provider=provider,
        )
        assert result.text.strip(), "Expected non-empty final text"
        names = _tool_names(result)
        assert "get_weather" in names, f"get_weather not called; got {names}"
        assert "get_time" in names, f"get_time not called; got {names}"


# ================================================================== #
# Scenario 7 -- Multi-step Tool Loop
# ================================================================== #


class TestS07MultiStepToolLoop:
    """Parity row 7: model executes >=2 sequential tool-call rounds."""

    @_providers(
        [
            _OPENAI,
            _ANTHROPIC,
            _gemini_xfail(
                "Gemini may short-circuit multi-step tool chains and answer from memory."
            ),
        ]
    )
    async def test_multi_step(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "Step 1: call get_weather for London. "
            "Step 2: call get_time for UTC. "
            "Step 3: call get_stock_price for AAPL. "
            "Call each tool individually, then summarize all three results.",
            tools=[WEATHER_TOOL, TIME_TOOL, STOCK_TOOL],
            max_rounds=8,
            provider=provider,
        )
        assert result.text.strip(), "Expected non-empty final summary"
        names = _tool_names(result)
        assert len(names) >= 2, f"Expected >=2 distinct tools called; got {names}"


# ================================================================== #
# Scenario 8 -- Streaming + Tools
# ================================================================== #


class TestS08StreamingTools:
    """Parity row 8: stream_with_tools() emits START + content events."""

    @_providers(ALL_PROVIDERS)
    async def test_streaming_tools(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await stream_with_tools(
            provider_client,
            model,
            "What is 2 + 2? Answer directly without using any tools.",
            tools=[WEATHER_TOOL],
            provider=provider,
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

    @_providers(ALL_PROVIDERS)
    async def test_structured_output(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        obj = await generate_object(
            provider_client,
            model,
            "Alice is 30 years old.",
            schema=PERSON_SCHEMA,
            provider=provider,
        )
        assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
        assert "name" in obj and "age" in obj, f"Missing required keys; got {obj}"
        assert "alice" in obj["name"].lower(), f"Expected Alice in name; got {obj['name']}"
        assert obj["age"] == 30, f"Expected age=30; got {obj['age']}"


# ================================================================== #
# Scenario 10 -- Reasoning Tokens
# ================================================================== #


class TestS10ReasoningTokens:
    """Parity row 10: reasoning_effort populates reasoning_tokens."""

    @pytest.mark.parametrize(
        ("provider_client", "provider", "model", "effort"),
        [
            pytest.param(
                "openai",
                "openai",
                OPENAI_REASONING_MODEL,
                "medium",
                marks=[skip_no_openai],
                id="openai",
            ),
            pytest.param(
                "anthropic",
                "anthropic",
                ANTHROPIC_REASONING_MODEL,
                "low",
                marks=[skip_no_anthropic],
                id="anthropic",
            ),
            pytest.param(
                "gemini",
                "gemini",
                GEMINI_REASONING_MODEL,
                "low",
                marks=[
                    skip_no_gemini,
                    pytest.mark.xfail(
                        strict=False,
                        reason=(
                            "Gemini adapter places thinkingConfig at "
                            "top level but API requires it inside "
                            "generationConfig -- known limitation."
                        ),
                    ),
                ],
                id="gemini",
            ),
        ],
        indirect=["provider_client"],
    )
    async def test_reasoning(
        self,
        provider_client: Client,
        provider: str,
        model: str,
        effort: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "What is the 7th prime number? Think step by step.",
            reasoning_effort=effort,
            provider=provider,
        )
        assert result.text.strip(), f"Expected non-empty response from {provider} reasoning model"
        usage = result.total_usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert isinstance(usage.reasoning_tokens, int), (
            f"reasoning_tokens must be int; got {type(usage.reasoning_tokens)}"
        )
        assert usage.reasoning_tokens > 0, (
            f"Expected reasoning_tokens > 0; got {usage.reasoning_tokens}"
        )


# ================================================================== #
# Scenario 11 -- Max Rounds
# ================================================================== #


class TestS11MaxRounds:
    """Parity row 11: max_rounds=0 returns tool_calls without executing."""

    @_providers(
        [
            _OPENAI,
            _ANTHROPIC,
            _gemini_xfail("Gemini may answer from memory without calling the tool."),
        ]
    )
    async def test_max_rounds(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "What is the weather in Tokyo? You MUST use the get_weather tool.",
            tools=[WEATHER_TOOL],
            max_rounds=0,
            provider=provider,
        )
        assert len(result.steps) == 1, (
            f"Expected exactly 1 step with max_rounds=0; got {len(result.steps)}"
        )
        step = result.steps[0]
        assert step.response.finish_reason == FinishReason.TOOL_CALLS, (
            f"Expected TOOL_CALLS finish; got {step.response.finish_reason}"
        )
        assert len(step.response.tool_calls) >= 1, "Expected >=1 tool call in response"
        assert len(step.tool_results) == 0, "tool_results must be empty (tools not executed)"


# ================================================================== #
# Scenario 12 -- Error Handling
# ================================================================== #


class TestS12ErrorHandling:
    """Parity row 12: invalid model name raises a provider-level error."""

    @pytest.mark.parametrize(
        ("provider_client", "provider", "bad_model"),
        [
            pytest.param(
                "openai",
                "openai",
                "gpt-nonexistent-xyzzy-9999",
                marks=[skip_no_openai],
                id="openai",
            ),
            pytest.param(
                "anthropic",
                "anthropic",
                "claude-nonexistent-xyzzy-9999",
                marks=[skip_no_anthropic],
                id="anthropic",
            ),
            pytest.param(
                "gemini",
                "gemini",
                "gemini-nonexistent-xyzzy-9999",
                marks=[skip_no_gemini],
                id="gemini",
            ),
        ],
        indirect=["provider_client"],
    )
    async def test_invalid_model(
        self,
        provider_client: Client,
        provider: str,
        bad_model: str,
    ) -> None:
        with pytest.raises(
            (NotFoundError, InvalidRequestError, ProviderError),
        ):
            await generate(
                provider_client,
                bad_model,
                "Hello",
                provider=provider,
            )


# ================================================================== #
# Scenario 13 -- Finish Reasons
# ================================================================== #


class TestS13FinishReasons:
    """Parity row 13: text-only response -> finish_reason STOP."""

    @_providers(ALL_PROVIDERS)
    async def test_finish_stop(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "Say hello.",
            temperature=0,
            provider=provider,
        )
        assert len(result.steps) >= 1
        last = result.steps[-1]
        assert last.response.finish_reason == FinishReason.STOP, (
            f"Expected STOP finish; got {last.response.finish_reason}"
        )


# ================================================================== #
# Scenario 14 -- Usage Tracking
# ================================================================== #


class TestS14UsageTracking:
    """Parity row 14: tokens populated and internally consistent."""

    @_providers(ALL_PROVIDERS)
    async def test_usage(
        self,
        provider_client: Client,
        provider: str,
        model: str,
    ) -> None:
        result = await generate(
            provider_client,
            model,
            "What is 1+1?",
            temperature=0,
            provider=provider,
        )
        u = result.total_usage
        assert isinstance(u, Usage)
        assert u.input_tokens > 0, f"input_tokens should be >0; got {u.input_tokens}"
        assert u.output_tokens > 0, f"output_tokens should be >0; got {u.output_tokens}"
        assert u.total_tokens >= u.input_tokens + u.output_tokens, (
            f"total {u.total_tokens} < input {u.input_tokens} + output {u.output_tokens}"
        )


# ================================================================== #
# Scenario 15 -- Provider Options
# ================================================================== #
# This scenario has structurally different test logic per provider,
# so each method is written separately rather than parametrized.


class TestS15ProviderOptions:
    """Parity row 15: provider_options pass-through doesn't crash."""

    @skip_no_openai
    async def test_openai_provider_options(
        self,
        openai_client: Client,
    ) -> None:
        """OpenAI: empty provider_options passes through cleanly."""
        request = Request(
            model=OPENAI_MODEL,
            messages=[Message.user("What is 2+2?")],
            provider="openai",
            temperature=0.0,
            provider_options={"openai": {}},
        )
        response = await openai_client.complete(request)
        assert response.message.text, "Expected non-empty response text"
        assert response.usage.input_tokens > 0

    @skip_no_anthropic
    async def test_anthropic_provider_options(
        self,
        anthropic_client: Client,
    ) -> None:
        """Anthropic: provider_options passthrough with metadata doesn't crash."""
        request = Request(
            model=ANTHROPIC_MODEL,
            messages=[Message.user("What is 2+2?")],
            provider="anthropic",
            temperature=0.0,
            provider_options={"anthropic": {"metadata": {"user_id": "test-suite"}}},
        )
        response = await anthropic_client.complete(request)
        assert response.message.text, "Expected non-empty response text"
        assert response.usage.input_tokens > 0

    @skip_no_gemini
    async def test_gemini_provider_options(
        self,
        gemini_client: Client,
    ) -> None:
        """Gemini: safetySettings via provider_options don't crash."""
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
        assert response.message.text, "Expected non-empty response text"
        assert response.usage.input_tokens > 0
