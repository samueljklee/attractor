"""Tests for Wave 10: streaming generation, abort signal, and new types.

Covers:
  - StreamResult: text_stream, response(), __aiter__ backward compat
  - New StreamEventKind values (TEXT_START, TEXT_END, REASONING_START, etc.)
  - TimeoutConfig and AdapterTimeout dataclass fields
  - abort_signal on generate() raises AbortError
  - generate_object() uses schema parameter (mock test)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor_llm.client import Client
from attractor_llm.errors import AbortError
from attractor_llm.generate import generate, generate_object, stream
from attractor_llm.streaming import StreamAccumulator, StreamResult
from attractor_llm.types import (
    AdapterTimeout,
    ContentPart,
    FinishReason,
    GenerateResult,
    Message,
    Request,
    Response,
    StreamEvent,
    StreamEventKind,
    TimeoutConfig,
    Tool,
    Usage,
)

# ================================================================== #
# Helpers
# ================================================================== #


def _text_response(text: str) -> Response:
    """Create a simple text-only Response for mocking."""
    return Response(message=Message.assistant(text), usage=Usage())


async def _mock_event_stream(*events: StreamEvent):
    """Create an async iterator from a sequence of StreamEvents."""
    for event in events:
        yield event


# ================================================================== #
# StreamResult tests
# ================================================================== #


class TestStreamResult:
    """StreamResult wraps a live event stream with convenient access."""

    @pytest.mark.asyncio
    async def test_text_stream_yields_text_deltas(self):
        """text_stream yields only TEXT_DELTA text values."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="test", provider="test"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=" world"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))

        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_response_accumulates_full_text(self):
        """response() consumes remaining events and returns accumulated Response."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m1", provider="p1"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=" world"),
            StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))
        resp = await result.response()

        assert resp.text == "Hello world"
        assert resp.model == "m1"
        assert resp.provider == "p1"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_response_after_text_stream_returns_same_result(self):
        """After iterating text_stream, response() returns the accumulated data."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hi"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))

        # Consume via text_stream first
        chunks = []
        async for chunk in result.text_stream:
            chunks.append(chunk)
        assert chunks == ["Hi"]

        # Then get the response
        resp = await result.response()
        assert resp.text == "Hi"

    @pytest.mark.asyncio
    async def test_aiter_delegates_to_text_stream_for_backward_compat(self):
        """__aiter__ yields text chunks (backward compat with old stream())."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="abc"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="def"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))

        # Using `async for chunk in result:` should yield text strings
        chunks = []
        async for chunk in result:
            assert isinstance(chunk, str)
            chunks.append(chunk)

        assert chunks == ["abc", "def"]

    @pytest.mark.asyncio
    async def test_iter_events_yields_raw_events(self):
        """iter_events() yields raw StreamEvent objects."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="hi"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))

        collected = []
        async for event in result.iter_events():
            collected.append(event)

        assert len(collected) == 3
        assert collected[0].kind == StreamEventKind.START
        assert collected[1].kind == StreamEventKind.TEXT_DELTA
        assert collected[2].kind == StreamEventKind.FINISH

    @pytest.mark.asyncio
    async def test_response_idempotent_after_consumed(self):
        """Calling response() multiple times returns the same result."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="ok"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_mock_event_stream(*events))

        resp1 = await result.response()
        resp2 = await result.response()
        assert resp1.text == resp2.text == "ok"


# ================================================================== #
# StreamEventKind new values
# ================================================================== #


class TestStreamEventKindNewValues:
    """New StreamEventKind enum members added in Wave 10."""

    def test_text_start_exists(self):
        assert StreamEventKind.TEXT_START == "text_start"

    def test_text_end_exists(self):
        assert StreamEventKind.TEXT_END == "text_end"

    def test_reasoning_start_exists(self):
        assert StreamEventKind.REASONING_START == "reasoning_start"

    def test_reasoning_end_exists(self):
        assert StreamEventKind.REASONING_END == "reasoning_end"

    def test_provider_event_exists(self):
        assert StreamEventKind.PROVIDER_EVENT == "provider_event"

    def test_provider_event_has_raw_event_field(self):
        """StreamEvent with PROVIDER_EVENT kind can carry raw_event data."""
        event = StreamEvent(
            kind=StreamEventKind.PROVIDER_EVENT,
            raw_event={"type": "content_block_start"},
        )
        assert event.raw_event == {"type": "content_block_start"}


# ================================================================== #
# TimeoutConfig and AdapterTimeout
# ================================================================== #


class TestTimeoutConfig:
    """TimeoutConfig dataclass for high-level timeout settings."""

    def test_default_values(self):
        tc = TimeoutConfig()
        assert tc.total is None
        assert tc.per_step is None

    def test_custom_values(self):
        tc = TimeoutConfig(total=120.0, per_step=30.0)
        assert tc.total == 120.0
        assert tc.per_step == 30.0


class TestAdapterTimeout:
    """AdapterTimeout dataclass for low-level HTTP timeout settings."""

    def test_default_values(self):
        at = AdapterTimeout()
        assert at.connect == 10.0
        assert at.request == 120.0
        assert at.stream_read == 30.0

    def test_custom_values(self):
        at = AdapterTimeout(connect=5.0, request=60.0, stream_read=15.0)
        assert at.connect == 5.0
        assert at.request == 60.0
        assert at.stream_read == 15.0


# ================================================================== #
# abort_signal on generate()
# ================================================================== #


class TestGenerateAbortSignal:
    """abort_signal parameter on generate() raises AbortError when set."""

    @pytest.mark.asyncio
    async def test_abort_signal_raises_abort_error_after_complete(self):
        """If abort_signal is set after client.complete(), AbortError is raised."""
        abort = MagicMock()
        abort.is_set = True  # Already set

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hello"))

        with pytest.raises(AbortError, match="aborted"):
            await generate(
                client,
                "test-model",
                "Say hello",
                abort_signal=abort,
            )

    @pytest.mark.asyncio
    async def test_no_abort_signal_returns_normally(self):
        """Without abort signal, generate() returns normally."""
        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hello"))

        result = await generate(client, "test-model", "Say hello")
        assert result.text == "Hello"

    @pytest.mark.asyncio
    async def test_abort_signal_not_set_returns_normally(self):
        """If abort_signal exists but is_set is False, generate() returns normally."""
        abort = MagicMock()
        abort.is_set = False

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hello"))

        result = await generate(
            client, "test-model", "Say hello", abort_signal=abort
        )
        assert result.text == "Hello"

    @pytest.mark.asyncio
    async def test_abort_signal_checked_in_tool_loop(self):
        """abort_signal is checked after each complete() call in the tool loop."""
        call_count = 0
        abort = MagicMock()

        # Signal is not set on first call, set on second
        def _check_is_set():
            return call_count >= 1

        type(abort).is_set = property(lambda self: _check_is_set())

        # First response has tool calls, second would be text
        tool_call_response = Response(
            message=Message(
                role="assistant",
                content=[
                    ContentPart.tool_call_part("tc1", "my_tool", '{"x": 1}'),
                ],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
        )

        async def _mock_complete(request, **kwargs):
            nonlocal call_count
            call_count += 1
            return tool_call_response

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(side_effect=_mock_complete)

        async def _exec(**kwargs):
            return "result"

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            execute=_exec,
        )

        with pytest.raises(AbortError):
            await generate(
                client,
                "test-model",
                "Use the tool",
                tools=[tool],
                abort_signal=abort,
            )


# ================================================================== #
# generate_object() with schema (mock test)
# ================================================================== #


class TestGenerateObject:
    """generate_object() passes schema info to the LLM."""

    @pytest.mark.asyncio
    async def test_schema_is_included_in_system_prompt(self):
        """When schema is provided, it's included in the system prompt."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        captured_request: list[Request] = []

        async def _capture_complete(request, **kwargs):
            captured_request.append(request)
            return _text_response(json.dumps({"name": "Alice", "age": 30}))

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(side_effect=_capture_complete)

        result = await generate_object(
            client,
            "test-model",
            "Extract person info",
            schema=schema,
        )

        assert result == {"name": "Alice", "age": 30}

        # Verify schema was included in the system prompt
        req = captured_request[0]
        assert req.system is not None
        assert "JSON" in req.system
        assert '"name"' in req.system

    @pytest.mark.asyncio
    async def test_generate_object_without_schema(self):
        """generate_object() works without a schema (free-form JSON)."""
        async def _mock_complete(request, **kwargs):
            return _text_response('{"key": "value"}')

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(side_effect=_mock_complete)

        result = await generate_object(client, "test-model", "Return some JSON")
        assert result == {"key": "value"}


# ================================================================== #
# stream() returns StreamResult
# ================================================================== #


class TestStreamReturnsStreamResult:
    """stream() function returns StreamResult instead of raw AsyncIterator."""

    @pytest.mark.asyncio
    async def test_stream_returns_stream_result(self):
        """stream() returns a StreamResult instance."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hi"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]

        client = AsyncMock(spec=Client)
        client.stream = AsyncMock(return_value=_mock_event_stream(*events))

        result = await stream(client, "test-model", "Hello")
        assert isinstance(result, StreamResult)

    @pytest.mark.asyncio
    async def test_stream_result_backward_compat_iteration(self):
        """async for chunk in (await stream(...)): yields text strings."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=" world"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]

        client = AsyncMock(spec=Client)
        client.stream = AsyncMock(return_value=_mock_event_stream(*events))

        result = await stream(client, "test-model", "Hello")

        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]


# ================================================================== #
# Exports
# ================================================================== #


class TestExports:
    """New types are exported from attractor_llm package."""

    def test_stream_result_exported(self):
        from attractor_llm import StreamResult as SR

        assert SR is StreamResult

    def test_timeout_config_exported(self):
        from attractor_llm import TimeoutConfig as TC

        assert TC is TimeoutConfig

    def test_adapter_timeout_exported(self):
        from attractor_llm import AdapterTimeout as AT

        assert AT is AdapterTimeout
