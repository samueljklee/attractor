"""Audit 2 Wave 2 – Streaming Tool Loop tests.

Covers all 5 items:
  Item 1  §8.4.6   - TEXT_START/TEXT_END emitted by Anthropic adapter
  Item 2  §8.4.6   - TEXT_START/TEXT_END emitted by OpenAI adapter
  Item 3  §8.4.6   - TEXT_START/TEXT_END emitted by Gemini adapter
  Item 4  §8.1.6   - Middleware streaming gap documented (no regression)
  Items 5-9 §8.9.22-24  - stream_with_tools() agentic tool loop
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from attractor_llm.streaming import StreamAccumulator
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Tool,
    Usage,
)

# ================================================================== #
# Helpers
# ================================================================== #


async def _events(*evts: StreamEvent) -> AsyncIterator[StreamEvent]:
    """Tiny async generator from a fixed sequence of events."""
    for e in evts:
        yield e


def _kinds(events: list[StreamEvent]) -> list[StreamEventKind]:
    return [e.kind for e in events]


async def _collect(it: AsyncIterator[Any]) -> list[Any]:
    """Drain an async iterator into a list."""
    result = []
    async for item in it:
        result.append(item)
    return result


def _tool_response(name: str, call_id: str, args: dict[str, Any]) -> Response:
    """Build a Response containing a single tool call."""
    return Response(
        id="resp-tool",
        model="test-model",
        provider="test",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.tool_call_part(
                    tool_call_id=call_id,
                    name=name,
                    arguments=args,
                )
            ],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _text_response(text: str) -> Response:
    return Response(
        id="resp-text",
        model="test-model",
        provider="test",
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=5, output_tokens=10),
    )


# ================================================================== #
# Item 1 – Anthropic adapter TEXT_START / TEXT_END
# ================================================================== #


class TestAnthropicAdapterTextStartEnd:
    """Anthropic emits TEXT_START before text deltas and TEXT_END after."""

    def _make_adapter(self):
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.adapters.base import ProviderConfig

        config = ProviderConfig(api_key="test-key", timeout=30.0)
        return AnthropicAdapter(config)

    async def _run_handle_event(
        self,
        adapter: Any,
        event_type: str,
        data: dict[str, Any],
        block_type: str | None = None,
        block_id: str | None = None,
        block_name: str | None = None,
    ) -> list[StreamEvent]:
        collected: list[StreamEvent] = []
        async for ev in adapter._handle_sse_event(
            event_type,
            data,
            block_type,
            block_id,
            block_name,
            model="m",
            response_id="r",
        ):
            collected.append(ev)
        return collected

    @pytest.mark.asyncio
    async def test_anthropic_adapter_emits_text_start_before_deltas(self):
        """content_block_start with type='text' yields TEXT_START."""
        adapter = self._make_adapter()

        events = await self._run_handle_event(
            adapter,
            "content_block_start",
            {"content_block": {"type": "text", "id": "blk_0"}},
        )

        assert len(events) == 1
        assert events[0].kind == StreamEventKind.TEXT_START

    @pytest.mark.asyncio
    async def test_anthropic_adapter_emits_text_end_after_deltas(self):
        """content_block_stop with block_type='text' yields TEXT_END."""
        adapter = self._make_adapter()

        events = await self._run_handle_event(
            adapter,
            "content_block_stop",
            {},
            block_type="text",
            block_id="blk_0",
        )

        assert len(events) == 1
        assert events[0].kind == StreamEventKind.TEXT_END

    @pytest.mark.asyncio
    async def test_anthropic_tool_block_start_still_yields_tool_call_start(self):
        """content_block_start with type='tool_use' still yields TOOL_CALL_START."""
        adapter = self._make_adapter()

        events = await self._run_handle_event(
            adapter,
            "content_block_start",
            {"content_block": {"type": "tool_use", "id": "call_1", "name": "my_tool"}},
        )

        assert len(events) == 1
        assert events[0].kind == StreamEventKind.TOOL_CALL_START
        assert events[0].tool_call_id == "call_1"
        assert events[0].tool_name == "my_tool"

    @pytest.mark.asyncio
    async def test_anthropic_tool_block_stop_still_yields_tool_call_end(self):
        """content_block_stop with block_type='tool_use' still yields TOOL_CALL_END."""
        adapter = self._make_adapter()

        events = await self._run_handle_event(
            adapter,
            "content_block_stop",
            {},
            block_type="tool_use",
            block_id="call_1",
        )

        assert len(events) == 1
        assert events[0].kind == StreamEventKind.TOOL_CALL_END

    @pytest.mark.asyncio
    async def test_anthropic_full_text_stream_has_start_deltas_end(self):
        """A full simulated Anthropic SSE stream has TEXT_START…deltas…TEXT_END."""
        adapter = self._make_adapter()

        # Simulate the SSE lines for a simple text response
        def _d(obj: dict) -> str:
            return "data: " + json.dumps(obj)

        sse_lines = [
            "event: message_start",
            _d(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "claude",
                        "usage": {"input_tokens": 5},
                    },
                }
            ),
            "",
            "event: content_block_start",
            _d(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "id": "blk_0"},
                }
            ),
            "",
            "event: content_block_delta",
            _d(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                }
            ),
            "",
            "event: content_block_delta",
            _d(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                }
            ),
            "",
            "event: content_block_stop",
            _d({"type": "content_block_stop", "index": 0}),
            "",
            "event: message_delta",
            _d(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                }
            ),
            "",
            "event: message_stop",
            _d({"type": "message_stop"}),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_sse(mock_http):
            collected.append(ev)

        kinds = _kinds(collected)

        # TEXT_START must appear before any TEXT_DELTA
        assert StreamEventKind.TEXT_START in kinds
        assert StreamEventKind.TEXT_END in kinds

        start_idx = kinds.index(StreamEventKind.TEXT_START)
        end_idx = kinds.index(StreamEventKind.TEXT_END)
        delta_indices = [i for i, k in enumerate(kinds) if k == StreamEventKind.TEXT_DELTA]

        assert all(start_idx < i for i in delta_indices), "TEXT_START must precede all TEXT_DELTAs"
        assert all(end_idx > i for i in delta_indices), "TEXT_END must follow all TEXT_DELTAs"
        assert start_idx < end_idx, "TEXT_START must come before TEXT_END"

        # Finish must come after TEXT_END
        finish_idx = kinds.index(StreamEventKind.FINISH)
        assert finish_idx > end_idx, "FINISH must come after TEXT_END"

        # Text content is correct
        acc = StreamAccumulator()
        for ev in collected:
            acc.feed(ev)
        assert acc.response().text == "Hello world"


# ================================================================== #
# Item 2 – OpenAI adapter TEXT_START / TEXT_END
# ================================================================== #


class TestOpenAIAdapterTextStartEnd:
    """OpenAI adapter emits TEXT_START before text deltas and TEXT_END before FINISH."""

    def _make_adapter(self):
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        config = ProviderConfig(api_key="test-key", timeout=30.0)
        return OpenAIAdapter(config)

    @pytest.mark.asyncio
    async def test_openai_adapter_emits_text_start_end(self):
        """A simulated OpenAI SSE stream has TEXT_START…TEXT_DELTA…TEXT_END…FINISH."""
        adapter = self._make_adapter()
        request = Request(model="gpt-4o", messages=[Message.user("hi")])

        def _d(obj: dict) -> str:
            return "data: " + json.dumps(obj)

        sse_lines = [
            "event: response.created",
            _d(
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-4o",
                        "status": "in_progress",
                    },
                }
            ),
            "",
            "event: response.output_text.delta",
            _d(
                {
                    "type": "response.output_text.delta",
                    "delta": "Hello",
                    "output_index": 0,
                    "content_part_index": 0,
                }
            ),
            "",
            "event: response.output_text.delta",
            _d(
                {
                    "type": "response.output_text.delta",
                    "delta": " world",
                    "output_index": 0,
                    "content_part_index": 0,
                }
            ),
            "",
            "event: response.completed",
            _d(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-4o",
                        "status": "completed",
                        "usage": {"input_tokens": 5, "output_tokens": 2},
                    },
                }
            ),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_sse(mock_http, request):
            collected.append(ev)

        kinds = _kinds(collected)

        assert StreamEventKind.TEXT_START in kinds
        assert StreamEventKind.TEXT_END in kinds

        start_idx = kinds.index(StreamEventKind.TEXT_START)
        end_idx = kinds.index(StreamEventKind.TEXT_END)
        finish_idx = kinds.index(StreamEventKind.FINISH)
        delta_indices = [i for i, k in enumerate(kinds) if k == StreamEventKind.TEXT_DELTA]

        assert all(start_idx < i for i in delta_indices)
        assert all(end_idx > i for i in delta_indices)
        assert end_idx < finish_idx, "TEXT_END must come immediately before FINISH"

    @pytest.mark.asyncio
    async def test_openai_adapter_no_text_start_end_for_tool_call(self):
        """A tool-call-only stream must NOT emit TEXT_START / TEXT_END."""
        adapter = self._make_adapter()
        request = Request(model="gpt-4o", messages=[Message.user("hi")])

        def _d(obj: dict) -> str:
            return "data: " + json.dumps(obj)

        fc_item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "my_tool",
            "arguments": "",
        }
        fc_done_item = {**fc_item, "arguments": '{"x":1}'}
        sse_lines = [
            "event: response.created",
            _d({"type": "response.created", "response": {"id": "r1", "model": "gpt-4o"}}),
            "",
            "event: response.output_item.added",
            _d({"type": "response.output_item.added", "item": fc_item}),
            "",
            "event: response.function_call_arguments.delta",
            _d({"type": "response.function_call_arguments.delta", "delta": '{"x":1}'}),
            "",
            "event: response.output_item.done",
            _d({"type": "response.output_item.done", "item": fc_done_item}),
            "",
            "event: response.completed",
            _d(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r1",
                        "model": "gpt-4o",
                        "status": "completed",
                        "usage": {"input_tokens": 5, "output_tokens": 5},
                    },
                }
            ),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_sse(mock_http, request):
            collected.append(ev)

        kinds = _kinds(collected)
        assert StreamEventKind.TEXT_START not in kinds
        assert StreamEventKind.TEXT_END not in kinds


# ================================================================== #
# Item 3 – Gemini adapter TEXT_START / TEXT_END
# ================================================================== #


class TestGeminiAdapterTextStartEnd:
    """Gemini adapter emits TEXT_START before text deltas and TEXT_END before FINISH."""

    def _make_adapter(self):
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        config = ProviderConfig(api_key="test-key", timeout=30.0)
        return GeminiAdapter(config)

    @pytest.mark.asyncio
    async def test_gemini_adapter_emits_text_start_end(self):
        """A simulated Gemini SSE stream has TEXT_START…TEXT_DELTA…TEXT_END…FINISH."""
        adapter = self._make_adapter()
        request = Request(model="gemini-2.0-flash", messages=[Message.user("hi")])

        # Two text chunks + finish
        chunk1 = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}], "role": "model"},
                }
            ],
            "modelVersion": "gemini-2.0-flash",
            "responseId": "r1",
        }
        chunk2 = {
            "candidates": [
                {
                    "content": {"parts": [{"text": " world"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        }

        sse_lines = [
            "data: " + json.dumps(chunk1),
            "",
            "data: " + json.dumps(chunk2),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_stream(mock_http, request, first_chunk=True):
            collected.append(ev)

        kinds = _kinds(collected)

        assert StreamEventKind.TEXT_START in kinds
        assert StreamEventKind.TEXT_END in kinds

        start_idx = kinds.index(StreamEventKind.TEXT_START)
        end_idx = kinds.index(StreamEventKind.TEXT_END)
        finish_idx = kinds.index(StreamEventKind.FINISH)
        delta_indices = [i for i, k in enumerate(kinds) if k == StreamEventKind.TEXT_DELTA]

        assert all(start_idx < i for i in delta_indices)
        assert all(end_idx > i for i in delta_indices)
        assert end_idx < finish_idx, "TEXT_END must come before FINISH"

        acc = StreamAccumulator()
        for ev in collected:
            acc.feed(ev)
        assert acc.response().text == "Hello world"

    @pytest.mark.asyncio
    async def test_gemini_adapter_thinking_does_not_open_text_block(self):
        """Thought parts do NOT trigger TEXT_START / TEXT_END."""
        adapter = self._make_adapter()
        request = Request(model="gemini-2.0-flash-thinking", messages=[Message.user("hi")])

        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking...", "thought": True},
                            {"text": "answer"},
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.0-flash-thinking",
            "responseId": "r1",
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }

        async def _mock_aiter_lines():
            yield "data: " + json.dumps(chunk)
            yield ""

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_stream(mock_http, request, first_chunk=True):
            collected.append(ev)

        kinds = _kinds(collected)

        # THINKING_DELTA emitted for thought part
        assert StreamEventKind.THINKING_DELTA in kinds

        # TEXT_START/END are present (for the "answer" part)
        assert StreamEventKind.TEXT_START in kinds
        assert StreamEventKind.TEXT_END in kinds

        # THINKING_DELTA comes before TEXT_START
        thinking_idx = kinds.index(StreamEventKind.THINKING_DELTA)
        start_idx = kinds.index(StreamEventKind.TEXT_START)
        assert thinking_idx < start_idx


# ================================================================== #
# Items 5-9 – stream_with_tools() agentic tool loop
# ================================================================== #


class _MockClient:
    """Minimal mock client for stream_with_tools() tests."""

    def __init__(self, streams: list[list[StreamEvent]]) -> None:
        # Each call to stream() pops the next stream from the queue
        self._streams = list(streams)
        self.stream_calls: list[Request] = []

    async def stream(
        self, request: Request, abort_signal: Any = None
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        events = self._streams.pop(0)
        return _events(*events)


def _tool_call_stream(call_id: str, tool_name: str, args_json: str) -> list[StreamEvent]:
    """Build a stream of events representing a single tool call."""
    return [
        StreamEvent(kind=StreamEventKind.START, model="test", provider="test"),
        StreamEvent(
            kind=StreamEventKind.TOOL_CALL_START, tool_call_id=call_id, tool_name=tool_name
        ),
        StreamEvent(
            kind=StreamEventKind.TOOL_CALL_DELTA, tool_call_id=call_id, arguments_delta=args_json
        ),
        StreamEvent(kind=StreamEventKind.TOOL_CALL_END, tool_call_id=call_id),
        StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.TOOL_CALLS),
    ]


def _text_stream(text: str) -> list[StreamEvent]:
    """Build a stream of events representing a plain text reply."""
    return [
        StreamEvent(kind=StreamEventKind.START, model="test", provider="test"),
        StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=text),
        StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
    ]


class TestStreamWithTools:
    """stream_with_tools() agentic loop tests."""

    @pytest.mark.asyncio
    async def test_stream_with_tools_no_tools_returns_text(self):
        """When no tools are declared, stream_with_tools streams text normally."""
        from attractor_llm.generate import stream_with_tools

        text_events = _text_stream("Hello!")
        mock_client = _MockClient([text_events])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="hi",
        )

        chunks = await _collect(result.text_stream)
        assert "".join(chunks) == "Hello!"
        assert len(mock_client.stream_calls) == 1  # only one round

    @pytest.mark.asyncio
    async def test_stream_with_tools_executes_tool_and_continues(self):
        """Mock adapter returns tool call stream then text stream. Verify tools executed."""
        from attractor_llm.generate import stream_with_tools

        executed: list[str] = []

        async def my_tool(x: str) -> str:
            executed.append(x)
            return f"result:{x}"

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            execute=my_tool,
        )

        round1 = _tool_call_stream("call_1", "my_tool", '{"x": "hello"}')
        round2 = _text_stream("Done!")

        mock_client = _MockClient([round1, round2])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="use the tool",
            tools=[tool],
        )

        # Consume the full stream
        await result.response()

        # Tool was executed
        assert executed == ["hello"]
        # Two LLM calls were made
        assert len(mock_client.stream_calls) == 2
        # Second call includes the tool result message
        second_request = mock_client.stream_calls[1]
        tool_result_msgs = [
            m
            for m in second_request.messages
            if any(p.kind == ContentPartKind.TOOL_RESULT for p in m.content)
        ]
        assert len(tool_result_msgs) == 1, "Tool result must be appended to history"
        tool_result_part = tool_result_msgs[0].content[0]
        assert "result:hello" in (tool_result_part.output or "")

    @pytest.mark.asyncio
    async def test_stream_with_tools_text_stream_yields_all_text(self):
        """text_stream yields TEXT_DELTA text from ALL rounds (tool call then text)."""
        from attractor_llm.generate import stream_with_tools

        async def multiplier(n: int) -> str:
            return str(n * 2)

        tool = Tool(
            name="multiplier",
            description="Multiplies by 2",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
            execute=multiplier,
        )

        # Round 1: no text, just tool call
        round1 = _tool_call_stream("call_x", "multiplier", '{"n": 7}')
        # Round 2: final text with the answer
        round2 = _text_stream("The answer is 14")

        mock_client = _MockClient([round1, round2])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="what is 7*2?",
            tools=[tool],
        )

        chunks = await _collect(result.text_stream)
        full_text = "".join(chunks)
        assert "The answer is 14" in full_text

    @pytest.mark.asyncio
    async def test_stream_with_tools_max_rounds_limits(self):
        """max_rounds=0 prevents any tool execution; loop stops after first stream."""
        from attractor_llm.generate import stream_with_tools

        executed: list[str] = []

        async def my_tool(x: str) -> str:
            executed.append(x)
            return f"result:{x}"

        tool = Tool(
            name="my_tool",
            description="test",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            execute=my_tool,
        )

        # The adapter would return a tool call, but max_rounds=0 stops execution
        round1 = _tool_call_stream("call_1", "my_tool", '{"x": "hello"}')

        mock_client = _MockClient([round1])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="use tool",
            tools=[tool],
            max_rounds=0,
        )

        await result.response()

        # Tool was NOT executed
        assert executed == []
        # Only 1 LLM call was made (max_rounds=0 stops immediately)
        assert len(mock_client.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tools_max_rounds_limits_multi_turn(self):
        """max_rounds=1 caps the loop at range(2) iterations.

        The loop runs `for _round in range(max_rounds + 1)`.  With max_rounds=1
        that is exactly 2 LLM calls.  If both return tool calls, both execute and
        the third stream (text) is never consumed.
        """
        from attractor_llm.generate import stream_with_tools

        executed: list[int] = []

        async def counter() -> str:
            executed.append(len(executed) + 1)
            return "ok"

        tool = Tool(
            name="counter",
            description="counts calls",
            parameters={"type": "object", "properties": {}},
            execute=counter,
        )

        round1 = _tool_call_stream("call_1", "counter", "{}")
        round2 = _tool_call_stream("call_2", "counter", "{}")
        # round3 is text -- unreachable because the loop exhausted after round2
        round3 = _text_stream("unreachable")

        mock_client = _MockClient([round1, round2, round3])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="go",
            tools=[tool],
            max_rounds=1,
        )

        await result.response()

        # range(max_rounds+1) = range(2) → both rounds execute their tool calls
        assert len(executed) == 2
        # Exactly 2 LLM calls; round3 (text) is never reached
        assert len(mock_client.stream_calls) == 2
        # round3 still unconsumed in the queue
        assert len(mock_client._streams) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tools_unknown_tool_returns_error(self):
        """If the model calls an unknown tool, the error is fed back as tool result."""
        from attractor_llm.generate import stream_with_tools

        # No tools registered, but model asks for one
        round1 = _tool_call_stream("call_1", "nonexistent_tool", '{"x": 1}')
        round2 = _text_stream("I cannot do that.")

        mock_client = _MockClient([round1, round2])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="go",
            tools=[Tool(name="other_tool", description="another tool", execute=None)],
        )

        await result.response()

        # Two LLM calls made (error result fed back)
        assert len(mock_client.stream_calls) == 2
        # Second call has a tool result with an error
        second_msgs = mock_client.stream_calls[1].messages
        tool_result_parts = [
            p for m in second_msgs for p in m.content if p.kind == ContentPartKind.TOOL_RESULT
        ]
        assert len(tool_result_parts) == 1
        assert tool_result_parts[0].is_error is True

    @pytest.mark.asyncio
    async def test_stream_with_tools_invalid_prompt_and_messages_raises(self):
        """Providing both prompt and messages raises InvalidRequestError."""
        from attractor_llm.errors import InvalidRequestError
        from attractor_llm.generate import stream_with_tools

        mock_client = _MockClient([])  # type: ignore[arg-type]

        with pytest.raises(InvalidRequestError, match="Cannot provide both"):
            await stream_with_tools(
                mock_client,  # type: ignore[arg-type]
                model="test-model",
                prompt="hi",
                messages=[Message.user("hi")],
            )

    @pytest.mark.asyncio
    async def test_stream_with_tools_no_prompt_and_no_messages_raises(self):
        """Providing neither prompt nor messages raises InvalidRequestError."""
        from attractor_llm.errors import InvalidRequestError
        from attractor_llm.generate import stream_with_tools

        mock_client = _MockClient([])  # type: ignore[arg-type]

        with pytest.raises(InvalidRequestError, match="Must provide either"):
            await stream_with_tools(
                mock_client,  # type: ignore[arg-type]
                model="test-model",
            )

    @pytest.mark.asyncio
    async def test_stream_with_tools_using_messages_kwarg(self):
        """stream_with_tools accepts a messages= list instead of prompt=."""
        from attractor_llm.generate import stream_with_tools

        mock_client = _MockClient([_text_stream("hi back")])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            messages=[Message.user("hello")],
        )

        chunks = await _collect(result.text_stream)
        assert "hi back" in "".join(chunks)

    # ------------------------------------------------------------------ #
    # Swarm-review additions
    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_stream_with_tools_passive_tool_returns_early(self):
        """A tool with execute=None (passive) causes loop to stop after first round.

        §5.5: passive tools return control to the caller; stream_with_tools must
        not attempt execution and must not raise.
        """
        from attractor_llm.generate import stream_with_tools

        passive_tool = Tool(
            name="passive_tool",
            description="A tool with no execute handler",
            parameters={"type": "object", "properties": {}},
            execute=None,  # passive
        )

        # Round 1 returns a tool call for the passive tool; no round 2 should happen
        round1 = _tool_call_stream("call_p1", "passive_tool", "{}")

        mock_client = _MockClient([round1])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="use passive tool",
            tools=[passive_tool],
        )

        # Consuming the stream must not raise
        await result.response()

        # Only 1 LLM call -- loop exited early on passive tool
        assert len(mock_client.stream_calls) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tools_parallel_execution(self):
        """Two tool calls in one round are executed in parallel via asyncio.gather.

        Both tools must run; call-order tracking confirms both were invoked.
        """
        import asyncio

        from attractor_llm.generate import stream_with_tools

        call_log: list[str] = []

        async def tool_a() -> str:
            call_log.append("a_start")
            await asyncio.sleep(0)  # yield to event loop
            call_log.append("a_end")
            return "result_a"

        async def tool_b() -> str:
            call_log.append("b_start")
            await asyncio.sleep(0)
            call_log.append("b_end")
            return "result_b"

        empty_params: dict[str, Any] = {"type": "object", "properties": {}}
        tools = [
            Tool(name="tool_a", description="tool A", parameters=empty_params, execute=tool_a),
            Tool(name="tool_b", description="tool B", parameters=empty_params, execute=tool_b),
        ]

        # Build a stream that returns two tool calls in a single round
        two_tool_round: list[StreamEvent] = [
            StreamEvent(kind=StreamEventKind.START, model="test", provider="test"),
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_START, tool_call_id="c1", tool_name="tool_a"
            ),
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_DELTA, tool_call_id="c1", arguments_delta="{}"
            ),
            StreamEvent(kind=StreamEventKind.TOOL_CALL_END, tool_call_id="c1"),
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_START, tool_call_id="c2", tool_name="tool_b"
            ),
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_DELTA, tool_call_id="c2", arguments_delta="{}"
            ),
            StreamEvent(kind=StreamEventKind.TOOL_CALL_END, tool_call_id="c2"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.TOOL_CALLS),
        ]
        round2 = _text_stream("both done")

        mock_client = _MockClient([two_tool_round, round2])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="run both tools",
            tools=tools,
        )

        await result.response()

        # Both tools were called
        assert "a_start" in call_log
        assert "b_start" in call_log
        assert "a_end" in call_log
        assert "b_end" in call_log

        # Two LLM calls: tool round + final text round
        assert len(mock_client.stream_calls) == 2

        # Both tool results appear in the second request's history
        second_messages = mock_client.stream_calls[1].messages
        tool_result_parts = [
            p for m in second_messages for p in m.content if p.kind == ContentPartKind.TOOL_RESULT
        ]
        result_outputs = {p.output for p in tool_result_parts}
        assert "result_a" in result_outputs
        assert "result_b" in result_outputs

    @pytest.mark.asyncio
    async def test_stream_with_tools_abort_signal(self):
        """A pre-set abort signal causes AbortError to be raised while consuming stream."""
        from attractor_llm.errors import AbortError
        from attractor_llm.generate import stream_with_tools

        abort = MagicMock()
        abort.is_set = True  # pre-set before the call

        mock_client = _MockClient([_text_stream("hi")])  # type: ignore[arg-type]

        result = await stream_with_tools(
            mock_client,  # type: ignore[arg-type]
            model="test-model",
            prompt="hi",
            abort_signal=abort,
        )

        with pytest.raises(AbortError, match="aborted"):
            await result.response()


# ================================================================== #
# Swarm-review Fix 1 – TEXT_END ordering in OpenAI + Gemini adapters
# ================================================================== #


class TestOpenAITextEndBeforeToolCallStart:
    """TEXT_END must be emitted before TOOL_CALL_START when text precedes a tool call."""

    def _make_adapter(self):
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        config = ProviderConfig(api_key="test-key", timeout=30.0)
        return OpenAIAdapter(config)

    @pytest.mark.asyncio
    async def test_openai_text_end_before_tool_call_start(self):
        """Interleaved text + tool call: TEXT_END must immediately precede TOOL_CALL_START."""
        adapter = self._make_adapter()
        request = Request(model="gpt-4o", messages=[Message.user("hi")])

        def _d(obj: dict) -> str:
            return "data: " + json.dumps(obj)

        fc_item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "my_tool",
            "arguments": "",
        }
        fc_done_item = {**fc_item, "arguments": "{}"}

        sse_lines = [
            # Model starts with text
            "event: response.created",
            _d({"type": "response.created", "response": {"id": "r1", "model": "gpt-4o"}}),
            "",
            "event: response.output_text.delta",
            _d(
                {
                    "type": "response.output_text.delta",
                    "delta": "Let me check",
                    "output_index": 0,
                    "content_part_index": 0,
                }
            ),
            "",
            # Then switches to a tool call
            "event: response.output_item.added",
            _d({"type": "response.output_item.added", "item": fc_item}),
            "",
            "event: response.function_call_arguments.delta",
            _d({"type": "response.function_call_arguments.delta", "delta": "{}"}),
            "",
            "event: response.output_item.done",
            _d({"type": "response.output_item.done", "item": fc_done_item}),
            "",
            "event: response.completed",
            _d(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r1",
                        "model": "gpt-4o",
                        "status": "completed",
                        "usage": {"input_tokens": 5, "output_tokens": 5},
                    },
                }
            ),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_sse(mock_http, request):
            collected.append(ev)

        kinds = _kinds(collected)

        # TEXT_END must be present
        assert StreamEventKind.TEXT_END in kinds, "TEXT_END missing from interleaved stream"
        # TOOL_CALL_START must be present
        assert StreamEventKind.TOOL_CALL_START in kinds

        text_end_idx = kinds.index(StreamEventKind.TEXT_END)
        tool_call_start_idx = kinds.index(StreamEventKind.TOOL_CALL_START)

        assert text_end_idx < tool_call_start_idx, (
            f"TEXT_END (pos {text_end_idx}) must come before "
            f"TOOL_CALL_START (pos {tool_call_start_idx})"
        )


class TestGeminiTextEndBeforeToolCallStart:
    """TEXT_END must be emitted before TOOL_CALL_START when text precedes a function call."""

    def _make_adapter(self):
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        config = ProviderConfig(api_key="test-key", timeout=30.0)
        return GeminiAdapter(config)

    @pytest.mark.asyncio
    async def test_gemini_text_end_before_tool_call_start(self):
        """Interleaved text + functionCall: TEXT_END must precede TOOL_CALL_START."""
        adapter = self._make_adapter()
        request = Request(model="gemini-2.0-flash", messages=[Message.user("hi")])

        # Single chunk that has both a text part AND a functionCall part
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "I'll help you with that."},
                            {
                                "functionCall": {
                                    "name": "my_tool",
                                    "args": {"x": 1},
                                }
                            },
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.0-flash",
            "responseId": "r1",
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10},
        }

        sse_lines = [
            "data: " + json.dumps(chunk),
            "",
        ]

        async def _mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_http = MagicMock()
        mock_http.aiter_lines = _mock_aiter_lines

        collected: list[StreamEvent] = []
        async for ev in adapter._parse_stream(mock_http, request, first_chunk=True):
            collected.append(ev)

        kinds = _kinds(collected)

        # Both TEXT_END and TOOL_CALL_START must be present
        assert StreamEventKind.TEXT_END in kinds, "TEXT_END missing from interleaved stream"
        assert StreamEventKind.TOOL_CALL_START in kinds

        text_end_idx = kinds.index(StreamEventKind.TEXT_END)
        tool_call_start_idx = kinds.index(StreamEventKind.TOOL_CALL_START)

        assert text_end_idx < tool_call_start_idx, (
            f"TEXT_END (pos {text_end_idx}) must come before "
            f"TOOL_CALL_START (pos {tool_call_start_idx})"
        )
