"""Wave 3 spec compliance tests: SteeringTurn, TOOL_CALL_END events,
configurable truncation limits, and corrected default line limits.

Covers spec items #9-#12.
"""

from __future__ import annotations

import pytest

from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.session import Session, SessionConfig, SteeringTurn
from attractor_agent.tools.registry import ToolRegistry
from attractor_agent.truncation import TruncationLimits, truncate_output
from attractor_llm.client import Client
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    Message,
    Response,
    Role,
    Usage,
)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _text_response(text: str) -> Response:
    """Build a simple text-only Response."""
    return Response(
        message=Message.assistant(text),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _tool_call_response(name: str, args: dict | None = None) -> Response:
    """Build a Response containing a single tool call."""
    return Response(
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart(
                    kind=ContentPartKind.TOOL_CALL,
                    name=name,
                    tool_call_id=f"call_{name}",
                    arguments=args or {},
                ),
            ],
        ),
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class FakeClient(Client):
    """Deterministic LLM client for testing."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, request):  # noqa: ANN001
        resp = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        return resp

    async def stream(self, request):  # noqa: ANN001
        raise NotImplementedError


def _make_echo_tool(output: str = "ok") -> list:
    """Create a simple echo tool that returns *output*."""
    from attractor_llm.types import Tool

    async def _execute(**kwargs: object) -> str:  # noqa: ARG001
        return output

    return [Tool(name="echo", description="echo", parameters={}, execute=_execute)]


# ================================================================== #
# Item #9 -- SteeringTurn type
# ================================================================== #


class TestSteeringTurn:
    """SteeringTurn is stored in history, not plain Message."""

    def test_dataclass_fields(self):
        turn = SteeringTurn(content="focus on tests")
        assert turn.content == "focus on tests"
        assert turn.timestamp is None

    def test_dataclass_with_timestamp(self):
        turn = SteeringTurn(content="x", timestamp=1234567890.0)
        assert turn.timestamp == 1234567890.0

    @pytest.mark.asyncio
    async def test_steering_stored_as_steering_turn(self):
        """Steering messages should appear as SteeringTurn in history."""
        client = FakeClient(
            [
                _tool_call_response("echo"),
                _text_response("done"),
            ]
        )
        session = Session(client=client, tools=_make_echo_tool())
        session.steer("focus on tests")
        await session.submit("hello")

        steering_entries = [e for e in session.history if isinstance(e, SteeringTurn)]
        assert len(steering_entries) >= 1
        assert steering_entries[0].content == "focus on tests"

    @pytest.mark.asyncio
    async def test_steering_not_plain_message(self):
        """Steering messages must NOT be stored as plain Message(role=USER)."""
        client = FakeClient(
            [
                _tool_call_response("echo"),
                _text_response("done"),
            ]
        )
        session = Session(client=client, tools=_make_echo_tool())
        session.steer("focus on tests")
        await session.submit("hello")

        # The only USER message should be the original prompt
        user_messages = [
            e for e in session.history if isinstance(e, Message) and e.role == Role.USER
        ]
        assert len(user_messages) == 1
        assert user_messages[0].text == "hello"

    @pytest.mark.asyncio
    async def test_steering_converts_to_user_message_for_llm(self):
        """_build_messages should convert SteeringTurn to Message(role=USER)."""
        client = FakeClient([_text_response("ok")])
        session = Session(client=client)

        # Manually insert a SteeringTurn
        session._history.append(Message.user("hello"))
        session._history.append(SteeringTurn(content="be concise"))

        messages = session._build_messages()

        assert all(isinstance(m, Message) for m in messages)
        # Second message is the converted steering turn
        assert messages[1].role == Role.USER
        assert messages[1].text == "[STEERING] be concise"

    @pytest.mark.asyncio
    async def test_loop_detection_creates_steering_turn(self):
        """Loop detection warning should create a SteeringTurn entry."""
        # All responses are the same tool call -> triggers loop detection
        client = FakeClient([_tool_call_response("echo")] * 10)
        config = SessionConfig(
            loop_detection_window=4,
            loop_detection_threshold=3,
        )
        session = Session(client=client, config=config, tools=_make_echo_tool())
        result = await session.submit("go")

        assert "Loop detected" in result

        steering_entries = [e for e in session.history if isinstance(e, SteeringTurn)]
        assert len(steering_entries) >= 1
        assert "Loop detected" in steering_entries[0].content


# ================================================================== #
# Item #10 -- TOOL_CALL_END carries full untruncated output
# ================================================================== #


class TestToolCallEndEvent:
    """TOOL_CALL_END event should carry raw (untruncated) output."""

    @pytest.mark.asyncio
    async def test_event_carries_full_output(self):
        """Even when output is truncated, the event gets the raw string."""
        large_output = "x" * 100_000  # Way over default limits

        from attractor_llm.types import Tool

        async def _big(**kwargs: object) -> str:  # noqa: ARG001
            return large_output

        tool = Tool(name="shell", description="shell", parameters={}, execute=_big)

        emitter = EventEmitter()
        events: list[SessionEvent] = []

        async def handler(e: SessionEvent) -> None:
            if e.kind == EventKind.TOOL_CALL_END:
                events.append(e)

        emitter.on(handler)

        registry = ToolRegistry(event_emitter=emitter)
        registry.register(tool)

        tc = ContentPart(
            kind=ContentPartKind.TOOL_CALL,
            name="shell",
            tool_call_id="call_1",
            arguments={},
        )
        result = await registry.execute_tool_call(tc)

        # The result sent to LLM should be truncated
        assert len(result.output or "") < len(large_output)

        # The event should carry the full untruncated output
        assert len(events) == 1
        assert "output" in events[0].data
        assert events[0].data["output"] == large_output

    @pytest.mark.asyncio
    async def test_event_output_not_just_length(self):
        """The event data should have 'output' string, not 'output_length' int."""
        from attractor_llm.types import Tool

        async def _exec(**kwargs: object) -> str:  # noqa: ARG001
            return "hello world"

        tool = Tool(name="echo", description="echo", parameters={}, execute=_exec)

        emitter = EventEmitter()
        events: list[SessionEvent] = []

        async def handler(e: SessionEvent) -> None:
            if e.kind == EventKind.TOOL_CALL_END:
                events.append(e)

        emitter.on(handler)

        registry = ToolRegistry(event_emitter=emitter)
        registry.register(tool)

        tc = ContentPart(
            kind=ContentPartKind.TOOL_CALL,
            name="echo",
            tool_call_id="call_1",
            arguments={},
        )
        await registry.execute_tool_call(tc)

        assert len(events) == 1
        assert "output" in events[0].data
        assert isinstance(events[0].data["output"], str)
        assert "output_length" not in events[0].data


# ================================================================== #
# Item #11 -- SessionConfig truncation overrides
# ================================================================== #


class TestConfigurableTruncationLimits:
    """SessionConfig tool_output_limits / tool_line_limits override defaults."""

    def test_output_limits_override(self):
        limits = TruncationLimits.for_tool("shell", output_limits={"shell": 5_000})
        assert limits.max_chars == 5_000
        # Line limit should remain at default
        assert limits.max_lines == 256

    def test_line_limits_override(self):
        limits = TruncationLimits.for_tool("grep", line_limits={"grep": 50})
        assert limits.max_lines == 50
        # Char limit should remain at default
        assert limits.max_chars == 20_000

    def test_both_overrides(self):
        limits = TruncationLimits.for_tool(
            "shell",
            output_limits={"shell": 1_000},
            line_limits={"shell": 10},
        )
        assert limits.max_chars == 1_000
        assert limits.max_lines == 10

    def test_override_for_different_tool_ignored(self):
        """Overrides for other tools don't affect the queried tool."""
        limits = TruncationLimits.for_tool(
            "shell",
            output_limits={"grep": 1},
            line_limits={"grep": 1},
        )
        # Should get default shell limits
        assert limits.max_chars == 30_000
        assert limits.max_lines == 256

    def test_none_overrides_use_defaults(self):
        limits = TruncationLimits.for_tool("shell", output_limits=None, line_limits=None)
        assert limits.max_chars == 30_000
        assert limits.max_lines == 256

    @pytest.mark.asyncio
    async def test_config_flows_through_session(self):
        """SessionConfig limits should reach the ToolRegistry."""
        config = SessionConfig(
            tool_output_limits={"echo": 100},
            tool_line_limits={"echo": 5},
        )
        client = FakeClient([_text_response("ok")])
        session = Session(client=client, config=config)

        assert session._tool_registry._output_limits == {"echo": 100}
        assert session._tool_registry._line_limits == {"echo": 5}

    @pytest.mark.asyncio
    async def test_config_overrides_truncation_in_execution(self):
        """Tool output should respect config overrides during execution."""
        from attractor_llm.types import Tool

        # Tool returns 200 chars
        async def _exec(**kwargs: object) -> str:  # noqa: ARG001
            return "a" * 200

        tool = Tool(name="mytool", description="test", parameters={}, execute=_exec)

        registry = ToolRegistry(
            event_emitter=EventEmitter(),
            tool_output_limits={"mytool": 50},  # Only allow 50 chars
        )
        registry.register(tool)

        tc = ContentPart(
            kind=ContentPartKind.TOOL_CALL,
            name="mytool",
            tool_call_id="call_1",
            arguments={},
        )
        result = await registry.execute_tool_call(tc)

        # Output should be truncated because limit is 50 chars
        assert result.output is not None
        assert "truncated" in result.output


# ================================================================== #
# Item #12 -- Correct default line limits
# ================================================================== #


class TestDefaultLineLimits:
    """Verify spec-mandated default line limits: shell=256, grep=200, glob=500."""

    def test_shell_256(self):
        limits = TruncationLimits.for_tool("shell")
        assert limits.max_lines == 256

    def test_grep_200(self):
        limits = TruncationLimits.for_tool("grep")
        assert limits.max_lines == 200

    def test_glob_500(self):
        limits = TruncationLimits.for_tool("glob")
        assert limits.max_lines == 500

    def test_read_file_unchanged(self):
        limits = TruncationLimits.for_tool("read_file")
        assert limits.max_lines == 1000

    def test_write_file_unchanged(self):
        limits = TruncationLimits.for_tool("write_file")
        assert limits.max_lines == 50

    def test_unknown_tool_uses_global_default(self):
        limits = TruncationLimits.for_tool("unknown_tool_xyz")
        assert limits.max_lines == 500
        assert limits.max_chars == 30_000


# ================================================================== #
# Truncation ordering: chars first, then lines
# ================================================================== #


class TestTruncationOrdering:
    """Verify the two-pass truncation: chars first, then lines."""

    def test_chars_truncated_first(self):
        """When output exceeds char limit, chars are truncated before lines."""
        # 100 lines of 1000 chars each = 100,000 chars
        lines = ["x" * 1000 for _ in range(100)]
        output = "\n".join(lines)

        limits = TruncationLimits(max_chars=5_000, max_lines=50)
        result, was_truncated = truncate_output(output, limits)

        assert was_truncated is True
        assert len(result) < len(output)
        assert "characters were removed from the middle" in result

    def test_lines_truncated_after_chars(self):
        """Line truncation operates on the already char-truncated output."""
        # Many short lines -> only line limit triggers
        lines = [f"line {i}" for i in range(1000)]
        output = "\n".join(lines)

        limits = TruncationLimits(max_chars=100_000, max_lines=10)
        result, was_truncated = truncate_output(output, limits)

        assert was_truncated is True
        assert "lines were removed from the middle" in result

    def test_both_passes_can_trigger(self):
        """Both char and line truncation can trigger in sequence.

        The line pass may remove the char-omission marker, so we verify
        that the result is bounded by *both* limits.
        """
        # 50k short lines → exceeds both char and line limits
        lines = ["y" * 10 for _ in range(50_000)]
        output = "\n".join(lines)

        limits = TruncationLimits(max_chars=10_000, max_lines=20)
        result, was_truncated = truncate_output(output, limits)

        assert was_truncated is True
        # Line count bounded by max_lines (+ a few for omission markers and
        # the longer WARNING text which may wrap across multiple lines)
        assert len(result.split("\n")) <= 20 + 10
        # Char count well under original (both passes ran); the WARNING
        # marker text adds overhead so we allow up to 11k
        assert len(result) < 11_000

    def test_no_truncation_when_within_limits(self):
        result, was_truncated = truncate_output("short", TruncationLimits())
        assert was_truncated is False
        assert result == "short"
