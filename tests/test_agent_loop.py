"""Tests for the Coding Agent Loop: session, tools, events, truncation, abort."""

from __future__ import annotations

import os

import pytest

from attractor_agent.abort import AbortSignal
from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.session import Session, SessionState
from attractor_agent.tools.core import ALL_CORE_TOOLS, set_allowed_roots
from attractor_agent.tools.registry import ToolRegistry
from attractor_agent.truncation import TruncationLimits, truncate_output
from attractor_llm.client import Client
from attractor_llm.types import ContentPart

# ================================================================== #
# AbortSignal
# ================================================================== #


class TestAbortSignal:
    def test_initial_state(self):
        sig = AbortSignal()
        assert sig.is_set is False

    def test_set(self):
        sig = AbortSignal()
        sig.set()
        assert sig.is_set is True

    def test_double_set_is_noop(self):
        sig = AbortSignal()
        fired: list[bool] = []
        sig.on_abort(lambda: fired.append(True))
        sig.set()
        sig.set()
        assert fired == [True]  # only fired once

    def test_callback_fires_on_set(self):
        sig = AbortSignal()
        fired: list[bool] = []
        sig.on_abort(lambda: fired.append(True))
        sig.set()
        assert fired == [True]

    def test_late_registration_fires_immediately(self):
        sig = AbortSignal()
        sig.set()
        fired: list[bool] = []
        sig.on_abort(lambda: fired.append(True))
        assert fired == [True]


# ================================================================== #
# EventEmitter
# ================================================================== #


class TestEventEmitter:
    @pytest.mark.asyncio
    async def test_emit_and_receive(self):
        emitter = EventEmitter()
        received: list[EventKind] = []

        async def handler(e: SessionEvent) -> None:
            received.append(e.kind)

        emitter.on(handler)
        await emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
        assert received == [EventKind.SESSION_START]

    @pytest.mark.asyncio
    async def test_off_removes_handler(self):
        emitter = EventEmitter()
        received: list[EventKind] = []

        async def handler(e: SessionEvent) -> None:
            received.append(e.kind)

        emitter.on(handler)
        await emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
        emitter.off(handler)
        await emitter.emit(SessionEvent(kind=EventKind.SESSION_END))
        assert len(received) == 1  # only the first event

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_break_emit(self):
        emitter = EventEmitter()
        received: list[str] = []

        async def bad_handler(e: SessionEvent) -> None:
            raise RuntimeError("boom")

        async def good_handler(e: SessionEvent) -> None:
            received.append("ok")

        emitter.on(bad_handler)
        emitter.on(good_handler)
        await emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
        assert received == ["ok"]


# ================================================================== #
# Truncation
# ================================================================== #


class TestTruncation:
    def test_short_text_passthrough(self):
        result, truncated = truncate_output("hello")
        assert result == "hello"
        assert truncated is False

    def test_char_truncation(self):
        long_text = "x" * 50_000
        result, truncated = truncate_output(long_text, TruncationLimits(max_chars=1000))
        assert truncated is True
        assert len(result) < 1400
        assert "characters were removed from the middle" in result

    def test_line_truncation(self):
        many_lines = "\n".join(f"line {i}" for i in range(1000))
        result, truncated = truncate_output(many_lines, TruncationLimits(max_lines=100))
        assert truncated is True
        assert "lines were removed from the middle" in result

    def test_empty_string(self):
        result, truncated = truncate_output("")
        assert result == ""
        assert truncated is False

    def test_per_tool_presets(self):
        assert TruncationLimits.for_tool("read_file").max_chars == 50_000
        assert TruncationLimits.for_tool("shell").max_chars == 30_000
        assert TruncationLimits.for_tool("unknown").max_chars == 30_000


# ================================================================== #
# ToolRegistry
# ================================================================== #


class TestToolRegistry:
    def test_register_and_lookup(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)
        assert registry.has("read_file")
        assert registry.has("write_file")
        assert registry.has("edit_file")
        assert registry.has("shell")
        assert registry.has("grep")
        assert registry.has("glob")
        assert len(registry.definitions()) == 6

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        registry = ToolRegistry()
        tc = ContentPart.tool_call_part("tc-1", "nonexistent", "{}")
        result = await registry.execute_tool_call(tc)
        assert result.is_error
        assert "Unknown tool" in (result.output or "")


# ================================================================== #
# Core Tools (file ops, shell, grep, glob)
# ================================================================== #


class TestCoreTools:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path):
        """Create a temp directory and confine tools to it."""
        self.sandbox = tmp_path
        set_allowed_roots([str(tmp_path)])
        yield
        set_allowed_roots([os.getcwd()])

    @pytest.mark.asyncio
    async def test_read_file(self):
        test_file = self.sandbox / "test.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part("tc-1", "read_file", {"path": str(test_file)})
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert "line 1" in (result.output or "")
        assert "line 3" in (result.output or "")

    @pytest.mark.asyncio
    async def test_write_file(self):
        target = self.sandbox / "output.txt"

        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "write_file",
            {"path": str(target), "content": "hello world"},
        )
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert target.exists()
        assert target.read_text() == "hello world"

    @pytest.mark.asyncio
    async def test_edit_file(self):
        test_file = self.sandbox / "edit.txt"
        test_file.write_text("hello world")

        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "edit_file",
            {"path": str(test_file), "old_string": "hello", "new_string": "goodbye"},
        )
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert "goodbye" in test_file.read_text()

    @pytest.mark.asyncio
    async def test_shell_echo(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "shell",
            {"command": "echo hello_test", "working_dir": str(self.sandbox)},
        )
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert "hello_test" in (result.output or "")

    @pytest.mark.asyncio
    async def test_shell_timeout(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "shell",
            {"command": "sleep 10", "timeout": 1, "working_dir": str(self.sandbox)},
        )
        result = await registry.execute_tool_call(tc)
        assert result.is_error
        assert "timed out" in (result.output or "").lower()

    @pytest.mark.asyncio
    async def test_shell_deny_list(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "shell",
            {"command": "rm -rf /"},
        )
        result = await registry.execute_tool_call(tc)
        assert result.is_error
        assert "blocked" in (result.output or "").lower() or "Permission" in (result.output or "")

    @pytest.mark.asyncio
    async def test_path_confinement_blocks_outside(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "write_file",
            {"path": "/tmp/should_not_work.txt", "content": "bad"},
        )
        result = await registry.execute_tool_call(tc)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_shell_working_dir_confinement(self):
        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "shell",
            {"command": "echo bad", "working_dir": "/tmp"},
        )
        result = await registry.execute_tool_call(tc)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_grep(self):
        test_file = self.sandbox / "search.txt"
        test_file.write_text("foo bar\nbaz qux\nfoo again\n")

        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "grep",
            {"pattern": "foo", "path": str(test_file)},
        )
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert "foo" in (result.output or "")

    @pytest.mark.asyncio
    async def test_glob(self):
        (self.sandbox / "a.py").write_text("")
        (self.sandbox / "b.py").write_text("")
        (self.sandbox / "c.txt").write_text("")

        registry = ToolRegistry()
        registry.register_many(ALL_CORE_TOOLS)

        tc = ContentPart.tool_call_part(
            "tc-1",
            "glob",
            {"pattern": "*.py", "path": str(self.sandbox)},
        )
        result = await registry.execute_tool_call(tc)
        assert not result.is_error
        assert "a.py" in (result.output or "")
        assert "b.py" in (result.output or "")
        assert "c.txt" not in (result.output or "")


# ================================================================== #
# Session
# ================================================================== #


class TestSession:
    def test_initial_state(self):
        client = Client()
        session = Session(client=client)
        assert session.state == SessionState.IDLE
        assert session.turn_count == 0

    @pytest.mark.asyncio
    async def test_lifecycle_events(self):
        client = Client()
        session = Session(client=client)
        events: list[EventKind] = []
        session.events.on(lambda e: events.append(e.kind))

        async with session:
            assert EventKind.SESSION_START in events
        assert EventKind.SESSION_END in events
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_session_rejects_submit(self):
        client = Client()
        session = Session(client=client)
        await session.close()

        with pytest.raises(RuntimeError, match="closed"):
            await session.submit("test")

    @pytest.mark.asyncio
    async def test_abort_returns_immediately(self):
        client = Client()
        abort = AbortSignal()
        abort.set()
        session = Session(client=client, abort_signal=abort)

        result = await session.submit("test")
        assert result == "[Session aborted]"

    @pytest.mark.asyncio
    async def test_error_emits_error_and_turn_end(self):
        """When LLM call fails, both ERROR and TURN_END events must fire."""
        client = Client()  # No adapters registered -- will fail
        session = Session(client=client)
        events: list[EventKind] = []
        session.events.on(lambda e: events.append(e.kind))

        result = await session.submit("test")
        assert "[Error:" in result
        assert EventKind.TURN_START in events
        assert EventKind.ERROR in events
        assert EventKind.TURN_END in events

    def test_tools_registered_via_constructor(self):
        client = Client()
        session = Session(client=client, tools=ALL_CORE_TOOLS)
        assert session.tool_registry.has("read_file")
        assert len(session.tool_registry.definitions()) == 6
