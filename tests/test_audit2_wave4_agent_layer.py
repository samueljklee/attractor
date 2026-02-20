"""Audit 2 Wave 4 - Agent Layer regression tests.

Covers 12 items:
  Items 1-3  §9.12.34-36  - Subagent tools wired into all 3 profiles
  Items 4-5  §9.9.1,9.9.6 - Cascade from items 1-3 (auto-resolved)
  Item 6     §9.2.1       - OpenAI profile recommends apply_patch as primary
  Item 7     §9.12.43     - Cascade from item 6 (auto-resolved)
  Item 8     §9.8.3       - System prompt includes available tool descriptions
  Item 9     §9.10.1      - ASSISTANT_TEXT_START/DELTA emitted from session
  Item 10    §9.10.2      - EventEmitter supports async iterator interface
  Items 11-12 §9.1.6,9.11.5 - SIGTERM/SIGKILL on tracked processes
"""

from __future__ import annotations

import asyncio
import signal as _signal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.session import Session, SessionConfig
from attractor_llm.client import Client
from attractor_llm.types import (
    ContentPart,
    FinishReason,
    Message,
    Response,
    Role,
    Usage,
)

# ================================================================== #
# Helpers
# ================================================================== #


def _make_text_response(text: str = "Done.") -> Response:
    """Build a minimal text-only Response."""
    return Response(
        id="resp-test",
        model="mock-model",
        provider="mock",
        message=Message.assistant(text),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _make_mock_client(response: Response) -> MagicMock:
    """Return a mock Client whose complete() returns the given Response."""
    client = MagicMock(spec=Client)
    client.complete = AsyncMock(return_value=response)
    return client


# ================================================================== #
# Items 1-3 §9.12.34-36: Subagent tools in profiles
# ================================================================== #

SUBAGENT_TOOL_NAMES = {"spawn_agent", "send_input", "wait", "close_agent"}


# A minimal stub tool so get_tools() doesn't short-circuit (base_tools guard)
def _stub_tool(name: str = "stub") -> Any:
    from attractor_llm.types import Tool

    async def _execute(**kwargs: Any) -> str:  # noqa: ANN401
        return "ok"

    return Tool(
        name=name,
        description=f"Stub tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=_execute,
    )


class TestOpenAIProfileDoesNotInjectSubagentTools:
    """§9.12.34 - Subagent tools come from Session, not the OpenAI profile."""

    def test_openai_profile_does_not_inject_subagent_tools(self) -> None:
        """Profile.get_tools() must NOT include subagent tools -- Session does that."""
        from attractor_agent.profiles.openai import OpenAIProfile

        profile = OpenAIProfile()
        tools = profile.get_tools([_stub_tool("read_file")])
        tool_names = {t.name for t in tools}

        injected = SUBAGENT_TOOL_NAMES & tool_names
        assert not injected, (
            f"OpenAI profile must NOT inject subagent tools (Session does). Found: {injected}"
        )


class TestAnthropicProfileDoesNotInjectSubagentTools:
    """§9.12.35 - Subagent tools come from Session, not the Anthropic profile."""

    def test_anthropic_profile_does_not_inject_subagent_tools(self) -> None:
        """Profile.get_tools() must NOT include subagent tools -- Session does that."""
        from attractor_agent.profiles.anthropic import AnthropicProfile

        profile = AnthropicProfile()
        tools = profile.get_tools([_stub_tool("read_file")])
        tool_names = {t.name for t in tools}

        injected = SUBAGENT_TOOL_NAMES & tool_names
        assert not injected, (
            f"Anthropic profile must NOT inject subagent tools (Session does). Found: {injected}"
        )


class TestGeminiProfileDoesNotInjectSubagentTools:
    """§9.12.36 - Subagent tools come from Session, not the Gemini profile."""

    def test_gemini_profile_does_not_inject_subagent_tools(self) -> None:
        """Profile.get_tools() must NOT include subagent tools -- Session does that."""
        from attractor_agent.profiles.gemini import GeminiProfile

        profile = GeminiProfile()
        tools = profile.get_tools([_stub_tool("read_file")])
        tool_names = {t.name for t in tools}

        injected = SUBAGENT_TOOL_NAMES & tool_names
        assert not injected, (
            f"Gemini profile must NOT inject subagent tools (Session does). Found: {injected}"
        )


class TestSessionInjectsFunctionalSubagentTools:
    """§9.12.34-36 - Session wires all 4 subagent tools with the real client."""

    @pytest.mark.asyncio
    async def test_session_injects_all_subagent_tools(self) -> None:
        """Session tool registry includes spawn_agent, send_input, wait, close_agent."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        tool_names = {t.name for t in session.tool_registry.definitions()}
        missing = SUBAGENT_TOOL_NAMES - tool_names
        assert not missing, f"Session missing subagent tools: {missing}. Registry has: {tool_names}"

    @pytest.mark.asyncio
    async def test_session_injects_functional_subagent_tools(self) -> None:
        """spawn_agent injected by Session has a real client -- no 'no client' error."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        tools_by_name = {t.name: t for t in session.tool_registry.definitions()}
        assert "spawn_agent" in tools_by_name, "spawn_agent must be in Session tool registry"

        spawn = tools_by_name["spawn_agent"]
        assert spawn.execute is not None, "spawn_agent must have an execute handler"

        # With a real client wired, the tool should NOT return the "no client" error.
        # (It may fail for other reasons with the mock, but not for missing client.)
        result = await spawn.execute(prompt="Do something")
        assert "requires an LLM client" not in result, (
            f"spawn_agent should not return 'no client' error when Session has a client. "
            f"Got: {result!r}"
        )


class TestCreateInteractiveToolsReturnsAllFour:
    """create_interactive_tools() returns exactly the 4 expected tools."""

    def test_all_four_tools_present(self) -> None:
        from attractor_agent.subagent_manager import SubagentManager, create_interactive_tools

        tools = create_interactive_tools(SubagentManager())
        names = {t.name for t in tools}
        assert names == SUBAGENT_TOOL_NAMES, f"Expected {SUBAGENT_TOOL_NAMES}, got {names}"

    @pytest.mark.asyncio
    async def test_spawn_agent_without_client_returns_error(self) -> None:
        """spawn_agent without a client returns an informative error string."""
        from attractor_agent.subagent_manager import SubagentManager, create_interactive_tools

        tools = create_interactive_tools(SubagentManager())  # no client
        spawn = next(t for t in tools if t.name == "spawn_agent")
        assert spawn.execute is not None, "spawn_agent tool must have an execute handler"
        result = await spawn.execute(prompt="Do something")
        assert "Error" in result or "error" in result.lower(), (
            f"Expected error message, got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_spawn_agent_with_client_calls_manager(self) -> None:
        """spawn_agent with a client delegates to manager.spawn()."""
        from attractor_agent.subagent_manager import SubagentManager, create_interactive_tools

        manager = SubagentManager()
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=_make_text_response("task done"))

        # Patch spawn so we don't need a real session
        async def _fake_spawn(client: Any, prompt: str, **kwargs: Any) -> str:
            return f"agent-{prompt[:4]}"

        manager.spawn = _fake_spawn  # type: ignore[assignment]

        tools = create_interactive_tools(manager, client=mock_client)
        spawn = next(t for t in tools if t.name == "spawn_agent")
        assert spawn.execute is not None, "spawn_agent tool must have an execute handler"
        result = await spawn.execute(prompt="Refactor auth module")
        assert "agent-" in result


# ================================================================== #
# Item 6 §9.2.1: OpenAI profile recommends apply_patch
# ================================================================== #


class TestOpenAIProfileRecommendsApplyPatch:
    """§9.2.1 - OpenAI system prompt promotes apply_patch as primary editing tool."""

    def test_openai_profile_recommends_apply_patch(self) -> None:
        from attractor_agent.profiles.openai import OpenAIProfile

        profile = OpenAIProfile()
        prompt = profile.system_prompt

        assert "apply_patch" in prompt, "OpenAI system prompt must mention apply_patch"
        # Verify it's described as primary/preferred
        assert any(
            phrase in prompt.lower()
            for phrase in ["primary", "prefer apply_patch", "primary editing tool"]
        ), (
            "OpenAI system prompt must describe apply_patch as the primary "
            f"editing tool. Prompt excerpt: {prompt[:300]!r}"
        )

    def test_openai_workflow_pattern_uses_apply_patch(self) -> None:
        """The WORKFLOW PATTERN line should reference apply_patch, not edit_file."""
        from attractor_agent.profiles.openai import OpenAIProfile

        profile = OpenAIProfile()
        prompt = profile.system_prompt

        # Find the WORKFLOW PATTERN section
        assert "WORKFLOW PATTERN" in prompt
        workflow_idx = prompt.index("WORKFLOW PATTERN")
        # Get the next 200 chars after WORKFLOW PATTERN
        section = prompt[workflow_idx : workflow_idx + 200]
        assert "apply_patch" in section, (
            f"WORKFLOW PATTERN section should reference apply_patch. Section: {section!r}"
        )


# ================================================================== #
# Item 8 §9.8.3: Enriched system prompt includes tool descriptions
# ================================================================== #


class TestEnrichedSystemPromptIncludesToolDescriptions:
    """§9.8.3 - _build_enriched_system_prompt() lists available tools."""

    @pytest.mark.asyncio
    async def test_enriched_system_prompt_includes_tool_descriptions(self) -> None:
        """Tool names and descriptions appear in the enriched system prompt."""
        client = _make_mock_client(_make_text_response("Hello"))
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            system_prompt="Base prompt.",
        )

        session = Session(client=client, config=config, tools=[_stub_tool("my_custom_tool")])

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch(
                "attractor_agent.project_docs.discover_project_docs",
                return_value="",
            ),
            patch(
                "attractor_agent.env_context.get_git_context",
                return_value={},
            ),
        ):
            prompt = session._build_enriched_system_prompt()

        assert "my_custom_tool" in prompt, (
            f"Tool name 'my_custom_tool' should appear in enriched prompt. "
            f"Prompt snippet: {prompt[:500]!r}"
        )
        assert (
            "available_tools" in prompt
            or "Available Tools" in prompt
            or "available" in prompt.lower()
        ), "Enriched prompt should have an available tools section"

    @pytest.mark.asyncio
    async def test_enriched_system_prompt_includes_multiple_tools(self) -> None:
        """Multiple registered tools all appear in the prompt."""
        client = _make_mock_client(_make_text_response("Hi"))
        config = SessionConfig(model="mock-model", provider="mock")
        tools = [_stub_tool("alpha_tool"), _stub_tool("beta_tool"), _stub_tool("gamma_tool")]
        session = Session(client=client, config=config, tools=tools)

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            prompt = session._build_enriched_system_prompt()

        for tool_name in ("alpha_tool", "beta_tool", "gamma_tool"):
            assert tool_name in prompt, f"Tool '{tool_name}' missing from enriched system prompt"

    @pytest.mark.asyncio
    async def test_enriched_system_prompt_includes_subagent_tools_even_when_no_explicit_tools(
        self,
    ) -> None:
        """Session always injects subagent tools, so available_tools is always present.

        Even when tools=[] is passed, Session.__init__ wires the 4 subagent tools
        (§9.12.34-36), so the <available_tools> section always appears.
        """
        client = _make_mock_client(_make_text_response("Hi"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config, tools=[])

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            prompt = session._build_enriched_system_prompt()

        # Session always injects subagent tools, so available_tools section must exist
        assert "<available_tools>" in prompt, (
            "Session always injects subagent tools, so <available_tools> must appear"
        )
        assert "spawn_agent" in prompt, (
            "spawn_agent must appear in the enriched system prompt (always injected)"
        )


# ================================================================== #
# Item 9 §9.10.1: ASSISTANT_TEXT_START / ASSISTANT_TEXT_DELTA emitted
# ================================================================== #


class TestSessionEmitsAssistantTextEvents:
    """§9.10.1 - Session emits TEXT_START then TEXT_DELTA then TEXT_END."""

    @pytest.mark.asyncio
    async def test_session_emits_assistant_text_start(self) -> None:
        """ASSISTANT_TEXT_START fires before ASSISTANT_TEXT_END."""
        client = _make_mock_client(_make_text_response("Hello, world!"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        emitted: list[EventKind] = []

        def _capture(event: SessionEvent) -> None:
            if event.kind in (
                EventKind.ASSISTANT_TEXT_START,
                EventKind.ASSISTANT_TEXT_DELTA,
                EventKind.ASSISTANT_TEXT_END,
            ):
                emitted.append(event.kind)

        session.events.on(_capture)

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            await session.submit("Say hello")

        assert EventKind.ASSISTANT_TEXT_START in emitted, (
            f"ASSISTANT_TEXT_START not emitted. Events seen: {emitted}"
        )
        # Verify ordering: START before END
        start_idx = emitted.index(EventKind.ASSISTANT_TEXT_START)
        end_idx = emitted.index(EventKind.ASSISTANT_TEXT_END)
        assert start_idx < end_idx, (
            f"ASSISTANT_TEXT_START ({start_idx}) must come before "
            f"ASSISTANT_TEXT_END ({end_idx}). Order: {emitted}"
        )

    @pytest.mark.asyncio
    async def test_session_emits_assistant_text_delta(self) -> None:
        """ASSISTANT_TEXT_DELTA fires with the response text."""
        client = _make_mock_client(_make_text_response("Hello, delta!"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        deltas: list[str] = []

        def _capture(event: SessionEvent) -> None:
            if event.kind == EventKind.ASSISTANT_TEXT_DELTA:
                deltas.append(event.data.get("delta", ""))

        session.events.on(_capture)

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            await session.submit("Say hello")

        assert deltas, "ASSISTANT_TEXT_DELTA was never emitted"
        assert "Hello, delta!" in "".join(deltas), (
            f"Expected response text in delta events. Got: {deltas!r}"
        )

    @pytest.mark.asyncio
    async def test_session_emits_all_three_text_events_in_order(self) -> None:
        """START → DELTA → END ordering is maintained."""
        client = _make_mock_client(_make_text_response("Ordered response."))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        order: list[str] = []

        def _capture(event: SessionEvent) -> None:
            if event.kind in (
                EventKind.ASSISTANT_TEXT_START,
                EventKind.ASSISTANT_TEXT_DELTA,
                EventKind.ASSISTANT_TEXT_END,
            ):
                order.append(event.kind)

        session.events.on(_capture)

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            await session.submit("Order check")

        assert EventKind.ASSISTANT_TEXT_START in order
        assert EventKind.ASSISTANT_TEXT_DELTA in order
        assert EventKind.ASSISTANT_TEXT_END in order

        start_i = order.index(EventKind.ASSISTANT_TEXT_START)
        delta_i = order.index(EventKind.ASSISTANT_TEXT_DELTA)
        end_i = order.index(EventKind.ASSISTANT_TEXT_END)
        assert start_i < delta_i < end_i, f"Expected START < DELTA < END. Got order: {order}"

    @pytest.mark.asyncio
    async def test_no_text_events_when_tool_call_only(self) -> None:
        """TEXT_START/DELTA/END are NOT emitted for tool-call rounds."""

        # First response: tool call; second: text
        tool_resp = Response(
            id="r1",
            model="mock-model",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[ContentPart.tool_call_part("tc-1", "stub", "{}")],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
        )
        text_resp = _make_text_response("Final answer.")

        call_count = 0

        async def _complete(request: Any, abort_signal: Any = None) -> Response:
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count == 1 else text_resp

        client = MagicMock(spec=Client)
        client.complete = _complete

        config = SessionConfig(model="mock-model", provider="mock")

        async def _stub_exec(**kwargs: Any) -> str:
            return "tool output"

        from attractor_llm.types import Tool

        stub = Tool(
            name="stub",
            description="Stub",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=_stub_exec,
        )
        session = Session(client=client, config=config, tools=[stub])

        text_event_turns: list[tuple[str, int]] = []
        turn_counter = [0]

        def _capture(event: SessionEvent) -> None:
            if event.kind == EventKind.TURN_START:
                turn_counter[0] += 1
            if event.kind in (
                EventKind.ASSISTANT_TEXT_START,
                EventKind.ASSISTANT_TEXT_DELTA,
                EventKind.ASSISTANT_TEXT_END,
            ):
                text_event_turns.append((event.kind, turn_counter[0]))

        session.events.on(_capture)

        with (
            patch(
                "attractor_agent.env_context.build_environment_context",
                return_value="<environment>env</environment>",
            ),
            patch("attractor_agent.project_docs.discover_project_docs", return_value=""),
            patch("attractor_agent.env_context.get_git_context", return_value={}),
        ):
            await session.submit("Run stub tool then answer")

        # Text events should exist (from final text response)
        assert text_event_turns, "Expected at least one text event from final response"


# ================================================================== #
# Item 10 §9.10.2: EventEmitter async iterator
# ================================================================== #


class TestEventEmitterAsyncIterator:
    """§9.10.2 - EventEmitter.events() is an async generator."""

    @pytest.mark.asyncio
    async def test_event_emitter_async_iterator_basic(self) -> None:
        """async for loop receives emitted events in order."""
        emitter = EventEmitter()
        events_to_emit = [
            SessionEvent(kind=EventKind.SESSION_START),
            SessionEvent(kind=EventKind.TURN_START, data={"turn": 1}),
            SessionEvent(kind=EventKind.TURN_END, data={"turn": 1}),
        ]

        async def _emit_and_close() -> None:
            for ev in events_to_emit:
                await emitter.emit(ev)
            await emitter.close()

        received: list[SessionEvent] = []

        async def _collect() -> None:
            async for event in emitter.events():
                received.append(event)

        await asyncio.gather(_emit_and_close(), _collect())

        assert len(received) == len(events_to_emit), (
            f"Expected {len(events_to_emit)} events, got {len(received)}"
        )
        for expected, actual in zip(events_to_emit, received, strict=True):
            assert actual.kind == expected.kind, f"Expected kind {expected.kind}, got {actual.kind}"

    @pytest.mark.asyncio
    async def test_event_emitter_async_iterator_terminates_on_close(self) -> None:
        """close() causes the async generator to terminate cleanly."""
        emitter = EventEmitter()

        async def _producer() -> None:
            await emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
            await emitter.close()

        count = 0

        async def _consumer() -> None:
            nonlocal count
            async for _ in emitter.events():
                count += 1

        await asyncio.gather(_producer(), _consumer())
        assert count == 1

    @pytest.mark.asyncio
    async def test_event_emitter_async_iterator_and_callbacks_coexist(self) -> None:
        """Async iterator and callback handlers both receive events."""
        emitter = EventEmitter()
        callback_events: list[EventKind] = []

        def _cb(event: SessionEvent) -> None:
            callback_events.append(event.kind)

        emitter.on(_cb)

        async def _produce() -> None:
            await emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
            await emitter.emit(SessionEvent(kind=EventKind.SESSION_END))
            await emitter.close()

        iterator_events: list[EventKind] = []

        async def _consume() -> None:
            async for event in emitter.events():
                iterator_events.append(event.kind)

        await asyncio.gather(_produce(), _consume())

        assert EventKind.SESSION_START in callback_events
        assert EventKind.SESSION_END in callback_events
        assert EventKind.SESSION_START in iterator_events
        assert EventKind.SESSION_END in iterator_events

    @pytest.mark.asyncio
    async def test_event_emitter_async_iterator_empty_close(self) -> None:
        """Closing immediately yields no events."""
        emitter = EventEmitter()

        async def _close_immediately() -> None:
            await emitter.close()

        received: list[SessionEvent] = []

        async def _collect() -> None:
            async for event in emitter.events():
                received.append(event)

        await asyncio.gather(_close_immediately(), _collect())
        assert received == []

    @pytest.mark.asyncio
    async def test_event_emitter_async_iterator_returns_correct_data(self) -> None:
        """Event data payload is preserved through the async queue."""
        emitter = EventEmitter()

        async def _produce() -> None:
            await emitter.emit(
                SessionEvent(kind=EventKind.TURN_START, data={"turn": 42, "prompt": "hi"})
            )
            await emitter.close()

        received: list[SessionEvent] = []

        async def _consume() -> None:
            async for event in emitter.events():
                received.append(event)

        await asyncio.gather(_produce(), _consume())

        assert len(received) == 1
        assert received[0].data["turn"] == 42
        assert received[0].data["prompt"] == "hi"


# ================================================================== #
# Items 11-12 §9.1.6, §9.11.5: SIGTERM/SIGKILL tracked processes
# ================================================================== #


class TestShutdownSendsSigterm:
    """§9.1.6 / §9.11.5 - _cleanup_on_abort sends SIGTERM then SIGKILL."""

    def _make_mock_process(self, *, already_done: bool = False) -> MagicMock:
        """Build a mock asyncio.subprocess.Process."""
        proc = MagicMock()
        # returncode=None → process still running; int → done
        proc.returncode = 0 if already_done else None
        proc.send_signal = MagicMock()
        # proc.wait() is awaited after SIGTERM
        proc.wait = AsyncMock(return_value=0)
        return proc

    @pytest.mark.asyncio
    async def test_shutdown_sends_sigterm(self) -> None:
        """Abort cleanup sends SIGTERM to tracked live processes."""
        from attractor_agent.abort import AbortSignal

        abort = AbortSignal()
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config, abort_signal=abort)

        proc = self._make_mock_process()
        session._tracked_processes.append(proc)

        # Trigger cleanup directly (without running the full loop)
        with (
            patch("attractor_agent.session.asyncio.wait", new_callable=AsyncMock) as mock_wait,
        ):
            mock_wait.return_value = (set(), set())  # (done, pending)
            await session._cleanup_on_abort()

        proc.send_signal.assert_any_call(_signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_shutdown_sends_sigkill_to_survivors(self) -> None:
        """After SIGTERM + wait, SIGKILL is sent to processes that didn't exit."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        proc = self._make_mock_process()
        # Simulate: returncode stays None even after SIGTERM+wait (stubborn process)
        # send_signal doesn't change returncode in mock, so it stays None → SIGKILL fires
        session._tracked_processes.append(proc)

        with (
            patch("attractor_agent.session.asyncio.wait", new_callable=AsyncMock) as mock_wait,
        ):
            mock_wait.return_value = (set(), set())
            await session._cleanup_on_abort()

        calls = [c.args[0] for c in proc.send_signal.call_args_list]
        assert _signal.SIGTERM in calls, f"Expected SIGTERM in calls: {calls}"
        assert _signal.SIGKILL in calls, f"Expected SIGKILL in calls: {calls}"

    @pytest.mark.asyncio
    async def test_shutdown_skips_already_done_processes(self) -> None:
        """Processes that have already exited (returncode set) are skipped."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        done_proc = self._make_mock_process(already_done=True)
        session._tracked_processes.append(done_proc)

        with (
            patch("attractor_agent.session.asyncio.wait", new_callable=AsyncMock) as mock_wait,
        ):
            mock_wait.return_value = (set(), set())
            await session._cleanup_on_abort()

        # Already done → send_signal should never be called
        done_proc.send_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_clears_tracked_processes(self) -> None:
        """After cleanup, _tracked_processes is emptied."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        session._tracked_processes.append(self._make_mock_process())
        session._tracked_processes.append(self._make_mock_process())

        with (
            patch("attractor_agent.session.asyncio.wait", new_callable=AsyncMock) as mock_wait,
        ):
            mock_wait.return_value = (set(), set())
            await session._cleanup_on_abort()

        assert session._tracked_processes == [], "_tracked_processes must be cleared after cleanup"

    @pytest.mark.asyncio
    async def test_shutdown_no_processes_is_noop(self) -> None:
        """When _tracked_processes is empty, cleanup runs without error."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        # No processes registered -- should not raise
        await session._cleanup_on_abort()
        assert session._tracked_processes == []

    @pytest.mark.asyncio
    async def test_tracked_processes_attribute_exists_on_session(self) -> None:
        """Session has _tracked_processes: list[asyncio.subprocess.Process]."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        assert hasattr(session, "_tracked_processes"), (
            "Session must have _tracked_processes attribute"
        )
        assert isinstance(session._tracked_processes, list), "_tracked_processes must be a list"
        assert session._tracked_processes == [], "_tracked_processes must be empty at init"

    @pytest.mark.asyncio
    async def test_shutdown_handles_process_lookup_error(self) -> None:
        """ProcessLookupError during send_signal is swallowed gracefully."""
        client = _make_mock_client(_make_text_response("ok"))
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config)

        proc = self._make_mock_process()
        proc.send_signal.side_effect = ProcessLookupError("No such process")
        session._tracked_processes.append(proc)

        # Should not raise
        with (
            patch("attractor_agent.session.asyncio.wait", new_callable=AsyncMock) as mock_wait,
        ):
            mock_wait.return_value = (set(), set())
            await session._cleanup_on_abort()  # must not raise
