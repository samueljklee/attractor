"""Tests for Wave 13 -- Events & Lifecycle.

P22 (S9.10): Event types match the spec's event vocabulary.
  - ASSISTANT_TEXT_START and TOOL_CALL_OUTPUT_DELTA are present
  - 4 divergent names corrected: ASSISTANT_TEXT_END, STEERING_INJECTED,
    TURN_LIMIT, LOOP_DETECTION

F   (S9.11): Authentication errors (401/403) surface immediately with
  session -> CLOSED.

P23 (Appendix B): Graceful shutdown 8-step sequence.
  - Tracked asyncio tasks are cancelled on abort
  - SESSION_END is emitted on abort
"""

from __future__ import annotations

import asyncio

import pytest  # type: ignore[import-untyped]

from attractor_agent.abort import AbortSignal
from attractor_agent.events import EventKind, SessionEvent
from attractor_agent.session import Session, SessionConfig, SessionState
from attractor_llm.client import Client
from attractor_llm.errors import AccessDeniedError, AuthenticationError
from attractor_llm.types import Response
from tests.helpers import MockAdapter, make_text_response, make_tool_call_response

# ================================================================== #
# Helpers
# ================================================================== #


def _make_session(
    responses: list[Response | Exception],
    *,
    abort_signal: AbortSignal | None = None,
) -> tuple[Session, MockAdapter]:
    adapter = MockAdapter(responses=responses)
    client = Client()
    client.register_adapter("mock", adapter)
    config = SessionConfig(model="mock-model", provider="mock")
    session = Session(
        client=client,
        config=config,
        abort_signal=abort_signal,
    )
    return session, adapter


# ================================================================== #
# P22 -- Event vocabulary
# ================================================================== #


class TestP22EventKindVocabulary:
    """EventKind enum matches the spec S9.10 event vocabulary."""

    # --- Newly-added types ---

    def test_event_kind_has_assistant_text_start(self) -> None:
        """ASSISTANT_TEXT_START must exist with value 'assistant.text_start'."""
        assert EventKind.ASSISTANT_TEXT_START == "assistant.text_start"

    def test_event_kind_has_tool_call_output_delta(self) -> None:
        """TOOL_CALL_OUTPUT_DELTA must exist with value 'tool.call_output_delta'."""
        assert EventKind.TOOL_CALL_OUTPUT_DELTA == "tool.call_output_delta"

    # --- Renamed events ---

    def test_event_kind_assistant_text_end_renamed(self) -> None:
        """ASSISTANT_TEXT_END (was ASSISTANT_TEXT) must exist. Old name must not."""
        assert EventKind.ASSISTANT_TEXT_END == "assistant.text_end"
        assert not hasattr(EventKind, "ASSISTANT_TEXT"), (
            "Old name ASSISTANT_TEXT still present; should be ASSISTANT_TEXT_END"
        )

    def test_event_kind_steering_injected_renamed(self) -> None:
        """STEERING_INJECTED (was STEER_INJECTED) must exist. Old name must not."""
        assert EventKind.STEERING_INJECTED == "steering.injected"
        assert not hasattr(EventKind, "STEER_INJECTED"), (
            "Old name STEER_INJECTED still present; should be STEERING_INJECTED"
        )

    def test_event_kind_turn_limit_renamed(self) -> None:
        """TURN_LIMIT (was LIMIT_REACHED) must exist. Old name must not."""
        assert EventKind.TURN_LIMIT == "turn.limit"
        assert not hasattr(EventKind, "LIMIT_REACHED"), (
            "Old name LIMIT_REACHED still present; should be TURN_LIMIT"
        )

    def test_event_kind_loop_detection_renamed(self) -> None:
        """LOOP_DETECTION (was LOOP_DETECTED) must exist. Old name must not."""
        assert EventKind.LOOP_DETECTION == "loop.detection"
        assert not hasattr(EventKind, "LOOP_DETECTED"), (
            "Old name LOOP_DETECTED still present; should be LOOP_DETECTION"
        )

    # --- Pre-existing events are unaffected ---

    def test_existing_session_lifecycle_events_unchanged(self) -> None:
        assert EventKind.SESSION_START == "session.start"
        assert EventKind.SESSION_END == "session.end"
        assert EventKind.TURN_START == "turn.start"
        assert EventKind.TURN_END == "turn.end"

    def test_existing_tool_events_unchanged(self) -> None:
        assert EventKind.TOOL_CALL_START == "tool.call_start"
        assert EventKind.TOOL_CALL_END == "tool.call_end"

    @pytest.mark.asyncio
    async def test_assistant_text_end_event_fires_in_session(self) -> None:
        """Session emits ASSISTANT_TEXT_END when the model produces text."""
        session, _ = _make_session([make_text_response("hello")])
        kinds: list[EventKind] = []
        session.events.on(lambda e: kinds.append(e.kind))

        await session.submit("hi")

        assert EventKind.ASSISTANT_TEXT_END in kinds
        # Old string value must NOT appear as any emitted event value
        assert all(str(k) != "assistant.text" for k in kinds)

    @pytest.mark.asyncio
    async def test_steering_injected_event_fires_in_session(self) -> None:
        """Session emits STEERING_INJECTED when a steering message is drained."""
        from attractor_llm.types import Tool

        async def _noop(**kwargs: object) -> str:
            return "ok"

        session, _ = _make_session(
            [
                make_tool_call_response("noop", {}),
                make_text_response("done"),
            ]
        )
        session.tool_registry.register(
            Tool(
                name="noop",
                description="no-op",
                parameters={"type": "object", "properties": {}, "required": []},
                execute=_noop,
            )
        )

        kinds: list[EventKind] = []
        session.events.on(lambda e: kinds.append(e.kind))
        session.steer("focus on tests")

        await session.submit("go")

        assert EventKind.STEERING_INJECTED in kinds


# ================================================================== #
# F -- Auth errors -> CLOSED
# ================================================================== #


class TestAuthErrorTransitionsToClosed:
    """Authentication errors must close the session immediately (S9.11)."""

    @pytest.mark.asyncio
    async def test_auth_error_transitions_to_closed(self) -> None:
        """AuthenticationError during LLM call -> session.state == CLOSED."""
        auth_exc = AuthenticationError("Invalid API key", status_code=401)
        session, _ = _make_session([auth_exc])

        result = await session.submit("hello")

        assert session.state == SessionState.CLOSED
        assert "Authentication Error" in result

    @pytest.mark.asyncio
    async def test_access_denied_error_transitions_to_closed(self) -> None:
        """AccessDeniedError (403) during LLM call -> session.state == CLOSED."""
        denied_exc = AccessDeniedError("Forbidden", status_code=403)
        session, _ = _make_session([denied_exc])

        result = await session.submit("hello")

        assert session.state == SessionState.CLOSED
        assert "Authentication Error" in result

    @pytest.mark.asyncio
    async def test_auth_error_emits_error_event(self) -> None:
        """AuthenticationError must emit an ERROR event with auth_error=True."""
        auth_exc = AuthenticationError("Bad key", status_code=401)
        session, _ = _make_session([auth_exc])

        events: list[SessionEvent] = []
        session.events.on(lambda e: events.append(e))

        await session.submit("hello")

        error_events = [e for e in events if e.kind == EventKind.ERROR]
        assert error_events, "No ERROR event was emitted"
        assert error_events[0].data.get("auth_error") is True

    @pytest.mark.asyncio
    async def test_auth_error_rejects_new_inputs(self) -> None:
        """After an auth error the session is CLOSED and rejects new submit()."""
        auth_exc = AuthenticationError("Invalid API key", status_code=401)
        session, _ = _make_session([auth_exc])

        await session.submit("first")
        assert session.state == SessionState.CLOSED

        with pytest.raises(RuntimeError, match="closed"):
            await session.submit("second")

    @pytest.mark.asyncio
    async def test_non_auth_error_stays_idle(self) -> None:
        """A generic exception must still transition the session back to IDLE."""
        generic_exc = RuntimeError("some transient failure")
        session, _ = _make_session([generic_exc])

        result = await session.submit("hello")

        assert session.state == SessionState.IDLE
        assert "[Error:" in result

    @pytest.mark.asyncio
    async def test_auth_error_turn_end_still_emitted(self) -> None:
        """TURN_END must be emitted even when an auth error causes CLOSED."""
        auth_exc = AuthenticationError("Expired key", status_code=401)
        session, _ = _make_session([auth_exc])

        kinds: list[EventKind] = []
        session.events.on(lambda e: kinds.append(e.kind))

        await session.submit("hi")

        assert EventKind.TURN_END in kinds


# ================================================================== #
# P23 -- Graceful shutdown
# ================================================================== #


class TestGracefulShutdown:
    """Shutdown sequence: cancel tasks, flush events, CLOSED. Spec Appendix B."""

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_cancels_tracked_tasks(self) -> None:
        """Tasks registered in _subagent_tasks are cancelled on abort."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("hi")], abort_signal=abort)

        # Register a long-running task as a tracked subagent task
        async def _sleep_forever() -> None:
            await asyncio.sleep(9999)

        task: asyncio.Task[None] = asyncio.create_task(_sleep_forever())
        session._subagent_tasks.add(task)

        # Abort before submit so cleanup runs
        abort.set()
        await session.submit("run")

        # The tracked task must have been cancelled
        assert task.cancelled() or task.done(), "Tracked task was not cancelled on abort"
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_cancels_active_tasks(self) -> None:
        """Tasks registered in _active_tasks are cancelled on abort."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("hi")], abort_signal=abort)

        async def _long_op() -> str:
            await asyncio.sleep(9999)
            return "done"

        task: asyncio.Task[str] = asyncio.create_task(_long_op())
        session._active_tasks.add(task)  # type: ignore[arg-type]

        abort.set()
        await session.submit("run")

        assert task.cancelled() or task.done(), "_active_tasks task was not cancelled on abort"
        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_emits_session_end(self) -> None:
        """SESSION_END event is emitted when the session is aborted."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("hi")], abort_signal=abort)

        kinds: list[EventKind] = []
        session.events.on(lambda e: kinds.append(e.kind))

        abort.set()
        await session.submit("run")

        assert EventKind.SESSION_END in kinds, f"SESSION_END not emitted on abort; got: {kinds}"

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_emits_turn_end_before_session_end(self) -> None:
        """TURN_END is emitted before SESSION_END on an aborted session."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("hi")], abort_signal=abort)

        kinds: list[EventKind] = []
        session.events.on(lambda e: kinds.append(e.kind))

        abort.set()
        await session.submit("run")

        assert EventKind.TURN_END in kinds
        assert EventKind.SESSION_END in kinds
        turn_idx = next(i for i, k in enumerate(kinds) if k == EventKind.TURN_END)
        end_idx = next(i for i, k in enumerate(kinds) if k == EventKind.SESSION_END)
        assert turn_idx < end_idx, "TURN_END must precede SESSION_END"

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_drains_queues(self) -> None:
        """Pending steer/follow-up queues are cleared on abort."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("hi")], abort_signal=abort)

        session.steer("steer msg")
        session.follow_up("follow up")

        abort.set()
        await session.submit("run")

        assert session._steer_queue == []
        assert session._followup_queue == []

    @pytest.mark.asyncio
    async def test_normal_abort_transitions_to_closed(self) -> None:
        """Abort signal (without auth error) also transitions to CLOSED."""
        abort = AbortSignal()
        session, _ = _make_session([make_text_response("ok")], abort_signal=abort)

        abort.set()
        await session.submit("hi")

        assert session.state == SessionState.CLOSED
