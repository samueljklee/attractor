"""Tests for Wave 5 agent loop improvements.

Covers spec compliance items:
  #7  SIGTERM->SIGKILL escalation on shell timeout (Spec S9.4)
  #8  follow_up() method (Spec S2.6, S9.6)
  #13 Context window overflow detection (Spec S9.11)
  #29 Abort transitions session to CLOSED (Spec S9.1)
"""

from __future__ import annotations

import signal
import subprocess
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from attractor_agent.abort import AbortSignal
from attractor_agent.environment import LocalEnvironment, _sigterm_sigkill
from attractor_agent.events import EventKind, SessionEvent
from attractor_agent.session import Session, SessionConfig, SessionState
from attractor_llm.client import Client
from attractor_llm.types import Message, Response, Usage

# ================================================================== #
# Helpers
# ================================================================== #


def _text_response(text: str) -> Response:
    """Create a simple text-only Response for mocking."""
    return Response(message=Message.assistant(text), usage=Usage())


# ================================================================== #
# Item #7: SIGTERM -> SIGKILL escalation (Spec S9.4)
# ================================================================== #


class TestSigtermSigkillEscalation:
    """Timed-out commands: SIGTERM to process group, SIGKILL after 2 s."""

    def test_sigterm_then_sigkill_when_process_resists(self):
        """Full escalation: SIGTERM first, SIGKILL after 2 s grace period."""
        mock_proc: MagicMock = MagicMock()
        mock_proc.pid = 42
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 2),  # survives SIGTERM
            None,  # reaped after SIGKILL
        ]

        with (
            patch("attractor_agent.environment.os.getpgid", return_value=42),
            patch("attractor_agent.environment.os.killpg") as mock_killpg,
        ):
            _sigterm_sigkill(mock_proc)

        assert mock_killpg.call_args_list == [
            call(42, signal.SIGTERM),
            call(42, signal.SIGKILL),
        ]

    def test_sigterm_sufficient_skips_sigkill(self):
        """If process exits after SIGTERM within 2 s, SIGKILL is not sent."""
        mock_proc: MagicMock = MagicMock()
        mock_proc.pid = 42
        mock_proc.wait.return_value = None  # exits promptly after SIGTERM

        with (
            patch("attractor_agent.environment.os.getpgid", return_value=42),
            patch("attractor_agent.environment.os.killpg") as mock_killpg,
        ):
            _sigterm_sigkill(mock_proc)

        assert mock_killpg.call_args_list == [
            call(42, signal.SIGTERM),
        ]

    def test_already_exited_process_is_handled(self):
        """If process already exited, OSError from getpgid is caught."""
        mock_proc: MagicMock = MagicMock()
        mock_proc.pid = 42

        with (
            patch("attractor_agent.environment.os.getpgid", side_effect=OSError),
            patch("attractor_agent.environment.os.killpg") as mock_killpg,
        ):
            _sigterm_sigkill(mock_proc)

        mock_killpg.assert_not_called()

    async def test_exec_shell_timeout_uses_escalation(self):
        """LocalEnvironment.exec_shell invokes SIGTERM->SIGKILL on timeout."""
        mock_proc: MagicMock = MagicMock()
        mock_proc.pid = 42
        mock_proc.communicate.side_effect = [
            subprocess.TimeoutExpired("cmd", 1),  # first call times out
            ("", ""),  # second call after kill
        ]
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 2),  # resists SIGTERM
            None,  # reaped after SIGKILL
        ]

        with (
            patch("attractor_agent.environment.subprocess.Popen", return_value=mock_proc),
            patch("attractor_agent.environment.os.getpgid", return_value=42),
            patch("attractor_agent.environment.os.killpg") as mock_killpg,
        ):
            env = LocalEnvironment()
            result = await env.exec_shell("sleep 100", timeout=1)

        assert result.returncode == -1
        assert "timed out" in result.stderr
        assert mock_killpg.call_args_list == [
            call(42, signal.SIGTERM),
            call(42, signal.SIGKILL),
        ]


# ================================================================== #
# Item #8: follow_up() method (Spec S2.6, S9.6)
# ================================================================== #


class TestFollowUp:
    """session.follow_up(message) queues messages for post-loop processing."""

    def _make_session(self, responses: list[Response]) -> Session:
        """Session with a mock client returning canned responses in order."""
        client = Client()
        client.complete = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
        return Session(client=client)

    async def test_follow_up_queues_message(self):
        """follow_up() adds to the internal queue without immediate processing."""
        session = self._make_session([])
        session.follow_up("check the tests")
        assert len(session._followup_queue) == 1

    async def test_follow_up_processed_after_main_loop(self):
        """Follow-up messages trigger additional processing cycles."""
        responses = [
            _text_response("First response"),
            _text_response("Follow-up response"),
        ]
        session = self._make_session(responses)
        session.follow_up("now check the output")

        result = await session.submit("write a function")

        assert result == "Follow-up response"
        assert session._client.complete.call_count == 2  # type: ignore[union-attr]
        assert len(session._followup_queue) == 0

    async def test_multiple_follow_ups_processed_sequentially(self):
        """Multiple follow-ups are drained one at a time in order."""
        responses = [
            _text_response("Main"),
            _text_response("Follow-up 1"),
            _text_response("Follow-up 2"),
        ]
        session = self._make_session(responses)
        session.follow_up("first follow-up")
        session.follow_up("second follow-up")

        result = await session.submit("start")

        assert result == "Follow-up 2"
        assert session._client.complete.call_count == 3  # type: ignore[union-attr]

    async def test_no_follow_ups_normal_behavior(self):
        """Without follow-ups, submit() behaves exactly as before."""
        responses = [_text_response("Normal response")]
        session = self._make_session(responses)

        result = await session.submit("hello")

        assert result == "Normal response"
        assert session._client.complete.call_count == 1  # type: ignore[union-attr]

    async def test_follow_up_message_appears_in_history(self):
        """Follow-up messages are added to history as user messages."""
        responses = [
            _text_response("First"),
            _text_response("Second"),
        ]
        session = self._make_session(responses)
        session.follow_up("my follow-up")

        await session.submit("initial prompt")

        user_texts = [m.text for m in session.history if m.role.value == "user"]
        assert "my follow-up" in user_texts


# ================================================================== #
# Item #13: Context window overflow detection (Spec S9.11)
# ================================================================== #


class TestContextWindowOverflow:
    """Warn when conversation history approaches model's context window limit."""

    async def test_warning_emitted_at_80_percent(self):
        """CONTEXT_WINDOW_WARNING fires when estimated tokens > 80% of context."""
        events: list[SessionEvent] = []

        client = Client()
        config = SessionConfig(model="claude-sonnet-4-5")  # 200k context
        session = Session(client=client, config=config)
        session.events.on(lambda e: events.append(e))

        # 200k tokens * 4 chars/token = 800k chars; 80% threshold = 640k chars
        # Inject ~700k chars -> ~175k estimated tokens (> 160k threshold)
        big_text = "x" * 700_000
        session._history.append(Message.user(big_text))

        client.complete = AsyncMock(return_value=_text_response("ok"))  # type: ignore[method-assign]
        await session.submit("one more thing")

        warning_events = [e for e in events if e.kind == EventKind.CONTEXT_WINDOW_WARNING]
        assert len(warning_events) >= 1
        data = warning_events[0].data
        assert "estimated_tokens" in data
        assert "context_window" in data
        assert data["context_window"] == 200_000

    async def test_no_warning_under_threshold(self):
        """No warning when history is well under 80%."""
        events: list[SessionEvent] = []

        client = Client()
        config = SessionConfig(model="claude-sonnet-4-5")
        session = Session(client=client, config=config)
        session.events.on(lambda e: events.append(e))

        client.complete = AsyncMock(return_value=_text_response("ok"))  # type: ignore[method-assign]
        await session.submit("short message")

        warning_events = [e for e in events if e.kind == EventKind.CONTEXT_WINDOW_WARNING]
        assert len(warning_events) == 0

    async def test_unknown_model_skips_check(self):
        """Models not in the catalog don't trigger warnings."""
        events: list[SessionEvent] = []

        client = Client()
        config = SessionConfig(model="unknown-model-xyz")
        session = Session(client=client, config=config)
        session.events.on(lambda e: events.append(e))

        # Even with huge history, unknown model should not warn
        session._history.append(Message.user("x" * 700_000))

        client.complete = AsyncMock(return_value=_text_response("ok"))  # type: ignore[method-assign]
        await session.submit("test")

        warning_events = [e for e in events if e.kind == EventKind.CONTEXT_WINDOW_WARNING]
        assert len(warning_events) == 0


# ================================================================== #
# Item #29: Abort transitions session to CLOSED (Spec S9.1)
# ================================================================== #


class TestAbortTransitionsClosed:
    """When abort signal fires, session transitions to CLOSED (not IDLE)."""

    async def test_abort_sets_closed_state(self):
        """Session state is CLOSED after abort, not IDLE."""
        abort = AbortSignal()
        abort.set()
        client = Client()
        session = Session(client=client, abort_signal=abort)

        result = await session.submit("test")

        assert result == "[Session aborted]"
        assert session.state == SessionState.CLOSED

    async def test_abort_emits_session_end(self):
        """Abort emits SESSION_END event."""
        abort = AbortSignal()
        abort.set()
        client = Client()
        session = Session(client=client, abort_signal=abort)
        events: list[EventKind] = []
        session.events.on(lambda e: events.append(e.kind))

        await session.submit("test")

        assert EventKind.SESSION_END in events
        assert EventKind.TURN_END in events  # TURN_END still fires too

    async def test_abort_then_submit_raises_closed(self):
        """After abort, further submit() calls raise 'Session is closed'."""
        abort = AbortSignal()
        abort.set()
        client = Client()
        session = Session(client=client, abort_signal=abort)

        await session.submit("test")

        with pytest.raises(RuntimeError, match="closed"):
            await session.submit("should fail")

    async def test_normal_completion_stays_idle(self):
        """Without abort, session returns to IDLE after submit."""
        client = Client()
        session = Session(client=client)

        client.complete = AsyncMock(return_value=_text_response("done"))  # type: ignore[method-assign]
        result = await session.submit("hello")

        assert result == "done"
        assert session.state == SessionState.IDLE
