"""Tests for the 6 medium-complexity spec compliance fixes.

Covers:
  Fix #1: follow_up() emits TURN_START event + drains steering (Spec §2.5)
  Fix #2: Abort cleanup cancels subagent tasks + clears queues (Spec Appendix B)
  Fix #3: AudioData / DocumentData models + ContentPart fields (Spec §3.5)
  Fix #5: Adapter auto-detect local file paths in ImageData.url (Spec §3.5)
  Fix #8b: wait_for_output returns SubAgentResult JSON (Spec §7.3)
  Fix #8d: Subagents share parent's ExecutionEnvironment (Spec §7.1)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from attractor_agent.abort import AbortSignal
from attractor_agent.events import EventKind, SessionEvent
from attractor_agent.session import Session, SessionState
from attractor_agent.subagent_manager import SubagentManager, TrackedSubagent
from attractor_llm.client import Client
from attractor_llm.types import (
    AudioData,
    ContentPart,
    ContentPartKind,
    DocumentData,
    ImageData,
    Message,
    Response,
    Role,
    Usage,
)

# ================================================================== #
# Helpers
# ================================================================== #


def _text_response(text: str) -> Response:
    """Create a simple text-only Response for mocking."""
    return Response(message=Message.assistant(text), usage=Usage())


def _make_session(responses: list[Response]) -> Session:
    """Session with a mock client returning canned responses in order."""
    client = Client()
    client.complete = AsyncMock(side_effect=responses)  # type: ignore[method-assign]
    return Session(client=client)


def _register_fake_agent(
    manager: SubagentManager,
    agent_id: str,
    *,
    result: str = "test output",
    abort: AbortSignal | None = None,
    delay: float = 0,
) -> TrackedSubagent:
    """Register a fake tracked subagent directly in the manager."""
    abort = abort or AbortSignal()
    session = Session(client=Client(), abort_signal=abort)

    async def _fake_work() -> str:
        if delay > 0:
            await asyncio.sleep(delay)
        return result

    task = asyncio.create_task(_fake_work(), name=f"fake-{agent_id}")
    tracked = TrackedSubagent(
        agent_id=agent_id,
        session=session,
        task=task,
        abort_signal=abort,
    )
    manager._agents[agent_id] = tracked
    return tracked


def _register_failing_agent(
    manager: SubagentManager,
    agent_id: str,
    *,
    error: Exception | None = None,
) -> TrackedSubagent:
    """Register a subagent whose task will raise an exception."""
    abort = AbortSignal()
    session = Session(client=Client(), abort_signal=abort)

    async def _fail() -> str:
        raise (error or RuntimeError("boom"))

    task = asyncio.create_task(_fail(), name=f"fail-{agent_id}")
    tracked = TrackedSubagent(
        agent_id=agent_id,
        session=session,
        task=task,
        abort_signal=abort,
    )
    manager._agents[agent_id] = tracked
    return tracked


# ================================================================== #
# Fix #1: follow_up() emits TURN_START event + drains steering
# ================================================================== #


class TestFollowUpEventsAndSteering:
    """follow_up() must emit USER_INPUT and call _drain_steering per §2.5."""

    async def test_follow_up_emits_user_input(self):
        """Each follow-up message triggers a USER_INPUT event (not TURN_START)."""
        events: list[SessionEvent] = []

        responses = [
            _text_response("Main response"),
            _text_response("Follow-up response"),
        ]
        session = _make_session(responses)
        session.events.on(lambda e: events.append(e))
        session.follow_up("check the output")

        await session.submit("initial prompt")

        # Initial submit emits TURN_START; follow-up emits USER_INPUT
        turn_starts = [e for e in events if e.kind == EventKind.TURN_START]
        assert len(turn_starts) == 1

        user_inputs = [e for e in events if e.kind == EventKind.USER_INPUT]
        assert len(user_inputs) == 1
        # USER_INPUT carries full content (no truncation) under "content" key
        assert user_inputs[0].data["content"] == "check the output"

    async def test_follow_up_increments_turn_count(self):
        """Each follow-up increments the turn counter."""
        responses = [
            _text_response("Main"),
            _text_response("Follow-up 1"),
            _text_response("Follow-up 2"),
        ]
        session = _make_session(responses)
        session.follow_up("fu-1")
        session.follow_up("fu-2")

        await session.submit("start")

        # Initial turn + 2 follow-ups = 3
        assert session.turn_count == 3

    async def test_follow_up_drains_steering(self):
        """Steering messages queued before follow-up processing are drained."""
        events: list[SessionEvent] = []

        # We need the follow-up loop to pick up a steering message.
        # We'll inject one during the first response handling.
        responses = [
            _text_response("Main response"),
            _text_response("Follow-up response"),
        ]
        session = _make_session(responses)
        session.events.on(lambda e: events.append(e))

        # Queue a follow-up
        session.follow_up("my follow-up")

        # Queue a steering message that should be drained before the follow-up's _run_loop
        session.steer("mid-turn guidance")

        await session.submit("initial prompt")

        # The steering message should have been drained (either during
        # initial loop or follow-up processing)
        steer_events = [e for e in events if e.kind == EventKind.STEERING_INJECTED]
        assert len(steer_events) >= 1
        assert steer_events[0].data["message"] == "mid-turn guidance"

    async def test_follow_up_adds_user_message_to_history(self):
        """Follow-up messages appear as proper user turns in history."""
        responses = [
            _text_response("First"),
            _text_response("Second"),
        ]
        session = _make_session(responses)
        session.follow_up("my follow-up")

        await session.submit("initial prompt")

        user_texts = [
            m.text for m in session.history if isinstance(m, Message) and m.role == Role.USER
        ]
        assert "initial prompt" in user_texts
        assert "my follow-up" in user_texts

    async def test_multiple_follow_ups_each_emit_user_input(self):
        """Multiple follow-ups each get their own USER_INPUT event."""
        events: list[SessionEvent] = []

        responses = [
            _text_response("Main"),
            _text_response("FU-1"),
            _text_response("FU-2"),
        ]
        session = _make_session(responses)
        session.events.on(lambda e: events.append(e))
        session.follow_up("first")
        session.follow_up("second")

        await session.submit("start")

        # Only the initial submit emits TURN_START
        turn_starts = [e for e in events if e.kind == EventKind.TURN_START]
        assert len(turn_starts) == 1

        # Each follow-up emits USER_INPUT
        user_inputs = [e for e in events if e.kind == EventKind.USER_INPUT]
        assert len(user_inputs) == 2


# ================================================================== #
# Fix #2: Abort cleanup
# ================================================================== #


class TestAbortCleanup:
    """Abort triggers _cleanup_on_abort which clears queues and cancels tasks."""

    async def test_abort_clears_steering_queue(self):
        """Steering queue is emptied on abort."""
        abort = AbortSignal()
        abort.set()
        session = Session(client=Client(), abort_signal=abort)
        session.steer("should be cleared")

        await session.submit("test")

        assert session._steer_queue == []

    async def test_abort_clears_followup_queue(self):
        """Follow-up queue is emptied on abort."""
        abort = AbortSignal()
        abort.set()
        session = Session(client=Client(), abort_signal=abort)
        session.follow_up("should be cleared")

        await session.submit("test")

        assert session._followup_queue == []

    async def test_abort_sets_closed_state(self):
        """Session transitions to CLOSED after abort."""
        abort = AbortSignal()
        abort.set()
        session = Session(client=Client(), abort_signal=abort)

        result = await session.submit("test")

        assert result == "[Session aborted]"
        assert session.state == SessionState.CLOSED

    async def test_abort_emits_session_end(self):
        """Abort still emits SESSION_END event."""
        abort = AbortSignal()
        abort.set()
        events: list[EventKind] = []
        session = Session(client=Client(), abort_signal=abort)
        session.events.on(lambda e: events.append(e.kind))

        await session.submit("test")

        assert EventKind.SESSION_END in events

    async def test_cleanup_cancels_tracked_subagent_tasks(self):
        """Abort cancels only tasks registered in session._subagent_tasks."""
        abort = AbortSignal()
        client = Client()
        session = Session(client=client, abort_signal=abort)

        # Create a long-running task and register it with the session
        cancelled = False

        async def _long_running():
            nonlocal cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled = True

        task = asyncio.create_task(_long_running(), name="subagent-agent-abc")
        session._subagent_tasks.add(task)

        # Create an untracked task that should NOT be cancelled
        untracked_cancelled = False

        async def _untracked():
            nonlocal untracked_cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                untracked_cancelled = True

        untracked_task = asyncio.create_task(_untracked(), name="subagent-other")

        # Now abort and submit to trigger cleanup
        abort.set()
        await session.submit("test")

        # Give the event loop a tick to process cancellation
        await asyncio.sleep(0.01)

        assert cancelled or task.cancelled(), "Tracked subagent task should be cancelled"
        assert not untracked_cancelled, "Untracked task should NOT be cancelled"
        assert len(session._subagent_tasks) == 0, "Task set should be cleared after cleanup"

        # Clean up the untracked task
        untracked_task.cancel()
        await asyncio.sleep(0.01)


# ================================================================== #
# Fix #3: AudioData and DocumentData models
# ================================================================== #


class TestAudioData:
    """AudioData model per spec §3.5."""

    def test_url_valid(self):
        audio = AudioData(url="https://example.com/audio.mp3")
        assert audio.url == "https://example.com/audio.mp3"
        assert audio.data is None

    def test_data_valid(self):
        audio = AudioData(data=b"\xff\xfb\x90\x00", media_type="audio/mpeg")
        assert audio.data == b"\xff\xfb\x90\x00"
        assert audio.media_type == "audio/mpeg"

    def test_requires_data_or_url(self):
        with pytest.raises(ValueError, match="requires either"):
            AudioData()

    def test_both_set(self):
        audio = AudioData(url="https://example.com/a.mp3", data=b"audio", media_type="audio/mpeg")
        assert audio.url is not None
        assert audio.data is not None


class TestDocumentData:
    """DocumentData model per spec §3.5."""

    def test_url_valid(self):
        doc = DocumentData(url="https://example.com/doc.pdf")
        assert doc.url == "https://example.com/doc.pdf"

    def test_data_valid(self):
        doc = DocumentData(data=b"%PDF", media_type="application/pdf", file_name="report.pdf")
        assert doc.data == b"%PDF"
        assert doc.media_type == "application/pdf"
        assert doc.file_name == "report.pdf"

    def test_requires_data_or_url(self):
        with pytest.raises(ValueError, match="requires either"):
            DocumentData()

    def test_file_name_optional(self):
        doc = DocumentData(url="https://example.com/doc.pdf")
        assert doc.file_name is None


class TestContentPartAudioDocument:
    """ContentPart audio/document fields and factory methods."""

    def test_audio_part_factory(self):
        audio = AudioData(url="https://example.com/audio.mp3")
        part = ContentPart.audio_part(audio)
        assert part.kind == ContentPartKind.AUDIO
        assert part.audio is not None
        assert part.audio.url == "https://example.com/audio.mp3"

    def test_document_part_factory(self):
        doc = DocumentData(data=b"%PDF", media_type="application/pdf")
        part = ContentPart.document_part(doc)
        assert part.kind == ContentPartKind.DOCUMENT
        assert part.document is not None
        assert part.document.media_type == "application/pdf"

    def test_audio_part_requires_audio(self):
        with pytest.raises(ValueError, match="AUDIO content part requires"):
            ContentPart(kind=ContentPartKind.AUDIO, audio=None)

    def test_document_part_requires_document(self):
        with pytest.raises(ValueError, match="DOCUMENT content part requires"):
            ContentPart(kind=ContentPartKind.DOCUMENT, document=None)

    def test_audio_data_exported_from_init(self):
        from attractor_llm import AudioData as AD

        assert AD is AudioData

    def test_document_data_exported_from_init(self):
        from attractor_llm import DocumentData as DD

        assert DD is DocumentData


# ================================================================== #
# Fix #5: Adapter auto-detect local file paths in ImageData.url
# ================================================================== #


class TestResolveImageData:
    """resolve_image_data handles local file paths per spec §3.5."""

    def test_http_url_unchanged(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        img = ImageData(url="https://example.com/photo.png")
        result = resolve_image_data(img)
        assert result.url == "https://example.com/photo.png"
        assert result.data is None

    def test_data_already_set_unchanged(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        img = ImageData(data=b"\x89PNG", media_type="image/png")
        result = resolve_image_data(img)
        assert result.data == b"\x89PNG"

    def test_absolute_path_reads_file(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG_test_data")
            tmp_path = f.name

        try:
            img = ImageData(url=tmp_path)
            result = resolve_image_data(img)
            assert result.data == b"\x89PNG_test_data"
            assert result.media_type == "image/png"
            assert result.url is None  # url is cleared
        finally:
            Path(tmp_path).unlink()

    def test_relative_path_reads_file(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        with tempfile.NamedTemporaryFile(suffix=".jpg", dir=".", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0")
            rel_path = f"./{Path(f.name).name}"

        try:
            img = ImageData(url=rel_path)
            result = resolve_image_data(img)
            assert result.data == b"\xff\xd8\xff\xe0"
            assert result.media_type == "image/jpeg"
        finally:
            Path(rel_path).unlink()

    def test_tilde_path_resolves(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        # Just verify it doesn't crash -- ~ may not contain test images
        img = ImageData(url="~/nonexistent_image_12345.png")
        result = resolve_image_data(img)
        # File doesn't exist, so image should be returned unchanged
        assert result.url == "~/nonexistent_image_12345.png"

    def test_nonexistent_local_path_returns_unchanged(self):
        from attractor_llm.adapters.image_utils import resolve_image_data

        img = ImageData(url="/tmp/absolutely_not_a_real_file_xyz.png")
        result = resolve_image_data(img)
        assert result.url == "/tmp/absolutely_not_a_real_file_xyz.png"
        assert result.data is None

    def test_openai_adapter_uses_resolve(self):
        """OpenAI adapter integrates resolve_image_data for local paths."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG_data")
            tmp_path = f.name

        try:
            adapter = OpenAIAdapter(ProviderConfig(api_key="test-key"))
            msg = Message.user_parts(
                ContentPart.text_part("Look at this:"),
                ContentPart.image_part(ImageData(url=tmp_path)),
            )
            result = adapter._translate_user_content(msg)
            # Should be a list with text + image (base64-encoded, not file path)
            assert isinstance(result, list)
            assert len(result) == 2
            image_part = result[1]
            assert image_part["type"] == "input_image"
            # URL should be a data URI (base64), not the local file path
            assert image_part["image_url"].startswith("data:")
        finally:
            Path(tmp_path).unlink()

    def test_anthropic_adapter_uses_resolve(self):
        """Anthropic adapter integrates resolve_image_data for local paths."""
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.adapters.base import ProviderConfig

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG_data")
            tmp_path = f.name

        try:
            adapter = AnthropicAdapter(ProviderConfig(api_key="test-key"))
            part = ContentPart.image_part(ImageData(url=tmp_path))
            result = adapter._translate_content_part(part, Role.USER)
            # Should be base64 encoded, not a URL
            assert result["type"] == "image"
            assert result["source"]["type"] == "base64"
        finally:
            Path(tmp_path).unlink()

    def test_gemini_adapter_uses_resolve(self):
        """Gemini adapter integrates resolve_image_data for local paths."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG_data")
            tmp_path = f.name

        try:
            adapter = GeminiAdapter(ProviderConfig(api_key="test-key"))
            part = ContentPart.image_part(ImageData(url=tmp_path))
            result = adapter._translate_part(part)
            # Should be inline data, not fileData
            assert result is not None
            assert "inlineData" in result
        finally:
            Path(tmp_path).unlink()


# ================================================================== #
# Fix #8b: wait_for_output returns SubAgentResult JSON
# ================================================================== #


class TestWaitForOutputSubAgentResult:
    """wait_for_output returns JSON SubAgentResult per spec §7.3."""

    @pytest.mark.asyncio
    async def test_success_returns_json(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="task complete")

        output = await manager.wait_for_output("agent-001")
        parsed = json.loads(output)
        assert parsed["output"] == "task complete"
        assert parsed["success"] is True
        assert "turns_used" in parsed

    @pytest.mark.asyncio
    async def test_failure_returns_json_with_success_false(self):
        manager = SubagentManager()
        _register_failing_agent(manager, "agent-fail", error=ValueError("bad input"))

        output = await manager.wait_for_output("agent-fail")
        parsed = json.loads(output)
        assert parsed["success"] is False
        assert "ValueError" in parsed["output"]
        assert "bad input" in parsed["output"]
        assert "turns_used" in parsed

    @pytest.mark.asyncio
    async def test_nonexistent_agent_returns_json(self):
        manager = SubagentManager()

        output = await manager.wait_for_output("no-such-agent")
        parsed = json.loads(output)
        assert parsed["success"] is False
        assert "no-such-agent" in parsed["output"]
        assert parsed["turns_used"] == 0

    @pytest.mark.asyncio
    async def test_result_is_valid_json_string(self):
        """The raw return value is a valid JSON string."""
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="done")

        output = await manager.wait_for_output("agent-001")
        # Must be parseable JSON
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert set(parsed.keys()) == {"output", "success", "turns_used"}

    @pytest.mark.asyncio
    async def test_still_removes_agent_after_wait(self):
        """Agent is still removed from tracking after wait."""
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="done")

        await manager.wait_for_output("agent-001")
        assert "agent-001" not in manager.active_agents


# ================================================================== #
# Fix #8d: Subagents share parent's ExecutionEnvironment
# ================================================================== #


class TestSubagentSharedEnvironment:
    """Subagents share the parent's ExecutionEnvironment per spec §7.1."""

    def test_environment_is_module_singleton(self):
        """The execution environment is a module-level singleton shared by all sessions."""
        from attractor_agent.tools.core import get_environment

        # Get the current environment
        env1 = get_environment()
        env2 = get_environment()
        # Same object -- shared by all sessions in the process
        assert env1 is env2

    def test_spawn_accepts_working_dir(self):
        """SubagentManager.spawn accepts working_dir parameter."""
        import inspect

        sig = inspect.signature(SubagentManager.spawn)
        assert "working_dir" in sig.parameters

    @pytest.mark.asyncio
    async def test_spawn_propagates_working_dir(self):
        """working_dir is passed through to the child session's config."""
        manager = SubagentManager()
        from unittest.mock import AsyncMock as AM
        from unittest.mock import patch

        # Mock get_profile to return a simple passthrough
        class FakeProfile:
            def apply_to_config(self, config):
                return config

        with patch("attractor_agent.subagent_manager.get_profile", return_value=FakeProfile()):
            client = Client()
            client.complete = AM(  # type: ignore[method-assign]
                return_value=_text_response("done")
            )

            agent_id = await manager.spawn(
                client,
                "test task",
                model="mock-model",
                working_dir="/custom/workdir",
            )

            tracked = manager.active_agents[agent_id]
            assert tracked.session._config.working_dir == "/custom/workdir"

            # Cleanup
            tracked.abort_signal.set()
            manager._agents.clear()
