"""Tests for Section 9.6 event system: event types and EventEmitter."""

from __future__ import annotations

import asyncio

import pytest

from attractor_pipeline import (
    HandlerRegistry,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
    InterviewTimeout,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
    PipelineCompleted,
    PipelineEvent,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)


class TestEventTypes:
    """All 16 event types from Spec Section 9.6 are importable dataclasses."""

    def test_pipeline_started(self):
        e = PipelineStarted(name="MyPipeline", id="run-123")
        assert e.name == "MyPipeline"
        assert e.id == "run-123"
        assert isinstance(e, PipelineEvent)

    def test_pipeline_completed(self):
        e = PipelineCompleted(duration=12.5, artifact_count=3)
        assert e.duration == 12.5
        assert e.artifact_count == 3

    def test_pipeline_failed(self):
        e = PipelineFailed(error="boom", duration=1.0)
        assert e.error == "boom"

    def test_stage_started(self):
        e = StageStarted(name="build", index=0)
        assert e.name == "build"
        assert e.index == 0

    def test_stage_completed(self):
        e = StageCompleted(name="build", index=0, duration=2.5)
        assert e.duration == 2.5

    def test_stage_failed(self):
        e = StageFailed(name="build", index=0, error="timeout", will_retry=True)
        assert e.will_retry is True

    def test_stage_retrying(self):
        e = StageRetrying(name="build", index=0, attempt=2, delay=1.5)
        assert e.attempt == 2
        assert e.delay == 1.5

    def test_parallel_started(self):
        e = ParallelStarted(branch_count=3)
        assert e.branch_count == 3

    def test_parallel_branch_started(self):
        e = ParallelBranchStarted(branch="branch_0", index=0)
        assert e.branch == "branch_0"

    def test_parallel_branch_completed(self):
        e = ParallelBranchCompleted(branch="b0", index=0, duration=1.0, success=True)
        assert e.success is True

    def test_parallel_completed(self):
        e = ParallelCompleted(duration=5.0, success_count=2, failure_count=1)
        assert e.success_count == 2

    def test_interview_started(self):
        e = InterviewStarted(question="Approve?", stage="review")
        assert e.question == "Approve?"
        assert e.stage == "review"

    def test_interview_completed(self):
        e = InterviewCompleted(question="Approve?", answer="yes", duration=3.0)
        assert e.answer == "yes"

    def test_interview_timeout(self):
        e = InterviewTimeout(question="Approve?", stage="review", duration=60.0)
        assert e.duration == 60.0

    def test_checkpoint_saved(self):
        e = CheckpointSaved(node_id="build_step")
        assert e.node_id == "build_step"

    def test_all_events_are_pipeline_event_subtype(self):
        """Every event type is a subclass of PipelineEvent."""
        all_types = [
            PipelineStarted,
            PipelineCompleted,
            PipelineFailed,
            StageStarted,
            StageCompleted,
            StageFailed,
            StageRetrying,
            ParallelStarted,
            ParallelBranchStarted,
            ParallelBranchCompleted,
            ParallelCompleted,
            InterviewStarted,
            InterviewCompleted,
            InterviewTimeout,
            CheckpointSaved,
        ]
        assert len(all_types) == 15  # 15 concrete types (PipelineEvent is base)
        for cls in all_types:
            assert issubclass(cls, PipelineEvent), f"{cls.__name__} is not a PipelineEvent"

    def test_event_description_property(self):
        """Every event has a human-readable description property."""
        e = PipelineStarted(name="Test", id="abc")
        assert isinstance(e.description, str)
        assert len(e.description) > 0


class TestEventEmitter:
    """EventEmitter supports callback and async stream patterns (Spec 9.6 lines 1639-1649)."""

    def test_emit_without_callback_is_noop(self):
        """Emitting without a callback does not raise."""
        emitter = EventEmitter()
        emitter.emit(PipelineStarted(name="test", id="1"))  # no error

    def test_callback_receives_events(self):
        """Observer pattern: on_event callback receives emitted events."""
        received: list = []
        emitter = EventEmitter(on_event=received.append)
        event = PipelineStarted(name="test", id="1")
        emitter.emit(event)
        assert received == [event]

    def test_callback_receives_multiple_events(self):
        """Callback receives events in emission order."""
        received: list = []
        emitter = EventEmitter(on_event=received.append)
        e1 = PipelineStarted(name="test", id="1")
        e2 = StageStarted(name="build", index=0)
        emitter.emit(e1)
        emitter.emit(e2)
        assert received == [e1, e2]

    @pytest.mark.asyncio
    async def test_async_stream_receives_events(self):
        """Stream pattern: async for event in emitter.events() yields emitted events."""
        emitter = EventEmitter()
        e1 = PipelineStarted(name="test", id="1")
        e2 = StageStarted(name="build", index=0)

        # Emit events then close the stream
        emitter.emit(e1)
        emitter.emit(e2)
        emitter.close()

        collected: list = []
        async for event in emitter.events():
            collected.append(event)

        assert collected == [e1, e2]

    @pytest.mark.asyncio
    async def test_async_stream_terminates_on_close(self):
        """Stream terminates cleanly when emitter.close() is called."""
        emitter = EventEmitter()

        async def collect() -> list:
            result = []
            async for event in emitter.events():
                result.append(event)
            return result

        task = asyncio.create_task(collect())
        emitter.emit(PipelineStarted(name="test", id="1"))
        emitter.close()

        result = await asyncio.wait_for(task, timeout=2.0)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_both_patterns_simultaneously(self):
        """Callback and stream both receive the same events."""
        callback_events: list = []
        emitter = EventEmitter(on_event=callback_events.append)

        event = PipelineStarted(name="test", id="1")
        emitter.emit(event)
        emitter.close()

        stream_events: list = []
        async for e in emitter.events():
            stream_events.append(e)

        assert callback_events == [event]
        assert stream_events == [event]


# ------------------------------------------------------------------ #
# Runner integration tests
# ------------------------------------------------------------------ #


class TestRunnerEventEmission:
    """run_pipeline emits lifecycle events via on_event callback."""

    @pytest.mark.asyncio
    async def test_pipeline_started_and_completed_events(self):
        """A successful pipeline emits PipelineStarted and PipelineCompleted."""
        g = parse_dot("""
        digraph E {
            graph [goal="Event test"]
            start [shape=Mdiamond]
            done  [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.COMPLETED

        # First event is PipelineStarted
        assert isinstance(events[0], PipelineStarted)
        assert events[0].name == "E"

        # Last event is PipelineCompleted
        assert isinstance(events[-1], PipelineCompleted)
        assert events[-1].duration > 0

    @pytest.mark.asyncio
    async def test_stage_started_and_completed_events(self):
        """Each node emits StageStarted and StageCompleted."""
        g = parse_dot("""
        digraph S {
            graph [goal="Stage test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do it"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        await run_pipeline(g, registry, on_event=events.append)

        stage_started = [e for e in events if isinstance(e, StageStarted)]
        stage_completed = [e for e in events if isinstance(e, StageCompleted)]

        # 3 nodes: start, task, done
        assert len(stage_started) == 3
        assert len(stage_completed) == 3

        # Names match node IDs
        names = [e.name for e in stage_started]
        assert "start" in names
        assert "task" in names
        assert "done" in names

    @pytest.mark.asyncio
    async def test_pipeline_failed_event_on_no_start(self):
        """PipelineFailed emitted when pipeline fails."""
        from attractor_pipeline.graph import Graph

        g = Graph(name="empty")
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        result = await run_pipeline(g, registry, on_event=events.append)

        assert result.status == PipelineStatus.FAILED
        assert isinstance(events[0], PipelineStarted)
        failed = [e for e in events if isinstance(e, PipelineFailed)]
        assert len(failed) == 1

    @pytest.mark.asyncio
    async def test_on_event_none_is_safe(self):
        """Passing on_event=None (default) does not error."""
        g = parse_dot("""
        digraph Safe {
            graph [goal="No events"]
            start [shape=Mdiamond]
            done  [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # Default: no on_event -- must not raise
        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_checkpoint_saved_events(self):
        """CheckpointSaved event emitted after each node when logs_root is set."""
        import tempfile
        from pathlib import Path

        g = parse_dot("""
        digraph C {
            graph [goal="Checkpoint test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do it"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        events: list = []
        with tempfile.TemporaryDirectory() as tmp:
            await run_pipeline(
                g,
                registry,
                on_event=events.append,
                logs_root=Path(tmp),
            )

        ckpt_events = [e for e in events if isinstance(e, CheckpointSaved)]
        assert len(ckpt_events) >= 1
        node_ids = [e.node_id for e in ckpt_events]
        assert "task" in node_ids
