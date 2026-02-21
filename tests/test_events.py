"""Tests for Section 9.6 event system: event types and EventEmitter."""

from __future__ import annotations

from attractor_pipeline.engine.events import (
    CheckpointSaved,
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
