"""Section 9.6 Pipeline Event Types.

Typed, frozen dataclasses representing every observable event
in the pipeline lifecycle.  Each concrete type inherits from
`PipelineEvent` and exposes a human-readable `description` property.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineEvent:
    """Base class for all pipeline events."""

    @property
    def description(self) -> str:
        """Human-readable one-line summary of this event."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStarted(PipelineEvent):
    """Emitted when a pipeline run begins."""

    name: str
    id: str

    @property
    def description(self) -> str:
        return f"Pipeline '{self.name}' started (id={self.id})"


@dataclass(frozen=True)
class PipelineCompleted(PipelineEvent):
    """Emitted when a pipeline run finishes successfully."""

    duration: float
    artifact_count: int

    @property
    def description(self) -> str:
        return f"Pipeline completed in {self.duration:.1f}s ({self.artifact_count} artifacts)"


@dataclass(frozen=True)
class PipelineFailed(PipelineEvent):
    """Emitted when a pipeline run fails."""

    error: str
    duration: float

    @property
    def description(self) -> str:
        return f"Pipeline failed after {self.duration:.1f}s: {self.error}"


# ---------------------------------------------------------------------------
# Stage lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageStarted(PipelineEvent):
    """Emitted when a stage begins execution."""

    name: str
    index: int

    @property
    def description(self) -> str:
        return f"Stage '{self.name}' [{self.index}] started"


@dataclass(frozen=True)
class StageCompleted(PipelineEvent):
    """Emitted when a stage finishes successfully."""

    name: str
    index: int
    duration: float

    @property
    def description(self) -> str:
        return f"Stage '{self.name}' [{self.index}] completed in {self.duration:.1f}s"


@dataclass(frozen=True)
class StageFailed(PipelineEvent):
    """Emitted when a stage fails."""

    name: str
    index: int
    error: str
    will_retry: bool

    @property
    def description(self) -> str:
        retry = " (will retry)" if self.will_retry else ""
        return f"Stage '{self.name}' [{self.index}] failed: {self.error}{retry}"


@dataclass(frozen=True)
class StageRetrying(PipelineEvent):
    """Emitted when a stage is about to be retried."""

    name: str
    index: int
    attempt: int
    delay: float

    @property
    def description(self) -> str:
        return (
            f"Stage '{self.name}' [{self.index}] retrying"
            f" (attempt {self.attempt}, delay {self.delay:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelStarted(PipelineEvent):
    """Emitted when parallel execution begins."""

    branch_count: int

    @property
    def description(self) -> str:
        return f"Parallel execution started ({self.branch_count} branches)"


@dataclass(frozen=True)
class ParallelBranchStarted(PipelineEvent):
    """Emitted when a single parallel branch begins."""

    branch: str
    index: int

    @property
    def description(self) -> str:
        return f"Parallel branch '{self.branch}' [{self.index}] started"


@dataclass(frozen=True)
class ParallelBranchCompleted(PipelineEvent):
    """Emitted when a single parallel branch finishes."""

    branch: str
    index: int
    duration: float
    success: bool

    @property
    def description(self) -> str:
        status = "succeeded" if self.success else "failed"
        return f"Parallel branch '{self.branch}' [{self.index}] {status} in {self.duration:.1f}s"


@dataclass(frozen=True)
class ParallelCompleted(PipelineEvent):
    """Emitted when all parallel branches have finished."""

    duration: float
    success_count: int
    failure_count: int

    @property
    def description(self) -> str:
        return (
            f"Parallel execution completed in {self.duration:.1f}s "
            f"({self.success_count} succeeded, {self.failure_count} failed)"
        )


# ---------------------------------------------------------------------------
# Human interaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterviewStarted(PipelineEvent):
    """Emitted when the pipeline asks a human a question."""

    question: str
    stage: str

    @property
    def description(self) -> str:
        return f"Interview started in stage '{self.stage}': {self.question}"


@dataclass(frozen=True)
class InterviewCompleted(PipelineEvent):
    """Emitted when a human answers a pipeline question."""

    question: str
    answer: str
    duration: float

    @property
    def description(self) -> str:
        return f"Interview completed in {self.duration:.1f}s"


@dataclass(frozen=True)
class InterviewTimeout(PipelineEvent):
    """Emitted when a human does not answer in time."""

    question: str
    stage: str
    duration: float

    @property
    def description(self) -> str:
        return f"Interview timed out in stage '{self.stage}' after {self.duration:.1f}s"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSaved(PipelineEvent):
    """Emitted when pipeline state is checkpointed."""

    node_id: str

    @property
    def description(self) -> str:
        return f"Checkpoint saved at node '{self.node_id}'"
