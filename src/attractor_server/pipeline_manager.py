"""Pipeline lifecycle manager -- tracks running pipelines.

Manages the lifecycle of pipeline executions: start, track, cancel,
and clean up. Each pipeline runs as an asyncio.Task with its own
AbortSignal and event queue.

Spec reference: attractor-spec §9.5.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_pipeline import (
    HandlerRegistry,
    PipelineStatus,
    parse_dot,
    run_pipeline,
)
from attractor_pipeline.engine.runner import PipelineResult
from attractor_pipeline.graph import Graph


class RunStatus(StrEnum):
    """Pipeline run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SSEEvent:
    """A typed event for SSE streaming."""

    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PendingQuestion:
    """A pending human gate question."""

    qid: str
    question: str
    stage: str
    timestamp: float
    answer_event: asyncio.Event = field(default_factory=asyncio.Event)
    answer: str | None = None


@dataclass
class PipelineRun:
    """State for a single pipeline execution."""

    id: str
    graph: Graph
    status: RunStatus = RunStatus.PENDING
    abort_signal: AbortSignal = field(default_factory=AbortSignal)
    task: asyncio.Task[Any] | None = None
    result: PipelineResult | None = None
    context: dict[str, Any] = field(default_factory=dict)
    completed_nodes: list[str] = field(default_factory=list)
    current_node: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    error: str | None = None

    # SSE event queues (one per subscribed client)
    _event_subscribers: list[asyncio.Queue[SSEEvent | None]] = field(default_factory=list)
    # Done event -- checked by late-connecting SSE clients
    _done_event: asyncio.Event = field(default_factory=asyncio.Event)
    _is_closed: bool = False
    # Event history buffer -- replayed to late-subscribing SSE clients
    _event_history: list[SSEEvent] = field(default_factory=list)

    # Pending human gate questions
    pending_questions: dict[str, PendingQuestion] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether the pipeline has reached a terminal state."""
        return self.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)

    def subscribe(self) -> asyncio.Queue[SSEEvent | None]:
        """Create a new SSE subscriber queue.

        Replays buffered event history so late-subscribing clients
        don't miss events emitted before they connected (e.g.,
        pipeline.started emitted before SSE client opens connection).
        """
        q: asyncio.Queue[SSEEvent | None] = asyncio.Queue(maxsize=1000)
        # Replay history for late subscribers
        for event in self._event_history:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                break
        self._event_subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[SSEEvent | None]) -> None:
        """Remove an SSE subscriber."""
        if q in self._event_subscribers:
            self._event_subscribers.remove(q)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an SSE event to all subscribers and buffer for replay."""
        event = SSEEvent(event_type=event_type, data=data)
        self._event_history.append(event)
        for q in self._event_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if subscriber is slow

    def close_subscribers(self) -> None:
        """Signal all subscribers that the stream is done.

        Idempotent -- safe to call multiple times. Sets the done_event
        which SSE streams check alongside queue.get(). Does NOT drain
        queues (events may still be unread by slow consumers).
        Sends sentinel as best-effort; done_event is the primary signal.
        """
        if self._is_closed:
            return
        self._is_closed = True
        self._done_event.set()

        # Best-effort sentinel -- done_event is the reliable signal
        for q in self._event_subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass  # OK -- done_event will terminate the stream


class PipelineManager:
    """Manages pipeline lifecycle for the HTTP server.

    Thread-safe for asyncio (single event loop).
    """

    MAX_COMPLETED_RUNS = 100  # Evict oldest completed runs beyond this

    def __init__(
        self,
        handlers: HandlerRegistry | None = None,
        max_concurrent: int = 5,
    ) -> None:
        self._runs: dict[str, PipelineRun] = {}
        self._handlers = handlers or HandlerRegistry()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def handlers(self) -> HandlerRegistry:
        return self._handlers

    def set_handlers(self, handlers: HandlerRegistry) -> None:
        self._handlers = handlers

    def get_run(self, pipeline_id: str) -> PipelineRun | None:
        return self._runs.get(pipeline_id)

    def list_runs(self) -> list[PipelineRun]:
        return list(self._runs.values())

    async def start_pipeline(
        self,
        dot_source: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineRun:
        """Parse DOT and start pipeline execution in background."""
        # Parse the DOT source (fails fast on invalid DOT)
        graph = parse_dot(dot_source)

        # Create the run with full UUID to avoid collisions
        run_id = uuid.uuid4().hex[:12]
        run = PipelineRun(
            id=run_id,
            graph=graph,
            context=dict(context or {}),
        )
        self._runs[run_id] = run

        # Evict old completed runs to prevent memory leak
        self._evict_stale_runs()

        # Start execution in background
        run.task = asyncio.create_task(self._execute(run))
        return run

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel a running pipeline.

        Only sets the abort signal -- _execute owns all state transitions
        to prevent race conditions between cancel and completion.
        """
        run = self._runs.get(pipeline_id)
        if not run:
            return False
        if run.is_terminal:
            return False

        run.abort_signal.set()
        return True

    async def _execute(self, run: PipelineRun) -> None:
        """Execute the pipeline and track state.

        This method owns ALL state transitions (RUNNING, COMPLETED,
        FAILED, CANCELLED). cancel_pipeline only sets the abort signal;
        this method checks it and sets the final status.
        """
        async with self._semaphore:
            # Check if cancelled before we even started
            if run.abort_signal.is_set:
                run.status = RunStatus.CANCELLED
                run.end_time = time.time()
                run.emit("pipeline.cancelled", {"id": run.id})
                run.close_subscribers()
                return

            run.status = RunStatus.RUNNING
            run.emit(
                "pipeline.started",
                {
                    "id": run.id,
                    "name": run.graph.name,
                    "goal": run.graph.goal,
                },
            )

            # Wire a per-run WebInterviewer so human gates push
            # questions to this run's pending_questions dict
            from attractor_pipeline.handlers.human import HumanHandler
            from attractor_server.interviewer import WebInterviewer

            web_interviewer = WebInterviewer(run)
            self._handlers.register("wait.human", HumanHandler(interviewer=web_interviewer))

            try:
                result = await run_pipeline(
                    run.graph,
                    self._handlers,
                    context=run.context,
                    abort_signal=run.abort_signal,
                )

                run.result = result
                run.context = result.context
                run.completed_nodes = result.completed_nodes

                # Check abort signal AFTER run_pipeline returns -- it may
                # have completed naturally despite a cancel request
                if run.abort_signal.is_set:
                    run.status = RunStatus.CANCELLED
                    run.emit("pipeline.cancelled", {"id": run.id})
                elif result.status == PipelineStatus.COMPLETED:
                    run.status = RunStatus.COMPLETED
                    run.emit(
                        "pipeline.completed",
                        {
                            "id": run.id,
                            "duration": time.time() - run.start_time,
                            "nodes_completed": len(result.completed_nodes),
                        },
                    )
                elif result.status == PipelineStatus.CANCELLED:
                    run.status = RunStatus.CANCELLED
                    run.emit("pipeline.cancelled", {"id": run.id})
                else:
                    run.status = RunStatus.FAILED
                    run.error = result.error
                    run.emit(
                        "pipeline.failed",
                        {
                            "id": run.id,
                            "error": result.error or "Unknown error",
                            "duration": time.time() - run.start_time,
                        },
                    )

            except Exception as exc:  # noqa: BLE001
                run.status = RunStatus.FAILED
                run.error = f"{type(exc).__name__}: {exc}"
                run.emit(
                    "pipeline.failed",
                    {
                        "id": run.id,
                        "error": run.error,
                    },
                )

            finally:
                run.end_time = time.time()
                run.task = None  # Release task reference
                run.close_subscribers()

    def _evict_stale_runs(self) -> None:
        """Remove oldest completed runs if over the limit."""
        completed = [
            (rid, r) for rid, r in self._runs.items() if r.is_terminal and r.end_time is not None
        ]
        if len(completed) <= self.MAX_COMPLETED_RUNS:
            return

        # Sort by end_time, evict oldest
        completed.sort(key=lambda x: x[1].end_time or 0)
        to_evict = len(completed) - self.MAX_COMPLETED_RUNS
        for rid, _ in completed[:to_evict]:
            del self._runs[rid]
