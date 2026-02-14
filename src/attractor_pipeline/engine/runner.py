"""Pipeline execution engine -- the core run loop.

Implements the 5-phase pipeline lifecycle from attractor-spec §3:
1. PARSE: DOT source -> Graph (done by parser module)
2. VALIDATE: Check graph correctness (done by validator module)
3. INITIALIZE: Set up context, checkpoint, artifact store
4. EXECUTE: Core loop -- execute node -> select edge -> advance
5. FINALIZE: Clean up resources, save final state

Key design decisions implemented from our swarm analysis:
- Goal gate circuit breaker (Issue 2): max_goal_gate_redirects counter
- Cooperative cancellation (Issue 4): AbortSignal threaded through
- Edge selection: 5-step priority algorithm from spec §3.3

Spec reference: attractor-spec §3.1-3.8.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import anyio

from attractor_agent.abort import AbortSignal
from attractor_pipeline.conditions import evaluate_condition
from attractor_pipeline.graph import Edge, Graph, Node
from attractor_pipeline.stylesheet import apply_stylesheet

# Pipeline retry backoff constants (Spec §3.6)
_RETRY_INITIAL_DELAY = 0.2  # 200ms
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_MAX_DELAY = 60.0  # 60 seconds


def _compute_retry_delay(attempt: int) -> float:
    """Compute exponential backoff delay with jitter for pipeline retries."""
    delay = _RETRY_INITIAL_DELAY * (_RETRY_BACKOFF_FACTOR**attempt)
    delay = min(delay, _RETRY_MAX_DELAY)
    # Equal jitter: uniform in [delay/2, delay]
    delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
    return delay


# ------------------------------------------------------------------ #
# Handler protocol
# ------------------------------------------------------------------ #


class Outcome(StrEnum):
    """Node execution outcome status. Spec §5.2."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAIL = "fail"
    RETRY = "retry"


@dataclass
class HandlerResult:
    """Result returned by a node handler. Spec §5.2.

    Contains the outcome status, routing hints, context updates,
    and notes for downstream consumption.
    """

    status: Outcome = Outcome.SUCCESS
    preferred_label: str = ""
    suggested_next_ids: list[str] = field(default_factory=list)
    context_updates: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    failure_reason: str = ""
    output: str = ""  # The handler's text output


class Handler(Protocol):
    """Protocol for node handlers. Spec §4.1.

    Each handler type implements this protocol. The engine calls
    execute() and uses the returned HandlerResult for edge selection.
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult: ...


# ------------------------------------------------------------------ #
# Edge selection algorithm
# ------------------------------------------------------------------ #


def select_edge(
    node: Node,
    result: HandlerResult,
    graph: Graph,
    context: dict[str, Any],
) -> Edge | None:
    """Select the next edge using the 5-step priority algorithm. Spec §3.3.

    Priority order:
    1. Condition match: edges whose condition evaluates to true
    2. Preferred label: edges matching result.preferred_label
    3. Suggested IDs: edges targeting a node in result.suggested_next_ids
    4. Weight: highest-weight edge among remaining candidates
    5. Lexical: alphabetically first target node ID (deterministic tiebreak)

    Returns None if no outgoing edges exist (terminal node).
    """
    edges = graph.outgoing_edges(node.id)
    if not edges:
        return None

    # Build evaluation variables
    variables: dict[str, Any] = {
        "outcome": result.status.value,
        "preferred_label": result.preferred_label,
        **context,
    }

    # Step 1: Condition matching
    condition_matches = [
        e for e in edges if e.condition and evaluate_condition(e.condition, variables)
    ]
    if condition_matches:
        return condition_matches[0]

    # Step 2: Preferred label matching
    if result.preferred_label:
        # Strip accelerator keys (e.g., "&Yes" -> "Yes")
        clean_label = result.preferred_label.replace("&", "").strip()
        label_matches = [
            e for e in edges if e.label.replace("&", "").strip().lower() == clean_label.lower()
        ]
        if label_matches:
            return label_matches[0]

    # Step 3: Suggested next IDs
    if result.suggested_next_ids:
        id_set = set(result.suggested_next_ids)
        id_matches = [e for e in edges if e.target in id_set]
        if id_matches:
            return id_matches[0]

    # Step 4: Weight (edges already sorted by weight desc from outgoing_edges)
    # Take unconditional edges (no condition) sorted by weight
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        return unconditional[0]

    # Step 5: If all edges have conditions and none matched, this is a dead end.
    # Returning a false-condition edge would silently violate the graph author's guards.
    return None


# ------------------------------------------------------------------ #
# Pipeline result
# ------------------------------------------------------------------ #


class PipelineStatus(StrEnum):
    """Final pipeline execution status."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    status: PipelineStatus
    context: dict[str, Any] = field(default_factory=dict)
    completed_nodes: list[str] = field(default_factory=list)
    final_outcome: HandlerResult | None = None
    error: str | None = None
    duration_seconds: float = 0.0


# ------------------------------------------------------------------ #
# Checkpoint
# ------------------------------------------------------------------ #


@dataclass
class Checkpoint:
    """Serializable pipeline state for resume. Spec §5.3."""

    graph_name: str = ""
    current_node_id: str = ""
    context_values: dict[str, Any] = field(default_factory=dict)
    completed_nodes: list[dict[str, Any]] = field(default_factory=list)
    node_retry_counts: dict[str, int] = field(default_factory=dict)
    goal_gate_redirect_count: int = 0
    status: str = "running"

    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "graph_name": self.graph_name,
            "current_node_id": self.current_node_id,
            "context_values": self.context_values,
            "completed_nodes": self.completed_nodes,
            "node_retry_counts": self.node_retry_counts,
            "goal_gate_redirect_count": self.goal_gate_redirect_count,
            "status": self.status,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Load checkpoint from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ------------------------------------------------------------------ #
# Handler registry
# ------------------------------------------------------------------ #


class HandlerRegistry:
    """Registry mapping handler names to Handler implementations."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, name: str, handler: Handler) -> None:
        """Register a handler by name."""
        self._handlers[name] = handler

    def get(self, name: str) -> Handler | None:
        """Look up a handler by name."""
        return self._handlers.get(name)


# ------------------------------------------------------------------ #
# Core execution engine
# ------------------------------------------------------------------ #


async def run_pipeline(
    graph: Graph,
    handlers: HandlerRegistry,
    *,
    context: dict[str, Any] | None = None,
    abort_signal: AbortSignal | None = None,
    logs_root: Path | None = None,
    checkpoint: Checkpoint | None = None,
) -> PipelineResult:
    """Execute a pipeline graph. Spec §3.2.

    This is the core execution loop:
    1. Find the start node (or resume from checkpoint)
    2. Execute the current node's handler
    3. Select the next edge using the 5-step algorithm
    4. Check goal gates at exit nodes
    5. Repeat until terminal node or limit reached

    Args:
        graph: The parsed pipeline graph.
        handlers: Registry of node handlers.
        context: Initial pipeline context (mutable dict).
        abort_signal: Cooperative cancellation signal.
        logs_root: Directory for logs and artifacts.
        checkpoint: Resume from a saved checkpoint.

    Returns:
        PipelineResult with final status, context, and metadata.
    """
    abort = abort_signal or AbortSignal()
    ctx = dict(context or {})
    ctx["goal"] = graph.goal
    start_time = time.monotonic()

    # Apply model stylesheet to nodes before execution (Spec §8)
    apply_stylesheet(graph)

    # State tracking
    completed_nodes: list[str] = []
    node_retry_counts: dict[str, int] = {}
    goal_gate_redirect_count = 0

    # Resume from checkpoint if provided
    if checkpoint:
        ctx.update(checkpoint.context_values)
        completed_nodes = [n["node_id"] for n in checkpoint.completed_nodes]
        node_retry_counts = dict(checkpoint.node_retry_counts)
        goal_gate_redirect_count = checkpoint.goal_gate_redirect_count

        # Generate fidelity preamble so the LLM has context about
        # what happened before the checkpoint (Spec §5.4)
        from attractor_pipeline.engine.preamble import generate_resume_preamble

        preamble = generate_resume_preamble(graph, checkpoint)
        ctx["_resume_preamble"] = preamble

    # Find start node
    if checkpoint and checkpoint.current_node_id:
        current_node = graph.get_node(checkpoint.current_node_id)
    else:
        current_node = graph.get_start_node()

    if current_node is None:
        return PipelineResult(
            status=PipelineStatus.FAILED,
            error="No start node found in graph",
            duration_seconds=time.monotonic() - start_time,
        )

    # Global iteration cap: prevents infinite loops in cyclic graphs.
    # Ceiling = default_max_retry * number of nodes (generous but bounded).
    max_total_steps = graph.default_max_retry * max(len(graph.nodes), 1)
    total_steps = 0

    # === CORE EXECUTION LOOP (Spec §3.2) ===
    while True:
        total_steps += 1
        if total_steps > max_total_steps:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                error=(
                    f"Pipeline exceeded max total steps ({max_total_steps}). "
                    f"Possible infinite loop in graph."
                ),
                context=ctx,
                completed_nodes=completed_nodes,
                duration_seconds=time.monotonic() - start_time,
            )

        # Check abort
        if abort.is_set:
            # Save checkpoint on cancel (Issue 4 design)
            if logs_root:
                ckpt = Checkpoint(
                    graph_name=graph.name,
                    current_node_id=current_node.id,
                    context_values=ctx,
                    completed_nodes=[{"node_id": nid} for nid in completed_nodes],
                    node_retry_counts=node_retry_counts,
                    goal_gate_redirect_count=goal_gate_redirect_count,
                    status="cancelled",
                )
                ckpt.save(logs_root / "checkpoint.json")

            return PipelineResult(
                status=PipelineStatus.CANCELLED,
                context=ctx,
                completed_nodes=completed_nodes,
                duration_seconds=time.monotonic() - start_time,
            )

        # Resolve handler for current node
        handler_name = current_node.effective_handler
        handler = handlers.get(handler_name)

        if handler is None:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                error=f"No handler for '{handler_name}' (node: {current_node.id})",
                context=ctx,
                completed_nodes=completed_nodes,
                duration_seconds=time.monotonic() - start_time,
            )

        # Execute with retry (Spec §3.5)
        max_retries = (
            current_node.max_retries if current_node.max_retries >= 0 else graph.default_max_retry
        )
        retry_count = node_retry_counts.get(current_node.id, 0)

        try:
            result = await handler.execute(
                current_node,
                ctx,
                graph,
                logs_root,
                abort,
            )
        except Exception as exc:  # noqa: BLE001
            result = HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        # Apply context updates from handler
        ctx.update(result.context_updates)

        # Clear resume preamble after the first node executes post-resume.
        # Without this, every subsequent node gets the stale "[RESUME]
        # Resuming at node X" message even when it's now at node Y.
        ctx.pop("_resume_preamble", None)

        # Track completed nodes
        completed_nodes.append(current_node.id)

        # Handle retry on failure
        if result.status in (Outcome.FAIL, Outcome.RETRY):
            if retry_count < max_retries:
                node_retry_counts[current_node.id] = retry_count + 1
                # Exponential backoff with jitter (Spec §3.6)
                delay = _compute_retry_delay(retry_count)
                await anyio.sleep(delay)
                continue  # retry same node
            # Max retries exhausted -- fall through to edge selection

        # Save checkpoint after each node
        if logs_root:
            ckpt = Checkpoint(
                graph_name=graph.name,
                current_node_id=current_node.id,
                context_values=ctx,
                completed_nodes=[{"node_id": nid} for nid in completed_nodes],
                node_retry_counts=node_retry_counts,
                goal_gate_redirect_count=goal_gate_redirect_count,
                status="running",
            )
            ckpt.save(logs_root / "checkpoint.json")

        # Check goal gate on ANY node with goal_gate set (Spec §3.4)
        # Exit nodes handle their own goal gate below.
        if current_node.shape != "Msquare" and current_node.goal_gate:
            gate_vars = {
                "outcome": result.status.value,
                **ctx,
            }
            if not evaluate_condition(current_node.goal_gate, gate_vars):
                goal_gate_redirect_count += 1

                # Circuit breaker
                if (
                    graph.max_goal_gate_redirects > 0
                    and goal_gate_redirect_count >= graph.max_goal_gate_redirects
                ):
                    return PipelineResult(
                        status=PipelineStatus.FAILED,
                        error=(
                            f"Goal gate on node '{current_node.id}' unsatisfied after "
                            f"{goal_gate_redirect_count} redirects "
                            f"(limit: {graph.max_goal_gate_redirects})"
                        ),
                        context=ctx,
                        completed_nodes=completed_nodes,
                        final_outcome=result,
                        duration_seconds=time.monotonic() - start_time,
                    )

                # Redirect to retry target
                retry_target = current_node.retry_target
                if retry_target:
                    target_node = graph.get_node(retry_target)
                    if target_node:
                        current_node = target_node
                        continue

                # No retry target -- fall through to normal edge selection

        # Check if this is an exit node
        if current_node.shape == "Msquare":
            # Goal gate check (Spec §3.4 + Issue 2 circuit breaker)
            if current_node.goal_gate:
                gate_vars = {
                    "outcome": result.status.value,
                    **ctx,
                }
                if not evaluate_condition(current_node.goal_gate, gate_vars):
                    # Goal gate failed
                    goal_gate_redirect_count += 1

                    # Circuit breaker (Issue 2 design)
                    if (
                        graph.max_goal_gate_redirects > 0
                        and goal_gate_redirect_count >= graph.max_goal_gate_redirects
                    ):
                        return PipelineResult(
                            status=PipelineStatus.FAILED,
                            error=(
                                f"Goal gate unsatisfied after "
                                f"{goal_gate_redirect_count} redirects "
                                f"(limit: {graph.max_goal_gate_redirects})"
                            ),
                            context=ctx,
                            completed_nodes=completed_nodes,
                            final_outcome=result,
                            duration_seconds=time.monotonic() - start_time,
                        )

                    # Redirect to retry target
                    retry_target = current_node.retry_target
                    if retry_target:
                        target_node = graph.get_node(retry_target)
                        if target_node:
                            current_node = target_node
                            continue

                    # No retry target -- fail
                    return PipelineResult(
                        status=PipelineStatus.FAILED,
                        error="Goal gate unsatisfied and no retry target",
                        context=ctx,
                        completed_nodes=completed_nodes,
                        final_outcome=result,
                        duration_seconds=time.monotonic() - start_time,
                    )

            # Goal gate passed (or no gate) -- pipeline complete
            # Reset redirect counter on success
            goal_gate_redirect_count = 0

            return PipelineResult(
                status=PipelineStatus.COMPLETED,
                context=ctx,
                completed_nodes=completed_nodes,
                final_outcome=result,
                duration_seconds=time.monotonic() - start_time,
            )

        # Select next edge (Spec §3.3)
        next_edge = select_edge(current_node, result, graph, ctx)
        if next_edge is None:
            # Dead end -- no outgoing edges
            return PipelineResult(
                status=PipelineStatus.COMPLETED,
                context=ctx,
                completed_nodes=completed_nodes,
                final_outcome=result,
                duration_seconds=time.monotonic() - start_time,
            )

        # Advance to next node
        next_node = graph.get_node(next_edge.target)
        if next_node is None:
            return PipelineResult(
                status=PipelineStatus.FAILED,
                error=f"Edge target node not found: {next_edge.target}",
                context=ctx,
                completed_nodes=completed_nodes,
                duration_seconds=time.monotonic() - start_time,
            )

        current_node = next_node
