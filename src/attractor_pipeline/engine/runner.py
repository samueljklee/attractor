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
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import anyio

from attractor_agent.abort import AbortSignal
from attractor_llm.retry import RetryPolicy
from attractor_pipeline.conditions import evaluate_condition
from attractor_pipeline.graph import Edge, Graph, Node
from attractor_pipeline.stylesheet import apply_stylesheet

# ------------------------------------------------------------------ #
# Retry presets (Spec §3.6, §11.5)
# ------------------------------------------------------------------ #

RETRY_PRESETS: dict[str, RetryPolicy] = {
    "none": RetryPolicy(
        max_retries=0, initial_delay=0.0, backoff_factor=1.0, max_delay=0.0, jitter=False
    ),
    "standard": RetryPolicy(
        max_retries=3, initial_delay=1.0, backoff_factor=2.0, max_delay=30.0, jitter=True
    ),
    "aggressive": RetryPolicy(
        max_retries=5, initial_delay=0.5, backoff_factor=1.5, max_delay=10.0, jitter=True
    ),
    "linear": RetryPolicy(
        max_retries=3, initial_delay=2.0, backoff_factor=1.0, max_delay=60.0, jitter=False
    ),
    "patient": RetryPolicy(
        max_retries=10, initial_delay=5.0, backoff_factor=2.0, max_delay=120.0, jitter=True
    ),
}


def get_retry_preset(name: str) -> RetryPolicy | None:
    """Return a named retry preset, or None if the name is unknown.

    Args:
        name: One of 'none', 'standard', 'aggressive', 'linear', 'patient'.

    Returns:
        The matching RetryPolicy, or None for unrecognised names.
    """
    return RETRY_PRESETS.get(name)


# Pipeline retry backoff policy (Spec §3.6) -- uses 'standard' preset so that
# node-level retries get sensible exponential backoff by default.
_PIPELINE_RETRY = RETRY_PRESETS["standard"]


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
        abort_signal: AbortSignal | None = None,
        # NOTE: Deliberate extension beyond spec §3.3.
        # Spec defines 4 params: (node, context, graph, logs_root).
        # We add abort_signal as an optional 5th param for cooperative
        # cancellation (Issue 4 design).  Defaulting to None preserves
        # backward compatibility with the 4-param spec signature.
    ) -> HandlerResult: ...


# ------------------------------------------------------------------ #
# Edge selection algorithm
# ------------------------------------------------------------------ #


def _best_by_weight_then_lexical(edges: list[Edge]) -> Edge:
    """Select best edge: highest weight, then lexicographic tiebreak on target."""
    return sorted(edges, key=lambda e: (-e.weight, e.target))[0]


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
    4. Weight + lexical: highest-weight edge, lexicographic tiebreak on target
    5. Dead end: all edges have conditions and none matched

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
        return _best_by_weight_then_lexical(condition_matches)

    # Step 2: Preferred label matching
    if result.preferred_label:
        # Strip accelerator keys (e.g., "&Yes" -> "Yes")
        clean_label = result.preferred_label.replace("&", "").strip()
        label_matches = [
            e for e in edges if e.label.replace("&", "").strip().lower() == clean_label.lower()
        ]
        if label_matches:
            return _best_by_weight_then_lexical(label_matches)

    # Step 3: Suggested next IDs
    if result.suggested_next_ids:
        id_set = set(result.suggested_next_ids)
        id_matches = [e for e in edges if e.target in id_set]
        if id_matches:
            return _best_by_weight_then_lexical(id_matches)

    # Step 4/5: Weight + lexical tiebreak on unconditional edges
    unconditional = [e for e in edges if not e.condition]
    if unconditional:
        return _best_by_weight_then_lexical(unconditional)

    # All edges have conditions and none matched -- dead end.
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
# Pipeline context (Spec §5.1, §11.7)
# ------------------------------------------------------------------ #


@dataclass
class PipelineContext:
    """Typed context container for pipeline execution state. Spec §5.1.

    Wraps a plain dict but exposes a structured interface with snapshot,
    clone, and log-append operations.  The internal dict is the single
    source of truth so existing code that holds a reference to
    ``pipeline_ctx._data`` continues to work.

    Backward compatibility: ``run_pipeline()`` still accepts a bare
    ``dict[str, Any]`` and converts it internally.
    """

    _data: dict[str, Any] = field(default_factory=dict)

    # ---- read/write ------------------------------------------------ #

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*."""
        self._data[key] = value

    # ---- bulk operations ------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current context."""
        return dict(self._data)

    def clone(self) -> PipelineContext:
        """Return an independent copy of this context."""
        return PipelineContext(_data=dict(self._data))

    def apply_updates(self, updates: dict[str, Any]) -> None:
        """Merge *updates* into the context (last-write wins)."""
        self._data.update(updates)

    # ---- audit log -------------------------------------------------- #

    def append_log(self, entry: str) -> None:
        """Append *entry* to the internal ``_log`` list."""
        logs: list[str] = self._data.setdefault("_log", [])
        logs.append(entry)


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


def _check_goal_gate(
    node: Node,
    result: HandlerResult,
    ctx: dict[str, Any],
    graph: Graph,
    redirect_count: int,
    start_time: float,
    completed_nodes: list[str],
) -> tuple[str, int, Node | PipelineResult | None]:
    """Evaluate a node's goal gate and determine the action to take.

    Returns a 3-tuple of (action, new_redirect_count, payload):
    - ("pass", count, None): gate satisfied or no gate.
    - ("redirect", count, Node): gate failed; redirect to this target node.
    - ("fail", count, PipelineResult): circuit breaker tripped.
    - ("no_retry_target", count, None): gate failed but no retry target set.
    """
    if not node.goal_gate:
        return ("pass", redirect_count, None)

    gate_vars: dict[str, Any] = {
        "outcome": result.status.value,
        **ctx,
    }
    if evaluate_condition(node.goal_gate, gate_vars):
        return ("pass", redirect_count, None)

    # Gate failed
    redirect_count += 1

    # Circuit breaker (Spec §3.4 + Issue 2 design)
    if graph.max_goal_gate_redirects > 0 and redirect_count >= graph.max_goal_gate_redirects:
        return (
            "fail",
            redirect_count,
            PipelineResult(
                status=PipelineStatus.FAILED,
                error=(
                    f"Goal gate on node '{node.id}' unsatisfied after "
                    f"{redirect_count} redirects "
                    f"(limit: {graph.max_goal_gate_redirects})"
                ),
                context=ctx,
                completed_nodes=completed_nodes,
                final_outcome=result,
                duration_seconds=time.monotonic() - start_time,
            ),
        )

    # Redirect to retry target
    retry_target = node.retry_target
    if retry_target:
        target_node = graph.get_node(retry_target)
        if target_node:
            return ("redirect", redirect_count, target_node)

    # No retry target configured -- caller decides behavior
    return ("no_retry_target", redirect_count, None)


def _safe_node_id(node_id: str) -> str:
    """Sanitize node ID for use as a directory name (defense-in-depth)."""
    return node_id.replace("/", "_").replace("\\", "_").replace("..", "_")


def _write_node_artifacts(
    logs_root: Path,
    node: Node,
    result: HandlerResult,
    context: dict[str, Any],
) -> None:
    """Write per-node artifact files for ANY node. Spec §5.6, §11.3, §11.7.

    Every executed node gets ``status.json``.  Nodes that have a prompt
    and/or LLM response also get ``prompt.md`` and ``response.md``.

    Artifact writing is *observability*, not core logic -- I/O errors
    are swallowed so they never tank the pipeline.
    """
    try:
        safe_id = _safe_node_id(node.id)
        node_dir = logs_root / safe_id
        node_dir.mkdir(parents=True, exist_ok=True)

        # status.json -- always written for every node
        status_data = {
            "node_id": node.id,
            "status": result.status.value,
            "preferred_label": result.preferred_label,
            "failure_reason": result.failure_reason,
            "notes": result.notes,
        }
        (node_dir / "status.json").write_text(json.dumps(status_data, indent=2), encoding="utf-8")

        # prompt.md -- expanded prompt (codergen) or raw prompt (others)
        prompt = context.get(f"_artifact_prompt.{node.id}", "") or node.prompt or ""
        if prompt:
            (node_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        # response.md -- handler output text (if any)
        if result.output:
            (node_dir / "response.md").write_text(result.output, encoding="utf-8")
    except Exception:  # noqa: BLE001
        # Artifact writing is observability, not core logic.
        # Don't let I/O errors tank the pipeline.
        pass


def _check_aggregate_goal_gates(
    graph: Graph,
    completed_nodes: list[str],
    node_outcomes: dict[str, Outcome],
    ctx: dict[str, Any],
    redirect_count: int,
    start_time: float,
) -> tuple[str, int, Node | PipelineResult | None]:
    """Check ALL visited goal-gate nodes in aggregate at exit. Spec §3.4.

    When the traversal reaches a terminal node, every *visited* node
    whose ``goal_gate`` attribute is set is re-evaluated using its
    stored outcome.  If any gate is unsatisfied the pipeline redirects
    to that node's ``retry_target`` (or fails if none is configured).

    Returns the same 3-tuple as ``_check_goal_gate``:
    ``("pass" | "redirect" | "fail" | "no_retry_target", count, payload)``.
    """
    seen: set[str] = set()
    for node_id in completed_nodes:
        if node_id in seen:
            continue
        seen.add(node_id)

        node = graph.get_node(node_id)
        if node is None or not node.goal_gate:
            continue

        outcome = node_outcomes.get(node_id)
        if outcome is None:
            continue

        gate_vars: dict[str, Any] = {"outcome": outcome.value, **ctx}
        if evaluate_condition(node.goal_gate, gate_vars):
            continue  # gate satisfied

        # Gate failed
        redirect_count += 1

        # Circuit breaker (Spec §3.4 + Issue 2 design)
        if graph.max_goal_gate_redirects > 0 and redirect_count >= graph.max_goal_gate_redirects:
            return (
                "fail",
                redirect_count,
                PipelineResult(
                    status=PipelineStatus.FAILED,
                    error=(
                        f"Aggregate goal gate on node '{node_id}' unsatisfied after "
                        f"{redirect_count} redirects "
                        f"(limit: {graph.max_goal_gate_redirects})"
                    ),
                    context=ctx,
                    completed_nodes=completed_nodes,
                    duration_seconds=time.monotonic() - start_time,
                ),
            )

        # Redirect to retry target
        retry_target = node.retry_target
        if retry_target:
            target_node = graph.get_node(retry_target)
            if target_node:
                return ("redirect", redirect_count, target_node)

        # No retry target configured
        return ("no_retry_target", redirect_count, None)

    return ("pass", redirect_count, None)


async def run_pipeline(
    graph: Graph,
    handlers: HandlerRegistry,
    *,
    context: dict[str, Any] | PipelineContext | None = None,
    abort_signal: AbortSignal | None = None,
    logs_root: Path | None = None,
    checkpoint: Checkpoint | None = None,
    transforms: list[Any] | None = None,
) -> PipelineResult:
    """Execute a pipeline graph. Spec §3.2.

    This is the core execution loop:
    1. Find the start node (or resume from checkpoint)
    2. Apply graph transforms (Spec §9, §11.11)
    3. Execute the current node's handler
    4. Select the next edge using the 5-step algorithm
    5. Check goal gates at exit nodes
    6. Repeat until terminal node or limit reached

    Args:
        graph: The parsed pipeline graph.
        handlers: Registry of node handlers.
        context: Initial pipeline context -- accepts a bare ``dict[str, Any]``
            (legacy) or a ``PipelineContext`` (Spec §5.1, §11.7).  A bare dict
            is wrapped in a ``PipelineContext`` automatically so all downstream
            code can rely on the structured interface.
        abort_signal: Cooperative cancellation signal.
        logs_root: Directory for logs and artifacts.
        checkpoint: Resume from a saved checkpoint.
        transforms: Ordered list of GraphTransform implementations
            to apply between parsing and validation (Spec §9).
            Each must have an ``apply(graph) -> graph`` method.

    Returns:
        PipelineResult with final status, context, and metadata.
    """
    abort = abort_signal or AbortSignal()

    # Accept dict or PipelineContext; always work on the underlying dict so
    # existing engine helpers that expect a plain dict need no changes.
    if isinstance(context, PipelineContext):
        pipeline_context = context
    else:
        pipeline_context = PipelineContext(_data=dict(context or {}))
    ctx = pipeline_context._data  # noqa: SLF001  -- internal access by design
    start_time = time.monotonic()

    # Apply graph transforms before validation (Spec §9, §11.11)
    if transforms:
        from attractor_pipeline.transforms import apply_transforms

        graph = apply_transforms(graph, transforms)

    # Set goal in context AFTER transforms (transforms may modify graph.goal)
    ctx["goal"] = graph.goal

    # Apply model stylesheet to nodes before execution (Spec §8)
    apply_stylesheet(graph)

    # State tracking
    completed_nodes: list[str] = []
    node_retry_counts: dict[str, int] = {}
    node_outcomes: dict[str, Outcome] = {}  # for aggregate goal gate check (Spec §3.4)
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
                delay = _PIPELINE_RETRY.compute_delay(retry_count)
                await anyio.sleep(delay)
                continue  # retry same node
            # Max retries exhausted -- fall through to edge selection

        # Track final node outcome for aggregate goal gate check (Spec §3.4)
        node_outcomes[current_node.id] = result.status

        # Write per-node artifacts for EVERY node (Spec §5.6, §11.3, §11.7)
        if logs_root:
            _write_node_artifacts(logs_root, current_node, result, ctx)

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
            action, goal_gate_redirect_count, payload = _check_goal_gate(
                current_node,
                result,
                ctx,
                graph,
                goal_gate_redirect_count,
                start_time,
                completed_nodes,
            )
            if action == "fail":
                return payload  # type: ignore[return-value]
            if action == "redirect":
                current_node = payload  # type: ignore[assignment]
                continue
            # action == "no_retry_target": intentional fall-through to normal
            # edge selection. The gate failed but no retry_target is configured,
            # so we continue along the graph edges rather than silently stalling.

        # Check if this is an exit node
        if current_node.shape == "Msquare":
            # Aggregate goal gate check: verify ALL visited goal-gate
            # nodes before allowing exit (Spec §3.4, §11.4).
            agg_action, goal_gate_redirect_count, agg_payload = _check_aggregate_goal_gates(
                graph,
                completed_nodes,
                node_outcomes,
                ctx,
                goal_gate_redirect_count,
                start_time,
            )
            if agg_action == "fail":
                return agg_payload  # type: ignore[return-value]
            if agg_action == "redirect":
                current_node = agg_payload  # type: ignore[assignment]
                continue
            if agg_action == "no_retry_target":
                return PipelineResult(
                    status=PipelineStatus.FAILED,
                    error="Aggregate goal gate unsatisfied and no retry target",
                    context=ctx,
                    completed_nodes=completed_nodes,
                    final_outcome=result,
                    duration_seconds=time.monotonic() - start_time,
                )

            # Per-exit-node goal gate check (Spec §3.4 + Issue 2 circuit breaker)
            if current_node.goal_gate:
                action, goal_gate_redirect_count, payload = _check_goal_gate(
                    current_node,
                    result,
                    ctx,
                    graph,
                    goal_gate_redirect_count,
                    start_time,
                    completed_nodes,
                )
                if action == "fail":
                    return payload  # type: ignore[return-value]
                if action == "redirect":
                    current_node = payload  # type: ignore[assignment]
                    continue
                if action == "no_retry_target":
                    # Exit node with no retry target: pipeline fails
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
