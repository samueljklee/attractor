"""Subgraph execution for parallel branches.

Implements execute_subgraph() -- our Issue 1 design from the 5-model
swarm analysis. This function re-enters the core execution loop from
a branch start node and runs until hitting a boundary node (fan-in
or exit).

Key design decisions:
- Boundary: stops at fan-in (tripleoctagon) or exit (Msquare) nodes
  WITHOUT executing them. The fan-in handler collects results.
- Reuses existing machinery: execute_with_retry pattern, select_edge.
- Context isolation: each branch gets a dict copy. Mutations don't
  leak between branches.
- Checkpoints: branch checkpoints scoped under parent.

Spec reference: attractor-spec §4.8 + our Issue 1 design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import (
    HandlerRegistry,
    HandlerResult,
    Outcome,
    select_edge,
)
from attractor_pipeline.graph import Graph, Node

# Boundary shapes: subgraph execution stops here without executing
BOUNDARY_SHAPES = frozenset({"tripleoctagon", "Msquare"})


async def execute_subgraph(
    start_node: Node,
    context: dict[str, Any],
    graph: Graph,
    handlers: HandlerRegistry,
    *,
    logs_root: Path | None = None,
    abort_signal: AbortSignal | None = None,
    max_steps: int = 500,
) -> HandlerResult:
    """Execute a subgraph starting at start_node until a boundary.

    This is the parallel branch execution primitive. It re-enters
    the core execution loop logic but stops at fan-in or exit nodes
    instead of processing them.

    Args:
        start_node: The first node to execute in this branch.
        context: Cloned context dict for this branch (isolated).
        graph: The full pipeline graph (shared, read-only for routing).
        handlers: Handler registry (shared across branches).
        logs_root: Optional directory for branch-specific logs.
        abort_signal: Cooperative cancellation.
        max_steps: Safety cap on iterations within this branch.

    Returns:
        HandlerResult from the last executed node in the branch.
    """
    abort = abort_signal or AbortSignal()
    current_node = start_node
    steps = 0
    last_result = HandlerResult(status=Outcome.SUCCESS)

    while True:
        # Safety cap
        steps += 1
        if steps > max_steps:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(
                    f"Branch exceeded max steps ({max_steps}) at node '{current_node.id}'"
                ),
            )

        # Check abort
        if abort.is_set:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Branch cancelled",
            )

        # Boundary check: stop at fan-in or exit WITHOUT executing
        if current_node.shape in BOUNDARY_SHAPES:
            return last_result

        # Dead end: no outgoing edges and not a boundary
        if not graph.outgoing_edges(current_node.id) and current_node != start_node:
            return last_result

        # Resolve and execute handler
        handler_name = current_node.effective_handler
        handler = handlers.get(handler_name)

        if handler is None:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(f"No handler for '{handler_name}' (node: {current_node.id})"),
            )

        try:
            result = await handler.execute(
                current_node,
                context,
                graph,
                logs_root,
                abort,
            )
        except Exception as exc:  # noqa: BLE001
            result = HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        last_result = result

        # Apply context updates
        context.update(result.context_updates)

        # Select next edge
        next_edge = select_edge(current_node, result, graph, context)
        if next_edge is None:
            return result

        next_node = graph.get_node(next_edge.target)
        if next_node is None:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"Edge target not found: {next_edge.target}",
            )

        current_node = next_node
