"""Parallel and Fan-In handlers for concurrent branch execution.

ParallelHandler (shape=component): Fans out to multiple branches,
running them concurrently via asyncio.TaskGroup. Each branch gets
an isolated context clone and runs execute_subgraph() until hitting
a fan-in or exit boundary.

FanInHandler (shape=tripleoctagon): Collects results from parallel
branches and selects the best one using our Issue 5 design (outcome
rank + completion order, no score field).

Spec reference: attractor-spec §4.8-4.9 + our Issue 1 and 5 designs.
"""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.engine.subgraph import execute_subgraph
from attractor_pipeline.graph import Graph, Node

# ------------------------------------------------------------------ #
# Branch result tracking
# ------------------------------------------------------------------ #


@dataclass
class BranchResult:
    """Result from a single parallel branch."""

    branch_id: str
    target_node_id: str
    result: HandlerResult
    context: dict[str, Any] = field(default_factory=dict)
    completed_at: float = 0.0
    error: str | None = None


# ------------------------------------------------------------------ #
# Join policies
# ------------------------------------------------------------------ #


class JoinPolicy:
    """Determines when the parallel handler considers all branches done."""

    WAIT_ALL = "wait_all"  # Wait for every branch to finish
    FIRST_SUCCESS = "first_success"  # Return as soon as one succeeds


# ------------------------------------------------------------------ #
# ParallelHandler
# ------------------------------------------------------------------ #


class ParallelHandler:
    """Handler for parallel fan-out nodes (shape=component). Spec §4.8.

    When the engine reaches a component node, this handler:
    1. Identifies all outgoing edges as parallel branches
    2. Clones the context for each branch
    3. Runs all branches concurrently via asyncio.TaskGroup
    4. Collects results and stores them in context for the fan-in handler

    The parallel node itself doesn't produce LLM output -- it's a
    control-flow node that spawns concurrent execution paths.
    """

    def __init__(
        self,
        handlers: Any = None,  # HandlerRegistry, set during registration
    ) -> None:
        self._handlers = handlers

    def set_handlers(self, handlers: Any) -> None:
        """Set the handler registry (called during registration)."""
        self._handlers = handlers

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        """Execute parallel branches for all outgoing edges."""
        if self._handlers is None:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="ParallelHandler has no handler registry",
            )

        outgoing = graph.outgoing_edges(node.id)
        if not outgoing:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(f"Parallel node '{node.id}' has no outgoing edges"),
            )

        # Determine join policy from node attributes
        join_policy = node.attrs.get("join_policy", JoinPolicy.WAIT_ALL)

        # Get max parallelism (0 = unlimited)
        max_parallel = int(node.attrs.get("max_parallel", "0"))

        # Launch branches
        if max_parallel > 0:
            # Bounded parallelism via semaphore
            sem = asyncio.Semaphore(max_parallel)

            async def bounded_branch(edge_target: str, branch_id: str) -> BranchResult:
                async with sem:
                    return await self._run_branch(
                        edge_target,
                        branch_id,
                        context,
                        graph,
                        logs_root,
                        abort_signal,
                    )

            tasks = [bounded_branch(edge.target, f"branch_{i}") for i, edge in enumerate(outgoing)]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Unbounded: all branches run concurrently
            tasks = [
                self._run_branch(
                    edge.target,
                    f"branch_{i}",
                    context,
                    graph,
                    logs_root,
                    abort_signal,
                )
                for i, edge in enumerate(outgoing)
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any escaped exceptions into BranchResult failures
        branch_results: list[BranchResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, BaseException):
                branch_results.append(
                    BranchResult(
                        branch_id=f"branch_{i}",
                        target_node_id=outgoing[i].target if i < len(outgoing) else "?",
                        result=HandlerResult(
                            status=Outcome.FAIL,
                            failure_reason=f"{type(r).__name__}: {r}",
                        ),
                        error=str(r),
                    )
                )
            else:
                branch_results.append(r)

        # Store branch results in context for fan-in
        context[f"parallel.{node.id}.results"] = [
            {
                "branch_id": br.branch_id,
                "target": br.target_node_id,
                "status": br.result.status.value,
                "output": br.result.output[:1000] if br.result.output else "",
                "notes": br.result.notes,
                "completed_at": br.completed_at,
                "error": br.error,
            }
            for br in branch_results
        ]

        # Also merge branch context updates into parent
        # (later branches override earlier for same keys)
        for br in branch_results:
            for key, value in br.context.items():
                if key.startswith("codergen.") or key.startswith("tool."):
                    context[f"{br.branch_id}.{key}"] = value

        # Determine overall outcome
        successes = [br for br in branch_results if br.result.status == Outcome.SUCCESS]
        failures = [br for br in branch_results if br.result.status == Outcome.FAIL]

        if join_policy == JoinPolicy.FIRST_SUCCESS and successes:
            best = successes[0]
            return HandlerResult(
                status=Outcome.SUCCESS,
                output=best.result.output,
                notes=(
                    f"Parallel: {len(successes)}/{len(branch_results)} "
                    f"branches succeeded (first_success policy)"
                ),
                context_updates=best.context,
            )

        if not failures:
            # All succeeded
            return HandlerResult(
                status=Outcome.SUCCESS,
                notes=(f"Parallel: all {len(branch_results)} branches succeeded"),
            )

        if not successes:
            # All failed
            reasons = [f"{br.branch_id}: {br.result.failure_reason}" for br in failures]
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(
                    f"All {len(branch_results)} branches failed: " + "; ".join(reasons)
                ),
            )

        # Mixed: some succeeded, some failed
        return HandlerResult(
            status=Outcome.PARTIAL_SUCCESS,
            notes=(
                f"Parallel: {len(successes)}/{len(branch_results)} "
                f"branches succeeded, {len(failures)} failed"
            ),
        )

    async def _run_branch(
        self,
        target_node_id: str,
        branch_id: str,
        parent_context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> BranchResult:
        """Run a single parallel branch."""
        target_node = graph.get_node(target_node_id)
        if target_node is None:
            return BranchResult(
                branch_id=branch_id,
                target_node_id=target_node_id,
                result=HandlerResult(
                    status=Outcome.FAIL,
                    failure_reason=f"Branch target not found: {target_node_id}",
                ),
                error=f"Node not found: {target_node_id}",
            )

        # Deep-clone context for isolation -- prevents concurrent
        # branches from sharing mutable nested structures (lists, dicts)
        branch_context = copy.deepcopy(parent_context)
        branch_context["_branch_id"] = branch_id

        # Set up branch-specific logs directory (inside try/except
        # so mkdir failures don't escape to asyncio.gather)
        branch_logs = None
        start_time = time.monotonic()

        try:
            if logs_root:
                branch_logs = logs_root / "branches" / branch_id
                branch_logs.mkdir(parents=True, exist_ok=True)
            result = await execute_subgraph(
                start_node=target_node,
                context=branch_context,
                graph=graph,
                handlers=self._handlers,
                logs_root=branch_logs,
                abort_signal=abort_signal,
            )
        except Exception as exc:  # noqa: BLE001
            result = HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        return BranchResult(
            branch_id=branch_id,
            target_node_id=target_node_id,
            result=result,
            context=branch_context,
            completed_at=time.monotonic() - start_time,
        )


# ------------------------------------------------------------------ #
# FanInHandler
# ------------------------------------------------------------------ #


class FanInHandler:
    """Handler for fan-in/join nodes (shape=tripleoctagon). Spec §4.9.

    Collects results from parallel branches (stored in context by
    ParallelHandler) and selects the best one using our Issue 5 design:
    outcome rank + completion order (no score field).

    If the node has a prompt, it could be used for LLM-based evaluation
    (not yet implemented -- uses heuristic selection for now).
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        """Collect and evaluate parallel branch results."""
        # Find branch results in context (set by ParallelHandler).
        # Look for the MOST RECENT parallel results by iterating in
        # reverse insertion order. This handles sequential parallel
        # sections (fork1->join1->fork2->join2) correctly -- join2
        # picks up fork2's results, not fork1's stale results.
        parallel_results: list[dict[str, Any]] | None = None
        parallel_keys = [k for k in context if k.startswith("parallel.") and k.endswith(".results")]
        if parallel_keys:
            # Last inserted = most recent parallel execution
            parallel_results = context[parallel_keys[-1]]

        if parallel_results is None:
            # No parallel results found -- this fan-in might be reached
            # via a non-parallel path, which is valid
            return HandlerResult(
                status=Outcome.SUCCESS,
                notes=(
                    f"Fan-in node '{node.id}' reached without parallel results (non-parallel path)"
                ),
            )

        if not parallel_results:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Fan-in received empty branch results",
            )

        # Heuristic selection (our Issue 5 design)
        best = heuristic_select(parallel_results)

        return HandlerResult(
            status=Outcome(best["status"]),
            output=best.get("output", ""),
            notes=(
                f"Fan-in selected {best['branch_id']} "
                f"(status: {best['status']}, "
                f"completed in {best.get('completed_at', 0):.1f}s) "
                f"from {len(parallel_results)} branches"
            ),
        )


def heuristic_select(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best branch result using heuristic ranking.

    Our Issue 5 design: outcome rank + completion order + branch ID.
    No score field (was undefined in the spec, we dropped it).

    Priority:
    1. Best outcome status (success > partial > retry > fail)
    2. Among equal outcomes, latest completion wins (freshest result)
    3. Branch ID as deterministic tiebreak
    """
    outcome_rank = {
        "success": 0,
        "partial_success": 1,
        "retry": 2,
        "fail": 3,
    }

    def sort_key(c: dict[str, Any]) -> tuple[int, float, str]:
        rank = outcome_rank.get(c.get("status", "fail"), 3)
        # Negate completed_at so latest (highest) sorts first
        completed = -(c.get("completed_at", 0.0))
        branch_id = c.get("branch_id", "")
        return (rank, completed, branch_id)

    if not candidates:
        raise ValueError("heuristic_select requires at least one candidate")

    return sorted(candidates, key=sort_key)[0]
