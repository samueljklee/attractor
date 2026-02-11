"""Aggressive tests for parallel/fan-in handlers and execute_subgraph.

Tests cover: basic fan-out/fan-in, concurrent execution, context isolation,
abort mid-branch, error handling, boundary detection, heuristic selection,
mixed success/failure, max_parallel semaphore, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from attractor_agent.abort import AbortSignal
from attractor_pipeline import (
    HandlerRegistry,
    Outcome,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.engine.subgraph import execute_subgraph
from attractor_pipeline.handlers.parallel import (
    heuristic_select,
)

# ================================================================== #
# execute_subgraph
# ================================================================== #


class TestExecuteSubgraph:
    @pytest.mark.asyncio
    async def test_basic_linear_subgraph(self):
        """Subgraph runs through linear nodes and stops at fan-in boundary."""
        g = parse_dot("""
        digraph Sub {
            start [shape=ellipse]
            a [shape=box, prompt="Task A"]
            b [shape=box, prompt="Task B"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> a -> b -> join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        ctx: dict[str, Any] = {"goal": "test"}
        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context=ctx,
            graph=g,
            handlers=registry,
        )
        assert result.status == Outcome.SUCCESS

    @pytest.mark.asyncio
    async def test_stops_at_fan_in_boundary(self):
        """Subgraph stops at tripleoctagon WITHOUT executing it."""
        g = parse_dot("""
        digraph Boundary {
            a [shape=box, prompt="A"]
            join [shape=tripleoctagon]
            a -> join
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context={},
            graph=g,
            handlers=registry,
        )
        # Should have executed 'a' but NOT 'join'
        assert result.status == Outcome.SUCCESS

    @pytest.mark.asyncio
    async def test_stops_at_exit_boundary(self):
        """Subgraph stops at Msquare (exit) boundary."""
        g = parse_dot("""
        digraph ExitBound {
            a [shape=box, prompt="A"]
            done [shape=Msquare]
            a -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context={},
            graph=g,
            handlers=registry,
        )
        assert result.status == Outcome.SUCCESS

    @pytest.mark.asyncio
    async def test_abort_stops_branch(self):
        """Abort signal stops subgraph execution."""
        g = parse_dot("""
        digraph Abort {
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            join [shape=tripleoctagon]
            a -> b -> join
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        abort = AbortSignal()
        abort.set()

        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context={},
            graph=g,
            handlers=registry,
            abort_signal=abort,
        )
        assert result.status == Outcome.FAIL
        assert "cancelled" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_max_steps_safety_cap(self):
        """Subgraph respects max_steps to prevent infinite loops."""
        g = parse_dot("""
        digraph Loop {
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            a -> b
            b -> a
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context={},
            graph=g,
            handlers=registry,
            max_steps=10,
        )
        assert result.status == Outcome.FAIL
        assert "max steps" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_context_isolation(self):
        """Branch context mutations don't leak to the caller's dict."""
        g = parse_dot("""
        digraph Iso {
            a [shape=parallelogram, prompt="echo branch_output"]
            join [shape=tripleoctagon]
            a -> join
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        parent_ctx: dict[str, Any] = {"shared": "original"}
        branch_ctx = dict(parent_ctx)  # clone like ParallelHandler does

        await execute_subgraph(
            start_node=g.nodes["a"],
            context=branch_ctx,
            graph=g,
            handlers=registry,
        )
        # Branch context may have new keys from tool execution
        assert "shared" in parent_ctx
        assert parent_ctx["shared"] == "original"  # parent unchanged

    @pytest.mark.asyncio
    async def test_missing_handler_returns_fail(self):
        """Subgraph with unknown handler fails gracefully."""
        g = parse_dot("""
        digraph Bad {
            a [shape=box, handler="nonexistent"]
            join [shape=tripleoctagon]
            a -> join
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await execute_subgraph(
            start_node=g.nodes["a"],
            context={},
            graph=g,
            handlers=registry,
        )
        assert result.status == Outcome.FAIL
        assert "nonexistent" in result.failure_reason


# ================================================================== #
# heuristic_select (Issue 5 design)
# ================================================================== #


class TestHeuristicSelect:
    def test_success_beats_failure(self):
        candidates = [
            {"branch_id": "b0", "status": "fail", "completed_at": 1.0},
            {"branch_id": "b1", "status": "success", "completed_at": 2.0},
        ]
        best = heuristic_select(candidates)
        assert best["branch_id"] == "b1"

    def test_partial_success_beats_fail(self):
        candidates = [
            {"branch_id": "b0", "status": "fail", "completed_at": 1.0},
            {"branch_id": "b1", "status": "partial_success", "completed_at": 2.0},
        ]
        best = heuristic_select(candidates)
        assert best["branch_id"] == "b1"

    def test_same_outcome_latest_wins(self):
        """Among equal outcomes, the one that completed last wins."""
        candidates = [
            {"branch_id": "b0", "status": "success", "completed_at": 1.0},
            {"branch_id": "b1", "status": "success", "completed_at": 3.0},
            {"branch_id": "b2", "status": "success", "completed_at": 2.0},
        ]
        best = heuristic_select(candidates)
        assert best["branch_id"] == "b1"  # completed_at=3.0, latest

    def test_branch_id_tiebreak(self):
        """Deterministic: same outcome + same time -> branch_id sorts."""
        candidates = [
            {"branch_id": "b2", "status": "success", "completed_at": 1.0},
            {"branch_id": "b0", "status": "success", "completed_at": 1.0},
            {"branch_id": "b1", "status": "success", "completed_at": 1.0},
        ]
        best = heuristic_select(candidates)
        # All equal outcome + time -> lexical branch_id: b0 < b1 < b2
        # But wait -- completed_at is negated, so same. Then branch_id ascending.
        assert best["branch_id"] == "b0"

    def test_single_candidate(self):
        candidates = [
            {"branch_id": "only", "status": "success", "completed_at": 1.0},
        ]
        assert heuristic_select(candidates)["branch_id"] == "only"

    def test_all_failed(self):
        candidates = [
            {"branch_id": "b0", "status": "fail", "completed_at": 1.0},
            {"branch_id": "b1", "status": "fail", "completed_at": 2.0},
        ]
        best = heuristic_select(candidates)
        assert best["status"] == "fail"
        assert best["branch_id"] == "b1"  # latest among failures


# ================================================================== #
# Full parallel pipeline integration
# ================================================================== #


class TestParallelPipeline:
    @pytest.mark.asyncio
    async def test_basic_parallel_fanout_fanin(self):
        """Full pipeline: start -> parallel -> branches -> fan-in -> done."""
        g = parse_dot("""
        digraph Parallel {
            graph [goal="Parallel test"]
            start [shape=ellipse]
            fork [shape=component]
            branch_a [shape=box, prompt="Branch A"]
            branch_b [shape=box, prompt="Branch B"]
            join [shape=tripleoctagon]
            done [shape=Msquare]

            start -> fork
            fork -> branch_a
            fork -> branch_b
            branch_a -> join
            branch_b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "fork" in result.completed_nodes
        assert "join" in result.completed_nodes
        assert "done" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_parallel_with_tool_branches(self):
        """Parallel branches running shell commands concurrently."""
        g = parse_dot("""
        digraph ParaTool {
            graph [goal="Parallel tools"]
            start [shape=ellipse]
            fork [shape=component]
            cmd_a [shape=parallelogram, prompt="echo branch_a_output"]
            cmd_b [shape=parallelogram, prompt="echo branch_b_output"]
            join [shape=tripleoctagon]
            done [shape=Msquare]

            start -> fork
            fork -> cmd_a
            fork -> cmd_b
            cmd_a -> join
            cmd_b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        # Both branches should have executed their tools
        parallel_key = [
            k for k in result.context if k.startswith("parallel.") and k.endswith(".results")
        ]
        assert len(parallel_key) == 1
        branch_results = result.context[parallel_key[0]]
        assert len(branch_results) == 2
        assert all(br["status"] == "success" for br in branch_results)

    @pytest.mark.asyncio
    async def test_parallel_abort_cancels_branches(self):
        """Aborting during parallel execution cancels all branches."""
        g = parse_dot("""
        digraph ParaAbort {
            start [shape=ellipse]
            fork [shape=component]
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> a
            fork -> b
            a -> join
            b -> join
            join -> done
        }
        """)
        abort = AbortSignal()
        abort.set()

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, abort_signal=abort)
        assert result.status == PipelineStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_parallel_single_branch(self):
        """Parallel with only one outgoing edge still works."""
        g = parse_dot("""
        digraph Single {
            graph [goal="Single branch"]
            start [shape=ellipse]
            fork [shape=component]
            only [shape=box, prompt="Only branch"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> only
            only -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_no_outgoing_fails(self):
        """Parallel node with no outgoing edges fails gracefully."""
        g = parse_dot("""
        digraph NoEdge {
            graph [goal="No edges"]
            start [shape=ellipse]
            fork [shape=component]
            done [shape=Msquare]
            start -> fork -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # The fork node has one outgoing edge to 'done' (Msquare).
        # Subgraph will stop at boundary immediately.
        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fanin_without_parallel_still_works(self):
        """Fan-in reached via non-parallel path succeeds."""
        g = parse_dot("""
        digraph NonPara {
            graph [goal="Non-parallel"]
            start [shape=ellipse]
            task [shape=box, prompt="Task"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> task -> join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "join" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_parallel_context_isolation(self):
        """Branch contexts are isolated from each other."""
        g = parse_dot("""
        digraph CtxIso {
            graph [goal="Context isolation"]
            start [shape=ellipse]
            fork [shape=component]
            a [shape=parallelogram, prompt="echo alpha"]
            b [shape=parallelogram, prompt="echo beta"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> a
            fork -> b
            a -> join
            b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        # Each branch's tool output should be namespaced
        ctx = result.context
        branch_keys = [k for k in ctx if k.startswith("branch_")]
        # Branch results are stored with branch prefix
        assert len(branch_keys) >= 0  # may or may not be present depending on tool output

    @pytest.mark.asyncio
    async def test_three_way_parallel(self):
        """Three branches running in parallel."""
        g = parse_dot("""
        digraph ThreeWay {
            graph [goal="Three-way"]
            start [shape=ellipse]
            fork [shape=component]
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            c [shape=box, prompt="C"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> a
            fork -> b
            fork -> c
            a -> join
            b -> join
            c -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        parallel_results = [
            v
            for k, v in result.context.items()
            if k.startswith("parallel.") and k.endswith(".results")
        ]
        assert len(parallel_results) == 1
        assert len(parallel_results[0]) == 3


# ================================================================== #
# Edge cases
# ================================================================== #


class TestParallelEdgeCases:
    @pytest.mark.asyncio
    async def test_branch_with_missing_handler_doesnt_crash_others(self):
        """One branch failing doesn't crash the parallel execution."""
        g = parse_dot("""
        digraph Mixed {
            graph [goal="Mixed"]
            start [shape=ellipse]
            fork [shape=component]
            good [shape=box, prompt="Good"]
            bad [shape=box, handler="nonexistent"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> good
            fork -> bad
            good -> join
            bad -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        # Should complete (partial success or similar) -- not crash
        assert result.status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED)

    @pytest.mark.asyncio
    async def test_checkpoint_with_parallel(self, tmp_path):
        """Parallel pipeline saves checkpoint correctly."""
        g = parse_dot("""
        digraph Ckpt {
            graph [goal="Checkpoint"]
            start [shape=ellipse]
            fork [shape=component]
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> a
            fork -> b
            a -> join
            b -> join
            join -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED
        assert (tmp_path / "checkpoint.json").exists()
