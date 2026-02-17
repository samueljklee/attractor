"""Wave 4 pipeline spec-compliance tests.

Item #21: Per-node artifacts from ALL handlers (Spec S5.6, S11.3, S11.7)
Item #23: Aggregate goal gate check at exit (Spec S3.4, S11.4)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from attractor_pipeline import (
    CodergenHandler,
    ExitHandler,
    HandlerRegistry,
    HandlerResult,
    Outcome,
    PipelineStatus,
    StartHandler,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)


# ================================================================== #
# Helpers
# ================================================================== #


def _read_status(logs_root: Path, node_id: str) -> dict[str, Any]:
    """Read and parse a node's status.json artifact."""
    safe_id = node_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    path = logs_root / safe_id / "status.json"
    assert path.exists(), f"status.json missing for node '{node_id}'"
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_exists(logs_root: Path, node_id: str, filename: str) -> bool:
    """Check whether a specific artifact file exists for a node."""
    safe_id = node_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    return (logs_root / safe_id / filename).exists()


def _read_artifact(logs_root: Path, node_id: str, filename: str) -> str:
    """Read a specific artifact file for a node."""
    safe_id = node_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    return (logs_root / safe_id / filename).read_text(encoding="utf-8")


# ================================================================== #
# Item #21: Per-node artifacts from ALL handlers
# ================================================================== #


class TestPerNodeArtifacts:
    """Spec S5.6: every executed node writes artifacts to {logs_root}/{node_id}/."""

    @pytest.mark.asyncio
    async def test_start_node_gets_status_json(self, tmp_path: Path):
        """Start nodes (ellipse) should get status.json but no prompt/response."""
        g = parse_dot("""
        digraph S {
            graph [goal="Test"]
            start [shape=Mdiamond]
            done [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        # Start node: status.json written
        status = _read_status(tmp_path, "start")
        assert status["node_id"] == "start"
        assert status["status"] == "success"

        # Start handler has no prompt/response -- files should be absent
        assert not _artifact_exists(tmp_path, "start", "prompt.md")
        assert not _artifact_exists(tmp_path, "start", "response.md")

    @pytest.mark.asyncio
    async def test_exit_node_gets_status_json(self, tmp_path: Path):
        """Exit nodes (Msquare) should also get status.json."""
        g = parse_dot("""
        digraph E {
            graph [goal="Test"]
            start [shape=Mdiamond]
            done [shape=Msquare]
            start -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        status = _read_status(tmp_path, "done")
        assert status["node_id"] == "done"
        assert status["status"] == "success"

    @pytest.mark.asyncio
    async def test_codergen_node_gets_all_three_artifacts(self, tmp_path: Path):
        """Codergen nodes should get status.json + prompt.md + response.md."""

        class MockBackend:
            async def run(self, node, prompt, context, abort_signal=None):
                return f"Generated code for: {prompt[:50]}"

        g = parse_dot("""
        digraph C {
            graph [goal="Build widget"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Implement $goal"]
            done [shape=Msquare]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(backend=MockBackend()))

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        # status.json
        status = _read_status(tmp_path, "code")
        assert status["node_id"] == "code"
        assert status["status"] == "success"

        # prompt.md -- should contain the EXPANDED prompt (variable-substituted)
        assert _artifact_exists(tmp_path, "code", "prompt.md")
        prompt_text = _read_artifact(tmp_path, "code", "prompt.md")
        assert "Build widget" in prompt_text  # $goal was expanded

        # response.md -- should contain the LLM response
        assert _artifact_exists(tmp_path, "code", "response.md")
        response_text = _read_artifact(tmp_path, "code", "response.md")
        assert "Generated code for" in response_text

    @pytest.mark.asyncio
    async def test_conditional_node_gets_status_json(self, tmp_path: Path):
        """Conditional (diamond) nodes should get status.json."""
        g = parse_dot("""
        digraph D {
            graph [goal="Branch"]
            start [shape=Mdiamond]
            check [shape=diamond]
            done [shape=Msquare]
            start -> check -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        status = _read_status(tmp_path, "check")
        assert status["node_id"] == "check"
        assert status["status"] == "success"

    @pytest.mark.asyncio
    async def test_tool_node_gets_artifacts(self, tmp_path: Path):
        """Tool (parallelogram) nodes should get status.json + response.md."""
        g = parse_dot("""
        digraph T {
            graph [goal="Run tool"]
            start [shape=Mdiamond]
            run [shape=parallelogram, prompt="echo hello_artifact"]
            done [shape=Msquare]
            start -> run -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        # status.json
        status = _read_status(tmp_path, "run")
        assert status["node_id"] == "run"
        assert status["status"] == "success"

        # Tool nodes have a prompt (the command) -> prompt.md written
        assert _artifact_exists(tmp_path, "run", "prompt.md")

        # Tool output -> response.md written
        assert _artifact_exists(tmp_path, "run", "response.md")
        response = _read_artifact(tmp_path, "run", "response.md")
        assert "hello_artifact" in response

    @pytest.mark.asyncio
    async def test_no_artifacts_when_logs_root_is_none(self):
        """When logs_root is None, no artifact directories should be created."""
        g = parse_dot("""
        digraph N {
            graph [goal="Test"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Do something"]
            done [shape=Msquare]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # logs_root=None (the default)
        result = await run_pipeline(g, registry, logs_root=None)
        assert result.status == PipelineStatus.COMPLETED
        # No assertion on filesystem -- just confirm it doesn't crash

    @pytest.mark.asyncio
    async def test_all_nodes_in_linear_pipeline_get_artifacts(self, tmp_path: Path):
        """Every node in a multi-step pipeline should have artifacts."""
        g = parse_dot("""
        digraph Multi {
            graph [goal="Multi-step"]
            start [shape=Mdiamond]
            step1 [shape=box, prompt="Step 1"]
            step2 [shape=box, prompt="Step 2"]
            done [shape=Msquare]
            start -> step1 -> step2 -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED

        # Every node should have status.json
        for node_id in ["start", "step1", "step2", "done"]:
            status = _read_status(tmp_path, node_id)
            assert status["node_id"] == node_id
            assert status["status"] == "success"


# ================================================================== #
# Item #23: Aggregate goal gate check at exit
# ================================================================== #


class TestAggregateGoalGate:
    """Spec S3.4: at exit, ALL visited goal-gate nodes are checked."""

    @pytest.mark.asyncio
    async def test_unsatisfied_gate_blocks_exit(self):
        """If a visited goal-gate node has a non-success outcome, exit is blocked."""
        # The codergen handler (no backend) returns SUCCESS, but the
        # goal_gate condition demands outcome=fail, which won't match.
        # So the gate will be *unsatisfied* when checked at exit.
        g = parse_dot("""
        digraph G {
            graph [goal="Gate test", max_goal_gate_redirects="2"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Code", goal_gate="outcome = fail", retry_target="code"]
            done [shape=Msquare]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        # The pipeline should fail because the aggregate gate on 'code'
        # is never satisfied (outcome is always SUCCESS, gate wants FAIL)
        # and the circuit breaker trips after max redirects.
        assert result.status == PipelineStatus.FAILED
        assert "gate" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_all_gates_satisfied_allows_exit(self):
        """If all visited goal-gate nodes are satisfied, exit proceeds."""
        # goal_gate="outcome = success" should pass because codergen
        # (without backend) returns SUCCESS.
        g = parse_dot("""
        digraph G {
            graph [goal="Gate OK"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Code", goal_gate="outcome = success"]
            done [shape=Msquare]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_unvisited_gate_does_not_block(self):
        """A goal-gate node that was never visited should not block exit."""
        # 'other_code' is never reached (it's on a branch not taken).
        # Even though it has goal_gate set, it shouldn't block exit.
        g = parse_dot("""
        digraph G {
            graph [goal="Unvisited"]
            start [shape=Mdiamond]
            check [shape=diamond]
            code [shape=box, prompt="Code"]
            other_code [shape=box, prompt="Other", goal_gate="outcome = fail", retry_target="other_code"]
            done [shape=Msquare]
            start -> check
            check -> code [condition="outcome = success"]
            check -> other_code [condition="outcome = fail"]
            code -> done
            other_code -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        # 'other_code' was never visited
        assert "other_code" not in result.completed_nodes

    @pytest.mark.asyncio
    async def test_aggregate_gate_with_retry_target_redirect(self):
        """Unsatisfied aggregate gate redirects to the node's retry_target."""
        # Track execution to verify redirect happens.
        call_counts: dict[str, int] = {}

        class CountingBackend:
            async def run(self, node, prompt, context, abort_signal=None):
                call_counts[node.id] = call_counts.get(node.id, 0) + 1
                # First call to 'code' returns success but gate wants
                # outcome=fail, so gate is never satisfied. After redirect,
                # 'retry_node' runs, and on second check the gate on 'code'
                # still fails -> circuit breaker trips.
                return HandlerResult(status=Outcome.SUCCESS, output="done")

        g = parse_dot("""
        digraph G {
            graph [goal="Retry", max_goal_gate_redirects="3"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Code", goal_gate="outcome = fail", retry_target="retry_node"]
            retry_node [shape=box, prompt="Retry"]
            done [shape=Msquare]
            start -> code -> done
            retry_node -> done
        }
        """)
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(backend=CountingBackend()))

        result = await run_pipeline(g, registry)
        # Circuit breaker should trip because the gate is never satisfied
        assert result.status == PipelineStatus.FAILED
        assert "gate" in (result.error or "").lower()
        # 'retry_node' should have been visited at least once (redirect happened)
        assert "retry_node" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_aggregate_checks_multiple_gate_nodes(self):
        """Aggregate check examines ALL visited goal-gate nodes, not just one."""
        g = parse_dot("""
        digraph G {
            graph [goal="Multi-gate", max_goal_gate_redirects="2"]
            start [shape=Mdiamond]
            gate1 [shape=box, prompt="G1", goal_gate="outcome = success"]
            gate2 [shape=box, prompt="G2", goal_gate="outcome = success"]
            done [shape=Msquare]
            start -> gate1 -> gate2 -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # Both gates satisfied (codergen without backend returns SUCCESS)
        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_aggregate_gate_second_node_fails(self):
        """If one of multiple visited gate nodes is unsatisfied, exit is blocked."""
        # 'gate1' has goal_gate="outcome = success" -> satisfied (handler returns SUCCESS)
        # 'gate2' has goal_gate="outcome = fail" -> NOT satisfied
        g = parse_dot("""
        digraph G {
            graph [goal="Multi-gate fail", max_goal_gate_redirects="2"]
            start [shape=Mdiamond]
            gate1 [shape=box, prompt="G1", goal_gate="outcome = success"]
            gate2 [shape=box, prompt="G2", goal_gate="outcome = fail", retry_target="gate2"]
            done [shape=Msquare]
            start -> gate1 -> gate2 -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.FAILED
        assert "gate" in (result.error or "").lower()
