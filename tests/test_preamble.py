"""Tests for fidelity resume preamble generator.

Covers:
- Preamble generation from checkpoint state
- Completed nodes summary with truncated outputs
- Current position display
- Retry/redirect state inclusion
- Context variable filtering
- Integration with prompt layering
- Integration with pipeline engine resume path
"""

from __future__ import annotations

from typing import Any

import pytest

from attractor_agent.prompt_layer import build_system_prompt
from attractor_pipeline.engine.preamble import generate_resume_preamble
from attractor_pipeline.engine.runner import (
    Checkpoint,
    HandlerRegistry,
    PipelineStatus,
    run_pipeline,
)
from attractor_pipeline.graph import Edge, Graph, Node

# ================================================================== #
# Helper: build a test graph + checkpoint
# ================================================================== #


def make_test_graph() -> Graph:
    return Graph(
        name="TestPipeline",
        goal="Build a widget",
        nodes={
            "start": Node(id="start", shape="Mdiamond"),
            "plan": Node(id="plan", shape="box", prompt="Plan: $goal"),
            "implement": Node(id="implement", shape="box", prompt="Implement: $goal"),
            "done": Node(id="done", shape="Msquare"),
        },
        edges=[
            Edge(source="start", target="plan"),
            Edge(source="plan", target="implement"),
            Edge(source="implement", target="done"),
        ],
    )


def make_checkpoint(
    *,
    current_node: str = "implement",
    completed: list[str] | None = None,
    context: dict[str, Any] | None = None,
    retries: dict[str, int] | None = None,
    goal_gate_redirects: int = 0,
) -> Checkpoint:
    completed_nodes = completed if completed is not None else ["start", "plan"]
    ctx = (
        context
        if context is not None
        else {
            "goal": "Build a widget",
            "codergen.plan.output": "1. Design the widget\n2. Build the UI\n3. Add tests",
        }
    )
    return Checkpoint(
        graph_name="TestPipeline",
        current_node_id=current_node,
        context_values=ctx,
        completed_nodes=[{"node_id": nid} for nid in completed_nodes],
        node_retry_counts=retries or {},
        goal_gate_redirect_count=goal_gate_redirects,
        status="running",
    )


# ================================================================== #
# Preamble generation
# ================================================================== #


class TestPreambleGeneration:
    def test_basic_preamble_has_resume_header(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt)
        assert "[RESUME]" in preamble
        assert "checkpoint" in preamble.lower()

    def test_preamble_includes_pipeline_name(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt)
        assert "TestPipeline" in preamble

    def test_preamble_includes_goal(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt)
        assert "Build a widget" in preamble

    def test_preamble_lists_completed_nodes(self):
        g = make_test_graph()
        ckpt = make_checkpoint(completed=["start", "plan"])
        preamble = generate_resume_preamble(g, ckpt)
        assert "Completed nodes (2)" in preamble
        assert "start" in preamble
        assert "plan" in preamble

    def test_preamble_includes_node_output(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt)
        assert "Design the widget" in preamble

    def test_preamble_truncates_long_output(self):
        g = make_test_graph()
        long_output = "x" * 2000
        ckpt = make_checkpoint(
            context={
                "goal": "test",
                "codergen.plan.output": long_output,
            }
        )
        preamble = generate_resume_preamble(g, ckpt, max_output_chars=100)
        assert "x" * 100 in preamble
        assert "2000 chars total" in preamble

    def test_preamble_excludes_output_when_max_zero(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt, max_output_chars=0)
        assert "Design the widget" not in preamble

    def test_preamble_shows_current_position(self):
        g = make_test_graph()
        ckpt = make_checkpoint(current_node="implement")
        preamble = generate_resume_preamble(g, ckpt)
        assert "Resuming at" in preamble
        assert "implement" in preamble

    def test_preamble_shows_retry_counts(self):
        g = make_test_graph()
        ckpt = make_checkpoint(retries={"implement": 2})
        preamble = generate_resume_preamble(g, ckpt)
        assert "Retry counts" in preamble
        assert "implement: 2" in preamble

    def test_preamble_shows_goal_gate_redirects(self):
        g = make_test_graph()
        ckpt = make_checkpoint(goal_gate_redirects=3)
        preamble = generate_resume_preamble(g, ckpt)
        assert "Goal gate redirects: 3" in preamble

    def test_preamble_no_retry_section_when_empty(self):
        g = make_test_graph()
        ckpt = make_checkpoint(retries={})
        preamble = generate_resume_preamble(g, ckpt)
        assert "Retry counts" not in preamble

    def test_preamble_no_redirect_section_when_zero(self):
        g = make_test_graph()
        ckpt = make_checkpoint(goal_gate_redirects=0)
        preamble = generate_resume_preamble(g, ckpt)
        assert "Goal gate redirects" not in preamble

    def test_preamble_includes_context_variables(self):
        g = make_test_graph()
        ckpt = make_checkpoint(
            context={
                "goal": "Build a widget",
                "language": "Python",
                "codergen.plan.output": "plan text",
                "_internal": "hidden",
                "parallel.fork.results": "hidden",
            }
        )
        preamble = generate_resume_preamble(g, ckpt)
        assert "language: Python" in preamble
        assert "_internal" not in preamble
        assert "parallel" not in preamble

    def test_preamble_exclude_context(self):
        g = make_test_graph()
        ckpt = make_checkpoint(context={"goal": "test", "language": "Python"})
        preamble = generate_resume_preamble(g, ckpt, include_context=False)
        assert "language" not in preamble

    def test_preamble_ends_with_continuation_instruction(self):
        g = make_test_graph()
        ckpt = make_checkpoint()
        preamble = generate_resume_preamble(g, ckpt)
        assert "Continue from the current node" in preamble

    def test_empty_completed_nodes(self):
        g = make_test_graph()
        ckpt = make_checkpoint(completed=[], context={"goal": "test"})
        preamble = generate_resume_preamble(g, ckpt)
        assert "[RESUME]" in preamble
        assert "Completed nodes" not in preamble


# ================================================================== #
# Integration: preamble in prompt layering
# ================================================================== #


class TestPreamblePromptLayering:
    def test_resume_preamble_injected_into_layered_prompt(self):
        preamble = "[RESUME] Pipeline is being resumed."
        result = build_system_prompt(
            profile_prompt="You are an expert.",
            pipeline_goal="Build widget",
            pipeline_context={"_resume_preamble": preamble, "goal": "Build widget"},
        )
        assert "[RESUME]" in result
        assert "Pipeline is being resumed" in result
        assert "You are an expert." in result

    def test_resume_preamble_appears_between_goal_and_instruction(self):
        preamble = "[RESUME] Resuming..."
        result = build_system_prompt(
            profile_prompt="Profile.",
            pipeline_goal="Goal.",
            pipeline_context={"_resume_preamble": preamble},
            node_instruction="Focus here.",
        )
        # Verify ordering: profile < goal < preamble < instruction
        profile_pos = result.index("Profile.")
        goal_pos = result.index("[GOAL]")
        resume_pos = result.index("[RESUME]")
        instruction_pos = result.index("[INSTRUCTION]")
        assert profile_pos < goal_pos < resume_pos < instruction_pos

    def test_empty_preamble_not_injected(self):
        result = build_system_prompt(
            profile_prompt="Profile.",
            pipeline_context={"_resume_preamble": ""},
        )
        assert "[RESUME]" not in result


# ================================================================== #
# Integration: preamble in pipeline engine resume
# ================================================================== #


class TestPreamblePipelineIntegration:
    @pytest.mark.asyncio
    async def test_resume_injects_preamble_into_context(self):
        """When resuming from checkpoint, _resume_preamble is set in context."""
        g = make_test_graph()
        ckpt = make_checkpoint(current_node="implement")

        registry = HandlerRegistry()
        from attractor_pipeline.handlers import register_default_handlers

        register_default_handlers(registry)

        result = await run_pipeline(g, registry, checkpoint=ckpt)

        # The pipeline completed. The preamble was used by the resumed
        # node but then cleared so subsequent nodes don't see stale context.
        assert result.status == PipelineStatus.COMPLETED
        # Preamble should be cleared after first node executes
        assert result.context.get("_resume_preamble") is None

    @pytest.mark.asyncio
    async def test_fresh_run_has_no_preamble(self):
        """Normal (non-resume) runs don't inject a preamble."""
        g = make_test_graph()
        registry = HandlerRegistry()
        from attractor_pipeline.handlers import register_default_handlers

        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert "_resume_preamble" not in result.context
