"""Tests for partial-items quick fixes (spec compliance).

Fix #4:  Shell timeout_ms clamping to max_command_timeout_ms (Spec S2.2)
Fix #6:  QueueInterviewer returns SKIPPED on exhaustion (Spec S6.4)
Fix #7:  GraphTransform should not mutate input graph (Spec S9.1)
Fix #8a: Rename wait_agent tool to wait (Spec S7.2)
Fix #8c: Default max_subagent_depth = 1 (Spec S2.2)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ================================================================== #
# Fix #4: Shell timeout_ms clamping to max_command_timeout_ms
# ================================================================== #


class TestShellTimeoutClamping:
    """Spec S2.2: max_command_timeout_ms = 600000 (10 minutes)."""

    def test_session_config_has_max_command_timeout_ms(self) -> None:
        """SessionConfig exposes max_command_timeout_ms with default 600000."""
        from attractor_agent.session import SessionConfig

        config = SessionConfig()
        assert config.max_command_timeout_ms == 600_000

    def test_module_level_default_is_600s(self) -> None:
        """core._max_command_timeout_s defaults to 600 (= 600000ms)."""
        from attractor_agent.tools.core import _max_command_timeout_s

        assert _max_command_timeout_s == 600

    def test_set_max_command_timeout(self) -> None:
        """set_max_command_timeout() converts ms to seconds."""
        from attractor_agent.tools.core import (
            _max_command_timeout_s,
            set_max_command_timeout,
        )

        original = _max_command_timeout_s
        try:
            set_max_command_timeout(120_000)  # 2 minutes
            from attractor_agent.tools import core

            assert core._max_command_timeout_s == 120
        finally:
            set_max_command_timeout(original * 1000)

    @pytest.mark.asyncio
    async def test_timeout_clamped_to_max(self) -> None:
        """When timeout exceeds max_command_timeout, it is clamped."""
        from attractor_agent.tools import core
        from attractor_agent.tools.core import _shell, set_max_command_timeout

        original = core._max_command_timeout_s

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        try:
            set_max_command_timeout(60_000)  # 60 seconds max
            with (
                patch("attractor_agent.tools.core._environment", mock_env),
                patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
            ):
                await _shell("echo hi", timeout=120)  # 120s > 60s max

            call_kwargs = mock_env.exec_shell.call_args
            assert call_kwargs[1]["timeout"] == 60  # clamped to 60
        finally:
            set_max_command_timeout(original * 1000)

    @pytest.mark.asyncio
    async def test_timeout_ms_clamped_to_max(self) -> None:
        """timeout_ms is also clamped after conversion."""
        from attractor_agent.tools import core
        from attractor_agent.tools.core import _shell, set_max_command_timeout

        original = core._max_command_timeout_s

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        try:
            set_max_command_timeout(30_000)  # 30 seconds max
            with (
                patch("attractor_agent.tools.core._environment", mock_env),
                patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
            ):
                await _shell("echo hi", timeout_ms=120_000)  # 120s > 30s max

            call_kwargs = mock_env.exec_shell.call_args
            assert call_kwargs[1]["timeout"] == 30  # clamped to 30
        finally:
            set_max_command_timeout(original * 1000)

    @pytest.mark.asyncio
    async def test_timeout_below_max_unchanged(self) -> None:
        """Timeouts below the max are not affected by clamping."""
        from attractor_agent.tools.core import _shell

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        with (
            patch("attractor_agent.tools.core._environment", mock_env),
            patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
        ):
            await _shell("echo hi", timeout=30)  # well under 600s default

        call_kwargs = mock_env.exec_shell.call_args
        assert call_kwargs[1]["timeout"] == 30


# ================================================================== #
# Fix #7: GraphTransform should not mutate input graph
# ================================================================== #


class TestVariableExpansionNoMutation:
    """Spec S9.1: VariableExpansionTransform should not modify the input graph."""

    def test_input_graph_not_mutated(self) -> None:
        """apply() returns a new graph; the original is untouched."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.transforms import VariableExpansionTransform

        graph = Graph(name="test")
        graph.nodes["start"] = Node(
            id="start", shape="Mdiamond", label="Start", prompt="Goal is $goal"
        )
        graph.nodes["end"] = Node(id="end", shape="Msquare", label="End")
        graph.edges.append(Edge(source="start", target="end"))

        transform = VariableExpansionTransform({"goal": "expanded-value"})
        result = transform.apply(graph)

        # Result has expanded values
        assert result.nodes["start"].prompt == "Goal is expanded-value"

        # Original is NOT mutated
        assert graph.nodes["start"].prompt == "Goal is $goal"

        # Result is a different object
        assert result is not graph

    def test_returned_graph_is_independent_copy(self) -> None:
        """Modifications to the returned graph don't affect the input."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.transforms import VariableExpansionTransform

        graph = Graph(name="test")
        graph.nodes["a"] = Node(id="a", shape="Mdiamond", prompt="$x")
        graph.edges.append(Edge(source="a", target="a"))

        transform = VariableExpansionTransform({"x": "val"})
        result = transform.apply(graph)

        # Mutate the result
        result.nodes["a"].label = "CHANGED"
        result.name = "modified"

        # Original unaffected
        assert graph.nodes["a"].label == ""
        assert graph.name == "test"


# ================================================================== #
# Fix #8c: Default max_subagent_depth = 1
# ================================================================== #


class TestDefaultMaxSubagentDepth:
    """Spec S2.2: max_subagent_depth defaults to 1."""

    def test_spawn_subagent_default_max_depth(self) -> None:
        """spawn_subagent() default max_depth is 1."""
        import inspect

        from attractor_agent.subagent import spawn_subagent

        sig = inspect.signature(spawn_subagent)
        assert sig.parameters["max_depth"].default == 1

    def test_manager_spawn_default_max_depth(self) -> None:
        """SubagentManager.spawn() default max_depth is 1."""
        import inspect

        from attractor_agent.subagent_manager import SubagentManager

        sig = inspect.signature(SubagentManager.spawn)
        assert sig.parameters["max_depth"].default == 1
