"""Tests for Wave 14: Pipeline Graph & Validation (P24–P30).

Covers:
- P24: Graph.label property (§11.1)
- P25: Multi-class stylesheet matching with comma-separated classes (§2.12)
- P28: R05 reachability uses BFS from start, severity is ERROR (§7.2)
- P29: Handler-shape mapping correct: hexagon→manager, house→wait.human (§2.8)
- P30: Handler.execute() abort_signal is optional 5th param (§3.3 extension)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from attractor_pipeline.engine.runner import Handler, HandlerResult, Outcome
from attractor_pipeline.graph import Edge, Graph, Node, NodeShape
from attractor_pipeline.stylesheet import Selector, _selector_matches
from attractor_pipeline.validation import Severity, _rule_no_orphan_nodes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_graph(**kwargs: Any) -> Graph:
    """Return a minimal valid graph: start → exit, both nodes present."""
    start = Node(id="start", shape="Mdiamond")
    exit_ = Node(id="exit", shape="Msquare")
    edge = Edge(source="start", target="exit")
    return Graph(
        nodes={"start": start, "exit": exit_},
        edges=[edge],
        **kwargs,
    )


def _class_selector(class_name: str) -> Selector:
    """Convenience: build a class-kind Selector."""
    return Selector(kind="class", value=class_name, specificity=2)


# ---------------------------------------------------------------------------
# P24: Graph.label property (§11.1)
# ---------------------------------------------------------------------------


class TestGraphLabelProperty:
    """P24: Graph must expose a typed `label` property backed by attrs."""

    def test_graph_label_property(self) -> None:
        """label property returns the value stored in attrs['label']."""
        graph = _minimal_graph()
        graph.attrs["label"] = "My Pipeline"
        assert graph.label == "My Pipeline"

    def test_graph_label_empty_default(self) -> None:
        """label returns '' when attrs does not contain 'label'."""
        graph = _minimal_graph()
        assert graph.label == ""

    def test_graph_label_updates_with_attrs(self) -> None:
        """Mutating attrs['label'] is immediately visible via the property."""
        graph = _minimal_graph()
        graph.attrs["label"] = "First"
        assert graph.label == "First"
        graph.attrs["label"] = "Second"
        assert graph.label == "Second"

    def test_graph_label_independent_of_name(self) -> None:
        """label and name are independent: setting one does not affect the other."""
        graph = Graph(name="pipeline", attrs={"label": "Friendly Name"})
        assert graph.name == "pipeline"
        assert graph.label == "Friendly Name"


# ---------------------------------------------------------------------------
# P25: Multi-class stylesheet matching (§2.12)
# ---------------------------------------------------------------------------


class TestMultiClassStylesheetMatching:
    """P25: _selector_matches must handle comma-separated node_class values."""

    def test_exact_class_still_works(self) -> None:
        """Single-class node still matches its selector exactly."""
        node = Node(id="n1", node_class="code")
        assert _selector_matches(_class_selector("code"), node) is True

    def test_multi_class_node_matches_single_selector(self) -> None:
        """node_class='code,critical' matches selector '.code'."""
        node = Node(id="n1", node_class="code,critical")
        assert _selector_matches(_class_selector("code"), node) is True

    def test_comma_separated_classes_both_matchable(self) -> None:
        """Each class in a comma-separated list can be independently matched."""
        node = Node(id="n1", node_class="code,critical")
        assert _selector_matches(_class_selector("code"), node) is True
        assert _selector_matches(_class_selector("critical"), node) is True

    def test_non_member_class_does_not_match(self) -> None:
        """A selector for a class absent from the list returns False."""
        node = Node(id="n1", node_class="code,critical")
        assert _selector_matches(_class_selector("fast"), node) is False

    def test_classes_with_whitespace_are_trimmed(self) -> None:
        """Spaces around class names in the list are stripped before comparison."""
        node = Node(id="n1", node_class="code , critical")
        assert _selector_matches(_class_selector("code"), node) is True
        assert _selector_matches(_class_selector("critical"), node) is True

    def test_empty_node_class_does_not_match(self) -> None:
        """An empty node_class never matches any class selector."""
        node = Node(id="n1", node_class="")
        assert _selector_matches(_class_selector("code"), node) is False

    def test_partial_class_name_does_not_match(self) -> None:
        """Selector '.cod' does not match node_class='code'."""
        node = Node(id="n1", node_class="code")
        assert _selector_matches(_class_selector("cod"), node) is False

    def test_three_classes_all_matchable(self) -> None:
        """Three comma-separated classes are all independently matchable."""
        node = Node(id="n1", node_class="a,b,c")
        for cls in ("a", "b", "c"):
            assert _selector_matches(_class_selector(cls), node) is True


# ---------------------------------------------------------------------------
# P28: Reachability rule uses BFS, severity ERROR (§7.2)
# ---------------------------------------------------------------------------


class TestReachabilityRule:
    """P28: R05 uses BFS from start; unreachable nodes → ERROR."""

    def _graph_with_island(self) -> Graph:
        """Build: start → a → exit, plus 'island' with no edges."""
        start = Node(id="start", shape="Mdiamond")
        a = Node(id="a", shape="box")
        exit_ = Node(id="exit", shape="Msquare")
        island = Node(id="island", shape="box")
        return Graph(
            nodes={"start": start, "a": a, "exit": exit_, "island": island},
            edges=[
                Edge(source="start", target="a"),
                Edge(source="a", target="exit"),
                # island is disconnected
            ],
        )

    def test_reachability_bfs_finds_unreachable_nodes(self) -> None:
        """BFS detects nodes not reachable from start."""
        graph = self._graph_with_island()
        diagnostics = _rule_no_orphan_nodes(graph)
        unreachable_ids = {d.node_id for d in diagnostics}
        assert "island" in unreachable_ids

    def test_reachability_all_connected_passes(self) -> None:
        """Fully connected graph produces zero R05 diagnostics."""
        graph = _minimal_graph()
        diagnostics = _rule_no_orphan_nodes(graph)
        assert diagnostics == []

    def test_reachability_severity_is_error(self) -> None:
        """Every R05 diagnostic from an unreachable node has ERROR severity."""
        graph = self._graph_with_island()
        diagnostics = _rule_no_orphan_nodes(graph)
        assert diagnostics, "Expected at least one diagnostic"
        for d in diagnostics:
            assert d.severity == Severity.ERROR, (
                f"Got {d.severity!r} for node {d.node_id!r}; expected ERROR"
            )

    def test_reachability_rule_code_is_r05(self) -> None:
        """Diagnostics carry rule code 'R05'."""
        graph = self._graph_with_island()
        diagnostics = _rule_no_orphan_nodes(graph)
        for d in diagnostics:
            assert d.rule == "R05"

    def test_reachable_via_chain_not_flagged(self) -> None:
        """Nodes reachable through a multi-hop chain are not reported."""
        start = Node(id="start", shape="Mdiamond")
        mid1 = Node(id="mid1", shape="box")
        mid2 = Node(id="mid2", shape="box")
        exit_ = Node(id="exit", shape="Msquare")
        graph = Graph(
            nodes={"start": start, "mid1": mid1, "mid2": mid2, "exit": exit_},
            edges=[
                Edge(source="start", target="mid1"),
                Edge(source="mid1", target="mid2"),
                Edge(source="mid2", target="exit"),
            ],
        )
        diagnostics = _rule_no_orphan_nodes(graph)
        assert diagnostics == []

    def test_no_start_node_returns_empty(self) -> None:
        """When there is no start node, R05 returns [] (R01 handles that case)."""
        graph = Graph(
            nodes={"a": Node(id="a", shape="box")},
            edges=[],
        )
        diagnostics = _rule_no_orphan_nodes(graph)
        assert diagnostics == []

    def test_multiple_unreachable_nodes_all_reported(self) -> None:
        """Every disconnected node is reported, not just the first."""
        start = Node(id="start", shape="Mdiamond")
        exit_ = Node(id="exit", shape="Msquare")
        orphan1 = Node(id="orphan1", shape="box")
        orphan2 = Node(id="orphan2", shape="box")
        graph = Graph(
            nodes={
                "start": start,
                "exit": exit_,
                "orphan1": orphan1,
                "orphan2": orphan2,
            },
            edges=[Edge(source="start", target="exit")],
        )
        diagnostics = _rule_no_orphan_nodes(graph)
        reported_ids = {d.node_id for d in diagnostics}
        assert "orphan1" in reported_ids
        assert "orphan2" in reported_ids

    def test_bfs_terminates_on_cyclic_graph(self) -> None:
        """BFS must not loop infinitely when the graph contains a back-edge."""
        start = Node(id="start", shape="Mdiamond")
        a = Node(id="a", shape="box")
        b = Node(id="b", shape="box")
        exit_ = Node(id="exit", shape="Msquare")
        graph = Graph(
            nodes={"start": start, "a": a, "b": b, "exit": exit_},
            edges=[
                Edge(source="start", target="a"),
                Edge(source="a", target="b"),
                Edge(source="b", target="a"),  # back-edge / cycle
                Edge(source="a", target="exit"),
            ],
        )
        diagnostics = _rule_no_orphan_nodes(graph)
        assert diagnostics == []  # all nodes are reachable despite cycle

    def test_unreachable_cluster_with_internal_edges(self) -> None:
        """Nodes with incoming edges from other unreachable nodes are still flagged.

        This is the key case BFS catches that the old incoming-edge heuristic
        missed: 'c' has an incoming edge (from 'b'), but neither 'b' nor 'c'
        is reachable from start.
        """
        start = Node(id="start", shape="Mdiamond")
        exit_ = Node(id="exit", shape="Msquare")
        b = Node(id="b", shape="box")
        c = Node(id="c", shape="box")
        graph = Graph(
            nodes={"start": start, "exit": exit_, "b": b, "c": c},
            edges=[
                Edge(source="start", target="exit"),
                Edge(source="b", target="c"),  # internal edge in disconnected cluster
            ],
        )
        diagnostics = _rule_no_orphan_nodes(graph)
        unreachable = {d.node_id for d in diagnostics}
        assert "b" in unreachable
        assert "c" in unreachable  # old heuristic would miss this (c has incoming edge)


# ---------------------------------------------------------------------------
# P29: Handler-shape mapping correct (§2.8)
# ---------------------------------------------------------------------------


class TestHandlerShapeMapping:
    """P29: Verify shape→handler mapping matches spec §2.8 exactly."""

    def test_handler_shape_mapping_correct(self) -> None:
        """Critical spec §2.8 mappings: hexagon→manager, house→wait.human, Mdiamond→start."""
        assert NodeShape.handler_for_shape("hexagon") == "manager"
        assert NodeShape.handler_for_shape("house") == "wait.human"
        assert NodeShape.handler_for_shape("Mdiamond") == "start"

    def test_full_mapping_table(self) -> None:
        """All known shapes have the correct handler per spec §2.8."""
        expected = {
            "box": "codergen",
            "hexagon": "manager",
            "diamond": "conditional",
            "component": "parallel",
            "tripleoctagon": "fan_in",
            "parallelogram": "tool",
            "house": "wait.human",
            "Msquare": "exit",
            "Mdiamond": "start",
        }
        for shape, handler in expected.items():
            got = NodeShape.handler_for_shape(shape)
            assert got == handler, f"Shape {shape!r}: expected {handler!r}, got {got!r}"

    def test_unknown_shape_defaults_to_codergen(self) -> None:
        """An unrecognised shape safely falls back to 'codergen'."""
        assert NodeShape.handler_for_shape("totally_unknown") == "codergen"

    def test_node_effective_handler_uses_shape_mapping(self) -> None:
        """Node.effective_handler delegates to handler_for_shape when no override."""
        node = Node(id="n", shape="hexagon")
        assert node.effective_handler == "manager"

        node2 = Node(id="n2", shape="house")
        assert node2.effective_handler == "wait.human"

    def test_node_handler_override_takes_precedence(self) -> None:
        """An explicit handler= attribute on a node overrides shape-derived mapping."""
        node = Node(id="n", shape="hexagon", handler="custom_handler")
        assert node.effective_handler == "custom_handler"


# ---------------------------------------------------------------------------
# P30: Handler.execute() abort_signal is optional (§3.3 extension)
# ---------------------------------------------------------------------------


class TestHandlerSignature:
    """P30: abort_signal defaults to None so the 4-param spec signature still works."""

    def test_abort_signal_protocol_has_none_default(self) -> None:
        """Handler Protocol's execute() must have abort_signal defaulting to None."""
        sig = inspect.signature(Handler.execute)
        param = sig.parameters.get("abort_signal")
        assert param is not None, "abort_signal parameter missing from Handler.execute"
        assert param.default is None, f"Expected default=None, got {param.default!r}"

    async def test_handler_execute_works_with_4_params(self) -> None:
        """A handler implementation can be called with only 4 positional args."""

        class MinimalHandler:
            async def execute(
                self,
                node: Node,
                context: dict[str, Any],
                graph: Graph,
                logs_root: Path | None,
                abort_signal: Any = None,
            ) -> HandlerResult:
                return HandlerResult(status=Outcome.SUCCESS, notes="4-param-ok")

        handler = MinimalHandler()
        node = Node(id="n", shape="box")
        graph = _minimal_graph()

        # Call WITHOUT abort_signal -- must not raise
        result = await handler.execute(node, {}, graph, None)
        assert result.status == Outcome.SUCCESS
        assert result.notes == "4-param-ok"

    async def test_handler_execute_works_with_5_params(self) -> None:
        """A handler implementation can be called with the full 5-param signature."""
        from attractor_agent.abort import AbortSignal

        class FullHandler:
            def __init__(self) -> None:
                self.received_signal: Any = "NOT_SET"

            async def execute(
                self,
                node: Node,
                context: dict[str, Any],
                graph: Graph,
                logs_root: Path | None,
                abort_signal: Any = None,
            ) -> HandlerResult:
                self.received_signal = abort_signal
                return HandlerResult(status=Outcome.SUCCESS, notes="5-param-ok")

        handler = FullHandler()
        node = Node(id="n", shape="box")
        graph = _minimal_graph()
        signal = AbortSignal()

        result = await handler.execute(node, {}, graph, None, signal)
        assert result.status == Outcome.SUCCESS
        assert result.notes == "5-param-ok"
        assert handler.received_signal is signal

    async def test_handler_abort_signal_default_is_none_at_runtime(self) -> None:
        """When called without abort_signal the param is None inside the handler."""

        class SignalCapture:
            received: Any = "SENTINEL"

            async def execute(
                self,
                node: Node,
                context: dict[str, Any],
                graph: Graph,
                logs_root: Path | None,
                abort_signal: Any = None,
            ) -> HandlerResult:
                SignalCapture.received = abort_signal
                return HandlerResult(status=Outcome.SUCCESS)

        handler = SignalCapture()
        await handler.execute(Node(id="x", shape="box"), {}, _minimal_graph(), None)
        assert SignalCapture.received is None
