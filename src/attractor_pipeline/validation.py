"""Graph validation and lint rules for Attractor pipelines.

Validates DOT pipeline graphs before execution, catching structural
errors early with clear diagnostic messages. Implements the 14 built-in
rules from attractor-spec §7.2.

Each rule produces Diagnostic objects with severity (ERROR, WARNING, INFO)
and a human-readable message with the offending node/edge ID.
Graphs with ERROR-level diagnostics are rejected before execution.

Usage::

    diagnostics = validate(graph)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        for e in errors:
            print(f"{e.severity}: {e.message}")
        raise ValidationError("Graph has errors")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from attractor_pipeline.conditions import evaluate_condition
from attractor_pipeline.graph import Graph


class Severity(StrEnum):
    """Diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Diagnostic:
    """A single validation finding."""

    rule: str
    severity: Severity
    message: str
    node_id: str = ""
    edge_index: int = -1


def validate(graph: Graph) -> list[Diagnostic]:
    """Run all validation rules against a graph.

    Returns a list of diagnostics. Empty list = graph is valid.
    """
    diagnostics: list[Diagnostic] = []
    for rule in ALL_RULES:
        diagnostics.extend(rule(graph))
    return diagnostics


def validate_or_raise(graph: Graph) -> None:
    """Validate a graph and raise if any ERROR-level diagnostics found.

    Raises:
        ValueError: With all error messages concatenated.
    """
    diagnostics = validate(graph)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    if errors:
        messages = [f"  [{d.rule}] {d.message}" for d in errors]
        raise ValueError(
            f"Graph validation failed with {len(errors)} error(s):\n" + "\n".join(messages)
        )


# ------------------------------------------------------------------ #
# Individual lint rules
# ------------------------------------------------------------------ #


def _rule_has_start_node(graph: Graph) -> list[Diagnostic]:
    """R01: Graph must have exactly one start node (shape=Mdiamond)."""
    start_nodes = [n for n in graph.nodes.values() if n.shape == "Mdiamond"]
    if len(start_nodes) == 0:
        return [
            Diagnostic(
                rule="R01",
                severity=Severity.ERROR,
                message="Graph has no start node (shape=Mdiamond)",
            )
        ]
    if len(start_nodes) > 1:
        ids = ", ".join(n.id for n in start_nodes)
        return [
            Diagnostic(
                rule="R01",
                severity=Severity.ERROR,
                message=f"Graph has multiple start nodes: {ids}. Expected exactly one.",
            )
        ]
    return []


def _rule_has_exit_node(graph: Graph) -> list[Diagnostic]:
    """R02: Graph must have exactly one exit node (shape=Msquare). Spec §11.2.2."""
    exit_nodes = graph.get_exit_nodes()
    if not exit_nodes:
        return [
            Diagnostic(
                rule="R02",
                severity=Severity.ERROR,
                message="Graph has no exit node (shape=Msquare)",
            )
        ]
    if len(exit_nodes) > 1:
        n = len(exit_nodes)
        return [
            Diagnostic(
                rule="R02",
                severity=Severity.ERROR,
                message=f"Graph has {n} exit nodes but exactly one is required.",
            )
        ]
    return []


def _rule_start_has_no_incoming(graph: Graph) -> list[Diagnostic]:
    """R03: Start node must have no incoming edges."""
    start = graph.get_start_node()
    if start is None:
        return []
    incoming = graph.incoming_edges(start.id)
    if incoming:
        sources = ", ".join(e.source for e in incoming)
        return [
            Diagnostic(
                rule="R03",
                severity=Severity.ERROR,
                message=f"Start node '{start.id}' has incoming edges from: {sources}",
                node_id=start.id,
            )
        ]
    return []


def _rule_exit_has_no_outgoing(graph: Graph) -> list[Diagnostic]:
    """R04: Exit nodes must have no outgoing edges."""
    results: list[Diagnostic] = []
    for node in graph.get_exit_nodes():
        outgoing = graph.outgoing_edges(node.id)
        if outgoing:
            targets = ", ".join(e.target for e in outgoing)
            results.append(
                Diagnostic(
                    rule="R04",
                    severity=Severity.ERROR,
                    message=(f"Exit node '{node.id}' has outgoing edges to: {targets}"),
                    node_id=node.id,
                )
            )
    return results


def _rule_no_orphan_nodes(graph: Graph) -> list[Diagnostic]:
    """R05: Every node must be reachable from the start node via BFS.

    Spec §7.2 / §11.12.6: Uses BFS/DFS traversal from the start node rather
    than a simple incoming-edge heuristic.  Unreachable nodes are reported as
    WARNING (not ERROR) per spec DoD §11.12.6.
    """
    start = graph.get_start_node()
    if start is None:
        # R01 already reports the missing start node -- skip here.
        return []

    # BFS from start to collect all reachable node IDs
    visited: set[str] = set()
    queue: deque[str] = deque([start.id])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for edge in graph.outgoing_edges(current):
            if edge.target not in visited:
                queue.append(edge.target)

    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.id not in visited:
            results.append(
                Diagnostic(
                    rule="R05",
                    severity=Severity.WARNING,
                    message=f"Node '{node.id}' is not reachable from the start node",
                    node_id=node.id,
                )
            )
    return results


def _rule_edges_reference_existing_nodes(graph: Graph) -> list[Diagnostic]:
    """R06: All edge endpoints must reference existing nodes."""
    results: list[Diagnostic] = []
    for i, edge in enumerate(graph.edges):
        if edge.source not in graph.nodes:
            results.append(
                Diagnostic(
                    rule="R06",
                    severity=Severity.ERROR,
                    message=f"Edge references unknown source node: '{edge.source}'",
                    edge_index=i,
                )
            )
        if edge.target not in graph.nodes:
            results.append(
                Diagnostic(
                    rule="R06",
                    severity=Severity.ERROR,
                    message=f"Edge references unknown target node: '{edge.target}'",
                    edge_index=i,
                )
            )
    return results


def _rule_exit_reachable_from_start(graph: Graph) -> list[Diagnostic]:
    """R07: At least one exit node must be reachable from the start node."""
    start = graph.get_start_node()
    if start is None:
        return []

    exit_ids = {n.id for n in graph.get_exit_nodes()}
    if not exit_ids:
        return []

    # BFS from start
    visited: set[str] = set()
    queue: deque[str] = deque([start.id])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for edge in graph.outgoing_edges(current):
            if edge.target not in visited:
                queue.append(edge.target)

    if not visited & exit_ids:
        return [
            Diagnostic(
                rule="R07",
                severity=Severity.ERROR,
                message="No exit node is reachable from the start node",
            )
        ]
    return []


def _rule_conditional_has_outgoing(graph: Graph) -> list[Diagnostic]:
    """R08: Conditional nodes (diamond) must have at least 2 outgoing edges."""
    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.shape == "diamond":
            outgoing = graph.outgoing_edges(node.id)
            if len(outgoing) < 2:
                results.append(
                    Diagnostic(
                        rule="R08",
                        severity=Severity.WARNING,
                        message=(
                            f"Conditional node '{node.id}' has "
                            f"{len(outgoing)} outgoing edge(s) (expected >= 2)"
                        ),
                        node_id=node.id,
                    )
                )
    return results


def _rule_goal_gate_has_retry_target(graph: Graph) -> list[Diagnostic]:
    """R09: Exit nodes with goal_gate should have a retry_target."""
    results: list[Diagnostic] = []
    for node in graph.get_exit_nodes():
        if node.goal_gate and not node.retry_target:
            results.append(
                Diagnostic(
                    rule="R09",
                    severity=Severity.WARNING,
                    message=(
                        f"Exit node '{node.id}' has a goal_gate but no "
                        f"retry_target. Gate failure will halt the pipeline."
                    ),
                    node_id=node.id,
                )
            )
    return results


def _rule_retry_target_exists(graph: Graph) -> list[Diagnostic]:
    """R10: retry_target must reference an existing node."""
    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.retry_target and node.retry_target not in graph.nodes:
            results.append(
                Diagnostic(
                    rule="R10",
                    severity=Severity.ERROR,
                    message=(
                        f"Node '{node.id}' has retry_target "
                        f"'{node.retry_target}' which does not exist"
                    ),
                    node_id=node.id,
                )
            )
    return results


def _rule_no_self_loops(graph: Graph) -> list[Diagnostic]:
    """R11: Edges should not create self-loops (source == target)."""
    results: list[Diagnostic] = []
    for i, edge in enumerate(graph.edges):
        if edge.source == edge.target:
            results.append(
                Diagnostic(
                    rule="R11",
                    severity=Severity.WARNING,
                    message=f"Self-loop on node '{edge.source}'",
                    edge_index=i,
                    node_id=edge.source,
                )
            )
    return results


def _rule_has_goal(graph: Graph) -> list[Diagnostic]:
    """R12: Graph should have a goal attribute."""
    if not graph.goal:
        return [
            Diagnostic(
                rule="R12",
                severity=Severity.INFO,
                message="Graph has no 'goal' attribute. Consider adding one.",
            )
        ]
    return []


def _rule_prompt_on_llm_nodes(graph: Graph) -> list[Diagnostic]:
    """R13: LLM task nodes (shape=box) must have a prompt (§11.2.7)."""
    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        # §11.2.7: WARNING fires whenever prompt is absent on a box node.
        # label does NOT substitute for prompt.
        if node.shape == "box" and not node.prompt:
            results.append(
                Diagnostic(
                    rule="R13",
                    severity=Severity.WARNING,
                    message=(
                        f"LLM node '{node.id}' has no 'prompt' attribute. "
                        f"Add a 'prompt' so the handler knows what to do "
                        f"(note: 'label' does not substitute for 'prompt')."
                    ),
                    node_id=node.id,
                )
            )
    return results


def _rule_condition_syntax(graph: Graph) -> list[Diagnostic]:
    """R14: Edge condition expressions must parse without errors."""
    results: list[Diagnostic] = []
    dummy_vars: dict[str, str] = {"outcome": "success", "preferred_label": ""}
    for i, edge in enumerate(graph.edges):
        if not edge.condition:
            continue
        try:
            evaluate_condition(edge.condition, dummy_vars)
        except Exception as exc:  # noqa: BLE001
            results.append(
                Diagnostic(
                    rule="R14",
                    severity=Severity.ERROR,
                    message=(
                        f"Edge {edge.source} -> {edge.target} has invalid condition "
                        f"'{edge.condition}': {exc}"
                    ),
                    edge_index=i,
                )
            )
    return results


def _rule_manager_has_child_graph(graph: Graph) -> list[Diagnostic]:
    """R15: Manager nodes (shape=hexagon) must have a child_graph attribute.

    Hexagon maps to ManagerHandler which requires child_graph to know
    which sub-pipeline to orchestrate.  Without it the handler fails
    silently on every retry attempt (GitHub issue #36).
    """
    results: list[Diagnostic] = []
    for node in graph.nodes.values():
        if node.shape == "hexagon" and not node.attrs.get("child_graph"):
            results.append(
                Diagnostic(
                    rule="R15",
                    severity=Severity.ERROR,
                    message=(
                        f"Manager node '{node.id}' (shape=hexagon) has no "
                        f"'child_graph' attribute. The manager handler requires "
                        f"child_graph to specify which sub-pipeline to run. "
                        f"Did you mean shape=house for a human review gate?"
                    ),
                    node_id=node.id,
                )
            )
    return results


# ------------------------------------------------------------------ #
# Rule registry
# ------------------------------------------------------------------ #

ALL_RULES = [
    _rule_has_start_node,
    _rule_has_exit_node,
    _rule_start_has_no_incoming,
    _rule_exit_has_no_outgoing,
    _rule_no_orphan_nodes,
    _rule_edges_reference_existing_nodes,
    _rule_exit_reachable_from_start,
    _rule_conditional_has_outgoing,
    _rule_goal_gate_has_retry_target,
    _rule_retry_target_exists,
    _rule_no_self_loops,
    _rule_has_goal,
    _rule_prompt_on_llm_nodes,
    _rule_condition_syntax,
    _rule_manager_has_child_graph,
]
