"""Graph model for Attractor pipelines.

Defines the core data structures for DOT-based pipeline graphs:
Node, Edge, Graph. These are the in-memory representation produced
by the DOT parser and consumed by the execution engine.

Spec reference: attractor-spec §2.3-2.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeShape(StrEnum):
    """Known node shapes that map to handler types. Spec §2.4."""

    BOX = "box"  # codergen (LLM task)
    HEXAGON = "hexagon"  # codergen (LLM task, alternative)
    DIAMOND = "diamond"  # conditional (branching)
    COMPONENT = "component"  # parallel (fan-out)
    TRIPLEOCTAGON = "tripleoctagon"  # fan-in (join)
    PARALLELOGRAM = "parallelogram"  # tool (shell/script)
    HOUSE = "house"  # wait.human (human gate)
    MSQUARE = "Msquare"  # exit (terminal)
    ELLIPSE = "ellipse"  # start (entry point)

    @classmethod
    def handler_for_shape(cls, shape: str) -> str:
        """Map a shape to its default handler type."""
        mapping: dict[str, str] = {
            "box": "codergen",
            "hexagon": "manager",
            "diamond": "conditional",
            "component": "parallel",
            "tripleoctagon": "fan_in",
            "parallelogram": "tool",
            "house": "wait.human",
            "Msquare": "exit",
            "ellipse": "start",
        }
        return mapping.get(shape, "codergen")


@dataclass
class Node:
    """A node in the pipeline graph. Spec §2.5.

    Nodes represent tasks: LLM calls, human reviews, conditional branches,
    tool executions, etc. The `shape` determines the default handler.
    """

    id: str
    shape: str = "box"
    label: str = ""
    handler: str = ""  # Explicit handler override; empty = derive from shape

    # LLM configuration (can be set by stylesheet or explicit attributes)
    llm_model: str = ""
    llm_provider: str = ""
    reasoning_effort: str = ""

    # Execution configuration
    prompt: str = ""
    timeout: str = ""  # Duration string, e.g., "5m", "30s"
    max_retries: int = -1  # -1 = use graph default
    fidelity: str = ""  # "full", "compact", "summary:high", etc.
    thread_id: str = ""  # For session reuse across nodes

    # Goal gate (for exit nodes)
    goal_gate: str = ""  # Condition expression
    retry_target: str = ""  # Node ID to jump to on gate failure

    # Class for stylesheet matching
    node_class: str = ""

    # All raw attributes from DOT (superset of the typed fields above)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_handler(self) -> str:
        """The handler to use: explicit override or shape-derived."""
        if self.handler:
            return self.handler
        return NodeShape.handler_for_shape(self.shape)


@dataclass
class Edge:
    """An edge connecting two nodes. Spec §2.6.

    Edges define control flow. They can carry conditions (for branching),
    labels (for preferred-label matching), and weights (for priority).
    """

    source: str
    target: str
    label: str = ""
    condition: str = ""  # Condition expression for this edge
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """A complete pipeline graph. Spec §2.3.

    Contains nodes, edges, and graph-level attributes like the goal,
    default retry policy, and model stylesheet.
    """

    name: str = "pipeline"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # Graph-level attributes (Spec §2.7)
    goal: str = ""
    model_stylesheet: str = ""
    default_max_retry: int = 50
    max_goal_gate_redirects: int = 5  # Our Issue 2 design

    # All raw graph attributes
    attrs: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Node | None:
        """Look up a node by ID."""
        return self.nodes.get(node_id)

    def get_start_node(self) -> Node | None:
        """Find the start node (shape=ellipse only).

        The validator (R01) enforces that exactly one ellipse node exists.
        No name-based fallback -- the shape is the single source of truth.
        """
        for node in self.nodes.values():
            if node.shape == "ellipse":
                return node
        return None

    def get_exit_nodes(self) -> list[Node]:
        """Find all exit nodes (shape=Msquare)."""
        return [n for n in self.nodes.values() if n.shape == "Msquare"]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get all edges leaving a node, sorted by weight (desc)."""
        edges = [e for e in self.edges if e.source == node_id]
        return sorted(edges, key=lambda e: -e.weight)

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Get all edges entering a node."""
        return [e for e in self.edges if e.target == node_id]
