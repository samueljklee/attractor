"""Recursive-descent parser for Attractor's strict DOT subset.

Parses DOT digraph definitions into Graph/Node/Edge structures.
This is a custom parser for the restricted subset defined in
attractor-spec §2.2, NOT a full Graphviz parser. The spec's BNF
grammar is strict: digraph only, typed attributes, chained edge
expansion, and subgraph scope inheritance.

Key design decisions (from 5-model swarm consensus):
- Custom parser, not pydot/graphviz (spec subset is too strict)
- Good error messages with line/column numbers
- Handles: quoted strings, chained edges (a -> b -> c), subgraphs,
  default node/edge attributes, comments (// and /* */)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from attractor_pipeline.graph import Edge, Graph, Node


@dataclass
class _Token:
    """A lexer token with position tracking."""

    kind: str  # IDENT, STRING, NUMBER, SYMBOL, EOF
    value: str
    line: int
    col: int


class ParseError(Exception):
    """Error during DOT parsing with position info."""

    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Line {line}, col {col}: {message}")


# ------------------------------------------------------------------ #
# Lexer
# ------------------------------------------------------------------ #

# Token patterns (order matters -- first match wins)
_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("COMMENT_LINE", re.compile(r"//[^\n]*")),
    ("COMMENT_BLOCK", re.compile(r"/\*.*?\*/", re.DOTALL)),
    ("WS", re.compile(r"\s+")),
    ("STRING", re.compile(r'"(?:[^"\\]|\\.)*"')),
    ("NUMBER", re.compile(r"-?\d+(?:\.\d+)?")),
    ("ARROW", re.compile(r"->")),
    ("SYMBOL", re.compile(r"[{}\[\]=;,]")),
    ("IDENT", re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")),
]


def _tokenize(source: str) -> list[_Token]:
    """Tokenize a DOT source string."""
    tokens: list[_Token] = []
    pos = 0
    line = 1
    col = 1

    while pos < len(source):
        matched = False
        for kind, pattern in _TOKEN_PATTERNS:
            m = pattern.match(source, pos)
            if m:
                value = m.group(0)
                if kind not in ("WS", "COMMENT_LINE", "COMMENT_BLOCK"):
                    tokens.append(_Token(kind=kind, value=value, line=line, col=col))

                # Update position tracking
                newlines = value.count("\n")
                if newlines:
                    line += newlines
                    col = len(value) - value.rfind("\n")
                else:
                    col += len(value)

                pos = m.end()
                matched = True
                break

        if not matched:
            raise ParseError(
                f"Unexpected character: {source[pos]!r}",
                line=line,
                col=col,
            )

    tokens.append(_Token(kind="EOF", value="", line=line, col=col))
    return tokens


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #


class _Parser:
    """Recursive-descent parser for the DOT subset."""

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0

        # Accumulated graph state
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._graph_attrs: dict[str, Any] = {}
        self._default_node_attrs: dict[str, str] = {}
        self._default_edge_attrs: dict[str, str] = {}

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: str, value: str | None = None) -> _Token:
        tok = self._advance()
        if tok.kind != kind:
            raise ParseError(
                f"Expected {kind}, got {tok.kind} ({tok.value!r})",
                line=tok.line,
                col=tok.col,
            )
        if value is not None and tok.value != value:
            raise ParseError(
                f"Expected {value!r}, got {tok.value!r}",
                line=tok.line,
                col=tok.col,
            )
        return tok

    def _at(self, kind: str, value: str | None = None) -> bool:
        tok = self._peek()
        if tok.kind != kind:
            return False
        if value is not None and tok.value != value:
            return False
        return True

    def _unquote(self, s: str) -> str:
        """Remove surrounding quotes and unescape."""
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
            s = s.replace('\\"', '"')
            s = s.replace("\\n", "\n")
            s = s.replace("\\\\", "\\")
        return s

    # ---------------------------------------------------------------- #
    # Grammar productions
    # ---------------------------------------------------------------- #

    def parse(self) -> Graph:
        """Parse: digraph IDENT { stmt_list }"""
        self._expect("IDENT", "digraph")

        name = "pipeline"
        if self._at("IDENT") or self._at("STRING"):
            name_tok = self._advance()
            name = self._unquote(name_tok.value)

        self._expect("SYMBOL", "{")
        self._parse_stmt_list()
        self._expect("SYMBOL", "}")

        return self._build_graph(name)

    def _parse_stmt_list(self) -> None:
        """Parse a list of statements inside { }."""
        while not self._at("SYMBOL", "}") and not self._at("EOF"):
            self._parse_stmt()
            # Optional semicolon
            if self._at("SYMBOL", ";"):
                self._advance()

    def _parse_stmt(self) -> None:
        """Parse a single statement: node, edge, attr, subgraph, or default."""
        # graph/node/edge defaults: e.g., "graph [ key = value ]"
        if self._at("IDENT", "graph"):
            self._advance()
            if self._at("SYMBOL", "["):
                attrs = self._parse_attr_list()
                self._graph_attrs.update(attrs)
            return

        if self._at("IDENT", "node"):
            self._advance()
            if self._at("SYMBOL", "["):
                attrs = self._parse_attr_list()
                self._default_node_attrs.update(attrs)
            return

        if self._at("IDENT", "edge"):
            self._advance()
            if self._at("SYMBOL", "["):
                attrs = self._parse_attr_list()
                self._default_edge_attrs.update(attrs)
            return

        # Subgraph: subgraph name { ... }
        if self._at("IDENT", "subgraph"):
            self._parse_subgraph()
            return

        # Node or edge statement: starts with IDENT or STRING
        if self._at("IDENT") or self._at("STRING"):
            self._parse_node_or_edge()
            return

        # Skip unknown tokens
        self._advance()

    def _parse_subgraph(self) -> None:
        """Parse: subgraph IDENT? { stmt_list }

        Subgraph default attributes are inherited by nodes within.
        """
        self._expect("IDENT", "subgraph")

        # Optional subgraph name/class
        sub_class = ""
        if self._at("IDENT") or self._at("STRING"):
            name_tok = self._advance()
            sub_name = self._unquote(name_tok.value)
            # Convention: subgraph names starting with "cluster_" or
            # plain names become the class for contained nodes
            if sub_name.startswith("cluster_"):
                sub_class = sub_name[8:]
            else:
                sub_class = sub_name

        self._expect("SYMBOL", "{")

        # Save and restore default attrs for subgraph scope
        saved_node_attrs = dict(self._default_node_attrs)
        saved_edge_attrs = dict(self._default_edge_attrs)

        # Parse subgraph body (inherits parent defaults)
        nodes_before = set(self._nodes.keys())
        self._parse_stmt_list()
        self._expect("SYMBOL", "}")

        # Apply subgraph class to new nodes
        if sub_class:
            for nid in self._nodes:
                if nid not in nodes_before and not self._nodes[nid].node_class:
                    self._nodes[nid].node_class = sub_class

        # Restore parent defaults
        self._default_node_attrs = saved_node_attrs
        self._default_edge_attrs = saved_edge_attrs

    def _parse_node_or_edge(self) -> None:
        """Parse a node definition or edge chain.

        Determines which by looking ahead for '->'.
        Handles chained edges: a -> b -> c expands to a->b, b->c.
        """
        first_tok = self._advance()
        first_id = self._unquote(first_tok.value)

        # Check if this is an edge chain
        if self._at("ARROW"):
            # Edge chain: collect all node IDs
            node_ids = [first_id]
            while self._at("ARROW"):
                self._advance()  # consume ->
                next_tok = self._advance()
                node_ids.append(self._unquote(next_tok.value))

            # Optional edge attributes
            edge_attrs: dict[str, str] = {}
            if self._at("SYMBOL", "["):
                edge_attrs = self._parse_attr_list()

            # Ensure all nodes exist
            for nid in node_ids:
                self._ensure_node(nid)

            # Expand chain: a->b->c becomes edges a->b and b->c
            merged_attrs = {**self._default_edge_attrs, **edge_attrs}
            for i in range(len(node_ids) - 1):
                self._edges.append(
                    Edge(
                        source=node_ids[i],
                        target=node_ids[i + 1],
                        label=merged_attrs.get("label", ""),
                        condition=merged_attrs.get("condition", ""),
                        weight=float(merged_attrs.get("weight", "1.0")),
                        attrs=dict(merged_attrs),
                    )
                )
        else:
            # Node definition with optional attributes
            node_attrs: dict[str, str] = {}
            if self._at("SYMBOL", "["):
                node_attrs = self._parse_attr_list()
            self._ensure_node(first_id, node_attrs)

    def _parse_attr_list(self) -> dict[str, str]:
        """Parse: [ key = value, key = value, ... ]"""
        self._expect("SYMBOL", "[")
        attrs: dict[str, str] = {}

        while not self._at("SYMBOL", "]") and not self._at("EOF"):
            # key
            key_tok = self._advance()
            key = self._unquote(key_tok.value)

            # =
            self._expect("SYMBOL", "=")

            # value (can be STRING, NUMBER, IDENT)
            val_tok = self._advance()
            value = self._unquote(val_tok.value)

            attrs[key] = value

            # Optional comma or semicolon separator
            if self._at("SYMBOL", ",") or self._at("SYMBOL", ";"):
                self._advance()

        self._expect("SYMBOL", "]")
        return attrs

    def _ensure_node(self, node_id: str, extra_attrs: dict[str, str] | None = None) -> None:
        """Create or update a node, merging default + explicit attributes."""
        merged = {**self._default_node_attrs, **(extra_attrs or {})}

        if node_id in self._nodes:
            # Update existing node with new attributes
            node = self._nodes[node_id]
            self._apply_attrs_to_node(node, merged)
        else:
            # Create new node
            node = Node(
                id=node_id,
                label=merged.get("label", node_id),
            )
            self._apply_attrs_to_node(node, merged)
            self._nodes[node_id] = node

    def _apply_attrs_to_node(self, node: Node, attrs: dict[str, str]) -> None:
        """Apply parsed attributes to a Node's typed fields."""
        node.attrs.update(attrs)

        if "shape" in attrs:
            node.shape = attrs["shape"]
        if "label" in attrs:
            node.label = attrs["label"]
        if "handler" in attrs:
            node.handler = attrs["handler"]
        if "llm_model" in attrs:
            node.llm_model = attrs["llm_model"]
        if "llm_provider" in attrs:
            node.llm_provider = attrs["llm_provider"]
        if "reasoning_effort" in attrs:
            node.reasoning_effort = attrs["reasoning_effort"]
        if "prompt" in attrs:
            node.prompt = attrs["prompt"]
        if "timeout" in attrs:
            node.timeout = attrs["timeout"]
        if "max_retries" in attrs:
            try:
                node.max_retries = int(attrs["max_retries"])
            except ValueError:
                pass
        if "fidelity" in attrs:
            node.fidelity = attrs["fidelity"]
        if "thread_id" in attrs:
            node.thread_id = attrs["thread_id"]
        if "goal_gate" in attrs:
            node.goal_gate = attrs["goal_gate"]
        if "retry_target" in attrs:
            node.retry_target = attrs["retry_target"]
        if "class" in attrs:
            node.node_class = attrs["class"]

    def _build_graph(self, name: str) -> Graph:
        """Assemble the final Graph from parsed state."""
        graph = Graph(
            name=name,
            nodes=self._nodes,
            edges=self._edges,
            goal=self._graph_attrs.get("goal", ""),
            model_stylesheet=self._graph_attrs.get("model_stylesheet", ""),
            attrs=self._graph_attrs,
        )

        # Parse graph-level retry config
        if "default_max_retry" in self._graph_attrs:
            try:
                graph.default_max_retry = int(self._graph_attrs["default_max_retry"])
            except ValueError:
                pass

        if "max_goal_gate_redirects" in self._graph_attrs:
            try:
                graph.max_goal_gate_redirects = int(self._graph_attrs["max_goal_gate_redirects"])
            except ValueError:
                pass

        return graph


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def parse_dot(source: str) -> Graph:
    """Parse a DOT digraph string into a Graph.

    Args:
        source: DOT format string defining a digraph.

    Returns:
        Parsed Graph with nodes, edges, and attributes.

    Raises:
        ParseError: If the source is not valid DOT syntax.
    """
    tokens = _tokenize(source)
    parser = _Parser(tokens)
    return parser.parse()
