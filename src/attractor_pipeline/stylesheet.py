"""Model stylesheet parser and applicator.

Parses CSS-like stylesheets that assign LLM models, providers, and
reasoning effort to pipeline nodes based on selectors. This implements
attractor-spec §8 with our Issue 3 grammar extension (shape selectors).

Grammar (extended from spec §8.2):
    Stylesheet  ::= Rule*
    Rule        ::= Selector '{' Declaration* '}'
    Selector    ::= '*' | ShapeName | '.' ClassName | '#' NodeID
    Declaration ::= Property ':' Value ';'
    Property    ::= 'llm_model' | 'llm_provider' | 'reasoning_effort'

Specificity (from our Issue 3 design):
    Level 0: * (universal)
    Level 1: ShapeName (e.g., box, diamond, house)
    Level 2: .ClassName (e.g., .critical, .fast)
    Level 3: #NodeID (e.g., #final_review)

Later rules of equal specificity override earlier ones.
Explicit node attributes always override stylesheet rules.

Example::

    * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
    .critical { llm_model: claude-opus-4-6; reasoning_effort: high; }
    box { reasoning_effort: medium; }
    #final_review { llm_model: gpt-5.2; llm_provider: openai; }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from attractor_pipeline.graph import Graph, Node, NodeShape

logger = logging.getLogger(__name__)

# Derive known shapes from NodeShape enum (single source of truth)
KNOWN_SHAPES: frozenset[str] = frozenset(shape.value for shape in NodeShape)

# Properties that stylesheets can set
ALLOWED_PROPERTIES: frozenset[str] = frozenset(
    {
        "llm_model",
        "llm_provider",
        "reasoning_effort",
    }
)


@dataclass
class Selector:
    """A parsed CSS-like selector with specificity."""

    kind: str  # "universal", "shape", "class", "id"
    value: str  # "*", shape name, class name, or node ID
    specificity: int  # 0=universal, 1=shape, 2=class, 3=id


@dataclass
class Rule:
    """A parsed stylesheet rule: selector + declarations."""

    selector: Selector
    declarations: dict[str, str] = field(default_factory=dict)


@dataclass
class Stylesheet:
    """A parsed stylesheet containing multiple rules."""

    rules: list[Rule] = field(default_factory=list)


class StylesheetParseError(Exception):
    """Error during stylesheet parsing."""


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #

# Tokenization patterns
_WS = re.compile(r"\s+")
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def parse_stylesheet(source: str) -> Stylesheet:
    """Parse a CSS-like stylesheet string into a Stylesheet object.

    Args:
        source: The stylesheet string (from graph's model_stylesheet attribute).

    Returns:
        Parsed Stylesheet with rules ordered by appearance.

    Raises:
        StylesheetParseError: On invalid syntax.
    """
    if not source or not source.strip():
        return Stylesheet()

    # Strip comments
    source = _COMMENT.sub("", source)

    rules: list[Rule] = []
    pos = 0

    while pos < len(source):
        # Skip whitespace
        m = _WS.match(source, pos)
        if m:
            pos = m.end()
            continue

        if pos >= len(source):
            break

        # Parse selector
        selector, pos = _parse_selector(source, pos)

        # Skip whitespace
        m = _WS.match(source, pos)
        if m:
            pos = m.end()

        # Expect '{'
        if pos >= len(source) or source[pos] != "{":
            raise StylesheetParseError(
                f"Expected '{{' after selector '{selector.value}' at position {pos}"
            )
        pos += 1

        # Parse declarations until '}'
        declarations: dict[str, str] = {}
        while pos < len(source) and source[pos] != "}":
            # Skip whitespace
            m = _WS.match(source, pos)
            if m:
                pos = m.end()
                continue

            if pos >= len(source) or source[pos] == "}":
                break

            # Parse property: value;
            prop, val, pos = _parse_declaration(source, pos)
            if prop and val:
                if prop in ALLOWED_PROPERTIES:
                    declarations[prop] = val
                else:
                    logger.warning("Ignoring unknown stylesheet property '%s'", prop)

        # Expect '}'
        if pos >= len(source) or source[pos] != "}":
            raise StylesheetParseError(f"Expected '}}' at position {pos}")
        pos += 1

        if declarations:
            rules.append(Rule(selector=selector, declarations=declarations))

    return Stylesheet(rules=rules)


def _parse_selector(source: str, pos: int) -> tuple[Selector, int]:
    """Parse a single selector starting at pos."""
    # Skip whitespace
    m = _WS.match(source, pos)
    if m:
        pos = m.end()

    if pos >= len(source):
        raise StylesheetParseError(f"Expected selector at position {pos}")

    ch = source[pos]

    # Universal: *
    if ch == "*":
        return Selector(kind="universal", value="*", specificity=0), pos + 1

    # ID: #identifier
    if ch == "#":
        pos += 1
        ident, pos = _parse_identifier(source, pos)
        return Selector(kind="id", value=ident, specificity=3), pos

    # Class: .classname
    if ch == ".":
        pos += 1
        ident, pos = _parse_identifier(source, pos)
        return Selector(kind="class", value=ident, specificity=2), pos

    # Shape or error: bare identifier
    ident, pos = _parse_identifier(source, pos)
    if ident in KNOWN_SHAPES:
        return Selector(kind="shape", value=ident, specificity=1), pos

    raise StylesheetParseError(
        f"Unknown selector '{ident}' at position {pos}. "
        f"Expected *, #id, .class, or a shape name ({', '.join(sorted(KNOWN_SHAPES))})"
    )


def _parse_identifier(source: str, pos: int) -> tuple[str, int]:
    """Parse an identifier (letters, digits, underscores, hyphens)."""
    start = pos
    while pos < len(source) and (source[pos].isalnum() or source[pos] in "_-"):
        pos += 1
    if pos == start:
        raise StylesheetParseError(f"Expected identifier at position {pos}")
    return source[start:pos], pos


def _parse_declaration(source: str, pos: int) -> tuple[str, str, int]:
    """Parse a 'property: value;' declaration."""
    # Property name
    prop, pos = _parse_identifier(source, pos)

    # Skip whitespace
    m = _WS.match(source, pos)
    if m:
        pos = m.end()

    # Expect ':'
    if pos >= len(source) or source[pos] != ":":
        raise StylesheetParseError(f"Expected ':' after property '{prop}' at position {pos}")
    pos += 1

    # Skip whitespace
    m = _WS.match(source, pos)
    if m:
        pos = m.end()

    # Value (everything until ';' or '}')
    start = pos
    while pos < len(source) and source[pos] not in ";}\n":
        pos += 1
    value = source[start:pos].strip()

    # Require ';' between declarations. Last declaration before '}'
    # or newline-terminated declarations are allowed without ';'.
    if pos < len(source) and source[pos] == ";":
        pos += 1
    elif pos < len(source) and source[pos] == "}":
        pass  # last declaration before closing brace, ';' optional
    elif pos < len(source) and source[pos] == "\n":
        pos += 1  # newline acts as implicit terminator

    return prop, value, pos


# ------------------------------------------------------------------ #
# Applicator
# ------------------------------------------------------------------ #


def apply_stylesheet(graph: Graph) -> None:
    """Apply the graph's model_stylesheet to all nodes.

    For each node, finds all matching rules, sorts by specificity,
    and applies declarations in order. Explicit node attributes
    (set directly in DOT) always take precedence over stylesheet values.

    This mutates the graph's nodes in place.
    """
    if not graph.model_stylesheet:
        return

    stylesheet = parse_stylesheet(graph.model_stylesheet)
    if not stylesheet.rules:
        return

    for node in graph.nodes.values():
        _apply_rules_to_node(node, stylesheet.rules)


def _apply_rules_to_node(node: Node, rules: list[Rule]) -> None:
    """Apply matching stylesheet rules to a single node."""
    # Collect all matching rules with their specificity
    matching: list[Rule] = []
    for rule in rules:
        if _selector_matches(rule.selector, node):
            matching.append(rule)

    if not matching:
        return

    # Sort by specificity (lower first, so higher specificity overrides)
    # Rules of equal specificity: later in source wins (stable sort preserves order)
    matching.sort(key=lambda r: r.selector.specificity)

    # Build merged declarations (later/higher-specificity wins)
    merged: dict[str, str] = {}
    for rule in matching:
        merged.update(rule.declarations)

    # Apply to node, but ONLY if the node doesn't have an explicit value
    # set in DOT. We use node.attrs (the raw DOT attributes) as the source
    # of truth for "explicitly set" -- this distinguishes an attribute that
    # was set in DOT (even to "") from one that was never set (default "").
    for prop, value in merged.items():
        if prop not in node.attrs:
            # Not explicitly set in DOT -- stylesheet can apply
            if prop == "llm_model":
                node.llm_model = value
            elif prop == "llm_provider":
                node.llm_provider = value
            elif prop == "reasoning_effort":
                node.reasoning_effort = value


def _selector_matches(selector: Selector, node: Node) -> bool:
    """Check if a selector matches a node."""
    match selector.kind:
        case "universal":
            return True
        case "shape":
            return node.shape == selector.value
        case "class":
            # Spec §2.12: node classes are comma-separated (e.g., "code,critical").
            # Split and check membership so ".code" matches "code,critical".
            classes = [c.strip() for c in node.node_class.split(",") if c.strip()]
            return selector.value in classes
        case "id":
            return node.id == selector.value
        case _:
            return False
