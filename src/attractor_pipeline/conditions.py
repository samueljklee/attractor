"""Condition expression evaluator for edge selection and goal gates.

Implements the minimal condition language from attractor-spec §10:
- key = value (equality)
- key != value (inequality)
- expr && expr (conjunction -- AND only, no OR)

Variables are resolved from the pipeline Context (a dict-like store).
Special keys: "outcome", "preferred_label", "context.*".

Examples:
    "outcome = SUCCESS"
    "outcome = SUCCESS && context.tests_passed = true"
    "preferred_label = yes"
"""

from __future__ import annotations

from typing import Any


def evaluate_condition(
    expression: str,
    variables: dict[str, Any],
) -> bool:
    """Evaluate a condition expression against a variable dict.

    Args:
        expression: Condition string (e.g., "outcome = SUCCESS").
        variables: Dict of variable names to values. Supports dotted
            keys like "context.foo" which look up variables["context.foo"].

    Returns:
        True if the condition is satisfied, False otherwise.
        Empty/blank expressions are always True.
    """
    expression = expression.strip()
    if not expression:
        return True

    # Split on && (conjunction)
    clauses = [c.strip() for c in expression.split("&&")]

    return all(_evaluate_clause(clause, variables) for clause in clauses)


def _evaluate_clause(clause: str, variables: dict[str, Any]) -> bool:
    """Evaluate a single clause: 'key op value'."""
    # Try != first (before =)
    if "!=" in clause:
        parts = clause.split("!=", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            expected = parts[1].strip()
            actual = _resolve(key, variables)
            return str(actual).lower() != expected.lower()

    if "=" in clause:
        parts = clause.split("=", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            expected = parts[1].strip()
            actual = _resolve(key, variables)
            return str(actual).lower() == expected.lower()

    # Bare truthy check: "key" alone means "key is truthy"
    key = clause.strip()
    if key:
        actual = _resolve(key, variables)
        return bool(actual)

    return True


def _resolve(key: str, variables: dict[str, Any]) -> Any:
    """Resolve a variable key from the variables dict.

    Resolution order:
    1. Direct lookup: variables["context.foo"]
    2. Nested dict: variables["context"]["foo"]
    3. Bare-key fallback for "context.*" prefix: variables["foo"]
       This handles the common case where select_edge() spreads the
       pipeline context as {**context}, giving flat keys like "foo"
       rather than "context.foo". Spec §10.4 requires context.* to
       resolve against the pipeline context store.
    """
    # Direct lookup
    if key in variables:
        return variables[key]

    # Dotted path fallback
    if "." in key:
        parts = key.split(".", 1)
        parent = variables.get(parts[0])
        if isinstance(parent, dict):
            return parent.get(parts[1], "")

        # Bare-key fallback for "context.*" prefix (Spec §10.4)
        # select_edge() spreads context as {**context} so keys are flat.
        if parts[0] == "context" and parts[1] in variables:
            return variables[parts[1]]

    return ""
