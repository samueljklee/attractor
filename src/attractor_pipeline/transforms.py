"""Graph transform pipeline for Attractor pipelines.

Implements the registerable, ordered transform pipeline from
attractor-spec S9, S11.11. Transforms run between parsing/loading
and validation: ``transform(graph) -> graph``.

The pipeline is a simple ordered list of GraphTransform implementations.
Each transform receives a Graph and returns a (possibly modified) Graph.

Built-in transforms:
- VariableExpansionTransform: wraps ``expand_node_prompt`` as a
  graph-level transform (expands $variable references in all node
  prompts before execution).

Note: variable expansion is also applied at handler level (just before
each node executes) so the handler-level expansion sees runtime context.
The graph-level transform here is for static/pre-validation expansion
when the full context is known upfront.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from attractor_pipeline.graph import Graph
from attractor_pipeline.variable_expansion import expand_variables


@runtime_checkable
class GraphTransform(Protocol):
    """Protocol for graph transforms. Spec S9, S11.11.

    A transform receives a Graph and returns a (possibly modified) Graph.
    Transforms are applied in order between parsing and validation.
    """

    def apply(self, graph: Graph) -> Graph:
        """Apply this transform to the graph.

        Args:
            graph: The pipeline graph to transform.

        Returns:
            The transformed graph (may be the same object, mutated).
        """
        ...


def apply_transforms(graph: Graph, transforms: list[GraphTransform]) -> Graph:
    """Apply an ordered list of transforms to a graph.

    Each transform's ``apply()`` is called in sequence, passing the
    result of the previous transform to the next.

    Args:
        graph: The initial pipeline graph.
        transforms: Ordered list of transforms to apply.

    Returns:
        The final transformed graph.
    """
    for transform in transforms:
        graph = transform.apply(graph)
    return graph


class VariableExpansionTransform:
    """Expands $variable references in all node prompts. Spec S9.

    This is a graph-level counterpart to the per-node expansion
    done at handler execution time. Useful when the full context
    is known before pipeline execution begins.
    """

    def __init__(self, context: dict[str, Any]) -> None:
        self._context = context

    def apply(self, graph: Graph) -> Graph:
        """Expand variables in every node's prompt attribute."""
        for node in graph.nodes.values():
            if node.prompt:
                node.prompt = expand_variables(node.prompt, self._context, undefined="keep")
        return graph
