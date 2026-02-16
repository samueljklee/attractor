"""Codergen handler -- LLM-powered node execution.

The codergen handler is the primary integration point between the
Attractor pipeline and the Coding Agent Loop (or any other LLM backend).
It delegates to a CodergenBackend interface, which is deliberately
narrow by design (see our Issue 5 resolution).

Spec reference: attractor-spec §4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Graph, Node
from attractor_pipeline.variable_expansion import expand_node_prompt


class CodergenBackend(Protocol):
    """Backend interface for LLM-powered nodes. Spec §4.5.

    This is intentionally narrow -- the Node carries all configuration
    (llm_model, llm_provider, reasoning_effort, etc.) and the Context
    carries all runtime state. The backend can extract whatever it needs.

    Implementations:
    - Wrap a Coding Agent Session (primary use case)
    - Call the LLM SDK directly (for simple prompts)
    - Shell out to a CLI agent (Claude Code, Codex, etc.)

    Returns either a plain string (auto-wrapped as SUCCESS) or a
    full HandlerResult for richer control over routing and context.
    """

    async def run(
        self,
        node: Node,
        prompt: str,
        context: dict[str, Any],
        abort_signal: AbortSignal | None = None,
    ) -> str | HandlerResult: ...


class CodergenHandler:
    """Handler for codergen/LLM nodes (shape=box, hexagon). Spec §4.5.

    Expands the node's prompt template with context variables,
    then delegates to the CodergenBackend for LLM execution.
    """

    def __init__(self, backend: CodergenBackend | None = None) -> None:
        self._backend = backend

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        # Build prompt from node's prompt attribute + goal
        prompt = self._expand_prompt(node, context, graph)

        if not prompt:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(f"Codergen node '{node.id}' has no prompt"),
            )

        if self._backend is None:
            # No backend configured -- return placeholder
            return HandlerResult(
                status=Outcome.SUCCESS,
                output=f"[No backend configured] Prompt: {prompt[:200]}",
                notes="Codergen executed without backend (dry run)",
            )

        # Delegate to backend
        try:
            result = await self._backend.run(
                node,
                prompt,
                context,
                abort_signal,
            )
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        # Normalize result to HandlerResult
        if isinstance(result, str):
            # Plain string -> wrap as SUCCESS
            context[f"codergen.{node.id}.output"] = result
            handler_result = HandlerResult(
                status=Outcome.SUCCESS,
                output=result,
                notes=f"Codergen node '{node.id}' completed",
            )
        else:
            # Already a HandlerResult
            if result.output:
                context[f"codergen.{node.id}.output"] = result.output
            handler_result = result

        # Write per-node artifact files (Spec §5.6)
        if logs_root is not None:
            try:
                # Sanitize node.id to prevent path traversal (defense-in-depth)
                safe_id = node.id.replace("/", "_").replace("\\", "_").replace("..", "_")
                node_dir = Path(logs_root) / safe_id
                node_dir.mkdir(parents=True, exist_ok=True)

                # prompt.md -- the expanded prompt sent to the LLM
                (node_dir / "prompt.md").write_text(prompt, encoding="utf-8")

                # response.md -- the LLM response text
                (node_dir / "response.md").write_text(handler_result.output or "", encoding="utf-8")

                # status.json -- the handler result metadata
                status_data = {
                    "node_id": node.id,
                    "status": handler_result.status.value,
                    "preferred_label": handler_result.preferred_label,
                    "failure_reason": handler_result.failure_reason,
                    "notes": handler_result.notes,
                }
                (node_dir / "status.json").write_text(
                    json.dumps(status_data, indent=2), encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                # Artifact writing is observability, not core logic.
                # Don't let I/O errors tank the pipeline.
                pass

        return handler_result

    def _expand_prompt(self, node: Node, context: dict[str, Any], graph: Graph) -> str:
        """Expand template variables in the node's prompt.

        Uses the variable_expansion module for proper $var and ${var}
        expansion with escaped \\$ support. The graph goal is injected
        into the context for expansion.
        """
        prompt = node.prompt or node.label or ""

        # Merge goal into expansion context
        expand_ctx = dict(context)
        expand_ctx["goal"] = graph.goal

        return expand_node_prompt(prompt, expand_ctx)
