"""Human-in-the-loop handler for wait.human nodes.

Implements the Interviewer pattern from attractor-spec §6.
The handler delegates to an Interviewer implementation to ask
the human a question and wait for a response.

Default implementations:
- AutoApproveInterviewer: always approves (for testing)
- ConsoleInterviewer: prompts on stdin (for CLI use)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Graph, Node


class Interviewer(Protocol):
    """Interface for human-in-the-loop interactions. Spec §6.

    Implementations handle how questions are presented to and
    answered by humans: console prompts, web UI, Slack, etc.
    """

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
    ) -> str:
        """Ask the human a question and return their response.

        Args:
            question: The question text to present.
            options: Optional list of valid choices (e.g., ["yes", "no"]).
            node_id: The pipeline node ID (for context).

        Returns:
            The human's response string.
        """
        ...


class AutoApproveInterviewer:
    """Auto-approves all human gates. For testing and CI."""

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
    ) -> str:
        if options:
            return options[0]
        return "approved"


class ConsoleInterviewer:
    """Prompts on stdin for human input. For CLI use."""

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
    ) -> str:
        import asyncio

        prompt = f"\n[HUMAN GATE: {node_id}]\n{question}"
        if options:
            prompt += f"\nOptions: {', '.join(options)}"
        prompt += "\n> "

        # Run input() in a thread to avoid blocking the event loop
        return await asyncio.to_thread(input, prompt)


class HumanHandler:
    """Handler for wait.human nodes (shape=house). Spec §4.6.

    Presents a question to the human via the Interviewer interface
    and waits for a response. The response is stored in context
    and used for downstream edge selection.
    """

    def __init__(self, interviewer: Interviewer | None = None) -> None:
        self._interviewer = interviewer or AutoApproveInterviewer()

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        # Build question from node's prompt/label
        question = node.prompt or node.label or f"Approve '{node.id}'?"

        # Extract options from outgoing edge labels
        edges = graph.outgoing_edges(node.id)
        options = [e.label for e in edges if e.label] or None

        # Check abort before blocking on human input
        if abort_signal and abort_signal.is_set:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Aborted while waiting for human input",
            )

        # Ask the human
        try:
            response = await self._interviewer.ask(
                question=question,
                options=options,
                node_id=node.id,
            )
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"Interviewer error: {exc}",
            )

        # Store response in context
        context[f"human.{node.id}.response"] = response

        # Use response as preferred_label for edge selection
        return HandlerResult(
            status=Outcome.SUCCESS,
            preferred_label=response,
            output=response,
            notes=f"Human responded: {response}",
        )
