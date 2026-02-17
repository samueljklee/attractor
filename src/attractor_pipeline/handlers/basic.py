"""Basic node handlers: start, exit, conditional, tool.

These are the simple handlers that don't require external backends
(unlike codergen which needs an LLM, or human which needs an interviewer).

Spec reference: attractor-spec §4.3-4.10.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Graph, Node


class StartHandler:
    """Handler for start nodes (shape=Mdiamond). Spec §4.3.

    The start handler is a no-op -- it simply passes through
    with SUCCESS status. Its purpose is to mark the entry point.
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        return HandlerResult(
            status=Outcome.SUCCESS,
            notes=f"Pipeline started at node '{node.id}'",
        )


class ExitHandler:
    """Handler for exit nodes (shape=Msquare). Spec §4.4.

    The exit handler is a no-op -- it marks the terminal node.
    Goal gate checking happens in the engine, not the handler.
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        return HandlerResult(
            status=Outcome.SUCCESS,
            notes=f"Pipeline reached exit node '{node.id}'",
        )


class ConditionalHandler:
    """Handler for conditional/branching nodes (shape=diamond). Spec §4.7.

    The conditional handler is a no-op at the handler level -- the actual
    branching logic is in the edge selection algorithm (§3.3). The handler
    simply returns SUCCESS, and the edge selector uses conditions on
    outgoing edges to determine the next node.

    If the node has a prompt, it's stored in context for edge conditions
    to reference.
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        # If the node has a prompt, evaluate it as context
        if node.prompt:
            context[f"conditional.{node.id}"] = node.prompt

        return HandlerResult(
            status=Outcome.SUCCESS,
            notes=f"Conditional branch at '{node.id}'",
        )


class ToolHandler:
    """Handler for tool/script nodes (shape=parallelogram). Spec §4.10.

    Executes a shell command defined in the node's prompt attribute.
    The command runs in the pipeline's working directory.
    """

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        command = node.prompt or node.attrs.get("command", "")
        if not command:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"Tool node '{node.id}' has no command",
            )

        # Variable expansion in command (shell-safe quoting to prevent injection)
        for key, value in context.items():
            if isinstance(value, str):
                safe_value = shlex.quote(value)
                command = command.replace(f"${{{key}}}", safe_value)
                command = command.replace(f"${key}", safe_value)

        timeout_str = node.timeout or "120s"
        timeout_seconds = _parse_duration(timeout_str)

        try:
            result = await asyncio.to_thread(
                subprocess.run,  # noqa: S603
                ["bash", "-c", command],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=os.getcwd(),
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=(f"Command timed out after {timeout_seconds}s"),
                output=f"Timeout: {command}",
            )
        except OSError as e:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=str(e),
            )

        output = result.stdout or ""
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        if result.returncode == 0:
            # Store output in context for downstream nodes
            context[f"tool.{node.id}.output"] = output.strip()
            return HandlerResult(
                status=Outcome.SUCCESS,
                output=output,
                notes="Command succeeded (exit 0)",
            )

        return HandlerResult(
            status=Outcome.FAIL,
            failure_reason=f"Exit code {result.returncode}",
            output=output,
            notes=f"Command failed (exit {result.returncode})",
        )


def _parse_duration(duration_str: str) -> int:
    """Parse a duration string like '5m', '30s', '2h' to seconds."""
    duration_str = duration_str.strip().lower()

    if duration_str.endswith("h"):
        return int(float(duration_str[:-1]) * 3600)
    if duration_str.endswith("m"):
        return int(float(duration_str[:-1]) * 60)
    if duration_str.endswith("s"):
        return int(float(duration_str[:-1]))

    # Try plain integer (assume seconds)
    try:
        return int(duration_str)
    except ValueError:
        return 120  # default
