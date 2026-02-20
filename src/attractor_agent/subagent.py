"""Subagent spawning for delegated task execution.

Enables an agent session to spawn child sessions that share the
filesystem but have their own conversation history, tools, and
limits. Implements depth limiting to prevent infinite delegation.

Usage::

    from attractor_agent.subagent import spawn_subagent

    # From within a parent session's tool handler:
    result = await spawn_subagent(
        client=client,
        prompt="Refactor the auth module",
        parent_depth=0,
        max_depth=3,
        profile="anthropic",
    )

Spec reference: coding-agent-loop-spec §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_agent.profiles import get_profile
from attractor_agent.session import Session, SessionConfig
from attractor_agent.tools.core import ALL_CORE_TOOLS
from attractor_llm.types import Usage


class SubagentError(Exception):
    """Error during subagent execution."""


class MaxDepthError(SubagentError):
    """Subagent depth limit exceeded."""


async def spawn_subagent(
    client: Any,
    prompt: str,
    *,
    parent_depth: int = 0,
    max_depth: int = 1,
    model: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    max_turns: int = 20,
    max_tool_rounds: int = 15,
    abort_signal: AbortSignal | None = None,
    include_tools: bool = True,
    context: dict[str, Any] | None = None,
    working_dir: str | None = None,
) -> SubagentResult:
    """Spawn a child agent session to handle a delegated task.

    The child session shares the filesystem (same allowed_roots) but
    has its own conversation history, limits, and depth counter.

    Args:
        client: LLM Client with registered adapters.
        prompt: The task to delegate to the subagent.
        parent_depth: Current depth in the delegation chain (0 = root).
        max_depth: Maximum allowed depth. Raises MaxDepthError if exceeded.
        model: Model ID. If None, uses profile default.
        provider: Provider name. If None, auto-detected from model.
        system_prompt: Override system prompt. If None, uses profile default.
        max_turns: Maximum turns for the child session.
        max_tool_rounds: Maximum tool rounds per turn.
        abort_signal: Cooperative cancellation (shared with parent).
        include_tools: Whether to give the subagent developer tools.
        context: Optional context dict passed to the subagent's system prompt.
        working_dir: Working directory for the child agent (None = inherit cwd).

    Returns:
        SubagentResult with the child's response and metadata.

    Raises:
        MaxDepthError: If parent_depth >= max_depth.
        SubagentError: On unexpected errors during child execution.
    """
    # Depth check
    child_depth = parent_depth + 1
    if child_depth > max_depth:
        raise MaxDepthError(f"Subagent depth limit exceeded: depth {child_depth} > max {max_depth}")

    # Resolve profile
    profile = get_profile(provider or "")

    # Build config
    config = SessionConfig(
        model=model or "",
        provider=provider,
        system_prompt=system_prompt or "",
        max_turns=max_turns,
        max_tool_rounds_per_turn=max_tool_rounds,
        working_dir=working_dir,
    )

    # Apply profile defaults
    config = profile.apply_to_config(config)

    # Inject subagent context into system prompt
    depth_info = (
        f"\n\n[SUBAGENT] You are a subagent at depth {child_depth}/{max_depth}. "
        f"Focus on the delegated task. Be concise."
    )
    config.system_prompt = (config.system_prompt or "") + depth_info

    if context:
        context_info = "\n\n[CONTEXT] " + ", ".join(f"{k}={v}" for k, v in context.items())
        config.system_prompt += context_info

    # Build tools
    tools = list(ALL_CORE_TOOLS) if include_tools else []

    # If subagent can spawn further subagents, add the spawn tool
    if child_depth < max_depth:
        tools = _add_spawn_tool(tools, client, child_depth, max_depth, abort_signal)

    # Execute child session
    try:
        session = Session(
            client=client,
            config=config,
            tools=tools,
            abort_signal=abort_signal,
        )
        # Session.__init__ always injects subagent tools (§9.12.34-36).
        # At max depth the child must not be able to spawn further; remove it.
        if child_depth >= max_depth:
            session.tool_registry.unregister("spawn_agent")
            session.tool_registry.unregister("send_input")
            session.tool_registry.unregister("wait")
            session.tool_registry.unregister("close_agent")
        result_text = await session.submit(prompt)
    except Exception as exc:
        raise SubagentError(f"Subagent failed: {type(exc).__name__}: {exc}") from exc

    return SubagentResult(
        text=result_text,
        depth=child_depth,
        model=config.model,
        provider=config.provider or "",
        turn_count=session.turn_count,
        usage=session.total_usage,
    )


# ------------------------------------------------------------------ #
# Result type
# ------------------------------------------------------------------ #


@dataclass
class SubagentResult:
    """Result from a subagent execution."""

    text: str
    depth: int
    model: str
    provider: str
    turn_count: int
    usage: Usage


# ------------------------------------------------------------------ #
# Spawn tool (recursive delegation)
# ------------------------------------------------------------------ #


def _add_spawn_tool(
    tools: list[Any],
    client: Any,
    parent_depth: int,
    max_depth: int,
    abort_signal: AbortSignal | None,
) -> list[Any]:
    """Add a 'spawn_agent' tool that lets subagents spawn further subagents."""
    from attractor_llm.types import Tool

    async def spawn_agent_execute(
        task: str,
        model: str | None = None,
        provider: str | None = None,
        working_dir: str | None = None,
        max_turns: int = 20,
    ) -> str:
        try:
            result = await spawn_subagent(
                client=client,
                prompt=task,
                parent_depth=parent_depth,
                max_depth=max_depth,
                model=model,
                provider=provider,
                max_turns=max_turns,
                abort_signal=abort_signal,
                working_dir=working_dir,
            )
            return result.text
        except MaxDepthError:
            return f"[Cannot delegate: depth limit reached ({parent_depth + 1}/{max_depth})]"
        except SubagentError as e:
            return f"[Delegation failed: {e}]"

    spawn_tool = Tool(
        name="spawn_agent",
        description=(
            f"Spawn a child agent to handle a subtask. The child has its own "
            f"conversation and tools. Current depth: {parent_depth + 1}/{max_depth}. "
            f"Use for tasks that benefit from focused, isolated execution."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override",
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider override",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the child agent",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns for the child session. Default: 20",
                    "default": 20,
                },
            },
            "required": ["task"],
        },
        execute=spawn_agent_execute,
    )

    return [*tools, spawn_tool]
