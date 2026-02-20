"""Interactive subagent management for multi-turn agent communication.

Enables a parent agent session to spawn subagents that run in the
background and communicate with them interactively via send_input,
wait, and close_agent tools.

Unlike the fire-and-forget model in subagent.py, interactive subagents
run as background asyncio tasks and support mid-execution message
injection via the Session.steer() mechanism.

Spec reference: coding-agent-loop-spec §2.8, §9.9.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_agent.profiles import get_profile
from attractor_agent.session import Session, SessionConfig
from attractor_agent.subagent import MaxDepthError
from attractor_agent.tools.core import ALL_CORE_TOOLS
from attractor_llm.types import Tool

# ------------------------------------------------------------------ #
# Tracked subagent record
# ------------------------------------------------------------------ #


@dataclass
class TrackedSubagent:
    """A subagent running as a background asyncio task."""

    agent_id: str
    session: Session
    task: asyncio.Task[str]
    abort_signal: AbortSignal


# ------------------------------------------------------------------ #
# Manager
# ------------------------------------------------------------------ #


class SubagentManager:
    """Manages interactive subagents that run in the background.

    Tracks running subagents by ID and provides operations for
    inter-agent communication: send_input, wait_for_output, close_agent.

    Usage::

        manager = SubagentManager()
        agent_id = await manager.spawn(client, "Refactor auth module")
        manager.send_input(agent_id, "Also update the tests")
        result = await manager.wait_for_output(agent_id)
    """

    def __init__(self) -> None:
        self._agents: dict[str, TrackedSubagent] = {}

    @property
    def active_agents(self) -> dict[str, TrackedSubagent]:
        """Snapshot of currently tracked agents."""
        return dict(self._agents)

    async def spawn(
        self,
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
        working_dir: str | None = None,
    ) -> str:
        """Spawn an interactive subagent as a background task.

        The subagent shares the parent's ExecutionEnvironment (same
        filesystem) per spec §7.1. The environment is a module-level
        singleton in ``tools.core``, so all sessions in the same process
        automatically share it. The ``working_dir`` parameter propagates
        the parent's working directory so the child operates in the same
        filesystem context.

        Returns the agent_id for use with send_input / wait_for_output /
        close_agent.

        Raises:
            MaxDepthError: If parent_depth >= max_depth.
        """
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            raise MaxDepthError(
                f"Subagent depth limit exceeded: depth {child_depth} > max {max_depth}"
            )

        # Resolve profile and build config
        profile = get_profile(provider or "")
        config = SessionConfig(
            model=model or "",
            provider=provider,
            system_prompt=system_prompt or "",
            max_turns=max_turns,
            max_tool_rounds_per_turn=max_tool_rounds,
            working_dir=working_dir,
        )
        config = profile.apply_to_config(config)

        depth_info = (
            f"\n\n[SUBAGENT] You are an interactive subagent at depth "
            f"{child_depth}/{max_depth}. Focus on the delegated task. Be concise."
        )
        config.system_prompt = (config.system_prompt or "") + depth_info

        # Build tools and session
        tools = list(ALL_CORE_TOOLS) if include_tools else []
        agent_abort = abort_signal or AbortSignal()
        session = Session(
            client=client,
            config=config,
            tools=tools,
            abort_signal=agent_abort,
        )

        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(
            session.submit(prompt),
            name=f"subagent-{agent_id}",
        )

        self._agents[agent_id] = TrackedSubagent(
            agent_id=agent_id,
            session=session,
            task=task,
            abort_signal=agent_abort,
        )
        return agent_id

    # ------------------------------------------------------------------ #
    # Interactive operations
    # ------------------------------------------------------------------ #

    def send_input(self, agent_id: str, message: str) -> str:
        """Send a message to a running subagent via steering.

        The message is injected into the subagent's conversation loop
        using Session.steer(), appearing as a user directive before
        the next LLM call.
        """
        tracked = self._agents.get(agent_id)
        if tracked is None:
            return f"Error: No agent found with ID '{agent_id}'"

        if tracked.task.done():
            return f"Error: Agent '{agent_id}' has already completed"

        tracked.session.steer(message)
        return f"Message sent to agent '{agent_id}'"

    async def wait_for_output(self, agent_id: str) -> str:
        """Wait for a subagent to complete and return its output.

        Returns a JSON-serialized SubAgentResult dict per spec §7.3:
        ``{"output": "...", "success": true/false, "turns_used": N}``

        The agent is removed from tracking after this call returns.
        """
        import json

        tracked = self._agents.get(agent_id)
        if tracked is None:
            return json.dumps(
                {
                    "output": f"Error: No agent found with ID '{agent_id}'",
                    "success": False,
                    "turns_used": 0,
                }
            )

        success = True
        turns_used = 0
        try:
            result = await tracked.task
            turns_used = tracked.session.turn_count
        except Exception as exc:
            result = f"Error: Agent '{agent_id}' failed: {type(exc).__name__}: {exc}"
            success = False
            turns_used = tracked.session.turn_count
        finally:
            self._agents.pop(agent_id, None)

        return json.dumps({"output": result, "success": success, "turns_used": turns_used})

    def close_agent(self, agent_id: str) -> str:
        """Terminate a running subagent by setting its abort signal.

        The subagent will exit cooperatively at the next loop-iteration
        check.  It is removed from tracking immediately.
        """
        tracked = self._agents.get(agent_id)
        if tracked is None:
            return f"Error: No agent found with ID '{agent_id}'"

        tracked.abort_signal.set()
        self._agents.pop(agent_id, None)
        return f"Agent '{agent_id}' terminated"


# ------------------------------------------------------------------ #
# Tool definitions
# ------------------------------------------------------------------ #


def create_interactive_tools(manager: SubagentManager, client: Any = None) -> list[Tool]:
    """Create the four interactive subagent Tool objects.

    The returned tools close over *manager* (and optionally *client*) so
    they can be registered on any ToolRegistry or Session.

    Args:
        manager: The SubagentManager instance to delegate operations to.
        client: Optional LLM client used by spawn_agent to create subagents.
                If None, spawn_agent returns an informative error message.
    """

    async def _spawn_agent(
        prompt: str,
        model: str = "",
        provider: str = "",
        system_prompt: str = "",
        max_turns: int = 20,
    ) -> str:
        if client is None:
            return (
                "Error: spawn_agent requires an LLM client. "
                "Pass client= when calling create_interactive_tools(), "
                "or use SubagentManager.spawn() directly."
            )
        return await manager.spawn(
            client,
            prompt,
            model=model or None,
            provider=provider or None,
            system_prompt=system_prompt or None,
            max_turns=max_turns,
        )

    async def _send_input(agent_id: str, message: str) -> str:
        return manager.send_input(agent_id, message)

    async def _wait(agent_id: str) -> str:
        return await manager.wait_for_output(agent_id)

    async def _close_agent(agent_id: str) -> str:
        return manager.close_agent(agent_id)

    spawn_agent_tool = Tool(
        name="spawn_agent",
        description=(
            "Spawn an interactive subagent as a background task. "
            "Returns an agent_id used with send_input, wait, and "
            "close_agent to communicate with the subagent. "
            "The subagent runs autonomously on the given prompt."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The initial prompt / task for the subagent",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override (e.g. 'claude-sonnet-4-5')",
                },
                "provider": {
                    "type": "string",
                    "description": "Optional provider override (e.g. 'anthropic', 'openai')",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt override for the subagent",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum turns the subagent may take (default 20)",
                },
            },
            "required": ["prompt"],
        },
        execute=_spawn_agent,
    )

    send_input_tool = Tool(
        name="send_input",
        description=(
            "Send a message to a running interactive subagent. "
            "The message is injected as a steering directive into "
            "the subagent's conversation loop."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the target subagent",
                },
                "message": {
                    "type": "string",
                    "description": "The message to send to the subagent",
                },
            },
            "required": ["agent_id", "message"],
        },
        execute=_send_input,
    )

    wait_tool = Tool(
        name="wait",
        description=(
            "Wait for an interactive subagent to complete and return "
            "its final output. The agent is removed from tracking "
            "after this call."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the subagent to wait for",
                },
            },
            "required": ["agent_id"],
        },
        execute=_wait,
    )

    close_agent_tool = Tool(
        name="close_agent",
        description=(
            "Terminate a running interactive subagent. Sets the abort "
            "signal to cooperatively cancel the subagent's execution."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the subagent to terminate",
                },
            },
            "required": ["agent_id"],
        },
        execute=_close_agent,
    )

    return [spawn_agent_tool, send_input_tool, wait_tool, close_agent_tool]
