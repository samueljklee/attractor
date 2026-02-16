"""Coding Agent Session -- the core agentic loop.

The Session is the central orchestrator for the Coding Agent Loop.
It manages the conversation history, drives the LLM → tool execution
→ steering → loop detection cycle, and emits events throughout.

Spec reference: coding-agent-loop §2.1-2.10.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from attractor_agent.abort import AbortSignal
from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.tools.registry import ToolRegistry
from attractor_llm.client import Client
from attractor_llm.types import (
    Message,
    Request,
    Response,
    Role,
    Tool,
    Usage,
)


class SessionState(StrEnum):
    """Session lifecycle states. Spec §2.3."""

    IDLE = "idle"
    PROCESSING = "processing"
    CLOSED = "closed"


@dataclass
class SteeringTurn:
    """A steering message in conversation history. Spec §2.4, §2.6, §9.6.

    Stored distinctly from plain Message to preserve the steering/meta intent
    in history. When building the LLM request, converted to Message(role=USER).
    """

    content: str
    timestamp: float | None = None


@dataclass
class SessionConfig:
    """Configuration for an agent session. Spec §2.1-2.2.

    Controls model selection, limits, and behavior.
    """

    model: str = "claude-sonnet-4-5"
    provider: str | None = None
    system_prompt: str = ""
    max_turns: int = 50
    max_tool_rounds_per_turn: int = 25
    temperature: float | None = None
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] | None = None

    # Working directory for environment context (None = os.getcwd())
    working_dir: str | None = None

    # Loop detection (None = use defaults: window=4, threshold=3)
    loop_detection_window: int | None = None
    loop_detection_threshold: int | None = None

    # Per-tool truncation overrides (Spec §2.2, §5.2, §5.3)
    tool_output_limits: dict[str, int] | None = None
    tool_line_limits: dict[str, int] | None = None

    # Execution environment: "local" (default) or "docker"
    environment: str = "local"
    docker_image: str = "python:3.12-slim"


@dataclass
class _LoopDetector:
    """Detects stuck tool-call loops. Spec §2.10.

    Tracks recent tool call signatures (name + truncated args) and
    triggers when the same signature appears too many times in the window.
    """

    window: int = 4
    threshold: int = 3
    _recent: list[str] = field(default_factory=list)

    def record(self, tool_name: str, arguments: str | dict[str, Any] | None) -> bool:
        """Record a tool call. Returns True if loop detected."""
        # Create a signature from tool name + first 200 chars of args
        args_str = str(arguments)[:200] if arguments else ""
        sig = f"{tool_name}:{args_str}"

        self._recent.append(sig)
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window :]

        # Check if any signature appears >= threshold times in the window
        from collections import Counter

        counts = Counter(self._recent)
        return any(count >= self.threshold for count in counts.values())

    def reset(self) -> None:
        self._recent.clear()


class Session:
    """Coding Agent Session. Spec §2.1.

    The main entry point for running an autonomous coding agent.
    Manages conversation history, tool execution, steering, and events.

    Usage::

        async with Session(client=client, config=config) as session:
            result = await session.submit("Create a hello.py file")
            print(result)
    """

    def __init__(
        self,
        *,
        client: Client,
        config: SessionConfig | None = None,
        tools: Sequence[Tool] | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> None:
        self._client = client
        self._config = config or SessionConfig()
        self._abort = abort_signal or AbortSignal()
        self._state = SessionState.IDLE
        self._history: list[Message | SteeringTurn] = []
        self._total_usage = Usage()
        self._turn_count = 0

        # Events
        self._emitter = EventEmitter()

        # Tools
        self._tool_registry = ToolRegistry(
            event_emitter=self._emitter,
            tool_output_limits=self._config.tool_output_limits,
            tool_line_limits=self._config.tool_line_limits,
        )
        if tools:
            self._tool_registry.register_many(list(tools))

        # Steering queue: messages injected between tool rounds
        self._steer_queue: list[str] = []

        # Loop detection
        self._loop_detector = _LoopDetector(
            window=self._config.loop_detection_window or 4,
            threshold=self._config.loop_detection_threshold or 3,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def history(self) -> list[Message | SteeringTurn]:
        return list(self._history)

    @property
    def total_usage(self) -> Usage:
        return self._total_usage

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def events(self) -> EventEmitter:
        """Access the event emitter to register handlers."""
        return self._emitter

    @property
    def tool_registry(self) -> ToolRegistry:
        """Access the tool registry to add/remove tools."""
        return self._tool_registry

    def steer(self, message: str) -> None:
        """Inject a steering message into the next tool round. Spec §2.6.

        The message will be added as a user message before the next LLM call,
        allowing mid-turn guidance without interrupting the tool loop.
        """
        self._steer_queue.append(message)

    async def submit(self, prompt: str) -> str:
        """Submit a user prompt and run the agentic loop to completion.

        This is the main entry point. It:
        1. Adds the user message to history
        2. Calls the LLM
        3. Executes any tool calls
        4. Drains steering messages
        5. Checks for loops and limits
        6. Repeats until the model produces a text-only response

        Args:
            prompt: The user's input text.

        Returns:
            The model's final text response.

        Raises:
            RuntimeError: If session is not in IDLE state.
        """
        if self._state == SessionState.CLOSED:
            raise RuntimeError("Session is closed")
        if self._state == SessionState.PROCESSING:
            raise RuntimeError("Session is already processing a request")

        self._state = SessionState.PROCESSING
        self._turn_count += 1

        await self._emitter.emit(
            SessionEvent(
                kind=EventKind.TURN_START,
                data={"turn": self._turn_count, "prompt": prompt[:200]},
            )
        )

        # Add user message to history
        self._history.append(Message.user(prompt))

        try:
            result = await self._run_loop()
        except Exception as exc:
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.ERROR,
                    data={"error": str(exc), "turn": self._turn_count},
                )
            )
            result = f"[Error: {exc}]"
        finally:
            self._state = SessionState.IDLE
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.TURN_END,
                    data={
                        "turn": self._turn_count,
                        "usage": self._total_usage.model_dump(),
                    },
                )
            )

        return result

    async def close(self) -> None:
        """Close the session and release resources."""
        self._state = SessionState.CLOSED
        await self._emitter.emit(SessionEvent(kind=EventKind.SESSION_END))

    async def __aenter__(self) -> Session:
        await self._emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Core agentic loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self) -> str:
        """The core agentic loop. Spec §2.5.

        LLM call → tool execution → steering drain → loop detection → repeat
        until the model produces a text-only response or a limit is hit.
        """
        tool_round = 0

        while True:
            # Check abort
            if self._abort.is_set:
                return "[Session aborted]"

            # Check turn limit
            if self._turn_count > self._config.max_turns:
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.LIMIT_REACHED,
                        data={"limit": "max_turns", "value": self._config.max_turns},
                    )
                )
                return "[Turn limit reached]"

            # Check tool round limit
            if tool_round >= self._config.max_tool_rounds_per_turn:
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.LIMIT_REACHED,
                        data={
                            "limit": "max_tool_rounds",
                            "value": self._config.max_tool_rounds_per_turn,
                        },
                    )
                )
                return "[Tool round limit reached]"

            # Build and send LLM request
            response = await self._call_llm()

            # Track usage
            self._total_usage = self._total_usage + response.usage

            # Add assistant message to history
            self._history.append(response.message)

            # Check if the model wants to call tools
            tool_calls = response.tool_calls
            if tool_calls:
                tool_round += 1

                # Loop detection BEFORE execution (avoid wasted work)
                for tc in tool_calls:
                    if self._loop_detector.record(tc.name or "", tc.arguments):
                        warning = f"Loop detected: {tc.name} called repeatedly"
                        self._history.append(SteeringTurn(content=warning))
                        await self._emitter.emit(
                            SessionEvent(
                                kind=EventKind.LOOP_DETECTED,
                                data={"tool": tc.name},
                            )
                        )
                        return "[Loop detected: repeated tool calls]"

                # Execute all tool calls
                results = await self._tool_registry.execute_tool_calls(tool_calls)

                # Add tool results to history
                for result in results:
                    self._history.append(
                        Message(
                            role=Role.TOOL,
                            content=[result],
                        )
                    )

                # Drain steering queue
                await self._drain_steering()

                # Continue the loop for another LLM call
                continue

            # No tool calls -- model produced a text response
            text = response.text or ""

            if text:
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.ASSISTANT_TEXT,
                        data={"text": text[:500]},
                    )
                )

            # Reset loop detector on successful text completion
            self._loop_detector.reset()

            return text

    def _build_messages(self) -> list[Message]:
        """Convert history entries to Message list for the LLM.

        SteeringTurn entries are converted to Message(role=USER) with a
        [STEERING] prefix. The LLM sees them as user messages but the
        history preserves the distinction.
        """
        messages: list[Message] = []
        for entry in self._history:
            if isinstance(entry, SteeringTurn):
                messages.append(Message.user(f"[STEERING] {entry.content}"))
            else:
                messages.append(entry)
        return messages

    async def _call_llm(self) -> Response:
        """Build a request from current history and call the LLM."""
        tools = self._tool_registry.definitions()

        # Build enriched system prompt with environment context
        # and project documentation (Spec §6.3-6.5)
        system = self._build_enriched_system_prompt()

        request = Request(
            model=self._config.model,
            provider=self._config.provider,
            messages=self._build_messages(),
            system=system or None,
            tools=tools if tools else None,
            temperature=self._config.temperature,
            reasoning_effort=self._config.reasoning_effort,
            provider_options=self._config.provider_options,
        )

        return await self._client.complete(request)

    def _build_enriched_system_prompt(self) -> str:
        """Compose the system prompt with environment context and project docs.

        Layers (in order, per spec §6.1):
        1. Base system prompt from config (provider-specific instructions)
        2. ``<environment>`` block with runtime context
        3. ``<project_instructions>`` from discovered docs

        Spec reference: coding-agent-loop-spec §6.3-6.5.
        """
        from attractor_agent.env_context import (
            build_environment_context,
            get_git_context,
        )
        from attractor_agent.project_docs import discover_project_docs

        working_dir = self._config.working_dir or os.getcwd()

        # Fetch git context once and share between env block and project docs
        git_info = get_git_context(working_dir)
        git_root = str(git_info.get("git_root", "")) or None

        parts: list[str] = []

        # 1. Base system prompt (provider-specific instructions)
        if self._config.system_prompt:
            parts.append(self._config.system_prompt)

        # 2. Environment context
        # TODO: wire knowledge_cutoff from model metadata (catalog) when available
        env_block = build_environment_context(
            working_dir=working_dir,
            model=self._config.model,
            git_info=git_info,
        )
        parts.append(env_block)

        # 3. Project documentation (appended)
        project_docs = discover_project_docs(
            working_dir=working_dir,
            provider_id=self._config.provider,
            git_root=git_root,
        )
        if project_docs:
            parts.append(project_docs)

        return "\n\n".join(parts)

    async def _drain_steering(self) -> None:
        """Inject any queued steering messages into history. Spec §2.6."""
        while self._steer_queue:
            msg = self._steer_queue.pop(0)
            self._history.append(SteeringTurn(content=msg))
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.STEER_INJECTED,
                    data={"message": msg[:200]},
                )
            )
