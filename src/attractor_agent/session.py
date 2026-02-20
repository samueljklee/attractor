"""Coding Agent Session -- the core agentic loop.

The Session is the central orchestrator for the Coding Agent Loop.
It manages the conversation history, drives the LLM → tool execution
→ steering → loop detection cycle, and emits events throughout.

Spec reference: coding-agent-loop §2.1-2.10.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import signal as _signal
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from attractor_agent.abort import AbortSignal
from attractor_agent.environment import ExecutionEnvironment
from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.tools.core import set_environment, set_max_command_timeout

# ProviderProfile lives in profiles.base which imports SessionConfig from this
# module -- guard with TYPE_CHECKING to break the circular import.
# `from __future__ import annotations` ensures the annotation in __init__ is
# never evaluated at runtime, so this is fully safe.
if TYPE_CHECKING:
    from attractor_agent.profiles.base import ProviderProfile
from attractor_agent.tools.registry import ToolRegistry
from attractor_llm.catalog import get_model_info
from attractor_llm.client import Client
from attractor_llm.errors import AccessDeniedError, AuthenticationError
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

    # User-supplied instruction override -- appended LAST (layer 5, §9.8)
    user_instructions: str = ""

    # Working directory for environment context (None = os.getcwd())
    working_dir: str | None = None

    # Loop detection (None = use defaults: window=4, threshold=3)
    loop_detection_window: int | None = None
    loop_detection_threshold: int | None = None

    # Per-tool truncation overrides (Spec §2.2, §5.2, §5.3)
    tool_output_limits: dict[str, int] | None = None
    tool_line_limits: dict[str, int] | None = None

    # Shell command timeout ceiling (Spec §2.2)
    max_command_timeout_ms: int = 600_000  # 10 minutes

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
        profile: ProviderProfile | None = None,
        environment: ExecutionEnvironment | None = None,
    ) -> None:
        """Initialise a Session.

        Note: environment= sets a process-wide singleton. Concurrent sessions
        with different environments will interfere with each other.
        """
        self._client = client
        self._config = config or SessionConfig()
        self._abort = abort_signal or AbortSignal()
        self._state = SessionState.IDLE
        self._history: list[Message | SteeringTurn] = []
        self._total_usage = Usage()
        self._turn_count = 0

        # P11: If a ProviderProfile is supplied, extract system prompt and tools.
        # profile.apply_to_config() fills missing config fields (system_prompt, model…)
        # then profile.get_tools() filters/augments the base tool list. Spec §9.1.
        if profile is not None:
            self._config = profile.apply_to_config(dataclasses.replace(self._config))
            # Merge profile-provided tools with any explicitly passed tools list.
            base_tools: list[Tool] = list(tools) if tools else []
            tools = profile.get_tools(base_tools)

        # Wire subagent tools with real client (§9.12.34-36)
        # Done here -- not in profiles -- so the tools close over the real client.
        from attractor_agent.subagent_manager import SubagentManager, create_interactive_tools

        manager = SubagentManager()
        subagent_tools = create_interactive_tools(manager, client=self._client)
        tools_list: list[Tool] = list(tools) if tools else []
        existing_names = {t.name for t in tools_list}
        for t in subagent_tools:
            if t.name not in existing_names:
                tools_list.append(t)
                existing_names.add(t.name)
        tools = tools_list

        # P11: If an ExecutionEnvironment is supplied, install it as the
        # module-level environment used by all tool implementations. Spec §9.1.
        if environment is not None:
            set_environment(environment)

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

        # Wire config timeout ceiling to the shell tool's clamping logic
        set_max_command_timeout(self._config.max_command_timeout_ms)

        # Steering queue: messages injected between tool rounds
        self._steer_queue: list[str] = []

        # Follow-up queue: messages processed after main loop completes (Spec §9.6)
        self._followup_queue: list[str] = []

        # Tracked subagent tasks for safe cleanup (Spec Appendix B)
        self._subagent_tasks: set[asyncio.Task[object]] = set()

        # TODO: Populated when streaming LLM tasks are created (§9.11 Step 1).
        # Currently scaffolding -- no production code populates this set yet.
        # The cleanup mechanism works but is inert until client.stream() is wired.
        self._active_tasks: set[asyncio.Task[object]] = set()

        # Tracked OS-level processes for graceful shutdown (§9.1.6, §9.11.5).
        # Populated by callers that spawn asyncio subprocesses and want them
        # cleaned up on abort. Full integration requires env protocol changes
        # (see _cleanup_on_abort Steps 2-4 comments).
        self._tracked_processes: list[asyncio.subprocess.Process] = []

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

    @property
    def reasoning_effort(self) -> str | None:
        """Current reasoning_effort setting. Spec §2.7."""
        return self._config.reasoning_effort

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Change reasoning_effort for subsequent LLM calls. Spec §2.7.

        The change takes effect on the next LLM call; in-flight requests are
        not affected.
        """
        self._config.reasoning_effort = effort

    def steer(self, message: str) -> None:
        """Inject a steering message into the next tool round. Spec §2.6.

        The message will be added as a user message before the next LLM call,
        allowing mid-turn guidance without interrupting the tool loop.
        """
        self._steer_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message for after the current loop completes. Spec §2.6, §9.6.

        The message will be processed as a new user input after the current
        submit() call's main loop returns a text response, triggering a new
        processing cycle.
        """
        self._followup_queue.append(message)

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

        _auth_error = False
        try:
            result = await self._run_loop()

            # Spec §9.6: process follow-up messages after main loop completes.
            # Each follow-up is treated like a recursive process_input() call:
            # emit TURN_START/USER_INPUT, drain steering, then run the loop.
            while self._followup_queue:
                msg = self._followup_queue.pop(0)
                self._turn_count += 1
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.USER_INPUT,
                        data={"turn": self._turn_count, "content": msg},
                    )
                )
                self._history.append(Message.user(msg))
                await self._drain_steering()
                result = await self._run_loop()
        except (AuthenticationError, AccessDeniedError) as exc:
            # Spec §9.11: auth errors (401/403) surface immediately → CLOSED.
            # The session must not accept new inputs after an auth failure.
            # Flag must be set BEFORE emit so the finally block sees it even
            # if emit() itself raises.
            _auth_error = True
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.ERROR,
                    data={"error": str(exc), "turn": self._turn_count, "auth_error": True},
                )
            )
            result = f"[Authentication Error: {exc}]"
        except Exception as exc:
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.ERROR,
                    data={"error": str(exc), "turn": self._turn_count},
                )
            )
            result = f"[Error: {exc}]"
        finally:
            # Spec §9.1: abort transitions to CLOSED (not IDLE).
            # Spec §9.11: auth errors also transition to CLOSED.
            if self._abort.is_set:
                await self._cleanup_on_abort()
                self._state = SessionState.CLOSED
            elif _auth_error:
                await self._cleanup_on_abort()
                self._state = SessionState.CLOSED
            else:
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
            if self._abort.is_set:
                await self._emitter.emit(SessionEvent(kind=EventKind.SESSION_END))

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
                        kind=EventKind.TURN_LIMIT,
                        data={"limit": "max_turns", "value": self._config.max_turns},
                    )
                )
                return "[Turn limit reached]"

            # Check tool round limit
            if tool_round >= self._config.max_tool_rounds_per_turn:
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.TURN_LIMIT,
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

                # Loop detection AFTER execution, per spec §2.5 step 8.
                # Spec §2.10: inject a SteeringTurn warning and CONTINUE the
                # loop -- do not exit -- so the model can self-correct.
                for tc in tool_calls:
                    if self._loop_detector.record(tc.name or "", tc.arguments):
                        warning = (
                            "[LOOP DETECTED] The conversation appears to be "
                            "repeating. Please try a different approach."
                        )
                        self._history.append(SteeringTurn(content=warning))
                        await self._emitter.emit(
                            SessionEvent(
                                kind=EventKind.LOOP_DETECTION,
                                data={"tool": tc.name},
                            )
                        )
                        # Reset so we don't fire again immediately, then
                        # let the model see the steering message.
                        self._loop_detector.reset()
                        break

                # Continue the loop for another LLM call
                continue

            # No tool calls -- model produced a text response
            text = response.text or ""

            if text:
                # Emit TEXT_START to signal the assistant has begun responding (§9.10.1)
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.ASSISTANT_TEXT_START,
                        data={"turn": self._turn_count},
                    )
                )
                # Emit TEXT_DELTA with the full text (single emission; not streaming) (§9.10.1)
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.ASSISTANT_TEXT_DELTA,
                        data={"delta": text},
                    )
                )
                # Emit TEXT_END with the complete assembled text (§9.10.1)
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.ASSISTANT_TEXT_END,
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

    def _estimate_history_tokens(self) -> int:
        """Rough token count estimate (~4 chars per token). Spec §9.11."""
        total_chars = len(self._config.system_prompt) if self._config.system_prompt else 0
        for entry in self._history:
            # SteeringTurn.content is a plain str -- count it directly.
            if isinstance(entry, SteeringTurn):
                total_chars += len(entry.content)
                continue
            # Message.content is list[ContentPart].
            for part in entry.content:
                if part.text:
                    total_chars += len(part.text)
                if part.output:
                    total_chars += len(part.output)
                if part.arguments:
                    total_chars += len(str(part.arguments))
        return total_chars // 4

    async def _call_llm(self) -> Response:
        """Build a request from current history and call the LLM."""
        # Spec §9.11: context window overflow detection
        model_info = get_model_info(self._config.model)
        if model_info:
            estimated_tokens = self._estimate_history_tokens()
            threshold = int(model_info.context_window * 0.8)
            if estimated_tokens > threshold:
                await self._emitter.emit(
                    SessionEvent(
                        kind=EventKind.CONTEXT_WINDOW_WARNING,
                        data={
                            "estimated_tokens": estimated_tokens,
                            "context_window": model_info.context_window,
                            "threshold_pct": 80,
                        },
                    )
                )

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

        return await self._client.complete(request, abort_signal=self._abort)

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

        # 4. Available tools -- list each tool's name and description (§9.8.3)
        tool_defs = self._tool_registry.definitions()
        if tool_defs:
            tool_lines = ["<available_tools>"]
            for tool in tool_defs:
                tool_lines.append(f"- {tool.name}: {tool.description}")
            tool_lines.append("</available_tools>")
            parts.append("\n".join(tool_lines))

        # 5. User instruction overrides -- appended LAST (layer 5, §9.8)
        if self._config.user_instructions:
            parts.append(self._config.user_instructions)

        return "\n\n".join(parts)

    async def _drain_steering(self) -> None:
        """Inject any queued steering messages into history. Spec §2.6."""
        while self._steer_queue:
            msg = self._steer_queue.pop(0)
            self._history.append(SteeringTurn(content=msg))
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.STEERING_INJECTED,
                    data={"message": msg[:200]},
                )
            )

    async def _cleanup_on_abort(self) -> None:
        """Best-effort resource cleanup on abort. Spec Appendix B.

        Implements the 8-step shutdown sequence:
          Step 1 (scaffolding): Cancel tracked asyncio tasks (LLM stream / active tasks).
          Step 2 (TODO): Send SIGTERM to running shell processes.
          Step 3 (TODO): Wait up to 2 seconds for processes to exit.
          Step 4 (TODO): Send SIGKILL to remaining processes.
          Step 5 (done): Flush pending work queues.
          Step 6 (done): Cancel and clear subagent tasks.
          Step 7 (done): Flush events -- caller emits TURN_END / SESSION_END.
          Step 8 (done): Transition to CLOSED -- handled by submit() finally block.

        Steps 2-4 (process SIGTERM/SIGKILL) require the ExecutionEnvironment to
        expose a process registry.  The LocalEnvironment runs each shell command
        in its own subprocess via asyncio.to_thread; individual PIDs are not
        accessible through the ExecutionEnvironment protocol.  Those steps are
        documented as TODOs pending an env.kill_all_processes() method.
        """
        # --- Step 1: Cancel tracked active tasks (e.g. LLM stream coroutines). ---
        # Tasks are added to _active_tasks when created via asyncio.create_task().
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        if self._active_tasks:
            # Await cancellations so they complete before we continue.
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

        # --- Step 2: SIGTERM running tracked processes (§9.1.6, §9.11.5). ---
        # _tracked_processes holds asyncio.subprocess.Process objects registered
        # by callers.  Full automatic tracking requires env protocol changes
        # (e.g. an env.kill_all_processes() method on LocalEnvironment).
        _alive: list[asyncio.subprocess.Process] = []
        for proc in self._tracked_processes:
            if proc.returncode is None:  # still running
                try:
                    proc.send_signal(_signal.SIGTERM)
                    _alive.append(proc)
                except (ProcessLookupError, PermissionError):
                    pass  # already exited or insufficient permissions

        # --- Step 3: Wait up to 2 seconds for processes to exit. ---
        if _alive:
            _done, _pending = await asyncio.wait(
                [asyncio.create_task(proc.wait()) for proc in _alive],
                timeout=2.0,
            )
            # Cancel the wait tasks for any that didn't finish in time
            for t in _pending:
                t.cancel()

        # --- Step 4: SIGKILL remaining processes. ---
        for proc in self._tracked_processes:
            if proc.returncode is None:  # still running after SIGTERM + wait
                try:
                    proc.send_signal(_signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # already exited or insufficient permissions
        self._tracked_processes.clear()

        # --- Step 5: Drain and discard pending work queues. ---
        self._steer_queue.clear()
        self._followup_queue.clear()

        # --- Step 6: Cancel subagent tasks. ---
        for task in self._subagent_tasks:
            if not task.done():
                task.cancel()
        if self._subagent_tasks:
            await asyncio.gather(*self._subagent_tasks, return_exceptions=True)
        self._subagent_tasks.clear()
        # TODO: SubagentManager integration — when subagents are spawned through
        # Session, register their tasks in self._subagent_tasks for cleanup.
