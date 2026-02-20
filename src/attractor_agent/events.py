"""Event system for the Coding Agent Loop.

Defines the 13 event kinds covering the full session lifecycle,
plus the event emitter used by Session to notify observers.

Spec reference: coding-agent-loop §2.9.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    """Session event types. Spec §2.9, §9.10."""

    # Session lifecycle
    SESSION_START = "session.start"
    SESSION_END = "session.end"

    # Turn lifecycle
    TURN_START = "turn.start"
    TURN_END = "turn.end"

    # User input (follow-up messages processed after main loop)
    USER_INPUT = "user.input"

    # Assistant output
    ASSISTANT_TEXT_START = "assistant.text_start"  # stream opening
    ASSISTANT_TEXT_END = "assistant.text_end"  # final assembled text (was ASSISTANT_TEXT)
    ASSISTANT_TEXT_DELTA = "assistant.text_delta"  # incremental chunk

    # Tool execution
    TOOL_CALL_START = "tool.call_start"
    TOOL_CALL_END = "tool.call_end"
    TOOL_CALL_OUTPUT_DELTA = "tool.call_output_delta"  # incremental tool output chunk

    # Steering
    STEERING_INJECTED = "steering.injected"  # was STEER_INJECTED

    # Limits and errors
    TURN_LIMIT = "turn.limit"  # was LIMIT_REACHED
    LOOP_DETECTION = "loop.detection"  # was LOOP_DETECTED
    CONTEXT_WINDOW_WARNING = "context_window.warning"
    ERROR = "error"


@dataclass
class SessionEvent:
    """A single event emitted by the agent session."""

    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)


# Callback type for event handlers
EventHandler = Callable[[SessionEvent], Awaitable[None] | None]


class EventEmitter:
    """Simple event emitter for session events.

    Supports both sync and async handlers. Handlers are called
    in registration order. Exceptions in handlers are caught and
    logged (they don't break the session loop).

    Also supports an async iterator interface via the ``events()`` method,
    backed by an ``asyncio.Queue``. Call ``close()`` to signal termination
    and unblock any pending ``async for`` loops.
    """

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []
        self._queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()

    def on(self, handler: EventHandler) -> None:
        """Register an event handler."""
        self._handlers.append(handler)

    def off(self, handler: EventHandler) -> None:
        """Remove an event handler."""
        self._handlers = [h for h in self._handlers if h is not handler]

    async def emit(self, event: SessionEvent) -> None:
        """Emit an event to all registered handlers and the async queue."""
        # Put into the async-iterator queue first (non-blocking; queue is unbounded).
        await self._queue.put(event)
        # Then notify callback-style handlers.
        for handler in self._handlers:
            try:
                result = handler(event)
                if isinstance(result, Awaitable):
                    await result
            except Exception:  # noqa: BLE001
                # Don't let handler errors break the session loop.
                # In production, this should log the exception.
                pass

    async def close(self) -> None:
        """Signal termination to any ``async for`` consumers.

        Puts a sentinel ``None`` onto the queue so that the async generator
        returned by ``events()`` exits cleanly. Safe to call multiple times.

        Note: Unblocks one pending events() consumer. Multiple concurrent
        consumers are not supported.
        """
        await self._queue.put(None)

    async def events(self) -> AsyncGenerator[SessionEvent, None]:
        """Async generator that yields events as they are emitted.

        Usage::

            async for event in session.events.events():
                print(event.kind)

        The generator exits when ``close()`` is called (or when a ``None``
        sentinel is received from the queue).

        Note: Only one concurrent consumer is supported. If multiple coroutines
        call events(), each event goes to only one of them (queue semantics).
        The close() sentinel only unblocks one consumer.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                # Sentinel received -- stop iteration.
                break
            yield item
