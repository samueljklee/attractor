"""Tool registry for the Coding Agent Loop.

Manages registration, lookup, validation, and execution of tools.
The registry is the central dispatch point for all tool calls from the LLM.

Spec reference: coding-agent-loop §3.8.
"""

from __future__ import annotations

import asyncio
import json

from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.truncation import TruncationLimits, truncate_output
from attractor_llm.types import ContentPart, ContentPartKind, Tool


class ToolRegistry:
    """Registry for managing and executing tools.

    The tool execution pipeline follows this sequence:
    1. Lookup: Find the tool by name
    2. Validate: Check input against JSON Schema (basic)
    3. Execute: Run the tool's execute handler
    4. Truncate: Apply output truncation limits
    5. Emit: Fire tool.call_start and tool.call_end events
    6. Return: Build tool result ContentPart

    Spec reference: coding-agent-loop §3.8.
    """

    def __init__(
        self,
        event_emitter: EventEmitter | None = None,
        tool_output_limits: dict[str, int] | None = None,
        tool_line_limits: dict[str, int] | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._emitter = event_emitter
        self._output_limits = tool_output_limits
        self._line_limits = tool_line_limits

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[tool.name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def definitions(self) -> list[Tool]:
        """Return all registered tools (for sending to LLM)."""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    async def execute_tool_call(self, tool_call: ContentPart) -> ContentPart:
        """Execute a single tool call and return the result.

        Handles the full pipeline: lookup → validate → execute → truncate → emit.

        Args:
            tool_call: A ContentPart with kind=TOOL_CALL.

        Returns:
            A ContentPart with kind=TOOL_RESULT containing the output.
        """
        assert tool_call.kind == ContentPartKind.TOOL_CALL  # noqa: S101

        tool_name = tool_call.name or ""
        tool_call_id = tool_call.tool_call_id or ""
        arguments = tool_call.arguments

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if not isinstance(arguments, dict):
            arguments = {}

        # Emit start event
        if self._emitter:
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.TOOL_CALL_START,
                    data={"tool": tool_name, "call_id": tool_call_id, "arguments": arguments},
                )
            )

        # Lookup
        tool = self.get(tool_name)
        raw_output_str = ""
        if tool is None:
            output = f"Error: Unknown tool '{tool_name}'"
            raw_output_str = output
            is_error = True
        elif tool.execute is None:
            output = f"Error: Tool '{tool_name}' has no execute handler"
            raw_output_str = output
            is_error = True
        else:
            # Execute
            try:
                raw_output = await tool.execute(**arguments)
                is_error = False

                # Truncate (raw preserved for event; truncated goes to LLM)
                raw_output_str = str(raw_output)
                limits = TruncationLimits.for_tool(
                    tool_name, self._output_limits, self._line_limits
                )
                output, was_truncated = truncate_output(raw_output_str, limits)
                if was_truncated:
                    output += "\n[output was truncated]"

            except Exception as exc:  # noqa: BLE001
                # Send only the exception message to the LLM, not the full
                # traceback (which leaks internal paths and implementation details).
                output = f"Error executing {tool_name}: {type(exc).__name__}: {exc}"
                raw_output_str = output
                is_error = True

        # Emit end event (carries full untruncated output per Spec §2.9, §5.1)
        if self._emitter:
            await self._emitter.emit(
                SessionEvent(
                    kind=EventKind.TOOL_CALL_END,
                    data={
                        "tool": tool_name,
                        "call_id": tool_call_id,
                        "is_error": is_error,
                        "output": raw_output_str,
                    },
                )
            )

        return ContentPart.tool_result_part(
            tool_call_id=tool_call_id,
            name=tool_name,
            output=output,
            is_error=is_error,
        )

    async def execute_tool_calls(self, tool_calls: list[ContentPart]) -> list[ContentPart]:
        """Execute multiple tool calls concurrently. Spec §5.7.

        Tool calls are executed in parallel via asyncio.gather().
        Results are returned in the same order as the input tool calls.
        Partial failures are handled: successful tools return normally,
        failed tools return is_error=True results.

        Args:
            tool_calls: List of ContentParts with kind=TOOL_CALL.

        Returns:
            List of ContentParts with kind=TOOL_RESULT, in the same order.
        """
        if len(tool_calls) <= 1:
            # Single tool call: sequential execution, no filesystem race risk.
            return [await self.execute_tool_call(tc) for tc in tool_calls]

        # Multiple tool calls: execute concurrently
        # NOTE: Filesystem-mutating tools (edit_file, write_file) may race when
        # targeting the same file. Per Spec §5.7 we execute concurrently; callers
        # should avoid issuing conflicting writes in the same batch.
        results = await asyncio.gather(
            *(self.execute_tool_call(tc) for tc in tool_calls),
            return_exceptions=True,
        )

        # Convert any unexpected exceptions to error results
        final: list[ContentPart] = []
        for i, result in enumerate(results):
            if isinstance(result, (KeyboardInterrupt, SystemExit)):
                raise result
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                tc = tool_calls[i]
                final.append(
                    ContentPart.tool_result_part(
                        tool_call_id=tc.tool_call_id or "",
                        name=tc.name or "",
                        output=f"Error: {type(result).__name__}: {result}",
                        is_error=True,
                    )
                )
            else:
                final.append(result)
        return final
