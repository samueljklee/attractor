"""Stream accumulator for building responses from streaming events.

Collects StreamEvents and assembles them into a complete Response.
"""

from __future__ import annotations

from attractor_llm.types import (
    ContentPart,
    FinishReason,
    Message,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)


class _ToolCallBuilder:
    """Accumulates deltas for a single tool call."""

    def __init__(self, tool_call_id: str, name: str) -> None:
        self.tool_call_id = tool_call_id
        self.name = name
        self.arguments_chunks: list[str] = []

    def feed_delta(self, delta: str) -> None:
        self.arguments_chunks.append(delta)

    def build(self) -> ContentPart:
        return ContentPart.tool_call_part(
            tool_call_id=self.tool_call_id,
            name=self.name,
            arguments="".join(self.arguments_chunks),
        )


class StreamAccumulator:
    """Accumulates StreamEvents into a complete Response.

    Usage::

        acc = StreamAccumulator()
        async for event in stream:
            acc.feed(event)
        response = acc.response()
    """

    def __init__(self) -> None:
        self._text_chunks: list[str] = []
        self._thinking_chunks: list[str] = []
        self._thinking_signature: str | None = None
        self._tool_builders: dict[str, _ToolCallBuilder] = {}
        self._usage: Usage = Usage()
        self._finish_reason: FinishReason = FinishReason.STOP
        self._model: str = ""
        self._response_id: str = ""
        self._provider: str = ""
        self._error: str | None = None
        self._started: bool = False

    def feed(self, event: StreamEvent) -> None:
        """Process a single stream event."""
        match event.kind:
            case StreamEventKind.START:
                self._started = True
                if event.model:
                    self._model = event.model
                if event.response_id:
                    self._response_id = event.response_id
                if event.provider:
                    self._provider = event.provider

            case StreamEventKind.TEXT_DELTA:
                if event.text:
                    self._text_chunks.append(event.text)

            case StreamEventKind.THINKING_DELTA:
                if event.text:
                    self._thinking_chunks.append(event.text)
                if event.thinking_signature:
                    self._thinking_signature = event.thinking_signature

            case StreamEventKind.TOOL_CALL_START:
                if event.tool_call_id and event.tool_name:
                    self._tool_builders[event.tool_call_id] = _ToolCallBuilder(
                        tool_call_id=event.tool_call_id,
                        name=event.tool_name,
                    )

            case StreamEventKind.TOOL_CALL_DELTA:
                if event.tool_call_id and event.arguments_delta:
                    builder = self._tool_builders.get(event.tool_call_id)
                    if builder:
                        builder.feed_delta(event.arguments_delta)

            case StreamEventKind.TOOL_CALL_END:
                pass  # Builder already has all data

            case StreamEventKind.USAGE:
                if event.usage:
                    self._usage = self._usage + event.usage

            case StreamEventKind.FINISH:
                if event.finish_reason:
                    self._finish_reason = event.finish_reason

            case StreamEventKind.ERROR:
                self._finish_reason = FinishReason.ERROR
                self._error = event.error

    def response(self) -> Response:
        """Build the final Response from accumulated events."""
        content: list[ContentPart] = []
        warnings: list[str] = []

        if self._thinking_chunks:
            content.append(
                ContentPart.thinking_part(
                    text="".join(self._thinking_chunks),
                    signature=self._thinking_signature,
                )
            )

        if self._text_chunks:
            content.append(ContentPart.text_part("".join(self._text_chunks)))

        for builder in self._tool_builders.values():
            content.append(builder.build())

        if self._error:
            warnings.append(f"Stream error: {self._error}")

        return Response(
            id=self._response_id or "stream",
            model=self._model or "unknown",
            provider=self._provider or "unknown",
            message=Message(role=Role.ASSISTANT, content=content),
            finish_reason=self._finish_reason,
            usage=self._usage,
            warnings=warnings,
        )

    @property
    def started(self) -> bool:
        """Whether a START event has been received."""
        return self._started
