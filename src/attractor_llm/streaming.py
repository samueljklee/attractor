"""Stream accumulator and StreamResult for building responses from streaming events.

Collects StreamEvents and assembles them into a complete Response.
StreamResult wraps a live event stream with convenient access patterns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

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
            case StreamEventKind.START | StreamEventKind.STREAM_START:
                # §8.4.5: Accept both legacy START and canonical STREAM_START.
                # Adapters now emit STREAM_START; StreamAccumulator handles both
                # so that older producers and newer producers work without any
                # migration window.
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


class StreamResult:
    """Result of a streaming generation. Spec §4.4.

    Wraps a live stream of events and provides convenient access patterns:

    - Async iterate for raw StreamEvents: ``async for event in result: ...``
    - Use ``text_stream`` for text-only chunks: ``async for chunk in result.text_stream: ...``
    - Call ``response()`` to consume remaining events and get the final Response.

    The underlying stream can only be consumed once. Use one iteration pattern,
    then call ``response()`` to get the accumulated result.

    Usage::

        result = await stream(client, model, prompt)

        # Pattern 1: text chunks
        async for chunk in result.text_stream:
            print(chunk, end="")
        resp = await result.response()

        # Pattern 2: raw events
        async for event in result:
            handle(event)
        resp = await result.response()

        # Pattern 3: just get the response
        resp = await result.response()
    """

    def __init__(self, event_stream: AsyncIterator[StreamEvent]) -> None:
        self._event_stream = event_stream
        self._accumulator = StreamAccumulator()
        self._consumed = False

    async def response(self) -> Response:
        """Consume any remaining stream events and return the accumulated Response."""
        if not self._consumed:
            async for event in self._event_stream:
                self._accumulator.feed(event)
            self._consumed = True
        return self._accumulator.response()

    @property
    def text_stream(self) -> AsyncIterator[str]:
        """Async iterator yielding text chunks as they arrive from the provider."""
        return self._text_stream_impl()

    async def _text_stream_impl(self) -> AsyncIterator[str]:
        async for event in self._event_stream:
            self._accumulator.feed(event)
            if event.kind == StreamEventKind.TEXT_DELTA and event.text:
                yield event.text
        self._consumed = True

    def __aiter__(self) -> AsyncIterator[str]:
        """Backward-compat: iterating a StreamResult yields text chunks.

        This allows ``async for chunk in result:`` to work the same as
        ``async for chunk in result.text_stream:``, preserving the old
        pattern where ``stream()`` returned an ``AsyncIterator[str]``.
        """
        return self._text_stream_impl()

    async def iter_events(self) -> AsyncIterator[StreamEvent]:
        """Iterate raw StreamEvents (for advanced consumers)."""
        async for event in self._event_stream:
            self._accumulator.feed(event)
            yield event
        self._consumed = True
