"""High-level generation API for the Unified LLM Client.

Provides convenient functions that wrap Client.complete() and
Client.stream() with automatic tool execution loops, structured
output parsing, and retry handling.

Usage::

    from attractor_llm.generate import generate, stream, generate_object

    # Simple text generation
    text = await generate(client, "claude-sonnet-4-5", "Explain recursion")

    # With automatic tool loop
    text = await generate(client, "claude-sonnet-4-5", "Read config.py",
                          tools=[read_file_tool], max_rounds=5)

    # Streaming
    async for chunk in stream(client, "claude-sonnet-4-5", "Write a poem"):
        print(chunk, end="")

    # Structured output (JSON)
    data = await generate_object(client, "claude-sonnet-4-5",
                                  "Extract entities from: ...",
                                  schema={"type": "object", ...})

Spec reference: unified-llm-spec §4.3-4.6.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from attractor_llm.client import Client
from attractor_llm.streaming import StreamAccumulator, StreamResult
from attractor_llm.types import (
    ContentPart,
    FinishReason,
    GenerateObjectResult,
    GenerateResult,
    Message,
    Request,
    Response,
    StepResult,
    StreamEvent,
    TimeoutConfig,
    Tool,
    Usage,
)


async def generate(
    client: Client,
    model: str,
    prompt: str | None = None,
    *,
    system: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
    tool_choice: str | None = None,
    max_rounds: int = 10,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    abort_signal: Any | None = None,
    timeout: float | TimeoutConfig | None = None,
) -> GenerateResult:
    """Generate text with automatic tool execution loop. Spec §4.3.

    If the model returns tool calls, executes them and feeds results
    back until the model produces a text response or max_rounds is hit.

    Args:
        client: The LLM client with registered adapters.
        model: Model ID (e.g., "claude-sonnet-4-5").
        prompt: User prompt string (mutually exclusive with messages).
        system: Optional system prompt.
        messages: Optional conversation history (mutually exclusive with prompt).
        tools: Optional tools for the model to call.
        max_rounds: Max tool-call rounds before stopping.
        temperature: Optional temperature override.
        reasoning_effort: Optional reasoning effort.
        provider: Optional provider override.
        abort_signal: Optional abort signal; if set, raises AbortError.
        timeout: Optional timeout. A float is treated as a total-timeout in
            seconds. Pass a TimeoutConfig for separate total/per_step control.
            When TimeoutConfig.total is set, the entire generate() call is
            wrapped with asyncio.wait_for(). When TimeoutConfig.per_step is
            set, each individual client.complete() call is wrapped.

    Returns:
        GenerateResult with text, step history, and aggregated token usage.
        Backward-compatible: str(result) returns text, result == "string" works.
    """
    # Normalise timeout. Spec §8.4.10.
    tc: TimeoutConfig | None = None
    if isinstance(timeout, (int, float)):
        tc = TimeoutConfig(total=float(timeout))
    elif isinstance(timeout, TimeoutConfig):
        tc = timeout

    # Spec §4.3: Cannot provide both prompt and messages
    if prompt is not None and messages is not None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Cannot provide both 'prompt' and 'messages'")
    if prompt is None and messages is None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Must provide either 'prompt' or 'messages'")

    if messages is not None:
        history = list(messages)
    else:
        history = [Message.user(prompt)]  # type: ignore[arg-type]

    async def _complete_with_timeout(request: Request) -> Response:
        """Wrap client.complete() with optional per-step timeout."""
        if tc and tc.per_step is not None:
            return await asyncio.wait_for(client.complete(request), timeout=tc.per_step)
        return await client.complete(request)

    async def _core() -> GenerateResult:
        steps: list[StepResult] = []
        total_usage = Usage()
        _tools_list: list[Tool] = tools or []

        async def _exec_one(tc_part: ContentPart) -> tuple[ContentPart, str, bool]:
            """Execute a single tool call, return (tool_call, output, is_error)."""
            tool = _find_tool(_tools_list, tc_part.name or "")
            if tool and tool.execute:
                try:
                    args = tc_part.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as exc:
                            return (
                                tc_part,
                                f"ToolArgError: failed to parse JSON arguments: {exc}",
                                True,
                            )
                    # P9: Validate required fields against tool parameter schema (§8.7)
                    if not isinstance(args, dict):
                        kind = type(args).__name__
                        return (
                            tc_part,
                            f"ToolArgError: arguments must be a JSON object, got {kind}",
                            True,
                        )
                    validation_error = _validate_tool_args(tool, args)
                    if validation_error:
                        return tc_part, validation_error, True
                    raw = await tool.execute(**args)
                    output = str(raw) if not isinstance(raw, str) else raw
                    return tc_part, output, False
                except Exception as exc:  # noqa: BLE001
                    return tc_part, f"{type(exc).__name__}: {exc}", True
            else:
                return tc_part, f"Unknown tool: {tc_part.name}", True

        response: Response | None = None
        for _round in range(max_rounds + 1):
            request = Request(
                model=model,
                provider=provider,
                messages=history,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

            response = await _complete_with_timeout(request)
            total_usage = total_usage + response.usage
            history.append(response.message)

            # Check abort signal after each LLM call
            if abort_signal is not None and abort_signal.is_set:
                from attractor_llm.errors import AbortError

                raise AbortError("Generation aborted by signal")

            # If no tool calls, return text
            if response.finish_reason != FinishReason.TOOL_CALLS:
                steps.append(StepResult(response=response))
                return GenerateResult(
                    text=response.text or "",
                    steps=steps,
                    total_usage=total_usage,
                )

            if not tools:
                steps.append(StepResult(response=response))
                return GenerateResult(
                    text=response.text or "",
                    steps=steps,
                    total_usage=total_usage,
                )

            # Item #6: max_rounds=0 means no automatic tool execution (Spec §4.3)
            if max_rounds == 0:
                steps.append(StepResult(response=response))
                return GenerateResult(
                    text=response.text or "",
                    steps=steps,
                    total_usage=total_usage,
                )

            # Item #5: Passive tools (no execute handler) return to caller (Spec §5.5)
            has_passive = False
            for tool_call in response.tool_calls:
                matched = _find_tool(tools or [], tool_call.name or "")
                if matched is not None and not matched.execute:
                    has_passive = True
                    break
            if has_passive:
                steps.append(StepResult(response=response))
                return GenerateResult(
                    text=response.text or "",
                    steps=steps,
                    total_usage=total_usage,
                )

            # Execute tool calls in parallel (Spec §5.7)
            if len(response.tool_calls) == 1:
                exec_results = [await _exec_one(response.tool_calls[0])]
            else:
                exec_results = await asyncio.gather(
                    *(_exec_one(tc_part) for tc_part in response.tool_calls),
                    return_exceptions=True,
                )
                # Re-raise fatal exceptions; convert others to error results
                final_results: list[tuple[ContentPart, str, bool]] = []
                for i, r in enumerate(exec_results):
                    if isinstance(r, (KeyboardInterrupt, SystemExit)):
                        raise r
                    if isinstance(r, asyncio.CancelledError):
                        raise r
                    if isinstance(r, BaseException):
                        tc_part = response.tool_calls[i]
                        final_results.append((tc_part, f"{type(r).__name__}: {r}", True))
                    else:
                        final_results.append(r)
                exec_results = final_results

            for tc_part, output, is_error in exec_results:
                history.append(
                    Message.tool_result(
                        tc_part.tool_call_id or "",
                        tc_part.name or "",
                        output,
                        is_error=is_error,
                    )
                )

            # Record this step
            tool_results = [
                ContentPart.tool_result_part(
                    tc_part.tool_call_id or "", tc_part.name or "", output, is_error=is_error
                )
                for tc_part, output, is_error in exec_results
            ]
            steps.append(StepResult(response=response, tool_results=tool_results))

        text = "[No response generated]"
        if response is not None:
            text = response.text or "[Max tool rounds reached]"
        return GenerateResult(text=text, steps=steps, total_usage=total_usage)

    # Apply total timeout wrapper if requested. Spec §8.4.10.
    if tc and tc.total is not None:
        return await asyncio.wait_for(_core(), timeout=tc.total)
    return await _core()


async def stream(
    client: Client,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    abort_signal: Any | None = None,
    timeout: float | TimeoutConfig | None = None,
) -> StreamResult:
    """Stream text generation, returning a StreamResult. Spec §4.4.

    This is a simple streaming wrapper -- no tool loop. For tool-using
    streams, use generate() which handles the full loop.

    The returned StreamResult supports multiple consumption patterns:
    - ``async for chunk in result.text_stream:`` for text chunks
    - ``async for event in result:`` for raw StreamEvents
    - ``await result.response()`` for the accumulated Response

    For backward compatibility, ``async for chunk in result:`` also
    yields text chunks (StreamResult implements __aiter__ for str).

    Args:
        client: The LLM client with registered adapters.
        model: Model ID.
        prompt: User prompt.
        system: Optional system prompt.
        temperature: Optional temperature.
        provider: Optional provider override.
        abort_signal: Optional abort signal for cancellation.
        timeout: Optional timeout (float = total seconds, or TimeoutConfig).

    Returns:
        StreamResult wrapping the live event stream.

    Note:
        timeout applies to stream connection setup, not full stream consumption.
        For full-stream timeouts, use abort_signal with an external timer.
    """
    from attractor_llm.streaming import StreamResult

    # Normalise timeout. Spec §8.4.10.
    tc: TimeoutConfig | None = None
    if isinstance(timeout, (int, float)):
        tc = TimeoutConfig(total=float(timeout))
    elif isinstance(timeout, TimeoutConfig):
        tc = timeout

    request = Request(
        model=model,
        provider=provider,
        messages=[Message.user(prompt)],
        system=system,
        temperature=temperature,
    )

    timeout_s = (tc.total if tc.total is not None else tc.per_step) if tc else None
    if timeout_s is not None:
        event_stream = await asyncio.wait_for(
            client.stream(request, abort_signal=abort_signal), timeout=timeout_s
        )
    else:
        event_stream = await client.stream(request, abort_signal=abort_signal)
    return StreamResult(event_stream)


async def stream_with_tools(
    client: Client,
    model: str,
    prompt: str | None = None,
    *,
    system: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
    tool_choice: str | None = None,
    max_rounds: int = 10,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    abort_signal: Any | None = None,
    timeout: float | TimeoutConfig | None = None,
) -> StreamResult:
    """Stream text generation with an automatic tool execution loop. Spec §8.9.22-24.

    Like ``stream()``, but supports agentic tool use:

    1. Streams from the provider, accumulating events via StreamAccumulator.
    2. When the accumulated response has tool calls (finish_reason == TOOL_CALLS),
       executes them and appends tool results to the conversation.
    3. Starts a new stream with the updated history.
    4. Repeats until the model produces a non-tool-call response or
       ``max_rounds`` is exhausted.

    All StreamEvents from every round are yielded in order, so callers see a
    single contiguous event stream covering all turns.

    Args:
        client: The LLM client with registered adapters.
        model: Model ID.
        prompt: User prompt string (mutually exclusive with messages).
        system: Optional system prompt.
        messages: Conversation history (mutually exclusive with prompt).
        tools: Tools the model may call.
        tool_choice: Optional tool-choice override.
        max_rounds: Maximum tool-execution rounds (0 = no tool execution).
        temperature: Optional temperature override.
        reasoning_effort: Optional reasoning effort.
        provider: Optional provider override.
        abort_signal: Optional abort signal for cancellation.
        timeout: Optional timeout (float = total seconds, or TimeoutConfig).

    Returns:
        StreamResult wrapping a multi-turn streaming loop.

    Note:
        timeout applies to stream connection setup per round, not full stream
        consumption. For full-stream timeouts, use abort_signal with an external timer.
    """
    # Normalise timeout. Spec §8.4.10.
    tc: TimeoutConfig | None = None
    if isinstance(timeout, (int, float)):
        tc = TimeoutConfig(total=float(timeout))
    elif isinstance(timeout, TimeoutConfig):
        tc = timeout

    # Validate input
    if prompt is not None and messages is not None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Cannot provide both 'prompt' and 'messages'")
    if prompt is None and messages is None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Must provide either 'prompt' or 'messages'")

    history: list[Message] = (
        list(messages) if messages is not None else [Message.user(prompt)]  # type: ignore[arg-type]
    )

    async def _stream_one(request: Request) -> AsyncIterator[StreamEvent]:
        """Start a stream, wrapping with per_step timeout if configured."""
        if tc and tc.per_step is not None:
            return await asyncio.wait_for(
                client.stream(request, abort_signal=abort_signal), timeout=tc.per_step
            )
        return await client.stream(request, abort_signal=abort_signal)

    async def _loop() -> AsyncIterator[StreamEvent]:
        # Iterate up to max_rounds+1 total LLM calls (mirrors generate() semantics)
        _st_tools: list[Tool] = tools or []

        async def _exec_one_st(tc_part: ContentPart) -> tuple[ContentPart, str, bool]:
            """Execute one tool call in the streaming loop."""
            tool_obj = _find_tool(_st_tools, tc_part.name or "")
            if tool_obj and tool_obj.execute:
                try:
                    args = tc_part.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError as exc:
                            return tc_part, f"ToolArgError: failed to parse JSON: {exc}", True
                    if not isinstance(args, dict):
                        kind = type(args).__name__
                        return (
                            tc_part,
                            f"ToolArgError: arguments must be a JSON object, got {kind}",
                            True,
                        )
                    validation_error = _validate_tool_args(tool_obj, args)
                    if validation_error:
                        return tc_part, validation_error, True
                    raw = await tool_obj.execute(**args)
                    output = str(raw) if not isinstance(raw, str) else raw
                    return tc_part, output, False
                except Exception as exc:  # noqa: BLE001
                    return tc_part, f"{type(exc).__name__}: {exc}", True
            else:
                return tc_part, f"Unknown tool: {tc_part.name}", True

        for _round in range(max_rounds + 1):
            request = Request(
                model=model,
                provider=provider,
                messages=history,
                system=system,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

            event_stream = await _stream_one(request)
            acc = StreamAccumulator()

            async for event in event_stream:
                acc.feed(event)
                yield event

            response = acc.response()
            history.append(response.message)

            # Check abort signal after each LLM call
            if abort_signal is not None and abort_signal.is_set:
                from attractor_llm.errors import AbortError

                raise AbortError("Generation aborted by signal")

            # If no tool calls, or no tools provided, we are done
            if response.finish_reason != FinishReason.TOOL_CALLS or not tools:
                return

            # max_rounds=0 means no automatic tool execution (mirrors generate())
            if max_rounds == 0:
                return

            # §5.5: passive tools (no execute handler) — return control to caller
            has_passive = any(
                (t := _find_tool(tools or [], tc_part.name or "")) is not None and not t.execute
                for tc_part in response.tool_calls
            )
            if has_passive:
                return  # caller handles tool calls

            # Execute tool calls in parallel (§5.7) -- mirrors generate()
            if len(response.tool_calls) == 1:
                exec_results: list[tuple[ContentPart, str, bool]] = [
                    await _exec_one_st(response.tool_calls[0])
                ]
            else:
                gathered = await asyncio.gather(
                    *(_exec_one_st(tc_part) for tc_part in response.tool_calls),
                    return_exceptions=True,
                )
                exec_results = []
                for i, r in enumerate(gathered):
                    if isinstance(r, (KeyboardInterrupt, SystemExit)):
                        raise r
                    if isinstance(r, asyncio.CancelledError):
                        raise r
                    if isinstance(r, BaseException):
                        tc_part = response.tool_calls[i]
                        exec_results.append((tc_part, f"{type(r).__name__}: {r}", True))
                    else:
                        exec_results.append(r)

            # Append tool results to history for next round
            for tc_part, output, is_error in exec_results:
                history.append(
                    Message.tool_result(
                        tc_part.tool_call_id or "",
                        tc_part.name or "",
                        output,
                        is_error=is_error,
                    )
                )

        # max_rounds exhausted without a terminal response -- stop cleanly

    return StreamResult(_loop())


async def generate_object(
    client: Client,
    model: str,
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> GenerateObjectResult:
    """Generate structured JSON output. Spec §8.4.7.

    Instructs the model to respond with JSON matching the given schema.
    For OpenAI and Gemini, uses native response_format for structured output.
    For Anthropic and unknown providers, falls back to prompt injection.

    Args:
        client: The LLM client.
        model: Model ID.
        prompt: User prompt.
        schema: Optional JSON schema for the response.
        system: Optional system prompt.
        temperature: Optional temperature.
        provider: Optional provider override.

    Returns:
        GenerateObjectResult with text (raw JSON), steps, usage, and
        parsed_object (the parsed dict).

    Raises:
        NoObjectGeneratedError: If the response is not valid JSON.
    """
    # §8.4.7: Use native response_format for providers that support it.
    # Anthropic has no native structured-output mode; unknown providers also
    # fall back to prompt injection.
    response_format: dict[str, Any] | None = None
    schema_instruction = ""

    if schema:
        if provider == "openai":
            # OpenAI Responses API: json_schema with nested json_schema object
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        elif provider == "gemini":
            # Gemini: responseMimeType + responseSchema via generationConfig
            response_format = {
                "type": "json_schema",
                "schema": schema,
            }
        else:
            # Anthropic and unknown providers: prompt injection fallback
            schema_instruction = (
                f"\n\nRespond with a JSON object matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                f"Output ONLY the JSON object, no other text."
            )

    full_system = ((system or "") + schema_instruction).strip() or None

    request = Request(
        model=model,
        provider=provider,
        messages=[Message.user(prompt)],
        system=full_system,
        temperature=temperature,
        response_format=response_format,
    )

    response = await client.complete(request)
    text = response.text or ""

    # Try to extract JSON from the response
    # Handle markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fence
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        from attractor_llm.errors import NoObjectGeneratedError

        raise NoObjectGeneratedError(
            f"Response is not valid JSON: {e}\nResponse was: {text[:500]}"
        ) from e

    return GenerateObjectResult(
        text=text,
        steps=[StepResult(response=response)],
        total_usage=response.usage,
        parsed_object=parsed,
    )


def _find_tool(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name in the tools list."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _validate_tool_args(tool: Tool, args: dict[str, Any]) -> str | None:
    """Validate tool call arguments against the tool's parameter schema. Spec §8.7.

    Checks that all fields listed in ``parameters.required`` are present in
    ``args``, and performs basic type checking for top-level declared properties.

    Args:
        tool: The Tool whose ``parameters`` schema to validate against.
        args: Parsed (dict) arguments from the model's tool call.

    Returns:
        An error message string if validation fails, or ``None`` if valid.
    """
    schema = tool.parameters
    if not schema:
        return None  # no schema, nothing to validate

    # Check required fields
    required: list[str] = schema.get("required", [])
    if not isinstance(required, list):
        return None  # malformed schema, skip validation

    missing = [field for field in required if field not in args]
    if missing:
        fields = ", ".join(repr(f) for f in missing)
        return f"ToolArgError: missing required argument(s) {fields} for tool '{tool.name}'"

    # Check types for declared properties
    properties: dict[str, Any] = schema.get("properties", {})
    for key, value in args.items():
        if key not in properties:
            continue  # extra keys are tolerated
        expected_type_name = properties[key].get("type")
        if not expected_type_name or expected_type_name not in _TYPE_MAP:
            continue  # unknown/missing type -- skip
        expected = _TYPE_MAP[expected_type_name]
        # bool is a subclass of int in Python -- reject bools for integer/number
        if expected_type_name in ("integer", "number") and isinstance(value, bool):
            return (
                f"ToolArgError: argument '{key}' expected {expected_type_name}, got boolean"
                f" for tool '{tool.name}'"
            )
        if not isinstance(value, expected):
            return (
                f"ToolArgError: argument '{key}' expected {expected_type_name},"
                f" got {type(value).__name__} for tool '{tool.name}'"
            )

    return None
