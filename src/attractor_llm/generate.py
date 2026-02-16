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
from attractor_llm.types import (
    ContentPart,
    FinishReason,
    GenerateResult,
    Message,
    Request,
    Response,
    StepResult,
    StreamEventKind,
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
    max_rounds: int = 10,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
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

    Returns:
        GenerateResult with text, step history, and aggregated token usage.
        Backward-compatible: str(result) returns text, result == "string" works.
    """
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

    steps: list[StepResult] = []
    total_usage = Usage()

    response: Response | None = None
    for _round in range(max_rounds + 1):
        request = Request(
            model=model,
            provider=provider,
            messages=history,
            system=system,
            tools=tools,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        response = await client.complete(request)
        total_usage = total_usage + response.usage
        history.append(response.message)

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

        # Execute tool calls in parallel (Spec §5.7)
        async def _exec_one(tc: ContentPart) -> tuple[ContentPart, str, bool]:
            """Execute a single tool call, return (tool_call, output, is_error)."""
            tool = _find_tool(tools, tc.name or "")
            if tool and tool.execute:
                try:
                    args = tc.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    raw = await tool.execute(**args)
                    output = str(raw) if not isinstance(raw, str) else raw
                    return tc, output, False
                except Exception as exc:  # noqa: BLE001
                    return tc, f"{type(exc).__name__}: {exc}", True
            else:
                return tc, f"Unknown tool: {tc.name}", True

        if len(response.tool_calls) == 1:
            exec_results = [await _exec_one(response.tool_calls[0])]
        else:
            exec_results = await asyncio.gather(
                *(_exec_one(tc) for tc in response.tool_calls),
                return_exceptions=True,
            )
            # Convert any unexpected BaseExceptions to error results
            final_results: list[tuple[ContentPart, str, bool]] = []
            for i, r in enumerate(exec_results):
                if isinstance(r, BaseException):
                    tc = response.tool_calls[i]
                    final_results.append(
                        (tc, f"{type(r).__name__}: {r}", True)
                    )
                else:
                    final_results.append(r)
            exec_results = final_results

        for tc, output, is_error in exec_results:
            history.append(
                Message.tool_result(
                    tc.tool_call_id or "",
                    tc.name or "",
                    output,
                    is_error=is_error,
                )
            )

        # Record this step
        tool_results = [
            ContentPart.tool_result_part(
                tc.tool_call_id or "", tc.name or "", output, is_error=is_error
            )
            for tc, output, is_error in exec_results
        ]
        steps.append(StepResult(response=response, tool_results=tool_results))

    text = "[No response generated]"
    if response is not None:
        text = response.text or "[Max tool rounds reached]"
    return GenerateResult(text=text, steps=steps, total_usage=total_usage)


async def stream(
    client: Client,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> AsyncIterator[str]:
    """Stream text generation, yielding chunks as they arrive.

    This is a simple streaming wrapper -- no tool loop. For tool-using
    streams, use generate() which handles the full loop.

    Args:
        client: The LLM client with registered adapters.
        model: Model ID.
        prompt: User prompt.
        system: Optional system prompt.
        temperature: Optional temperature.
        provider: Optional provider override.

    Yields:
        Text chunks as they arrive from the provider.
    """
    request = Request(
        model=model,
        provider=provider,
        messages=[Message.user(prompt)],
        system=system,
        temperature=temperature,
    )

    event_stream = await client.stream(request)
    async for event in event_stream:
        if event.kind == StreamEventKind.TEXT_DELTA and event.text:
            yield event.text


async def generate_object(
    client: Client,
    model: str,
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Generate structured JSON output.

    Instructs the model to respond with JSON matching the given schema.
    Parses and returns the JSON object.

    Args:
        client: The LLM client.
        model: Model ID.
        prompt: User prompt.
        schema: Optional JSON schema for the response.
        system: Optional system prompt.
        temperature: Optional temperature.
        provider: Optional provider override.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    schema_instruction = ""
    if schema:
        schema_instruction = (
            f"\n\nRespond with a JSON object matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```\n"
            f"Output ONLY the JSON object, no other text."
        )

    full_system = (system or "") + schema_instruction

    request = Request(
        model=model,
        provider=provider,
        messages=[Message.user(prompt)],
        system=full_system.strip() or None,
        temperature=temperature,
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
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Response is not valid JSON: {e}\nResponse was: {text[:500]}") from e


def _find_tool(tools: list[Tool], name: str) -> Tool | None:
    """Find a tool by name in the tools list."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None
