"""OpenAI-compatible adapter for local and proxy LLM servers.

Connects to any server implementing the OpenAI Chat Completions API
(the /v1/chat/completions endpoint). This covers:

- Ollama (with OpenAI compatibility mode)
- vLLM
- LiteLLM proxy
- LocalAI
- llama.cpp server
- Any other OpenAI-compatible endpoint

Unlike the OpenAI adapter (which uses the Responses API), this uses
the standard Chat Completions API format that most local servers implement.

Usage::

    from attractor_llm.adapters.openai_compat import OpenAICompatAdapter
    from attractor_llm.adapters.base import ProviderConfig

    # Ollama
    adapter = OpenAICompatAdapter(ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't check keys
    ))

    # vLLM
    adapter = OpenAICompatAdapter(ProviderConfig(
        base_url="http://localhost:8000/v1",
        api_key="token-xyz",
    ))

    client.register_adapter("local", adapter)

Spec reference: unified-llm-spec S7.10.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.errors import (
    InvalidRequestError,
    classify_http_error,
)
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)


class OpenAICompatAdapter:
    """Adapter for OpenAI-compatible Chat Completions API servers.

    Uses the standard /v1/chat/completions format (not the Responses API).
    Compatible with Ollama, vLLM, LiteLLM, LocalAI, llama.cpp, etc.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        base_url = (config.base_url or "http://localhost:11434/v1").rstrip("/")
        self._endpoint = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.api_key or ''}",
            "Content-Type": "application/json",
        }
        # Merge user-provided default_headers from config
        if hasattr(config, "default_headers") and config.default_headers:
            headers.update(config.default_headers)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout if config.timeout is not None else 120.0),
            headers=headers,
        )

    @property
    def provider_name(self) -> str:
        return "openai-compat"

    async def complete(self, request: Request) -> Response:
        """Send a chat completion request."""
        body = self._build_request_body(request)

        resp = await self._client.post(self._endpoint, json=body)

        if resp.status_code != 200:
            error_text = resp.text
            try:
                error_data = resp.json()
                error_text = error_data.get("error", {}).get("message", resp.text)
            except Exception:  # noqa: BLE001
                pass
            raise classify_http_error(
                resp.status_code,
                error_text,
                "openai-compat",
                headers=dict(resp.headers),
            )

        return self._parse_response(resp.json(), request)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion response via SSE."""
        body = self._build_request_body(request)
        body["stream"] = True

        async with self._client.stream("POST", self._endpoint, json=body) as resp:
            if resp.status_code != 200:
                error_text = await resp.aread()
                raise classify_http_error(
                    resp.status_code,
                    error_text.decode("utf-8", errors="replace"),
                    "openai-compat",
                )

            yield StreamEvent(
                kind=StreamEventKind.START,
                model=request.model,
                provider="openai-compat",
            )

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")

                # Text delta
                if content := delta.get("content"):
                    yield StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=content)

                # Tool calls
                if tool_calls := delta.get("tool_calls"):
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name"):
                            yield StreamEvent(
                                kind=StreamEventKind.TOOL_CALL_START,
                                tool_call_id=tc.get("id", ""),
                                tool_name=tc["function"]["name"],
                            )
                        if args := tc.get("function", {}).get("arguments"):
                            yield StreamEvent(
                                kind=StreamEventKind.TOOL_CALL_DELTA,
                                tool_call_id=tc.get("id", ""),
                                arguments_delta=args,
                            )
                        # Emit TOOL_CALL_END to match other adapters' contract
                        if finish == "tool_calls" or not tc.get("function", {}).get("arguments"):
                            yield StreamEvent(
                                kind=StreamEventKind.TOOL_CALL_END,
                                tool_call_id=tc.get("id", ""),
                            )

                if finish:
                    yield StreamEvent(
                        kind=StreamEventKind.FINISH,
                        finish_reason=self._map_finish_reason(finish),
                    )

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Request building
    # ------------------------------------------------------------------ #

    def _build_request_body(self, request: Request) -> dict[str, Any]:
        """Build a Chat Completions API request body."""
        messages: list[dict[str, Any]] = []

        # System message
        if request.system:
            messages.append(
                {
                    "role": "system",
                    "content": request.system,
                }
            )

        # Conversation messages
        for msg in request.messages:
            match msg.role:
                case Role.USER:
                    for part in msg.content:
                        if part.kind == ContentPartKind.AUDIO:
                            raise InvalidRequestError(
                                "OpenAI-compat does not support audio content input",
                                provider="openai-compat",
                            )
                        if part.kind == ContentPartKind.DOCUMENT:
                            raise InvalidRequestError(
                                "OpenAI-compat does not support document content input",
                                provider="openai-compat",
                            )
                    messages.append(
                        {
                            "role": "user",
                            "content": msg.text or "",
                        }
                    )

                case Role.ASSISTANT:
                    assistant_msg: dict[str, Any] = {"role": "assistant"}
                    # Check for tool calls
                    tool_calls = [p for p in msg.content if p.kind == ContentPartKind.TOOL_CALL]
                    if tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.tool_call_id or "",
                                "type": "function",
                                "function": {
                                    "name": tc.name or "",
                                    "arguments": (
                                        json.dumps(tc.arguments)
                                        if isinstance(tc.arguments, dict)
                                        else tc.arguments or "{}"
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ]
                        # Include text content if present alongside tool calls
                        text = msg.text
                        assistant_msg["content"] = text or ""
                    else:
                        assistant_msg["content"] = msg.text or ""
                    messages.append(assistant_msg)

                case Role.TOOL:
                    for part in msg.content:
                        if part.kind == ContentPartKind.TOOL_RESULT:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": part.tool_call_id or "",
                                    "content": part.output or "",
                                }
                            )

        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.max_tokens:
            body["max_tokens"] = request.max_tokens

        # Tools
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]

        return body

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _parse_response(self, data: dict[str, Any], request: Request) -> Response:
        """Parse a Chat Completions API response."""
        choices = data.get("choices", [])
        if not choices:
            raise InvalidRequestError("No choices in response", provider="openai-compat")

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason_str = choice.get("finish_reason", "stop")

        # Build content parts
        content_parts: list[ContentPart] = []

        # Text content
        if text := message.get("content"):
            content_parts.append(ContentPart.text_part(text))

        # Tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            content_parts.append(
                ContentPart.tool_call_part(
                    tool_call_id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                )
            )

        # Determine finish reason
        has_tool_calls = len(tool_calls) > 0
        if has_tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        else:
            finish_reason = self._map_finish_reason(finish_reason_str)

        # Parse usage
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        )

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider="openai-compat",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _map_finish_reason(reason: str) -> FinishReason:
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.MAX_TOKENS,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason, FinishReason.STOP)
