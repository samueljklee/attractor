"""OpenAI Responses API adapter.

Translates unified Request/Response types to/from the OpenAI Responses API.
Uses the native Responses API (/v1/responses) per spec §2.7, NOT the older
Chat Completions API. The Responses API properly surfaces reasoning tokens,
supports built-in tools, and is OpenAI's forward-looking API.

Key OpenAI-specific behaviors:
- Uses "input" items instead of "messages" (§7.2)
- Reasoning tokens via output_tokens_details (§3.9)
- reasoning.effort parameter for thinking depth (§3.9)
- Automatic prompt caching (no client-side injection needed, §2.10)
- SSE with response.* event types (§7.7)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from attractor_llm.errors import classify_http_error
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

from .base import ProviderConfig

DEFAULT_BASE_URL = "https://api.openai.com"


class OpenAIAdapter:
    """OpenAI Responses API adapter. Spec §7.3.

    Implements ProviderAdapter protocol using OpenAI's native Responses API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                **config.default_headers,
            },
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    # ------------------------------------------------------------------ #
    # Request translation (Unified -> OpenAI Responses API)
    # ------------------------------------------------------------------ #

    def _translate_request(self, request: Request) -> dict[str, Any]:
        """Translate unified Request to OpenAI Responses API body. Spec §7.2."""
        messages = request.effective_messages()
        input_items = self._translate_input_items(messages)

        body: dict[str, Any] = {
            "model": request.model,
            "input": input_items,
        }

        # Max tokens
        if request.max_tokens is not None:
            body["max_output_tokens"] = request.max_tokens

        # Temperature
        if request.temperature is not None:
            body["temperature"] = request.temperature

        # Top P
        if request.top_p is not None:
            body["top_p"] = request.top_p

        # Note: Responses API does not support `seed`. Callers who need
        # deterministic output should use provider_options["openai"]["seed"]
        # with Chat Completions via the openai_compat adapter (not this one).

        # Tools
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
            ]

        # Tool choice
        if request.tool_choice:
            if request.tool_choice in ("auto", "none", "required"):
                body["tool_choice"] = request.tool_choice
            else:
                body["tool_choice"] = {
                    "type": "function",
                    "name": request.tool_choice,
                }

        # Reasoning effort (maps to reasoning.effort in Responses API)
        if request.reasoning_effort:
            body["reasoning"] = {"effort": request.reasoning_effort}

        # Response format (Responses API uses text.format, not response_format)
        if request.response_format:
            body["text"] = {"format": request.response_format}

        # Provider-specific options
        openai_opts = (request.provider_options or {}).get("openai", {})
        for key, value in openai_opts.items():
            if key not in body:  # don't override explicit fields
                body[key] = value

        return body

    def _translate_input_items(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate unified messages to Responses API input items.

        The Responses API uses a different message format than Chat Completions:
        - System messages become {"role": "system", "content": "..."}
        - User messages become {"role": "user", "content": [...]}
        - Assistant messages become {"role": "assistant", "content": [...]}
        - Tool results become {"type": "function_call_output", ...}
        """
        items: list[dict[str, Any]] = []

        for msg in messages:
            match msg.role:
                case Role.SYSTEM:
                    text = msg.text or ""
                    items.append(
                        {
                            "role": "system",
                            "content": text,
                        }
                    )

                case Role.DEVELOPER:
                    text = msg.text or ""
                    items.append(
                        {
                            "role": "developer",
                            "content": text,
                        }
                    )

                case Role.USER:
                    content = self._translate_user_content(msg)
                    items.append(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )

                case Role.ASSISTANT:
                    # Assistant messages may contain text and/or tool calls
                    for part in msg.content:
                        match part.kind:
                            case ContentPartKind.TEXT:
                                items.append(
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [
                                            {"type": "output_text", "text": part.text or ""}
                                        ],
                                    }
                                )
                            case ContentPartKind.TOOL_CALL:
                                args = part.arguments
                                if isinstance(args, dict):
                                    args = json.dumps(args)
                                resolved_id = part.tool_call_id or str(uuid.uuid4())
                                # Responses API requires function_call id to start with "fc_"
                                fc_id = (
                                    resolved_id
                                    if resolved_id.startswith("fc_")
                                    else f"fc_{resolved_id}"
                                )
                                items.append(
                                    {
                                        "type": "function_call",
                                        "id": fc_id,
                                        "call_id": resolved_id,
                                        "name": part.name or "",
                                        "arguments": args or "{}",
                                    }
                                )
                            case ContentPartKind.THINKING:
                                # OpenAI doesn't expose reasoning text; skip
                                pass
                            case _:
                                pass

                case Role.TOOL:
                    for part in msg.content:
                        if part.kind == ContentPartKind.TOOL_RESULT:
                            items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": part.tool_call_id or "",
                                    "output": part.output or "",
                                }
                            )

        return items

    def _translate_user_content(self, msg: Message) -> str | list[dict[str, Any]]:
        """Translate user message content. Simple text -> string, multi-part -> array."""
        if len(msg.content) == 1 and msg.content[0].kind == ContentPartKind.TEXT:
            return msg.content[0].text or ""

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            match part.kind:
                case ContentPartKind.TEXT:
                    parts.append({"type": "input_text", "text": part.text or ""})
                case ContentPartKind.IMAGE:
                    image = part.image
                    if image:
                        from attractor_llm.adapters.image_utils import resolve_image_data

                        image = resolve_image_data(image)
                    if image and image.url:
                        parts.append(
                            {
                                "type": "input_image",
                                "image_url": image.url,
                            }
                        )
                    elif image and image.data:
                        import base64

                        data_uri = (
                            f"data:{image.media_type};base64,"
                            f"{base64.b64encode(image.data).decode()}"
                        )
                        parts.append(
                            {
                                "type": "input_image",
                                "image_url": data_uri,
                            }
                        )
                case _:
                    parts.append({"type": "input_text", "text": f"[unsupported: {part.kind}]"})

        return parts

    # ------------------------------------------------------------------ #
    # Response translation (OpenAI -> Unified)
    # ------------------------------------------------------------------ #

    def _translate_response(self, data: dict[str, Any], request: Request) -> Response:
        """Translate OpenAI Responses API response to unified Response."""
        content_parts: list[ContentPart] = []

        # Responses API returns output as a list of items
        for item in data.get("output", []):
            item_type = item.get("type")

            match item_type:
                case "message":
                    # Message items contain content array
                    for content_item in item.get("content", []):
                        if content_item.get("type") == "output_text":
                            text = content_item.get("text", "")
                            if text:
                                content_parts.append(ContentPart.text_part(text))

                case "function_call":
                    content_parts.append(
                        ContentPart.tool_call_part(
                            tool_call_id=item.get("call_id", item.get("id", "")),
                            name=item.get("name", ""),
                            arguments=item.get("arguments", "{}"),
                        )
                    )

                case "reasoning":
                    # Reasoning summary (OpenAI doesn't expose full thinking text)
                    for summary in item.get("summary", []):
                        if summary.get("type") == "summary_text":
                            content_parts.append(
                                ContentPart.thinking_part(
                                    text=summary.get("text", ""),
                                )
                            )

        # Map finish reason -- detect tool calls from content inspection
        # since Responses API uses "completed" for both text and tool-call completions
        status = data.get("status", "completed")
        has_tool_calls = any(p.kind == ContentPartKind.TOOL_CALL for p in content_parts)
        if status == "completed" and has_tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        else:
            finish_reason = self._map_finish_reason(status, data)

        # Parse usage
        usage_data = data.get("usage", {})
        output_details = usage_data.get("output_tokens_details", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            reasoning_tokens=output_details.get("reasoning_tokens", 0),
            # OpenAI caching is automatic; tokens reported in input_tokens_details
            cache_read_tokens=usage_data.get("input_tokens_details", {}).get("cached_tokens", 0),
        )

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider="openai",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw=data,
        )

    def _map_finish_reason(self, status: str, data: dict[str, Any] | None = None) -> FinishReason:
        """Map OpenAI Responses API status to unified FinishReason."""
        match status:
            case "completed":
                return FinishReason.STOP
            case "incomplete":
                # Distinguish content-filter truncation from max-tokens
                if data:
                    reason = data.get("incomplete_details", {}).get("reason", "")
                    if reason == "content_filter":
                        return FinishReason.CONTENT_FILTER
                return FinishReason.MAX_TOKENS
            case "failed":
                return FinishReason.ERROR
            case "cancelled":
                return FinishReason.ERROR
            case _:
                return FinishReason.STOP

    # ------------------------------------------------------------------ #
    # complete() -- blocking call
    # ------------------------------------------------------------------ #

    async def complete(self, request: Request) -> Response:
        """Send a request and return the complete response."""
        body = self._translate_request(request)

        http_response = await self._client.post("/v1/responses", json=body)

        if http_response.status_code != 200:
            headers_dict = dict(http_response.headers)
            raise classify_http_error(
                http_response.status_code,
                http_response.text,
                "openai",
                headers=headers_dict,
            )

        data = http_response.json()
        return self._translate_response(data, request)

    # ------------------------------------------------------------------ #
    # stream() -- SSE streaming
    # ------------------------------------------------------------------ #

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and yield streaming events.

        OpenAI Responses API streaming uses events like:
        - response.created: stream start with metadata
        - response.output_item.added: new output item
        - response.content_part.added: new content part
        - response.output_text.delta: text chunk
        - response.function_call_arguments.delta: tool args chunk
        - response.completed: stream done with full response
        """
        body = self._translate_request(request)
        body["stream"] = True

        async with self._client.stream("POST", "/v1/responses", json=body) as http_response:
            if http_response.status_code != 200:
                error_body = await http_response.aread()
                headers_dict = dict(http_response.headers)
                raise classify_http_error(
                    http_response.status_code,
                    error_body.decode("utf-8", errors="replace"),
                    "openai",
                    headers=headers_dict,
                )

            async for event in self._parse_sse(http_response, request):
                yield event

    async def _parse_sse(
        self, http_response: httpx.Response, request: Request
    ) -> AsyncIterator[StreamEvent]:
        """Parse OpenAI Responses API SSE stream into unified StreamEvents."""
        event_type: str | None = None

        # Track current function call for delta accumulation
        current_fc_id: str | None = None
        current_fc_name: str | None = None
        has_seen_tool_call: bool = False

        async for line in http_response.aiter_lines():
            line = line.strip()

            if line.startswith("event:"):
                event_type = line[6:].strip()
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event_type is None:
                    continue

                async for ev in self._handle_sse_event(
                    event_type,
                    data,
                    request,
                    current_fc_id,
                    current_fc_name,
                    has_seen_tool_call,
                ):
                    yield ev

                # Track function call state
                if event_type == "response.output_item.added":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        has_seen_tool_call = True
                        current_fc_id = item.get("call_id", item.get("id"))
                        current_fc_name = item.get("name")
                elif event_type == "response.output_item.done":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        current_fc_id = None
                        current_fc_name = None

                event_type = None

    async def _handle_sse_event(  # noqa: C901, PLR0912
        self,
        event_type: str,
        data: dict[str, Any],
        request: Request,
        fc_id: str | None,
        fc_name: str | None,
        has_seen_tool_call: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Convert a single OpenAI SSE event to unified StreamEvents."""
        match event_type:
            case "response.created":
                resp = data.get("response", data)
                yield StreamEvent(
                    kind=StreamEventKind.START,
                    model=resp.get("model", request.model),
                    response_id=resp.get("id", ""),
                    provider="openai",
                )

            case "response.output_text.delta":
                yield StreamEvent(
                    kind=StreamEventKind.TEXT_DELTA,
                    text=data.get("delta", ""),
                )

            case "response.output_item.added":
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_CALL_START,
                        tool_call_id=item.get("call_id", item.get("id", "")),
                        tool_name=item.get("name", ""),
                    )

            case "response.function_call_arguments.delta":
                yield StreamEvent(
                    kind=StreamEventKind.TOOL_CALL_DELTA,
                    tool_call_id=fc_id,
                    arguments_delta=data.get("delta", ""),
                )

            case "response.output_item.done":
                item = data.get("item", {})
                if item.get("type") == "function_call":
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_CALL_END,
                        tool_call_id=item.get("call_id", item.get("id")),
                    )

            case "response.completed":
                resp = data.get("response", data)
                # Extract usage
                usage_data = resp.get("usage", {})
                output_details = usage_data.get("output_tokens_details", {})
                if usage_data:
                    yield StreamEvent(
                        kind=StreamEventKind.USAGE,
                        usage=Usage(
                            input_tokens=usage_data.get("input_tokens", 0),
                            output_tokens=usage_data.get("output_tokens", 0),
                            reasoning_tokens=output_details.get("reasoning_tokens", 0),
                            cache_read_tokens=usage_data.get("input_tokens_details", {}).get(
                                "cached_tokens", 0
                            ),
                        ),
                    )

                status = resp.get("status", "completed")
                fr = self._map_finish_reason(status, resp)
                if fr == FinishReason.STOP and has_seen_tool_call:
                    fr = FinishReason.TOOL_CALLS
                yield StreamEvent(
                    kind=StreamEventKind.FINISH,
                    finish_reason=fr,
                )

            case "error":
                err = data.get("error", {})
                yield StreamEvent(
                    kind=StreamEventKind.ERROR,
                    error=err.get("message", str(data)),
                )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
