"""Anthropic Messages API adapter.

Translates unified Request/Response types to/from the Anthropic Messages API.
Uses the native HTTP API directly (not the Anthropic SDK) per spec §2.7.

Key Anthropic-specific behaviors:
- Strict user/assistant message alternation (§7.3)
- Automatic cache_control injection for prompt caching (§2.10)
- Extended thinking with signature round-tripping (§3.9)
- SSE streaming with content_block_start/delta/stop events (§7.7)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

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

from .base import ProviderConfig

# Anthropic API constants
DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192


class AnthropicAdapter:
    """Anthropic Messages API adapter. Spec §7.3.

    Implements ProviderAdapter protocol using Anthropic's native Messages API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")

        # Use AdapterTimeout if provided, else fall back to legacy timeout. Spec §8.4.10
        at = config.adapter_timeout
        if at:
            timeout = httpx.Timeout(at.request, connect=at.connect, read=at.stream_read)
        else:
            timeout = httpx.Timeout(config.timeout, connect=10.0)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
                **config.default_headers,
            },
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    # ------------------------------------------------------------------ #
    # Request translation (Unified -> Anthropic)
    # ------------------------------------------------------------------ #

    def _translate_request(self, request: Request) -> dict[str, Any]:
        """Translate unified Request to Anthropic Messages API body. Spec §7.2."""
        messages = request.effective_messages()
        system_parts, conversation = self._split_system(messages)
        anthropic_messages = self._translate_messages(conversation)

        # Enforce alternation: merge consecutive same-role messages
        anthropic_messages = self._enforce_alternation(anthropic_messages)

        body: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }

        # System message (Anthropic uses top-level "system" field, not in messages)
        if system_parts:
            body["system"] = system_parts

        # Tools
        if request.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
            ]

        # Tool choice
        if request.tool_choice:
            if request.tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif request.tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                # Anthropic does not support tool_choice={type: none} when
                # tools are present.  Omit the tools array instead (§8.7).
                body.pop("tools", None)
            else:
                body["tool_choice"] = {"type": "tool", "name": request.tool_choice}

        # Temperature
        if request.temperature is not None:
            body["temperature"] = request.temperature

        # Top P
        if request.top_p is not None:
            body["top_p"] = request.top_p

        # Stop sequences
        if request.stop:
            body["stop_sequences"] = request.stop

        # Extended thinking (via provider_options or reasoning_effort)
        anthropic_opts = (request.provider_options or {}).get("anthropic", {})
        if request.reasoning_effort or anthropic_opts.get("thinking"):
            thinking_config = anthropic_opts.get("thinking", {})
            budget = thinking_config.get(
                "budget_tokens", self._thinking_budget(request.reasoning_effort)
            )
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # Thinking requires removing temperature
            body.pop("temperature", None)

        # Beta headers
        beta_headers = anthropic_opts.get("beta_headers", [])
        if beta_headers:
            # Will be set on the request headers, not body
            body["_beta_headers"] = beta_headers

        # Apply cache_control for prompt caching (§2.10)
        # Can be disabled via provider_options.anthropic.auto_cache = false (Spec §8.6.6)
        if anthropic_opts.get("auto_cache", True):
            self._inject_cache_control(body)

        return body

    def _split_system(self, messages: list[Message]) -> tuple[list[dict[str, Any]], list[Message]]:
        """Extract system messages into Anthropic's top-level system field."""
        system_parts: list[dict[str, Any]] = []
        conversation: list[Message] = []

        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                for part in msg.content:
                    if part.kind == ContentPartKind.TEXT and part.text:
                        system_parts.append({"type": "text", "text": part.text})
            else:
                conversation.append(msg)

        return system_parts, conversation

    def _translate_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate unified messages to Anthropic format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            role = "user" if msg.role in (Role.USER, Role.TOOL) else "assistant"
            content: list[dict[str, Any]] = []

            for part in msg.content:
                content.append(self._translate_content_part(part, msg.role))

            if content:
                result.append({"role": role, "content": content})

        return result

    def _translate_content_part(self, part: ContentPart, msg_role: Role) -> dict[str, Any]:
        """Translate a single content part to Anthropic format."""
        match part.kind:
            case ContentPartKind.TEXT:
                return {"type": "text", "text": part.text or ""}

            case ContentPartKind.IMAGE:
                image = part.image
                if image:
                    from attractor_llm.adapters.image_utils import resolve_image_data

                    image = resolve_image_data(image)
                if image and image.data:
                    import base64

                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.media_type,
                            "data": base64.b64encode(image.data).decode(),
                        },
                    }
                elif image and image.url:
                    return {
                        "type": "image",
                        "source": {"type": "url", "url": image.url},
                    }
                return {"type": "text", "text": "[image data missing]"}

            case ContentPartKind.TOOL_CALL:
                args = part.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                return {
                    "type": "tool_use",
                    "id": part.tool_call_id or str(uuid.uuid4()),
                    "name": part.name or "",
                    "input": args or {},
                }

            case ContentPartKind.TOOL_RESULT:
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id or "",
                    "content": part.output or "",
                }
                if part.is_error:
                    result["is_error"] = True
                return result

            case ContentPartKind.THINKING:
                return {
                    "type": "thinking",
                    "thinking": part.text or "",
                    **({"signature": part.signature} if part.signature else {}),
                }

            case ContentPartKind.REDACTED_THINKING:
                return {
                    "type": "redacted_thinking",
                    "data": part.redacted_data or "",
                }

            case ContentPartKind.DOCUMENT:
                if not part.document:
                    raise InvalidRequestError(
                        "DOCUMENT content part has no document payload", provider="anthropic"
                    )
                if part.document.data:
                    import base64

                    return {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": part.document.media_type or "application/pdf",
                            "data": base64.b64encode(part.document.data).decode(),
                        },
                    }
                raise InvalidRequestError(
                    "Anthropic requires document data as base64;"
                    " URL-only documents are not supported",
                    provider="anthropic",
                )

            case ContentPartKind.AUDIO:
                raise InvalidRequestError(
                    "Anthropic does not support audio content input",
                    provider="anthropic",
                )

            case _:
                raise InvalidRequestError(
                    f"Unsupported content part kind for Anthropic: {part.kind}",
                    provider="anthropic",
                )

    def _enforce_alternation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages for strict alternation.

        Anthropic requires strict user/assistant alternation. If we have
        consecutive messages with the same role, merge their content arrays.
        """
        if not messages:
            return messages

        # Copy first message to avoid mutating the original content list
        merged: list[dict[str, Any]] = [{**messages[0], "content": list(messages[0]["content"])}]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Same role -- merge content into the copy
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append(msg)

        # Anthropic requires first message to be user role
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": [{"type": "text", "text": "..."}]})

        return merged

    def _inject_cache_control(self, body: dict[str, Any]) -> None:
        """Auto-inject cache_control breakpoints for prompt caching. Spec §2.10.

        Anthropic prompt caching requires explicit cache_control markers.
        We inject them at three strategic positions:
        1. End of system message (stable across turns)
        2. End of tool definitions (stable across turns)
        3. Last user message (changes each turn -- ephemeral cache)

        This is the "single highest-ROI optimization for agentic workloads"
        per the spec.
        """
        # 1. Cache system message -- persistent (stable across turns)
        if "system" in body and body["system"]:
            body["system"][-1]["cache_control"] = {"type": "ephemeral"}

        # 2. Cache tool definitions -- persistent (stable across turns)
        if "tools" in body and body["tools"]:
            body["tools"][-1]["cache_control"] = {"type": "ephemeral"}

        # 3. Cache last user message -- ephemeral (changes each turn)
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

        # Auto-add prompt-caching beta header (Spec §2.10)
        beta_headers: list[str] = body.get("_beta_headers", [])
        if "prompt-caching-2024-07-31" not in beta_headers:
            beta_headers.append("prompt-caching-2024-07-31")
        body["_beta_headers"] = beta_headers

    def _thinking_budget(self, effort: str | None) -> int:
        """Map reasoning_effort to Anthropic thinking budget tokens."""
        match effort:
            case "low":
                return 2048
            case "medium":
                return 8192
            case "high":
                return 32768
            case _:
                return 8192  # default medium

    # ------------------------------------------------------------------ #
    # Response translation (Anthropic -> Unified)
    # ------------------------------------------------------------------ #

    def _translate_response(self, data: dict[str, Any], request: Request) -> Response:
        """Translate Anthropic Messages API response to unified Response."""
        content_parts: list[ContentPart] = []

        for block in data.get("content", []):
            part = self._translate_response_block(block)
            if part:
                content_parts.append(part)

        # Map finish reason
        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = self._map_finish_reason(stop_reason)

        # Parse usage
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
        )

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider="anthropic",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw=data,
        )

    def _translate_response_block(self, block: dict[str, Any]) -> ContentPart | None:
        """Translate a single Anthropic content block to a ContentPart."""
        block_type = block.get("type")

        match block_type:
            case "text":
                return ContentPart.text_part(block.get("text", ""))

            case "tool_use":
                args = block.get("input", {})
                return ContentPart.tool_call_part(
                    tool_call_id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=args if isinstance(args, dict) else str(args),
                )

            case "thinking":
                return ContentPart.thinking_part(
                    text=block.get("thinking", ""),
                    signature=block.get("signature"),
                )

            case "redacted_thinking":
                return ContentPart.redacted_thinking_part(
                    redacted_data=block.get("data", ""),
                )

            case _:
                return None

    def _map_finish_reason(self, stop_reason: str) -> FinishReason:
        """Map Anthropic stop_reason to unified FinishReason."""
        match stop_reason:
            case "end_turn" | "stop_sequence":
                return FinishReason.STOP
            case "tool_use":
                return FinishReason.TOOL_CALLS
            case "max_tokens":
                return FinishReason.MAX_TOKENS
            case _:
                return FinishReason.STOP

    # ------------------------------------------------------------------ #
    # complete() -- blocking call
    # ------------------------------------------------------------------ #

    async def complete(self, request: Request) -> Response:
        """Send a request and return the complete response."""
        body = self._translate_request(request)

        # Extract beta headers if present
        extra_headers: dict[str, str] = {}
        beta_headers = body.pop("_beta_headers", [])
        if beta_headers:
            extra_headers["anthropic-beta"] = ",".join(beta_headers)

        http_response = await self._client.post(
            "/v1/messages",
            json=body,
            headers=extra_headers,
        )

        if http_response.status_code != 200:
            headers_dict = dict(http_response.headers)
            raise classify_http_error(
                http_response.status_code,
                http_response.text,
                "anthropic",
                headers=headers_dict,
            )

        data = http_response.json()
        return self._translate_response(data, request)

    # ------------------------------------------------------------------ #
    # stream() -- SSE streaming
    # ------------------------------------------------------------------ #

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and yield streaming events.

        Anthropic uses SSE with typed events:
        - message_start: contains message metadata
        - content_block_start: new content block beginning
        - content_block_delta: incremental content
        - content_block_stop: content block complete
        - message_delta: final metadata (stop_reason, usage)
        - message_stop: stream complete
        """
        body = self._translate_request(request)
        body["stream"] = True

        extra_headers: dict[str, str] = {}
        beta_headers = body.pop("_beta_headers", [])
        if beta_headers:
            extra_headers["anthropic-beta"] = ",".join(beta_headers)

        async with self._client.stream(
            "POST",
            "/v1/messages",
            json=body,
            headers=extra_headers,
        ) as http_response:
            if http_response.status_code != 200:
                error_body = await http_response.aread()
                headers_dict = dict(http_response.headers)
                raise classify_http_error(
                    http_response.status_code,
                    error_body.decode("utf-8", errors="replace"),
                    "anthropic",
                    headers=headers_dict,
                )

            async for event in self._parse_sse(http_response):
                yield event

    async def _parse_sse(self, http_response: httpx.Response) -> AsyncIterator[StreamEvent]:
        """Parse Anthropic SSE stream into unified StreamEvents."""
        # Track state for content blocks
        current_block_type: str | None = None
        current_block_id: str | None = None
        current_block_name: str | None = None
        model: str = ""
        response_id: str = ""

        event_type: str | None = None

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

                # Guard: skip data lines with no preceding event type
                # (e.g., keep-alive pings or malformed SSE)
                if event_type is None:
                    continue

                async for ev in self._handle_sse_event(
                    event_type,
                    data,
                    current_block_type,
                    current_block_id,
                    current_block_name,
                    model,
                    response_id,
                ):
                    yield ev

                # Update tracking state
                if event_type == "message_start":
                    msg = data.get("message", {})
                    model = msg.get("model", "")
                    response_id = msg.get("id", "")
                elif event_type == "content_block_start":
                    cb = data.get("content_block", {})
                    current_block_type = cb.get("type")
                    current_block_id = cb.get("id")
                    current_block_name = cb.get("name")
                elif event_type == "content_block_stop":
                    current_block_type = None
                    current_block_id = None
                    current_block_name = None

                event_type = None

    async def _handle_sse_event(  # noqa: C901, PLR0911, PLR0912
        self,
        event_type: str,
        data: dict[str, Any],
        block_type: str | None,
        block_id: str | None,
        block_name: str | None,
        model: str,
        response_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Convert a single Anthropic SSE event to unified StreamEvents."""
        match event_type:
            case "message_start":
                msg = data.get("message", {})
                usage_data = msg.get("usage", {})
                yield StreamEvent(
                    kind=StreamEventKind.START,
                    model=msg.get("model", model),
                    response_id=msg.get("id", response_id),
                    provider="anthropic",
                )
                if usage_data:
                    yield StreamEvent(
                        kind=StreamEventKind.USAGE,
                        usage=Usage(
                            input_tokens=usage_data.get("input_tokens", 0),
                            cache_read_tokens=usage_data.get("cache_read_input_tokens", 0),
                            cache_write_tokens=usage_data.get("cache_creation_input_tokens", 0),
                        ),
                    )

            case "content_block_start":
                cb = data.get("content_block", {})
                cb_type = cb.get("type")
                if cb_type == "text":
                    # §3.14: bracket text blocks with TEXT_START / TEXT_END
                    yield StreamEvent(kind=StreamEventKind.TEXT_START)
                elif cb_type == "tool_use":
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_CALL_START,
                        tool_call_id=cb.get("id", ""),
                        tool_name=cb.get("name", ""),
                    )

            case "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type")

                match delta_type:
                    case "text_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.TEXT_DELTA,
                            text=delta.get("text", ""),
                        )
                    case "thinking_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.THINKING_DELTA,
                            text=delta.get("thinking", ""),
                        )
                    case "input_json_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_DELTA,
                            tool_call_id=block_id,
                            arguments_delta=delta.get("partial_json", ""),
                        )
                    case "signature_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.THINKING_DELTA,
                            thinking_signature=delta.get("signature", ""),
                        )

            case "content_block_stop":
                if block_type == "text":
                    # §3.14: close the text block
                    yield StreamEvent(kind=StreamEventKind.TEXT_END)
                elif block_type == "tool_use":
                    yield StreamEvent(
                        kind=StreamEventKind.TOOL_CALL_END,
                        tool_call_id=block_id,
                    )

            case "message_delta":
                delta = data.get("delta", {})
                usage_data = data.get("usage", {})

                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    yield StreamEvent(
                        kind=StreamEventKind.FINISH,
                        finish_reason=self._map_finish_reason(stop_reason),
                    )

                if usage_data:
                    yield StreamEvent(
                        kind=StreamEventKind.USAGE,
                        usage=Usage(
                            output_tokens=usage_data.get("output_tokens", 0),
                        ),
                    )

            case "message_stop":
                pass  # Stream complete, no action needed

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
