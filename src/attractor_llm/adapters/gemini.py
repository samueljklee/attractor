"""Google Gemini API adapter.

Translates unified Request/Response types to/from the native Gemini API
(v1beta/models/*/generateContent). Uses the native API per spec §2.7,
NOT the OpenAI-compatible shim.

Key Gemini-specific behaviors:
- System instructions as a separate top-level field (§7.2)
- Synthetic tool call IDs (Gemini doesn't always provide them, §7.8)
- thinkingConfig for reasoning control (§3.9)
- Automatic prompt caching (no client-side injection needed, §2.10)
- JSON chunk streaming (not SSE -- uses chunked transfer encoding)
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiAdapter:
    """Google Gemini native API adapter. Spec §7.4.

    Implements ProviderAdapter protocol using Google's native Gemini API.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": config.api_key,
                **config.default_headers,
            },
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _endpoint(self, model: str, method: str = "generateContent") -> str:
        """Build the Gemini API endpoint URL."""
        # API key is in the x-goog-api-key header (set in __init__), not the URL.
        # This avoids leaking the key into logs, tracebacks, and proxy access logs.
        return f"{self._base_url}/v1beta/models/{model}:{method}"

    # ------------------------------------------------------------------ #
    # Request translation (Unified -> Gemini)
    # ------------------------------------------------------------------ #

    def _translate_request(self, request: Request) -> dict[str, Any]:
        """Translate unified Request to Gemini generateContent body. Spec §7.2."""
        messages = request.effective_messages()
        system_text, conversation = self._split_system(messages)
        contents = self._translate_contents(conversation)

        body: dict[str, Any] = {
            "contents": contents,
        }

        # System instruction (top-level field, not in contents)
        if system_text:
            body["systemInstruction"] = {
                "parts": [{"text": system_text}],
            }

        # Generation config
        gen_config: dict[str, Any] = {}
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gen_config["temperature"] = request.temperature
        if request.top_p is not None:
            gen_config["topP"] = request.top_p
        if request.stop:
            gen_config["stopSequences"] = request.stop
        if request.response_format:
            if request.response_format.get("type") == "json_object":
                gen_config["responseMimeType"] = "application/json"
            if "schema" in request.response_format:
                gen_config["responseSchema"] = request.response_format["schema"]

        if gen_config:
            body["generationConfig"] = gen_config

        # Tools
        if request.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters or {"type": "OBJECT", "properties": {}},
                        }
                        for t in request.tools
                    ],
                }
            ]

        # Tool config (tool_choice mapping)
        if request.tool_choice:
            tool_config: dict[str, Any] = {}
            match request.tool_choice:
                case "auto":
                    tool_config["functionCallingConfig"] = {"mode": "AUTO"}
                case "none":
                    tool_config["functionCallingConfig"] = {"mode": "NONE"}
                case "required":
                    tool_config["functionCallingConfig"] = {"mode": "ANY"}
                case _:
                    tool_config["functionCallingConfig"] = {
                        "mode": "ANY",
                        "allowedFunctionNames": [request.tool_choice],
                    }
            body["toolConfig"] = tool_config

        # Thinking config (reasoning_effort mapping)
        if request.reasoning_effort:
            body["thinkingConfig"] = {
                "thinkingBudget": self._thinking_budget(request.reasoning_effort),
            }

        # Provider-specific options
        gemini_opts = (request.provider_options or {}).get("gemini", {})
        for key, value in gemini_opts.items():
            if key not in body:
                body[key] = value

        return body

    def _split_system(self, messages: list[Message]) -> tuple[str, list[Message]]:
        """Extract system messages into a single system instruction string."""
        system_parts: list[str] = []
        conversation: list[Message] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.text:
                    system_parts.append(msg.text)
            else:
                conversation.append(msg)

        return "\n\n".join(system_parts), conversation

    def _translate_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate unified messages to Gemini contents format."""
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
            parts: list[dict[str, Any]] = []

            for part in msg.content:
                gemini_part = self._translate_part(part)
                if gemini_part:
                    parts.append(gemini_part)

            if parts:
                contents.append({"role": role, "parts": parts})

        return contents

    def _translate_part(self, part: ContentPart) -> dict[str, Any] | None:
        """Translate a single content part to Gemini format."""
        match part.kind:
            case ContentPartKind.TEXT:
                return {"text": part.text or ""}

            case ContentPartKind.IMAGE:
                if part.image and part.image.data:
                    import base64

                    return {
                        "inlineData": {
                            "mimeType": part.image.media_type,
                            "data": base64.b64encode(part.image.data).decode(),
                        },
                    }
                elif part.image and part.image.url:
                    return {
                        "fileData": {
                            "mimeType": part.image.media_type,
                            "fileUri": part.image.url,
                        },
                    }
                return None

            case ContentPartKind.TOOL_CALL:
                args = part.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                return {
                    "functionCall": {
                        "name": part.name or "",
                        "args": args or {},
                    },
                }

            case ContentPartKind.TOOL_RESULT:
                return {
                    "functionResponse": {
                        "name": part.name or "",
                        "response": {"result": part.output or ""},
                    },
                }

            case ContentPartKind.THINKING:
                return {"thought": True, "text": part.text or ""}

            case _:
                return None

    def _thinking_budget(self, effort: str | None) -> int:
        """Map reasoning_effort to Gemini thinkingBudget tokens."""
        match effort:
            case "low":
                return 1024
            case "medium":
                return 4096
            case "high":
                return 16384
            case _:
                return 4096

    # ------------------------------------------------------------------ #
    # Response translation (Gemini -> Unified)
    # ------------------------------------------------------------------ #

    def _translate_response(self, data: dict[str, Any], request: Request) -> Response:
        """Translate Gemini generateContent response to unified Response."""
        content_parts: list[ContentPart] = []

        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})

            for part in content.get("parts", []):
                translated = self._translate_response_part(part)
                if translated:
                    content_parts.append(translated)

            # Map finish reason -- detect tool calls from content inspection
            # since Gemini uses "STOP" for both text and function-call completions
            raw_reason = candidate.get("finishReason", "STOP")
            has_tool_calls = any(p.kind == ContentPartKind.TOOL_CALL for p in content_parts)
            if raw_reason == "STOP" and has_tool_calls:
                finish_reason = FinishReason.TOOL_CALLS
            else:
                finish_reason = self._map_finish_reason(raw_reason)
        else:
            # No candidates -- check for prompt-level safety block
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                finish_reason = FinishReason.CONTENT_FILTER
            else:
                finish_reason = FinishReason.ERROR

        # Parse usage
        usage_meta = data.get("usageMetadata", {})
        usage = Usage(
            input_tokens=usage_meta.get("promptTokenCount", 0),
            output_tokens=usage_meta.get("candidatesTokenCount", 0),
            reasoning_tokens=usage_meta.get("thoughtsTokenCount", 0),
            cache_read_tokens=usage_meta.get("cachedContentTokenCount", 0),
        )

        return Response(
            id=data.get("responseId", ""),
            model=data.get("modelVersion", request.model),
            provider="gemini",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw=data,
        )

    def _translate_response_part(self, part: dict[str, Any]) -> ContentPart | None:
        """Translate a single Gemini response part to ContentPart."""
        if "text" in part:
            if part.get("thought"):
                return ContentPart.thinking_part(text=part["text"])
            return ContentPart.text_part(part["text"])

        if "functionCall" in part:
            fc = part["functionCall"]
            # Gemini doesn't provide tool call IDs -- generate synthetic ones
            return ContentPart.tool_call_part(
                tool_call_id=f"gemini_{uuid.uuid4().hex[:12]}",
                name=fc.get("name", ""),
                arguments=fc.get("args", {}),
            )

        return None

    def _map_finish_reason(self, reason: str) -> FinishReason:
        """Map Gemini finishReason to unified FinishReason."""
        match reason:
            case "STOP" | "FINISH_REASON_STOP":
                return FinishReason.STOP
            case "MAX_TOKENS" | "FINISH_REASON_MAX_TOKENS":
                return FinishReason.MAX_TOKENS
            case (
                "SAFETY"
                | "FINISH_REASON_SAFETY"
                | "RECITATION"
                | "BLOCKLIST"
                | "PROHIBITED_CONTENT"
                | "SPII"
            ):
                return FinishReason.CONTENT_FILTER
            case "MALFORMED_FUNCTION_CALL":
                return FinishReason.ERROR
            case _:
                return FinishReason.STOP

    # ------------------------------------------------------------------ #
    # complete() -- blocking call
    # ------------------------------------------------------------------ #

    async def complete(self, request: Request) -> Response:
        """Send a request and return the complete response."""
        body = self._translate_request(request)
        url = self._endpoint(request.model)

        http_response = await self._client.post(url, json=body)

        if http_response.status_code != 200:
            headers_dict = dict(http_response.headers)
            raise classify_http_error(
                http_response.status_code,
                http_response.text,
                "gemini",
                headers=headers_dict,
            )

        data = http_response.json()
        return self._translate_response(data, request)

    # ------------------------------------------------------------------ #
    # stream() -- streaming (Gemini uses streamGenerateContent)
    # ------------------------------------------------------------------ #

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and yield streaming events.

        Gemini uses streamGenerateContent which returns newline-delimited
        JSON chunks (not SSE). Each chunk is a complete candidate response
        with incremental parts.
        """
        body = self._translate_request(request)
        url = self._endpoint(request.model, method="streamGenerateContent")
        # Gemini streaming uses alt=sse query parameter
        url += "&alt=sse"

        async with self._client.stream("POST", url, json=body) as http_response:
            if http_response.status_code != 200:
                error_body = await http_response.aread()
                headers_dict = dict(http_response.headers)
                raise classify_http_error(
                    http_response.status_code,
                    error_body.decode("utf-8", errors="replace"),
                    "gemini",
                    headers=headers_dict,
                )

            first_chunk = True
            async for event in self._parse_stream(http_response, request, first_chunk):
                yield event
                first_chunk = False

    async def _parse_stream(
        self,
        http_response: httpx.Response,
        request: Request,
        first_chunk: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Parse Gemini SSE stream into unified StreamEvents."""
        async for line in http_response.aiter_lines():
            line = line.strip()

            if line.startswith("event:"):
                # Gemini SSE event types (ignored -- we process data directly)
                continue

            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Emit START on first chunk
            if first_chunk:
                yield StreamEvent(
                    kind=StreamEventKind.START,
                    model=data.get("modelVersion", request.model),
                    response_id=data.get("responseId", ""),
                    provider="gemini",
                )
                first_chunk = False

            # Process candidate parts
            candidates = data.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                content = candidate.get("content", {})

                for part in content.get("parts", []):
                    if "text" in part:
                        if part.get("thought"):
                            yield StreamEvent(
                                kind=StreamEventKind.THINKING_DELTA,
                                text=part["text"],
                            )
                        else:
                            yield StreamEvent(
                                kind=StreamEventKind.TEXT_DELTA,
                                text=part["text"],
                            )

                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tc_id = f"gemini_{uuid.uuid4().hex[:12]}"
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_START,
                            tool_call_id=tc_id,
                            tool_name=fc.get("name", ""),
                        )
                        # Gemini sends complete function call in one chunk
                        args = fc.get("args", {})
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_DELTA,
                            tool_call_id=tc_id,
                            arguments_delta=json.dumps(args)
                            if isinstance(args, dict)
                            else str(args),
                        )
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_END,
                            tool_call_id=tc_id,
                        )

                # Check for finish
                finish_reason_str = candidate.get("finishReason")
                if finish_reason_str:
                    yield StreamEvent(
                        kind=StreamEventKind.FINISH,
                        finish_reason=self._map_finish_reason(finish_reason_str),
                    )

            # Usage metadata
            usage_meta = data.get("usageMetadata")
            if usage_meta:
                yield StreamEvent(
                    kind=StreamEventKind.USAGE,
                    usage=Usage(
                        input_tokens=usage_meta.get("promptTokenCount", 0),
                        output_tokens=usage_meta.get("candidatesTokenCount", 0),
                        reasoning_tokens=usage_meta.get("thoughtsTokenCount", 0),
                        cache_read_tokens=usage_meta.get("cachedContentTokenCount", 0),
                    ),
                )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
