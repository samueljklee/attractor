"""Audit 2 Wave 3 - Structured Output regression tests.

Covers 6 items:
  Item 1  s8.4.7     - generate_object() sets native response_format for OpenAI/Gemini
  Items 2-4 s8.9.25-27 - Adapter paths activated by response_format (cascade)
  Item 5  s8.7.10   - Full JSON Schema validation for tool args (type checking)
  Item 6  s8.3.3    - Anthropic accepts URL documents via source.type=url
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    DocumentData,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    Usage,
)


# ================================================================== #
# Helpers
# ================================================================== #


def _make_json_response(payload: dict[str, Any], provider: str = "openai") -> Response:
    """Build a Response whose text is a JSON string of payload."""
    text = json.dumps(payload)
    return Response(
        id="resp-test",
        model="test-model",
        provider=provider,
        message=Message(
            role=Role.ASSISTANT,
            content=[ContentPart.text_part(text)],
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(),
    )


def _make_mock_client(response: Response) -> MagicMock:
    """Return a mock Client whose complete() returns the given response."""
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    return client


SAMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name"],
}


# ================================================================== #
# Item 1 / Items 2-4: generate_object() response_format (s8.4.7)
# ================================================================== #


class TestGenerateObjectResponseFormat:
    """generate_object() sets request.response_format for native providers."""

    @pytest.mark.asyncio
    async def test_generate_object_sets_response_format_for_openai(self) -> None:
        """OpenAI provider: request.response_format must be json_schema with nested json_schema obj."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Alice"}, provider="openai")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gpt-4o",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="openai",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        assert request.response_format is not None, "response_format must be set for OpenAI"
        assert request.response_format["type"] == "json_schema"
        assert "json_schema" in request.response_format, (
            "OpenAI response_format must have nested 'json_schema' key"
        )
        js = request.response_format["json_schema"]
        assert js["name"] == "response"
        assert js["strict"] is True
        assert js["schema"] == SAMPLE_SCHEMA

    @pytest.mark.asyncio
    async def test_generate_object_openai_no_prompt_injection(self) -> None:
        """When using native response_format for OpenAI, schema NOT injected into system prompt."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Bob"}, provider="openai")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gpt-4o",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="openai",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        system = request.system or ""
        assert "```" not in system, "Markdown fence must not appear in system for OpenAI"
        assert "Output ONLY the JSON" not in system

    @pytest.mark.asyncio
    async def test_generate_object_sets_response_format_for_gemini(self) -> None:
        """Gemini provider: request.response_format must have top-level 'schema' key."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Carol"}, provider="gemini")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gemini-2.0-flash",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="gemini",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        assert request.response_format is not None, "response_format must be set for Gemini"
        assert request.response_format["type"] == "json_schema"
        assert request.response_format["schema"] == SAMPLE_SCHEMA, (
            "Gemini response_format must have top-level 'schema' key"
        )
        # Gemini format uses flat 'schema', NOT nested 'json_schema' (that is OpenAI's format)
        assert "json_schema" not in request.response_format

    @pytest.mark.asyncio
    async def test_generate_object_gemini_no_prompt_injection(self) -> None:
        """When using native response_format for Gemini, schema NOT injected into system prompt."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Dave"}, provider="gemini")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gemini-2.0-flash",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="gemini",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        system = request.system or ""
        assert "```" not in system
        assert "Output ONLY the JSON" not in system

    @pytest.mark.asyncio
    async def test_generate_object_uses_prompt_injection_for_anthropic(self) -> None:
        """Anthropic provider: no native support, schema injected into system prompt."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Eve"}, provider="anthropic")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "claude-sonnet-4-5",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="anthropic",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        # response_format must NOT be set for Anthropic
        assert request.response_format is None, (
            "Anthropic must not use response_format (no native support)"
        )

        # Schema must appear in system prompt
        system = request.system or ""
        assert json.dumps(SAMPLE_SCHEMA, indent=2) in system

    @pytest.mark.asyncio
    async def test_generate_object_unknown_provider_uses_prompt_injection(self) -> None:
        """Unknown provider falls back to prompt injection, no response_format set."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Frank"}, provider="unknown")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "some-model",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="unknown-provider",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        assert request.response_format is None, "Unknown provider must not set response_format"
        system = request.system or ""
        assert json.dumps(SAMPLE_SCHEMA, indent=2) in system

    @pytest.mark.asyncio
    async def test_generate_object_no_schema_no_response_format(self) -> None:
        """When schema=None, response_format is never set regardless of provider."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"result": "ok"}, provider="openai")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gpt-4o",
            "Return JSON",
            schema=None,
            provider="openai",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        assert request.response_format is None, "No schema -> response_format must remain None"

    @pytest.mark.asyncio
    async def test_generate_object_preserves_system_prompt_openai(self) -> None:
        """User-supplied system prompt preserved when using native response_format (OpenAI)."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Hank"}, provider="openai")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "gpt-4o",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            system="You are a helpful assistant.",
            provider="openai",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        assert request.system == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_object_preserves_system_prompt_anthropic(self) -> None:
        """User-supplied system prompt combined with schema injection for Anthropic."""
        from attractor_llm.generate import generate_object

        mock_resp = _make_json_response({"name": "Ivy"}, provider="anthropic")
        client = _make_mock_client(mock_resp)

        await generate_object(
            client,
            "claude-sonnet-4-5",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            system="You are an expert extractor.",
            provider="anthropic",
        )

        call_args = client.complete.call_args
        request: Request = call_args[0][0]

        system = request.system or ""
        assert "You are an expert extractor." in system
        assert json.dumps(SAMPLE_SCHEMA, indent=2) in system

    @pytest.mark.asyncio
    async def test_generate_object_returns_parsed_json(self) -> None:
        """generate_object() parses and returns the JSON dict from the response."""
        from attractor_llm.generate import generate_object

        payload = {"name": "Jack", "age": 30}
        mock_resp = _make_json_response(payload, provider="openai")
        client = _make_mock_client(mock_resp)

        result = await generate_object(
            client,
            "gpt-4o",
            "Extract entity",
            schema=SAMPLE_SCHEMA,
            provider="openai",
        )

        assert result == payload


# ================================================================== #
# Items 2-4 cascade: adapter native paths activated by response_format
# ================================================================== #


class TestAdapterNativePaths:
    """Setting response_format activates the correct adapter code paths."""

    def test_openai_adapter_response_format_path(self) -> None:
        """OpenAI adapter translates json_schema response_format to text.format in body."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        config = ProviderConfig(api_key="test-key")
        adapter = OpenAIAdapter(config)

        request = Request(
            model="gpt-4o",
            messages=[Message.user("hello")],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": SAMPLE_SCHEMA, "strict": True},
            },
        )

        body = adapter._translate_request(request)

        assert "text" in body, "OpenAI adapter must set body['text'] for response_format"
        assert "format" in body["text"]
        fmt = body["text"]["format"]
        assert fmt["type"] == "json_schema"

    def test_gemini_adapter_response_format_path(self) -> None:
        """Gemini adapter translates json_schema response_format to responseMimeType+responseSchema."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        config = ProviderConfig(api_key="test-key")
        adapter = GeminiAdapter(config)

        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("hello")],
            response_format={"type": "json_schema", "schema": SAMPLE_SCHEMA},
        )

        body = adapter._translate_request(request)

        gen_config = body.get("generationConfig", {})
        assert gen_config.get("responseMimeType") == "application/json", (
            "Gemini adapter must set responseMimeType for json_schema"
        )
        assert gen_config.get("responseSchema") == SAMPLE_SCHEMA, (
            "Gemini adapter must set responseSchema from response_format['schema']"
        )

    def test_openai_adapter_no_format_when_response_format_unset(self) -> None:
        """OpenAI adapter must not inject text.format when response_format is None."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        config = ProviderConfig(api_key="test-key")
        adapter = OpenAIAdapter(config)

        request = Request(
            model="gpt-4o",
            messages=[Message.user("hello")],
            response_format=None,
        )

        body = adapter._translate_request(request)

        if "text" in body:
            assert "format" not in body["text"]

    def test_gemini_adapter_no_mime_when_response_format_unset(self) -> None:
        """Gemini adapter must not set responseMimeType when response_format is None."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        config = ProviderConfig(api_key="test-key")
        adapter = GeminiAdapter(config)

        request = Request(
            model="gemini-2.0-flash",
            messages=[Message.user("hello")],
            response_format=None,
        )

        body = adapter._translate_request(request)
        gen_config = body.get("generationConfig", {})
        assert "responseMimeType" not in gen_config


# ================================================================== #
# Item 5: Tool arg type validation (s8.7.10)
# ================================================================== #


class TestToolArgTypeValidation:
    """validate_tool_arguments() in registry.py performs required + type checking."""

    def test_required_field_missing_returns_error(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        error = validate_tool_arguments({}, schema)
        assert error is not None
        assert "name" in error

    def test_all_required_present_returns_none(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert validate_tool_arguments({"name": "Alice"}, schema) is None

    def test_tool_arg_validation_checks_types_string_field_gets_int(self) -> None:
        """String field receiving int value must return a type error."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        error = validate_tool_arguments({"name": 42}, schema)
        assert error is not None, "Expected type error when string field gets int"
        assert "name" in error
        # Error message should mention type mismatch
        assert "int" in error or "string" in error

    def test_tool_arg_validation_checks_types_int_field_gets_string(self) -> None:
        """Integer field receiving string must return a type error."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        error = validate_tool_arguments({"count": "five"}, schema)
        assert error is not None, "Expected type error when integer field gets string"
        assert "count" in error

    def test_tool_arg_validation_bool_not_treated_as_int(self) -> None:
        """Boolean must NOT satisfy integer type (JSON booleans differ from integers)."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        error = validate_tool_arguments({"count": True}, schema)
        assert error is not None, "bool must not satisfy integer type check"

    def test_tool_arg_validation_correct_int_passes(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert validate_tool_arguments({"count": 5}, schema) is None

    def test_tool_arg_validation_extra_keys_tolerated(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assert validate_tool_arguments({"name": "Alice", "extra": 99}, schema) is None

    def test_tool_arg_validation_empty_schema_passes(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        assert validate_tool_arguments({"anything": "goes"}, {}) is None

    def test_tool_arg_validation_boolean_field_gets_string(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
        error = validate_tool_arguments({"flag": "yes"}, schema)
        assert error is not None
        assert "flag" in error

    def test_tool_arg_validation_number_allows_int_and_float(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }
        assert validate_tool_arguments({"score": 3.14}, schema) is None
        assert validate_tool_arguments({"score": 42}, schema) is None

    def test_tool_arg_validation_array_field_gets_dict(self) -> None:
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
        }
        error = validate_tool_arguments({"items": {"key": "val"}}, schema)
        assert error is not None


# ================================================================== #
# Item 6: Anthropic URL document support (s8.3.3)
# ================================================================== #


class TestAnthropicDocumentURL:
    """Anthropic adapter handles URL-only documents via source.type=url."""

    def _make_adapter(self):
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.adapters.base import ProviderConfig

        return AnthropicAdapter(ProviderConfig(api_key="test-key"))

    def _translate_document_part(self, part: ContentPart) -> dict[str, Any]:
        return self._make_adapter()._translate_content_part(part, Role.USER)

    def test_anthropic_document_url_translates(self) -> None:
        """URL-only document returns source.type=url block without raising."""
        part = ContentPart(
            kind=ContentPartKind.DOCUMENT,
            document=DocumentData(url="https://example.com/doc.pdf"),
        )
        result = self._translate_document_part(part)

        assert result["type"] == "document"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/doc.pdf"

    def test_anthropic_document_url_has_default_media_type(self) -> None:
        """URL document without explicit media_type defaults to application/pdf."""
        part = ContentPart(
            kind=ContentPartKind.DOCUMENT,
            document=DocumentData(url="https://example.com/report.pdf"),
        )
        result = self._translate_document_part(part)

        assert result["source"]["media_type"] == "application/pdf"

    def test_anthropic_document_url_has_media_type(self) -> None:
        """Explicit media_type is passed through to the source block."""
        part = ContentPart(
            kind=ContentPartKind.DOCUMENT,
            document=DocumentData(
                url="https://example.com/page.html",
                media_type="text/html",
            ),
        )
        result = self._translate_document_part(part)

        assert result["source"]["media_type"] == "text/html"
        assert result["source"]["url"] == "https://example.com/page.html"

    def test_anthropic_document_base64_still_works(self) -> None:
        """Existing base64 path is unaffected by the URL addition."""
        import base64

        raw = b"fake-pdf-bytes"
        part = ContentPart(
            kind=ContentPartKind.DOCUMENT,
            document=DocumentData(data=raw, media_type="application/pdf"),
        )
        result = self._translate_document_part(part)

        assert result["type"] == "document"
        assert result["source"]["type"] == "base64"
        assert result["source"]["data"] == base64.b64encode(raw).decode()
        assert result["source"]["media_type"] == "application/pdf"

    def test_anthropic_document_url_in_full_request_body(self) -> None:
        """URL document passes correctly through the full _translate_request pipeline."""
        adapter = self._make_adapter()

        msg = Message(
            role=Role.USER,
            content=[
                ContentPart.text_part("Please summarise this document:"),
                ContentPart(
                    kind=ContentPartKind.DOCUMENT,
                    document=DocumentData(
                        url="https://example.com/annual-report.pdf",
                        media_type="application/pdf",
                    ),
                ),
            ],
        )

        request = Request(model="claude-sonnet-4-5", messages=[msg])
        body = adapter._translate_request(request)

        messages = body["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        doc_part = next(c for c in user_msg["content"] if c.get("type") == "document")

        assert doc_part["source"]["type"] == "url"
        assert doc_part["source"]["url"] == "https://example.com/annual-report.pdf"
