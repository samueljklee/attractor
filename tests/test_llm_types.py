"""Tests for the Unified LLM Client SDK types, errors, retry, catalog, and streaming."""

from __future__ import annotations

import pytest

from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    ImageData,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)

# ================================================================== #
# Message
# ================================================================== #


class TestMessage:
    def test_user_factory(self):
        m = Message.user("hello")
        assert m.role == Role.USER
        assert m.text == "hello"
        assert len(m.content) == 1

    def test_assistant_factory(self):
        m = Message.assistant("reply")
        assert m.role == Role.ASSISTANT
        assert m.text == "reply"

    def test_system_factory(self):
        m = Message.system("be helpful")
        assert m.role == Role.SYSTEM
        assert m.text == "be helpful"

    def test_tool_result_factory(self):
        m = Message.tool_result("tc-1", "tool_name", "output")
        assert m.role == Role.TOOL
        assert m.content[0].kind == ContentPartKind.TOOL_RESULT
        assert m.content[0].tool_call_id == "tc-1"
        assert m.content[0].output == "output"

    def test_user_parts_multi_content(self):
        m = Message.user_parts(
            ContentPart.text_part("look"),
            ContentPart.image_part(ImageData(url="https://example.com/img.png")),
        )
        assert m.role == Role.USER
        assert len(m.content) == 2
        assert m.content[0].kind == ContentPartKind.TEXT
        assert m.content[1].kind == ContentPartKind.IMAGE

    def test_text_property_returns_first_text(self):
        m = Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.thinking_part("hmm"),
                ContentPart.text_part("answer"),
            ],
        )
        assert m.text == "answer"

    def test_text_property_returns_none_when_empty(self):
        m = Message(role=Role.ASSISTANT, content=[])
        assert m.text is None

    def test_tool_calls_property(self):
        m = Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.text_part("calling tools"),
                ContentPart.tool_call_part("tc-1", "grep", '{"pattern": "foo"}'),
                ContentPart.tool_call_part("tc-2", "read_file", '{"path": "x"}'),
            ],
        )
        assert len(m.tool_calls) == 2
        assert m.tool_calls[0].name == "grep"
        assert m.tool_calls[1].name == "read_file"

    def test_thinking_property(self):
        m = Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.thinking_part("deep thought", signature="sig-1"),
                ContentPart.text_part("answer"),
            ],
        )
        assert len(m.thinking) == 1
        assert m.thinking[0].signature == "sig-1"


# ================================================================== #
# ContentPart validation
# ================================================================== #


class TestContentPartValidation:
    def test_text_requires_text(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.TEXT, text=None)

    def test_image_requires_image(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.IMAGE, image=None)

    def test_tool_call_requires_id_and_name(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.TOOL_CALL, tool_call_id=None, name=None)

    def test_tool_result_requires_id_and_name(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.TOOL_RESULT, tool_call_id=None, name=None)

    def test_thinking_requires_text(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.THINKING, text=None)

    def test_redacted_thinking_requires_data(self):
        with pytest.raises(ValueError):
            ContentPart(kind=ContentPartKind.REDACTED_THINKING, redacted_data=None)

    def test_valid_text_part(self):
        p = ContentPart.text_part("hello")
        assert p.kind == ContentPartKind.TEXT
        assert p.text == "hello"

    def test_valid_tool_call_part(self):
        p = ContentPart.tool_call_part("tc-1", "grep", {"pattern": "foo"})
        assert p.tool_call_id == "tc-1"
        assert p.name == "grep"


# ================================================================== #
# ImageData validation
# ================================================================== #


class TestImageData:
    def test_requires_data_or_url(self):
        with pytest.raises(ValueError):
            ImageData()

    def test_url_valid(self):
        img = ImageData(url="https://example.com/img.png")
        assert img.url == "https://example.com/img.png"

    def test_data_valid(self):
        img = ImageData(data=b"\x89PNG")
        assert img.data == b"\x89PNG"


# ================================================================== #
# Usage
# ================================================================== #


class TestUsage:
    def test_total_tokens(self):
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == 150

    def test_addition(self):
        u1 = Usage(input_tokens=100, output_tokens=50, reasoning_tokens=10)
        u2 = Usage(input_tokens=200, output_tokens=100, reasoning_tokens=20)
        u3 = u1 + u2
        assert u3.input_tokens == 300
        assert u3.output_tokens == 150
        assert u3.reasoning_tokens == 30
        assert u3.total_tokens == 450

    def test_cache_tokens_aggregate(self):
        u1 = Usage(cache_read_tokens=50, cache_write_tokens=10)
        u2 = Usage(cache_read_tokens=30, cache_write_tokens=5)
        u3 = u1 + u2
        assert u3.cache_read_tokens == 80
        assert u3.cache_write_tokens == 15


# ================================================================== #
# Request
# ================================================================== #


class TestRequest:
    def test_simple_factory(self):
        r = Request.simple("claude-sonnet-4-5", "hello")
        assert r.model == "claude-sonnet-4-5"
        assert len(r.messages) == 1
        assert r.messages[0].text == "hello"

    def test_effective_messages_prepends_system(self):
        r = Request(
            model="claude-sonnet-4-5",
            system="be helpful",
            messages=[Message.user("hi")],
        )
        msgs = r.effective_messages()
        assert len(msgs) == 2
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].text == "be helpful"
        assert msgs[1].role == Role.USER

    def test_effective_messages_no_system(self):
        r = Request(model="claude-sonnet-4-5", messages=[Message.user("hi")])
        msgs = r.effective_messages()
        assert len(msgs) == 1


# ================================================================== #
# Response
# ================================================================== #


class TestResponse:
    def test_text_property(self):
        r = Response(
            message=Message.assistant("hello"),
            finish_reason=FinishReason.STOP,
        )
        assert r.text == "hello"

    def test_tool_calls_property(self):
        r = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[ContentPart.tool_call_part("tc-1", "grep", "{}")],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        assert len(r.tool_calls) == 1

    def test_reasoning_property(self):
        r = Response(
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart.thinking_part("thought"),
                    ContentPart.text_part("answer"),
                ],
            ),
        )
        assert len(r.reasoning) == 1


# ================================================================== #
# StreamEvent
# ================================================================== #


class TestStreamEvent:
    def test_start_event_metadata(self):
        e = StreamEvent(
            kind=StreamEventKind.START,
            model="claude-sonnet-4-5",
            provider="anthropic",
            response_id="resp-123",
        )
        assert e.model == "claude-sonnet-4-5"
        assert e.provider == "anthropic"
        assert e.response_id == "resp-123"

    def test_text_delta(self):
        e = StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="hello")
        assert e.text == "hello"

    def test_thinking_with_signature(self):
        e = StreamEvent(
            kind=StreamEventKind.THINKING_DELTA,
            text="thought",
            thinking_signature="sig-abc",
        )
        assert e.thinking_signature == "sig-abc"
