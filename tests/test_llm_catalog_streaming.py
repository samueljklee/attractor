"""Tests for LLM SDK model catalog and stream accumulator."""

from __future__ import annotations

from attractor_llm.catalog import get_default_model, get_model_info, list_models
from attractor_llm.streaming import StreamAccumulator
from attractor_llm.types import FinishReason, StreamEvent, StreamEventKind, Usage

# ================================================================== #
# Model Catalog
# ================================================================== #


class TestModelCatalog:
    def test_get_known_model(self):
        info = get_model_info("claude-opus-4-6")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.supports_tools is True

    def test_get_unknown_model(self):
        assert get_model_info("nonexistent-model") is None

    def test_list_all_models(self):
        models = list_models()
        assert len(models) == 7

    def test_list_by_provider(self):
        anthropic = list_models("anthropic")
        assert len(anthropic) == 2
        assert all(m.provider == "anthropic" for m in anthropic)

        openai = list_models("openai")
        assert len(openai) == 3
        assert all(m.provider == "openai" for m in openai)

        gemini = list_models("gemini")
        assert len(gemini) == 2

    def test_default_models(self):
        assert get_default_model("anthropic").id == "claude-sonnet-4-5"
        assert get_default_model("openai").id == "gpt-5.2"
        assert get_default_model("gemini").id == "gemini-3-flash-preview"

    def test_unknown_provider_raises(self):
        import pytest

        with pytest.raises(KeyError):
            get_default_model("unknown_provider")


# ================================================================== #
# StreamAccumulator
# ================================================================== #


class TestStreamAccumulator:
    def test_basic_text_accumulation(self):
        acc = StreamAccumulator()
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.START,
                model="claude-sonnet-4-5",
                provider="anthropic",
                response_id="resp-1",
            )
        )
        acc.feed(StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello "))
        acc.feed(StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="world!"))
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.FINISH,
                finish_reason=FinishReason.STOP,
            )
        )

        resp = acc.response()
        assert resp.text == "Hello world!"
        assert resp.model == "claude-sonnet-4-5"
        assert resp.provider == "anthropic"
        assert resp.id == "resp-1"
        assert resp.usage.total_tokens == 15
        assert resp.finish_reason == FinishReason.STOP

    def test_started_property(self):
        acc = StreamAccumulator()
        assert acc.started is False
        acc.feed(StreamEvent(kind=StreamEventKind.START))
        assert acc.started is True

    def test_tool_call_accumulation(self):
        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.START))
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_START,
                tool_call_id="tc-1",
                tool_name="grep",
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_DELTA,
                tool_call_id="tc-1",
                arguments_delta='{"pattern":',
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_DELTA,
                tool_call_id="tc-1",
                arguments_delta=' "foo"}',
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.TOOL_CALL_END,
                tool_call_id="tc-1",
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.FINISH,
                finish_reason=FinishReason.TOOL_CALLS,
            )
        )

        resp = acc.response()
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "grep"
        assert resp.tool_calls[0].arguments == '{"pattern": "foo"}'
        assert resp.finish_reason == FinishReason.TOOL_CALLS

    def test_thinking_accumulation(self):
        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.START))
        acc.feed(StreamEvent(kind=StreamEventKind.THINKING_DELTA, text="Let me think..."))
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.THINKING_DELTA,
                text=" More thinking.",
                thinking_signature="sig-abc",
            )
        )
        acc.feed(StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Answer"))
        acc.feed(StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP))

        resp = acc.response()
        assert resp.text == "Answer"
        assert len(resp.reasoning) == 1
        assert "Let me think" in resp.reasoning[0].text
        assert resp.reasoning[0].signature == "sig-abc"

    def test_error_event_captured_in_warnings(self):
        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.START))
        acc.feed(StreamEvent(kind=StreamEventKind.ERROR, error="connection lost"))

        resp = acc.response()
        assert resp.finish_reason == FinishReason.ERROR
        assert any("connection lost" in w for w in resp.warnings)

    def test_empty_accumulator(self):
        acc = StreamAccumulator()
        resp = acc.response()
        assert resp.text is None
        assert resp.model == "unknown"
        assert resp.id == "stream"

    def test_multiple_usage_events_aggregate(self):
        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.START))
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(input_tokens=100),
            )
        )
        acc.feed(
            StreamEvent(
                kind=StreamEventKind.USAGE,
                usage=Usage(output_tokens=50),
            )
        )
        acc.feed(StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP))

        resp = acc.response()
        assert resp.usage.input_tokens == 100
        assert resp.usage.output_tokens == 50
        assert resp.usage.total_tokens == 150
