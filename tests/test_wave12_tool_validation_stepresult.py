"""Tests for Wave 12 P9 (tool arg schema validation) and P10 (StepResult fields).

P9 -- section 8.7: Tool call arguments are validated against the tool's parameter
schema before passing to execute handlers. Only ``required`` field presence
is checked (no deep JSON Schema validation).

P10 -- section 4.3: StepResult exposes convenience properties that delegate to
``self.response`` so callers can write ``step.text`` instead of
``step.response.text``.
"""

from __future__ import annotations

import json

import pytest

from attractor_llm.client import Client
from attractor_llm.generate import _validate_tool_args, generate
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    Message,
    Response,
    Role,
    StepResult,
    Tool,
    Usage,
)
from tests.helpers import MockAdapter, make_text_response, make_tool_call_response

# ================================================================== #
# Helpers
# ================================================================== #


def _make_client(responses: list[Response]) -> tuple[Client, MockAdapter]:
    adapter = MockAdapter(responses=responses)
    client = Client()
    client.register_adapter("mock", adapter)
    return client, adapter


def _tool_with_schema(name: str, required: list[str], *, execute=None) -> Tool:
    """Build a Tool with a JSON Schema that marks some fields as required."""

    async def _default_execute(**kwargs):
        return "ok"

    return Tool(
        name=name,
        description=f"Test tool {name}",
        parameters={
            "type": "object",
            "properties": {f: {"type": "string"} for f in required},
            "required": required,
        },
        execute=execute or _default_execute,
    )


def _tool_no_schema(name: str, *, execute=None) -> Tool:
    """Build a Tool with NO parameters schema."""

    async def _default_execute(**kwargs):
        return "no-schema-ok"

    return Tool(
        name=name,
        description=f"Test tool {name} (no schema)",
        parameters={},
        execute=execute or _default_execute,
    )


# ================================================================== #
# P9: _validate_tool_args unit tests (pure function, no I/O)
# ================================================================== #


class TestValidateToolArgsUnit:
    """Unit tests for the _validate_tool_args helper directly."""

    def test_no_schema_returns_none(self):
        tool = Tool(name="t", description="d", parameters={})
        assert _validate_tool_args(tool, {"x": 1}) is None

    def test_no_required_field_returns_none(self):
        tool = Tool(
            name="t",
            description="d",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        # no "required" key -> nothing to check
        assert _validate_tool_args(tool, {}) is None

    def test_empty_required_list_returns_none(self):
        tool = Tool(
            name="t",
            description="d",
            parameters={"type": "object", "required": []},
        )
        assert _validate_tool_args(tool, {}) is None

    def test_all_required_present_returns_none(self):
        tool = _tool_with_schema("calc", ["a", "b"])
        assert _validate_tool_args(tool, {"a": "1", "b": "2"}) is None

    def test_extra_fields_allowed(self):
        tool = _tool_with_schema("calc", ["a"])
        # extra field 'z' not in schema should still pass
        assert _validate_tool_args(tool, {"a": "1", "z": "extra"}) is None

    def test_single_missing_field_returns_error(self):
        tool = _tool_with_schema("fetch", ["url"])
        err = _validate_tool_args(tool, {})
        assert err is not None
        assert "url" in err
        assert "fetch" in err
        assert "missing" in err.lower()

    def test_multiple_missing_fields_listed_in_error(self):
        tool = _tool_with_schema("create", ["name", "value", "type"])
        err = _validate_tool_args(tool, {"name": "x"})
        assert err is not None
        assert "value" in err
        assert "type" in err

    def test_partial_args_reports_only_missing(self):
        tool = _tool_with_schema("write", ["path", "content"])
        err = _validate_tool_args(tool, {"path": "/tmp/f"})
        assert err is not None
        assert "content" in err
        # 'path' was provided, should NOT appear in error
        assert "'path'" not in err


# ================================================================== #
# P9: Integration tests via generate() with MockAdapter
# ================================================================== #


class TestP9ToolArgValidationInGenerate:
    """P9: generate() validates tool args and injects errors into tool results."""

    @pytest.mark.asyncio
    async def test_generate_validates_tool_args_missing_required(self):
        """Tool with required param; model sends args without it -> error in result."""
        # The model calls 'read_file' but omits the required 'path' argument.
        tool_call_resp = make_tool_call_response(
            "read_file",
            arguments={"wrong_key": "oops"},  # 'path' is required but absent
            tool_call_id="tc-1",
        )
        final_text_resp = make_text_response("I got an error when reading the file.")

        client, adapter = _make_client([tool_call_resp, final_text_resp])

        tool = _tool_with_schema("read_file", required=["path"])

        result = await generate(client, "mock-model", "Read config.py", tools=[tool])

        # Two calls: 1st returns tool_call, 2nd returns text after seeing tool error
        assert adapter.call_count == 2

        # The step that executed tools should have an error tool result
        tool_step = result.steps[0]
        assert len(tool_step.tool_results) == 1
        tr = tool_step.tool_results[0]
        assert tr.is_error is True
        assert "path" in (tr.output or "")
        assert "missing" in (tr.output or "").lower()

    @pytest.mark.asyncio
    async def test_generate_passes_valid_args(self):
        """Tool with required param; model sends all required args -> success."""
        tool_call_resp = make_tool_call_response(
            "read_file",
            arguments={"path": "/etc/hosts", "encoding": "utf-8"},
            tool_call_id="tc-1",
        )
        final_text_resp = make_text_response("File contents: 127.0.0.1 localhost")

        client, _ = _make_client([tool_call_resp, final_text_resp])

        execute_called_with: dict = {}

        async def _execute(**kwargs):
            execute_called_with.update(kwargs)
            return "127.0.0.1 localhost"

        tool = _tool_with_schema("read_file", required=["path"], execute=_execute)

        result = await generate(client, "mock-model", "Read hosts file", tools=[tool])

        # execute was actually called (no validation error)
        assert execute_called_with.get("path") == "/etc/hosts"

        # The tool step has a successful (non-error) tool result
        tool_step = result.steps[0]
        assert len(tool_step.tool_results) == 1
        assert tool_step.tool_results[0].is_error is False

    @pytest.mark.asyncio
    async def test_generate_no_schema_no_validation(self):
        """Tool without a parameters schema -> no validation, args pass through."""
        tool_call_resp = make_tool_call_response(
            "freeform",
            arguments={"anything": "goes"},
            tool_call_id="tc-1",
        )
        final_text_resp = make_text_response("Done with freeform call.")

        client, _ = _make_client([tool_call_resp, final_text_resp])

        called: list[dict] = []

        async def _execute(**kwargs):
            called.append(kwargs)
            return "executed"

        tool = _tool_no_schema("freeform", execute=_execute)

        result = await generate(client, "mock-model", "Do freeform", tools=[tool])

        # execute must have been called; no error injected
        assert len(called) == 1
        assert called[0]["anything"] == "goes"

        tool_step = result.steps[0]
        assert tool_step.tool_results[0].is_error is False

    @pytest.mark.asyncio
    async def test_generate_validation_error_does_not_crash(self):
        """Validation failure returns an error tool result -- generate() does NOT raise."""
        tool_call_resp = make_tool_call_response(
            "write_file",
            arguments={},  # both 'path' and 'content' missing
            tool_call_id="tc-1",
        )
        final_text_resp = make_text_response("Could not write, missing arguments.")

        client, _ = _make_client([tool_call_resp, final_text_resp])

        tool = _tool_with_schema("write_file", required=["path", "content"])

        # Should not raise
        result = await generate(client, "mock-model", "Write a file", tools=[tool])
        assert result is not None

        tr = result.steps[0].tool_results[0]
        assert tr.is_error is True
        assert "path" in (tr.output or "")
        assert "content" in (tr.output or "")

    @pytest.mark.asyncio
    async def test_generate_args_as_json_string_validated(self):
        """Args delivered as a JSON string are parsed then validated."""
        # Arguments encoded as a JSON string (some adapters do this)
        args_str = json.dumps({"wrong": "field"})  # 'path' missing

        tool_call_resp = Response(
            id="r1",
            model="mock-model",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart.tool_call_part("tc-1", "read_file", args_str),
                ],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input_tokens=10, output_tokens=15),
        )
        final_text_resp = make_text_response("Got an error.")
        client, _ = _make_client([tool_call_resp, final_text_resp])

        tool = _tool_with_schema("read_file", required=["path"])
        result = await generate(client, "mock-model", "Read something", tools=[tool])

        tr = result.steps[0].tool_results[0]
        assert tr.is_error is True
        assert "path" in (tr.output or "")


# ================================================================== #
# P10: StepResult convenience property tests
# ================================================================== #


def _make_step_result(
    text: str = "hello",
    finish_reason: FinishReason = FinishReason.STOP,
    input_tokens: int = 5,
    output_tokens: int = 3,
    warnings: list[str] | None = None,
) -> StepResult:
    """Build a StepResult backed by a real Response for property testing."""
    response = Response(
        id="r1",
        model="mock-model",
        provider="mock",
        message=Message.assistant(text),
        finish_reason=finish_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        warnings=warnings or [],
    )
    return StepResult(response=response)


def _make_step_with_tool_calls() -> StepResult:
    """Build a StepResult whose response contains tool calls."""
    response = Response(
        id="r1",
        model="mock-model",
        provider="mock",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart.tool_call_part("tc-1", "my_tool", {"x": 1}),
                ContentPart.tool_call_part("tc-2", "other_tool", {"y": 2}),
            ],
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=10, output_tokens=20),
    )
    return StepResult(response=response)


def _make_step_with_reasoning() -> StepResult:
    """Build a StepResult whose response contains a THINKING part."""
    response = Response(
        id="r1",
        model="mock-model",
        provider="mock",
        message=Message(
            role=Role.ASSISTANT,
            content=[
                ContentPart(kind=ContentPartKind.THINKING, text="I need to think..."),
                ContentPart.text_part("Here is my answer."),
            ],
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=30),
    )
    return StepResult(response=response)


class TestP10StepResultProperties:
    """P10: StepResult convenience properties delegate to self.response."""

    def test_step_result_text_property(self):
        """step.text returns the same value as step.response.text."""
        step = _make_step_result(text="Hello, World!")
        assert step.text == "Hello, World!"
        assert step.text == step.response.text

    def test_step_result_text_none_when_no_text_content(self):
        """step.text is None when the response has no TEXT part."""
        step = _make_step_with_tool_calls()
        assert step.text is None
        assert step.text == step.response.text

    def test_step_result_reasoning_property(self):
        """step.reasoning returns THINKING parts from the response."""
        step = _make_step_with_reasoning()
        assert step.reasoning == step.response.reasoning
        assert len(step.reasoning) == 1
        assert step.reasoning[0].text == "I need to think..."

    def test_step_result_reasoning_empty_when_no_thinking(self):
        """step.reasoning is an empty list when there are no THINKING parts."""
        step = _make_step_result(text="plain answer")
        assert step.reasoning == []
        assert step.reasoning == step.response.reasoning

    def test_step_result_tool_calls_property(self):
        """step.tool_calls returns all TOOL_CALL parts from the response."""
        step = _make_step_with_tool_calls()
        assert step.tool_calls == step.response.tool_calls
        assert len(step.tool_calls) == 2
        names = {tc.name for tc in step.tool_calls}
        assert names == {"my_tool", "other_tool"}

    def test_step_result_tool_calls_empty_on_text_response(self):
        """step.tool_calls is an empty list when response has no tool calls."""
        step = _make_step_result(text="just text")
        assert step.tool_calls == []
        assert step.tool_calls == step.response.tool_calls

    def test_step_result_finish_reason_property(self):
        """step.finish_reason mirrors response.finish_reason."""
        step = _make_step_result(finish_reason=FinishReason.STOP)
        assert step.finish_reason == FinishReason.STOP
        assert step.finish_reason == step.response.finish_reason

    def test_step_result_finish_reason_tool_calls(self):
        step = _make_step_with_tool_calls()
        assert step.finish_reason == FinishReason.TOOL_CALLS

    def test_step_result_usage_property(self):
        """step.usage mirrors response.usage."""
        step = _make_step_result(input_tokens=42, output_tokens=17)
        assert step.usage.input_tokens == 42
        assert step.usage.output_tokens == 17
        assert step.usage is step.response.usage

    def test_step_result_warnings_property(self):
        """step.warnings returns an empty list when no warnings are present."""
        step = _make_step_result()
        assert step.warnings == []
        assert isinstance(step.warnings, list)

    def test_step_result_warnings_property_populated(self):
        """step.warnings returns the warnings from the underlying response."""
        step = _make_step_result(warnings=["deprecated param used", "rate limit near"])
        assert step.warnings == ["deprecated param used", "rate limit near"]
        assert step.warnings == step.response.warnings

    def test_step_result_backward_compat(self):
        """Existing code using step.response.text still works unchanged."""
        step = _make_step_result(text="backward compatible")
        # Old access pattern still works
        assert step.response.text == "backward compatible"
        # New access pattern works too
        assert step.text == "backward compatible"
        # They are identical
        assert step.text == step.response.text

    def test_step_result_backward_compat_usage(self):
        step = _make_step_result(input_tokens=100, output_tokens=200)
        assert step.response.usage.input_tokens == 100
        assert step.usage.input_tokens == 100

    def test_step_result_backward_compat_finish_reason(self):
        step = _make_step_result(finish_reason=FinishReason.MAX_TOKENS)
        assert step.response.finish_reason == FinishReason.MAX_TOKENS
        assert step.finish_reason == FinishReason.MAX_TOKENS

    def test_step_result_properties_are_not_stored_fields(self):
        """Properties must not shadow or replace the dataclass fields."""
        step = _make_step_result(text="check fields")
        # The two real dataclass fields must still be accessible
        assert hasattr(step, "response")
        assert hasattr(step, "tool_results")
        # Changing response propagates through the property
        original = step.response.text
        assert step.text == original


# ================================================================== #
# P10: Integration -- properties work on real generate() output
# ================================================================== #


class TestP10StepResultIntegration:
    """P10: StepResult properties work on steps from a real generate() call."""

    @pytest.mark.asyncio
    async def test_step_properties_on_generate_result(self):
        """Steps returned by generate() expose all P10 convenience properties."""
        resp = make_text_response("Integration answer")
        client, _ = _make_client([resp])

        result = await generate(client, "mock-model", "What is the answer?")

        assert len(result.steps) == 1
        step = result.steps[0]

        # All properties must be accessible without AttributeError
        assert step.text == "Integration answer"
        assert step.finish_reason == FinishReason.STOP
        assert step.tool_calls == []
        assert step.reasoning == []
        assert isinstance(step.usage, Usage)
        assert isinstance(step.warnings, list)

    @pytest.mark.asyncio
    async def test_step_properties_on_tool_step(self):
        """Tool-execution steps also expose all properties correctly."""
        tool_resp = make_tool_call_response("my_tool", {"arg": "val"})
        final_resp = make_text_response("Done.")

        client, _ = _make_client([tool_resp, final_resp])

        async def _execute(**kwargs):
            return "result"

        tool = Tool(
            name="my_tool",
            description="test",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
            execute=_execute,
        )

        result = await generate(client, "mock-model", "Run my tool", tools=[tool])

        # First step is the tool-call step
        tool_step = result.steps[0]
        assert tool_step.finish_reason == FinishReason.TOOL_CALLS
        assert len(tool_step.tool_calls) == 1
        assert tool_step.tool_calls[0].name == "my_tool"
        assert len(tool_step.tool_results) == 1
        assert not tool_step.tool_results[0].is_error

        # Last step is the final text step
        final_step = result.steps[-1]
        assert final_step.text == "Done."
        assert final_step.finish_reason == FinishReason.STOP
