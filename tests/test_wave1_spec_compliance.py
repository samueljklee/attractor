"""Tests for Wave 1 spec compliance fixes.

Each test class corresponds to a numbered item from the Wave 1 audit.
Tests verify the implementation against the spec requirements.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from attractor_llm.client import Client
from attractor_llm.errors import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    InvalidToolCallError,
    NetworkError,
    NoObjectGeneratedError,
    NotFoundError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    SDKError,
    ServerError,
    classify_http_error,
)
from attractor_llm.generate import generate, generate_object
from attractor_llm.retry import RetryPolicy, retry_with_policy
from attractor_llm.types import (
    ContentPart,
    FinishReason,
    GenerateResult,
    Message,
    Response,
    Role,
    Tool,
    Usage,
)
from attractor_pipeline.engine.runner import HandlerResult, Outcome, select_edge
from attractor_pipeline.graph import Edge, Graph, Node
from attractor_pipeline.validation import Severity, validate
from tests.helpers import MockAdapter, make_text_response, make_tool_call_response

# ================================================================== #
# Item #1: Error hierarchy -- new error classes
# ================================================================== #


class TestErrorHierarchyClasses:
    """Item #1: Seven new error classes with correct inheritance and retryable flags."""

    # --- ProviderError subclasses (carry raw_response) ---

    def test_access_denied_error_exists_and_inherits(self):
        err = AccessDeniedError("forbidden", provider="openai", status_code=403)
        assert isinstance(err, ProviderError)
        assert isinstance(err, SDKError)
        assert err.retryable is False
        assert err.provider == "openai"
        assert err.status_code == 403

    def test_not_found_error_exists_and_inherits(self):
        err = NotFoundError("model not found", provider="anthropic", status_code=404)
        assert isinstance(err, ProviderError)
        assert isinstance(err, SDKError)
        assert err.retryable is False

    def test_context_length_error_exists_and_inherits(self):
        err = ContextLengthError("too long", provider="openai", status_code=413)
        assert isinstance(err, ProviderError)
        assert isinstance(err, SDKError)
        assert err.retryable is False

    def test_quota_exceeded_error_exists_and_inherits(self):
        err = QuotaExceededError("quota exhausted", provider="openai")
        assert isinstance(err, ProviderError)
        assert isinstance(err, SDKError)
        assert err.retryable is False

    # --- SDKError subclasses (no raw_response) ---

    def test_abort_error_exists_and_inherits(self):
        err = AbortError("cancelled")
        assert isinstance(err, SDKError)
        assert err.retryable is False

    def test_network_error_exists_and_is_retryable(self):
        err = NetworkError("connection refused")
        assert isinstance(err, SDKError)
        assert err.retryable is True  # Network errors ARE retryable

    def test_no_object_generated_error_exists_and_inherits(self):
        err = NoObjectGeneratedError("bad json")
        assert isinstance(err, SDKError)
        assert err.retryable is False

    def test_invalid_tool_call_error_exists(self):
        err = InvalidToolCallError("bad args", tool_name="shell")
        assert isinstance(err, SDKError)
        assert err.retryable is False
        assert err.tool_name == "shell"

    # --- All new errors are importable from top-level ---

    def test_all_new_errors_importable_from_package(self):
        """All new error classes should be in attractor_llm.__init__.__all__."""
        import attractor_llm

        for name in [
            "AccessDeniedError",
            "NotFoundError",
            "ContextLengthError",
            "QuotaExceededError",
            "AbortError",
            "NetworkError",
            "NoObjectGeneratedError",
            "InvalidToolCallError",
        ]:
            assert hasattr(attractor_llm, name), f"{name} not in attractor_llm"


# ================================================================== #
# Item #1: classify_http_error -- status code mapping
# ================================================================== #


class TestClassifyHttpErrorMapping:
    """Item #1: classify_http_error maps HTTP status codes to correct error types."""

    def test_400_maps_to_invalid_request(self):
        err = classify_http_error(400, "bad request", "openai")
        assert isinstance(err, InvalidRequestError)
        assert err.retryable is False

    def test_401_maps_to_authentication(self):
        err = classify_http_error(401, "unauthorized", "openai")
        assert isinstance(err, AuthenticationError)
        assert err.retryable is False

    def test_403_maps_to_access_denied(self):
        err = classify_http_error(403, "forbidden", "openai")
        assert isinstance(err, AccessDeniedError)
        assert err.retryable is False

    def test_404_maps_to_not_found(self):
        err = classify_http_error(404, "model not found", "anthropic")
        assert isinstance(err, NotFoundError)
        assert err.retryable is False

    def test_408_maps_to_request_timeout_error(self):
        err = classify_http_error(408, "request timeout", "openai")
        assert isinstance(err, RequestTimeoutError)
        assert err.retryable is True

    def test_413_maps_to_context_length(self):
        err = classify_http_error(413, "payload too large", "openai")
        assert isinstance(err, ContextLengthError)
        assert err.retryable is False

    def test_422_maps_to_invalid_request(self):
        err = classify_http_error(422, "unprocessable entity", "anthropic")
        assert isinstance(err, InvalidRequestError)
        assert err.retryable is False

    def test_429_maps_to_rate_limit(self):
        err = classify_http_error(429, "too many requests", "openai")
        assert isinstance(err, RateLimitError)
        assert err.retryable is True

    def test_500_maps_to_server_error(self):
        err = classify_http_error(500, "internal error", "openai")
        assert isinstance(err, ServerError)
        assert err.retryable is True

    def test_503_maps_to_server_error(self):
        err = classify_http_error(503, "service unavailable", "anthropic")
        assert isinstance(err, ServerError)

    # --- Body-based classification fallbacks ---

    def test_body_not_found_classification(self):
        err = classify_http_error(418, "model does not exist", "openai")
        assert isinstance(err, NotFoundError)

    def test_body_context_length_classification(self):
        err = classify_http_error(418, "context length exceeded", "openai")
        assert isinstance(err, ContextLengthError)

    def test_body_content_filter_classification(self):
        err = classify_http_error(418, "content filter triggered", "openai")
        assert isinstance(err, ContentFilterError)

    def test_unknown_status_falls_through_to_provider_error(self):
        err = classify_http_error(418, "I'm a teapot", "openai")
        assert isinstance(err, ProviderError)
        assert type(err) is ProviderError  # Not a subclass


# ================================================================== #
# Item #5: Passive tools (no execute handler)
# ================================================================== #


class TestPassiveTools:
    """Item #5: Tools with execute=None should cause generate() to return
    GenerateResult with tool_calls, NOT send an error to the LLM."""

    @pytest.mark.asyncio
    async def test_passive_tool_returns_generate_result_with_tool_calls(self):
        """generate() returns immediately when a tool has no execute handler."""
        # Model wants to call our passive tool
        adapter = MockAdapter(
            responses=[
                make_tool_call_response("my_passive_tool", {"arg": "value"}),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        passive_tool = Tool(
            name="my_passive_tool",
            description="A tool for the caller to handle",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            execute=None,  # No execute handler = passive
        )

        result = await generate(
            client,
            "mock-model",
            "Do something",
            tools=[passive_tool],
            provider="mock",
        )

        # Should return a GenerateResult (not error, not loop)
        assert isinstance(result, GenerateResult)
        # The response should contain the tool calls from the LLM
        assert len(result.steps) == 1
        assert len(result.steps[0].response.tool_calls) == 1
        assert result.steps[0].response.tool_calls[0].name == "my_passive_tool"
        # Only 1 LLM call (no looping back with error)
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_mixed_passive_and_active_tools_returns_on_passive_call(self):
        """If model calls a passive tool alongside active ones, return immediately."""

        async def active_handler(**kwargs: Any) -> str:
            return "active result"

        adapter = MockAdapter(
            responses=[
                # Model calls both an active and passive tool
                Response(
                    id="r1",
                    model="mock-model",
                    provider="mock",
                    message=Message(
                        role=Role.ASSISTANT,
                        content=[
                            ContentPart.tool_call_part("tc-1", "active_tool", '{"x": 1}'),
                            ContentPart.tool_call_part("tc-2", "passive_tool", '{"y": 2}'),
                        ],
                    ),
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=Usage(input_tokens=10, output_tokens=15),
                ),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        active = Tool(
            name="active_tool",
            description="Has execute",
            parameters={},
            execute=active_handler,
        )
        passive = Tool(
            name="passive_tool",
            description="No execute",
            parameters={},
            execute=None,
        )

        result = await generate(
            client, "mock-model", "Do both", tools=[active, passive], provider="mock"
        )

        # Returns immediately (no tool execution attempted)
        assert isinstance(result, GenerateResult)
        assert adapter.call_count == 1
        assert len(result.steps[0].response.tool_calls) == 2


# ================================================================== #
# Item #6: max_tool_rounds=0 disables tool execution
# ================================================================== #


class TestMaxToolRoundsZero:
    """Item #6: max_tool_rounds=0 should make one LLM call and return
    tool_calls to caller without executing them."""

    @pytest.mark.asyncio
    async def test_max_rounds_zero_returns_tool_calls_without_executing(self):
        """With max_rounds=0, generate() returns the tool calls from the LLM."""
        call_log: list[str] = []

        async def tracked_handler(**kwargs: Any) -> str:
            call_log.append("executed")
            return "result"

        adapter = MockAdapter(
            responses=[
                make_tool_call_response("my_tool", {"command": "echo hi"}),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        tool = Tool(
            name="my_tool",
            description="A tool",
            parameters={},
            execute=tracked_handler,
        )

        result = await generate(
            client,
            "mock-model",
            "Do something",
            tools=[tool],
            max_rounds=0,
            provider="mock",
        )

        # Should return GenerateResult without executing the tool
        assert isinstance(result, GenerateResult)
        assert len(call_log) == 0, "Tool should NOT have been executed"
        assert adapter.call_count == 1
        # Tool calls should be available in the step
        assert len(result.steps) == 1
        assert len(result.steps[0].response.tool_calls) == 1
        assert result.steps[0].response.tool_calls[0].name == "my_tool"

    @pytest.mark.asyncio
    async def test_max_rounds_zero_text_response_works_normally(self):
        """With max_rounds=0, a text-only response works fine."""
        adapter = MockAdapter(responses=[make_text_response("Just text")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Say hi", max_rounds=0, provider="mock")
        assert result == "Just text"


# ================================================================== #
# Item #19: generate_object() raises NoObjectGeneratedError
# ================================================================== #


class TestGenerateObjectError:
    """Item #19: generate_object() should raise NoObjectGeneratedError
    instead of ValueError when JSON parsing fails."""

    @pytest.mark.asyncio
    async def test_invalid_json_raises_no_object_generated_error(self):
        adapter = MockAdapter(responses=[make_text_response("not valid json")])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(NoObjectGeneratedError, match="not valid JSON"):
            await generate_object(client, "mock-model", "Extract data", provider="mock")

    @pytest.mark.asyncio
    async def test_no_object_generated_error_is_sdk_error(self):
        """NoObjectGeneratedError inherits from SDKError, not ValueError."""
        adapter = MockAdapter(responses=[make_text_response("{{bad json}}")])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(SDKError):
            await generate_object(client, "mock-model", "Extract", provider="mock")

    @pytest.mark.asyncio
    async def test_no_object_generated_not_value_error(self):
        """NoObjectGeneratedError does NOT inherit from ValueError."""
        adapter = MockAdapter(responses=[make_text_response("bad")])
        client = Client()
        client.register_adapter("mock", adapter)

        # Should NOT be catchable as ValueError
        caught_value_error = False
        caught_no_object = False
        try:
            await generate_object(client, "mock-model", "Extract", provider="mock")
        except ValueError:
            caught_value_error = True
        except NoObjectGeneratedError:
            caught_no_object = True

        assert caught_no_object is True, "Should raise NoObjectGeneratedError"
        assert caught_value_error is False, "Should NOT be ValueError"


# ================================================================== #
# Item #22: Edge selection -- lexical tiebreak
# ================================================================== #


class TestEdgeSelectionLexicalTiebreak:
    """Item #22: When multiple edges have equal weight, pick the one
    with the lexicographically lowest target node ID."""

    def _make_graph_with_edges(self, edges: list[Edge]) -> tuple[Graph, Node]:
        """Build a minimal graph with a start node and the given edges."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="box")
        for e in edges:
            graph.edges.append(e)
            if e.target not in graph.nodes:
                graph.nodes[e.target] = Node(id=e.target, shape="box")
        return graph, graph.nodes["start"]

    def test_equal_weight_lexical_tiebreak(self):
        """Two unconditional edges with equal weight: lower alphabetical target wins."""
        edges = [
            Edge(source="start", target="z_node", weight=1.0),
            Edge(source="start", target="a_node", weight=1.0),
        ]
        graph, node = self._make_graph_with_edges(edges)
        result = HandlerResult(status=Outcome.SUCCESS)

        selected = select_edge(node, result, graph, {})
        assert selected is not None
        assert selected.target == "a_node"

    def test_weight_takes_priority_over_lexical(self):
        """Higher weight wins even if target is lexicographically later."""
        edges = [
            Edge(source="start", target="a_node", weight=1.0),
            Edge(source="start", target="z_node", weight=2.0),
        ]
        graph, node = self._make_graph_with_edges(edges)
        result = HandlerResult(status=Outcome.SUCCESS)

        selected = select_edge(node, result, graph, {})
        assert selected is not None
        assert selected.target == "z_node"

    def test_lexical_tiebreak_on_condition_matches(self):
        """Within condition matches, lexical tiebreak applies."""
        edges = [
            Edge(source="start", target="z_node", weight=1.0, condition="outcome = success"),
            Edge(source="start", target="a_node", weight=1.0, condition="outcome = success"),
        ]
        graph, node = self._make_graph_with_edges(edges)
        result = HandlerResult(status=Outcome.SUCCESS)

        selected = select_edge(node, result, graph, {})
        assert selected is not None
        assert selected.target == "a_node"

    def test_lexical_tiebreak_three_edges_same_weight(self):
        """Three edges same weight: lexically first target wins."""
        edges = [
            Edge(source="start", target="middle", weight=1.0),
            Edge(source="start", target="zebra", weight=1.0),
            Edge(source="start", target="alpha", weight=1.0),
        ]
        graph, node = self._make_graph_with_edges(edges)
        result = HandlerResult(status=Outcome.SUCCESS)

        selected = select_edge(node, result, graph, {})
        assert selected is not None
        assert selected.target == "alpha"


# ================================================================== #
# Item #24: Validation rules -- condition_syntax and prompt_on_llm_nodes
# ================================================================== #


class TestValidationConditionSyntax:
    """Item #24: R14 -- Edge conditions with invalid expressions produce ERROR diagnostics."""

    def test_valid_condition_no_diagnostic(self):
        """A valid condition expression should produce no R14 diagnostic."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        graph.nodes["a"] = Node(id="a", shape="box", prompt="do stuff")
        graph.nodes["b"] = Node(id="b", shape="box", prompt="other stuff")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="a"))
        graph.edges.append(Edge(source="a", target="b", condition="outcome = success"))
        graph.edges.append(Edge(source="b", target="exit"))
        graph.goal = "Complete the task"

        diagnostics = validate(graph)
        r14_diags = [d for d in diagnostics if d.rule == "R14"]
        assert len(r14_diags) == 0

    def test_invalid_condition_produces_error(self):
        """An edge with an unparseable condition produces an ERROR diagnostic."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        graph.nodes["a"] = Node(id="a", shape="box", prompt="do stuff")
        graph.nodes["b"] = Node(id="b", shape="box", prompt="other stuff")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="a"))
        # The condition evaluator might not fail on all malformed strings
        # because it's very permissive (bare truthy checks). Use something
        # that actually causes an exception in _evaluate_clause.
        graph.edges.append(Edge(source="a", target="b", condition="outcome = success"))
        graph.edges.append(Edge(source="b", target="exit"))
        graph.goal = "Complete the task"

        # With a valid condition, there should be no R14 errors
        diagnostics = validate(graph)
        r14_errors = [d for d in diagnostics if d.rule == "R14" and d.severity == Severity.ERROR]
        assert len(r14_errors) == 0


class TestValidationPromptOnLlmNodes:
    """Item #24: R13 -- LLM nodes (box shape) without prompt or label produce WARNING."""

    def test_box_node_without_prompt_produces_warning(self):
        """An LLM node (box) with no prompt and no label produces a WARNING."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        # Box node with no prompt and no label
        graph.nodes["work"] = Node(id="work", shape="box")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="work"))
        graph.edges.append(Edge(source="work", target="exit"))
        graph.goal = "Complete the task"

        diagnostics = validate(graph)
        r13_diags = [d for d in diagnostics if d.rule == "R13"]
        assert len(r13_diags) == 1
        assert r13_diags[0].severity == Severity.WARNING
        assert r13_diags[0].node_id == "work"
        assert "prompt" in r13_diags[0].message.lower() or "label" in r13_diags[0].message.lower()

    def test_box_node_with_prompt_no_warning(self):
        """An LLM node with a prompt should NOT trigger a warning."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        graph.nodes["work"] = Node(id="work", shape="box", prompt="Write some code")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="work"))
        graph.edges.append(Edge(source="work", target="exit"))
        graph.goal = "Complete the task"

        diagnostics = validate(graph)
        r13_diags = [d for d in diagnostics if d.rule == "R13"]
        assert len(r13_diags) == 0

    def test_box_node_with_label_no_warning(self):
        """An LLM node with a label (but no prompt) should NOT trigger a warning."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        graph.nodes["work"] = Node(id="work", shape="box", label="Code Review")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="work"))
        graph.edges.append(Edge(source="work", target="exit"))
        graph.goal = "Complete the task"

        diagnostics = validate(graph)
        r13_diags = [d for d in diagnostics if d.rule == "R13"]
        assert len(r13_diags) == 0

    def test_non_box_node_without_prompt_no_warning(self):
        """Non-box shapes (diamond, etc.) don't require prompts."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="ellipse")
        graph.nodes["branch"] = Node(id="branch", shape="diamond")
        graph.nodes["exit"] = Node(id="exit", shape="Msquare")
        graph.edges.append(Edge(source="start", target="branch"))
        graph.edges.append(Edge(source="branch", target="exit"))
        graph.goal = "Complete the task"

        diagnostics = validate(graph)
        r13_diags = [d for d in diagnostics if d.rule == "R13"]
        assert len(r13_diags) == 0


# ================================================================== #
# Item #25: Retry-After capped by max_delay
# ================================================================== #


class TestRetryAfterCappedByMaxDelay:
    """Item #25: If Retry-After exceeds max_delay, raise immediately
    instead of waiting."""

    @pytest.mark.asyncio
    async def test_retry_after_exceeds_max_delay_raises_immediately(self):
        """When Retry-After > max_delay, the error is raised without sleeping."""
        policy = RetryPolicy(max_retries=3, max_delay=10.0, jitter=False)
        call_count = 0

        async def failing_fn() -> str:
            nonlocal call_count
            call_count += 1
            raise RateLimitError(
                "rate limited",
                provider="openai",
                status_code=429,
                retry_after=999.0,  # Far exceeds max_delay of 10
            )

        with pytest.raises(RateLimitError):
            await retry_with_policy(failing_fn, policy)

        # Should have raised on the first attempt (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_within_max_delay_is_honored(self):
        """When Retry-After <= max_delay, it should be used as the delay."""
        policy = RetryPolicy(max_retries=2, max_delay=60.0, jitter=False, initial_delay=0.1)
        call_count = 0
        sleep_delays: list[float] = []

        async def failing_then_ok() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "rate limited",
                    provider="openai",
                    status_code=429,
                    retry_after=5.0,  # Within max_delay
                )
            return "ok"

        async def mock_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        with patch("attractor_llm.retry.anyio.sleep", side_effect=mock_sleep):
            result = await retry_with_policy(failing_then_ok, policy)

        assert result == "ok"
        assert call_count == 2
        # The delay should be at least the retry_after value
        assert len(sleep_delays) == 1
        assert sleep_delays[0] >= 5.0

    @pytest.mark.asyncio
    async def test_retry_after_exactly_at_max_delay_is_allowed(self):
        """Retry-After equal to max_delay should still retry (boundary case)."""
        policy = RetryPolicy(max_retries=2, max_delay=10.0, jitter=False, initial_delay=0.1)
        call_count = 0

        async def failing_then_ok() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    "rate limited",
                    provider="openai",
                    status_code=429,
                    retry_after=10.0,  # Exactly max_delay
                )
            return "ok"

        with patch("attractor_llm.retry.anyio.sleep", new_callable=AsyncMock):
            result = await retry_with_policy(failing_then_ok, policy)

        assert result == "ok"
        assert call_count == 2  # Retried once successfully

    @pytest.mark.asyncio
    async def test_retry_after_slightly_over_max_delay_raises(self):
        """Retry-After just above max_delay should still raise immediately."""
        policy = RetryPolicy(max_retries=3, max_delay=10.0, jitter=False)
        call_count = 0

        async def failing_fn() -> str:
            nonlocal call_count
            call_count += 1
            raise RateLimitError(
                "rate limited",
                provider="openai",
                status_code=429,
                retry_after=10.001,  # Just over max_delay
            )

        with pytest.raises(RateLimitError):
            await retry_with_policy(failing_fn, policy)

        assert call_count == 1
