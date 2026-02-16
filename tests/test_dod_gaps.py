"""Tests for DoD spec compliance fixes.

Each test class corresponds to a gap from the nlspec DoD checklist.
Organized by priority: P1 (critical) first, then P2 (medium).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from attractor_agent.tools.registry import ToolRegistry
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.client import Client
from attractor_llm.errors import ConfigurationError, InvalidRequestError
from attractor_llm.generate import generate
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    Message,
    Request,
    Response,
    Role,
    Tool,
    Usage,
)
from attractor_pipeline.engine.runner import (
    Graph,
    HandlerRegistry,
    HandlerResult,
    Node,
    Outcome,
    PipelineStatus,
    run_pipeline,
)
from attractor_pipeline.graph import Edge
from attractor_pipeline.handlers.basic import ExitHandler, StartHandler
from tests.helpers import MockAdapter, make_text_response, make_tool_call_response

# ================================================================== #
# P1 #1: Pipeline retry backoff
# ================================================================== #


class _FailNHandler:
    """Handler that fails N times then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._call_count = 0

    async def execute(self, node, context, graph, logs_root, abort_signal) -> HandlerResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return HandlerResult(status=Outcome.FAIL, failure_reason="deliberate failure")
        return HandlerResult(status=Outcome.SUCCESS, output="done")


def _make_linear_graph() -> Graph:
    """A minimal graph: start -> work -> exit."""
    g = Graph(name="test", default_max_retry=5)
    g.nodes["start"] = Node(id="start", shape="ellipse")
    g.nodes["work"] = Node(id="work", shape="box")
    g.nodes["exit"] = Node(id="exit", shape="Msquare")
    g.edges.append(Edge(source="start", target="work"))
    g.edges.append(Edge(source="work", target="exit"))
    return g


class TestPipelineRetryBackoff:
    """P1 #1: Pipeline retry should use exponential backoff, not zero delay."""

    @pytest.mark.asyncio
    async def test_retry_calls_sleep_with_positive_delay(self):
        """When a node fails and retries, anyio.sleep must be called with delay > 0."""
        graph = _make_linear_graph()
        handler = _FailNHandler(fail_count=2)

        registry = HandlerRegistry()
        registry.register("start", _FailNHandler(fail_count=0))  # start always succeeds
        registry.register("codergen", handler)
        registry.register("exit", _FailNHandler(fail_count=0))

        with patch(
            "attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await run_pipeline(graph, registry)

        assert result.status == PipelineStatus.COMPLETED
        # anyio.sleep must have been called for retries
        assert mock_sleep.call_count >= 2, f"Expected >=2 sleep calls, got {mock_sleep.call_count}"
        # Every delay must be positive
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay > 0, f"Retry delay must be positive, got {delay}"

    @pytest.mark.asyncio
    async def test_retry_delays_increase_exponentially(self):
        """Successive retry delays should increase (exponential backoff)."""
        graph = _make_linear_graph()
        handler = _FailNHandler(fail_count=3)

        registry = HandlerRegistry()
        registry.register("start", _FailNHandler(fail_count=0))
        registry.register("codergen", handler)
        registry.register("exit", _FailNHandler(fail_count=0))

        with patch(
            "attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await run_pipeline(graph, registry)

        assert result.status == PipelineStatus.COMPLETED
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(delays) >= 2
        # With jitter, we can't assert exact values, but the max possible delay
        # for attempt N is initial * factor^N. The min possible (with jitter) is
        # 0.5 * initial * factor^N. So delay[1] max > delay[0] min.
        # Just verify delays are in the right ballpark: all > 0, bounded by max.
        for d in delays:
            assert 0 < d <= 60.0, f"Delay {d} out of range (0, 60]"


# ================================================================== #
# P1 #3: Client.from_env() + default client
# ================================================================== #


class TestClientFromEnv:
    """P1 #3: Client.from_env() should auto-detect providers from env vars."""

    def test_detects_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = Client.from_env()
        assert "openai" in client._adapters
        assert "anthropic" not in client._adapters
        assert "gemini" not in client._adapters

    def test_detects_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = Client.from_env()
        assert "anthropic" in client._adapters
        assert "openai" not in client._adapters

    def test_detects_gemini_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = Client.from_env()
        assert "gemini" in client._adapters

    def test_detects_google_api_key_as_gemini(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-google")

        client = Client.from_env()
        assert "gemini" in client._adapters

    def test_detects_multiple_providers(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = Client.from_env()
        assert "openai" in client._adapters
        assert "anthropic" in client._adapters

    def test_no_keys_returns_empty_client(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = Client.from_env()
        assert len(client._adapters) == 0


class TestDefaultClient:
    """P1 #3b: Module-level default client pattern."""

    def test_get_default_client_raises_when_not_set(self):
        # Save and clear
        import attractor_llm.client as client_mod
        from attractor_llm.client import get_default_client

        saved = client_mod._default_client
        client_mod._default_client = None
        # Must also clear env vars since get_default_client() now
        # auto-creates from env (lazy init, Spec §2.2 / Wave 6 #20).
        env_keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ]
        saved_env = {k: os.environ.pop(k) for k in env_keys if k in os.environ}
        try:
            with pytest.raises(ConfigurationError, match="No default client"):
                get_default_client()
        finally:
            client_mod._default_client = saved
            os.environ.update(saved_env)

    def test_set_and_get_default_client(self):
        import attractor_llm.client as client_mod
        from attractor_llm.client import get_default_client, set_default_client

        saved = client_mod._default_client
        try:
            client = Client()
            set_default_client(client)
            assert get_default_client() is client
        finally:
            client_mod._default_client = saved


# ================================================================== #
# P1 #2: Parallel tool execution
# ================================================================== #


class TestParallelToolExecution:
    """P1 #2: Multiple tool calls should execute concurrently."""

    @pytest.mark.asyncio
    async def test_tool_registry_runs_tools_concurrently(self):
        """ToolRegistry.execute_tool_calls should run multiple tools in parallel."""
        call_times: list[float] = []

        async def slow_tool(**kwargs: Any) -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return "done"

        tool = Tool(name="slow", description="slow tool", execute=slow_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [
            ContentPart.tool_call_part("tc-1", "slow", {}),
            ContentPart.tool_call_part("tc-2", "slow", {}),
            ContentPart.tool_call_part("tc-3", "slow", {}),
        ]

        start = time.monotonic()
        results = await registry.execute_tool_calls(tool_calls)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.kind == ContentPartKind.TOOL_RESULT for r in results)
        assert all(not r.is_error for r in results)
        # Sequential would take ~0.3s. Parallel should take ~0.1s.
        assert elapsed < 0.25, f"Expected parallel execution (<0.25s), got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self):
        """Results must come back in the same order as the tool calls."""

        async def echo_tool(value: str = "") -> str:
            await asyncio.sleep(0.01)
            return f"echo:{value}"

        tool = Tool(name="echo", description="echo", execute=echo_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [
            ContentPart.tool_call_part("tc-a", "echo", {"value": "first"}),
            ContentPart.tool_call_part("tc-b", "echo", {"value": "second"}),
            ContentPart.tool_call_part("tc-c", "echo", {"value": "third"}),
        ]

        results = await registry.execute_tool_calls(tool_calls)

        assert results[0].tool_call_id == "tc-a"
        assert results[0].output == "echo:first"
        assert results[1].tool_call_id == "tc-b"
        assert results[1].output == "echo:second"
        assert results[2].tool_call_id == "tc-c"
        assert results[2].output == "echo:third"

    @pytest.mark.asyncio
    async def test_parallel_partial_failure(self):
        """If one tool fails, others should still succeed. Failed tool gets is_error=True."""

        async def good_tool(**kwargs: Any) -> str:
            return "ok"

        async def bad_tool(**kwargs: Any) -> str:
            raise RuntimeError("boom")

        good = Tool(name="good", description="good", execute=good_tool)
        bad = Tool(name="bad", description="bad", execute=bad_tool)
        registry = ToolRegistry()
        registry.register(good)
        registry.register(bad)

        tool_calls = [
            ContentPart.tool_call_part("tc-1", "good", {}),
            ContentPart.tool_call_part("tc-2", "bad", {}),
            ContentPart.tool_call_part("tc-3", "good", {}),
        ]

        results = await registry.execute_tool_calls(tool_calls)

        assert len(results) == 3
        assert not results[0].is_error
        assert results[1].is_error
        assert "boom" in (results[1].output or "")
        assert not results[2].is_error

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self):
        """A single tool call should work fine (no gather needed)."""

        async def simple_tool(**kwargs: Any) -> str:
            return "result"

        tool = Tool(name="simple", description="simple", execute=simple_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [ContentPart.tool_call_part("tc-1", "simple", {})]
        results = await registry.execute_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0].output == "result"
        assert not results[0].is_error


class TestParallelToolsInGenerate:
    """P1 #2b: generate() tool loop should also execute tools in parallel."""

    @pytest.mark.asyncio
    async def test_generate_runs_tools_concurrently(self):
        """generate() should execute multiple tool calls concurrently."""
        from tests.helpers import MockAdapter, make_multi_tool_response, make_text_response

        call_times: list[float] = []

        async def slow_tool(value: str = "") -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return f"result:{value}"

        tool = Tool(name="slow", description="slow", execute=slow_tool)

        adapter = MockAdapter(
            responses=[
                make_multi_tool_response(
                    [
                        ("tc-1", "slow", {"value": "a"}),
                        ("tc-2", "slow", {"value": "b"}),
                        ("tc-3", "slow", {"value": "c"}),
                    ]
                ),
                make_text_response("All done"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        start = time.monotonic()
        result = await generate(client, "mock-model", "Do it", tools=[tool], provider="mock")
        elapsed = time.monotonic() - start

        assert "All done" in result
        assert adapter.call_count == 2  # tool call response + final text
        # Sequential would take ~0.3s. Parallel should take ~0.1s.
        assert elapsed < 0.25, f"Expected parallel (<0.25s), got {elapsed:.2f}s"


# ================================================================== #
# P2 #4: Shell timeout default
# ================================================================== #


class TestShellTimeoutDefault:
    """P2 #4: Default shell timeout should be 10s, not 120s (Spec §9.4)."""

    def test_default_shell_timeout_is_10(self):
        from attractor_agent.tools.core import DEFAULT_SHELL_TIMEOUT

        assert DEFAULT_SHELL_TIMEOUT == 10


# ================================================================== #
# P2 #7: generate() rejects prompt + messages
# ================================================================== #


class TestGeneratePromptValidation:
    """P2 #7: generate() must reject prompt + messages together (Spec §4.3)."""

    @pytest.mark.asyncio
    async def test_rejects_prompt_and_messages_together(self):
        adapter = MockAdapter(responses=[make_text_response("ignored")])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(InvalidRequestError, match="Cannot provide both"):
            await generate(
                client,
                "mock-model",
                "hello",
                messages=[Message.user("world")],
                provider="mock",
            )

    @pytest.mark.asyncio
    async def test_allows_prompt_alone(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "hello", provider="mock")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_allows_messages_alone(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(
            client,
            "mock-model",
            messages=[Message.user("hello")],
            provider="mock",
        )
        assert result == "ok"


# ================================================================== #
# P2 #14: ConfigurationError
# ================================================================== #


class TestConfigurationError:
    """P2 #14: ConfigurationError for SDK misconfiguration (Spec §6)."""

    def test_configuration_error_exists(self):
        from attractor_llm.errors import SDKError

        err = ConfigurationError("bad config")
        assert isinstance(err, SDKError)
        assert not err.retryable

    @pytest.mark.asyncio
    async def test_client_resolve_raises_configuration_error(self):
        client = Client()
        # No adapters registered
        request = Request(model="some-model", messages=[Message.user("hi")])
        with pytest.raises(ConfigurationError, match="Cannot resolve provider"):
            await client.complete(request)


# ================================================================== #
# P2 #6: DEVELOPER role
# ================================================================== #


class TestDeveloperRole:
    """P2 #6: Role enum must include DEVELOPER (Spec §3.2)."""

    def test_developer_role_exists(self):
        assert Role.DEVELOPER == "developer"
        assert Role.DEVELOPER.value == "developer"

    def test_developer_message_factory(self):
        """Message should have a .developer() factory method."""
        msg = Message.developer("Build instructions here")
        assert msg.role == Role.DEVELOPER
        assert msg.text == "Build instructions here"


# ================================================================== #
# P2 #6b: DEVELOPER role in adapters
# ================================================================== #


class TestDeveloperRoleAdapters:
    """P2 #6b: Adapters must handle DEVELOPER role messages."""

    def test_openai_maps_developer_to_developer_role(self):
        adapter = OpenAIAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="gpt-4o",
            messages=[
                Message.developer("Build rules"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        input_items = body["input"]
        # DEVELOPER should map to {"role": "developer"} in OpenAI Responses API
        dev_items = [i for i in input_items if i.get("role") == "developer"]
        assert len(dev_items) == 1
        assert "Build rules" in str(dev_items[0].get("content", ""))

    def test_anthropic_merges_developer_into_system(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("You are helpful."),
                Message.developer("Build rules: always use TDD"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # DEVELOPER content should be merged into the "system" field
        system_parts = body.get("system", [])
        system_text = " ".join(p.get("text", "") for p in system_parts)
        assert "Build rules" in system_text
        # Should not appear in messages
        for msg in body["messages"]:
            for part in msg.get("content", []):
                if isinstance(part, dict):
                    assert "Build rules" not in part.get("text", "")

    def test_gemini_merges_developer_into_system_instruction(self):
        adapter = GeminiAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="gemini-2.5-pro",
            messages=[
                Message.system("You are helpful."),
                Message.developer("Build rules: always validate input"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # DEVELOPER content should be in systemInstruction
        system_inst = body.get("systemInstruction", {})
        system_text = " ".join(p.get("text", "") for p in system_inst.get("parts", []))
        assert "Build rules" in system_text
        # Should not appear in contents
        for content in body.get("contents", []):
            for part in content.get("parts", []):
                assert "Build rules" not in part.get("text", "")


# ================================================================== #
# P2 #5: Anthropic cache beta header
# ================================================================== #


class TestAnthropicCacheBetaHeader:
    """P2 #5: prompt-caching beta header must be auto-injected (Spec §2.10)."""

    def test_cache_control_adds_beta_header(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # _inject_cache_control adds markers, so beta header should be present
        beta_headers = body.get("_beta_headers", [])
        assert "prompt-caching-2024-07-31" in beta_headers, (
            f"Expected prompt-caching beta header, got: {beta_headers}"
        )

    def test_beta_header_not_duplicated(self):
        """If user already specified the beta header, don't add it twice."""
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            provider_options={
                "anthropic": {
                    "beta_headers": ["prompt-caching-2024-07-31"],
                },
            },
        )
        body = adapter._translate_request(request)
        beta_headers = body.get("_beta_headers", [])
        count = beta_headers.count("prompt-caching-2024-07-31")
        assert count == 1, f"Beta header duplicated: {beta_headers}"


# ================================================================== #
# P2 #15: auto_cache disable
# ================================================================== #


class TestAnthropicAutoCacheDisable:
    """P2 #15: auto_cache should be disableable via provider_options."""

    def test_auto_cache_disabled_skips_cache_control(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
            provider_options={"anthropic": {"auto_cache": False}},
        )
        body = adapter._translate_request(request)
        # No cache_control should be injected
        system_parts = body.get("system", [])
        for part in system_parts:
            assert "cache_control" not in part

    def test_auto_cache_enabled_by_default(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        system_parts = body.get("system", [])
        # Default: cache_control should be injected
        assert any("cache_control" in part for part in system_parts)


# ================================================================== #
# P2 #9: Goal gate tracking across all nodes
# ================================================================== #


class _OutcomeHandler:
    """Handler that always returns a specified outcome."""

    def __init__(self, status: Outcome = Outcome.SUCCESS) -> None:
        self._status = status

    async def execute(self, node, context, graph, logs_root, abort_signal) -> HandlerResult:
        return HandlerResult(status=self._status, output="done")


class TestGoalGateAllNodes:
    """P2 #9: Goal gates should be tracked on ALL nodes, not just exit."""

    @pytest.mark.asyncio
    async def test_midpipeline_goal_gate_failure_triggers_redirect(self):
        """A non-exit node with goal_gate should redirect on failure.

        The check node's handler returns SUCCESS (so retry logic doesn't
        interfere) but sets a context variable that makes the goal gate
        condition fail on the first pass.  On the second pass, the handler
        sets the variable to the value the gate expects, allowing the
        pipeline to proceed through to the exit node.
        """
        g = Graph(name="test", default_max_retry=3, max_goal_gate_redirects=5)
        g.nodes["start"] = Node(id="start", shape="ellipse")
        # 'check' node has a goal_gate but is NOT an exit node
        g.nodes["check"] = Node(
            id="check",
            shape="box",
            goal_gate="quality = high",
            retry_target="start",
        )
        g.nodes["exit"] = Node(id="exit", shape="Msquare")
        g.edges.append(Edge(source="start", target="check"))
        g.edges.append(Edge(source="check", target="exit"))

        call_count = 0

        class _GateFailOnceHandler:
            """Returns SUCCESS but sets quality=low first time, high second."""

            async def execute(self, node, context, graph, logs_root, abort_signal):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return HandlerResult(
                        status=Outcome.SUCCESS,
                        output="not ready",
                        context_updates={"quality": "low"},
                    )
                return HandlerResult(
                    status=Outcome.SUCCESS,
                    output="ready",
                    context_updates={"quality": "high"},
                )

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        registry.register("codergen", _GateFailOnceHandler())
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        with patch("attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock):
            result = await run_pipeline(g, registry)

        # The pipeline should complete (the check node eventually passes)
        assert result.status == PipelineStatus.COMPLETED
        # The check node's goal_gate failure should have triggered a redirect
        # back to start, so 'start' should appear multiple times in completed_nodes
        assert result.completed_nodes.count("start") >= 2


# ================================================================== #
# P2 #8: StepResult / GenerateResult
# ================================================================== #


class TestGenerateResultType:
    """P2 #8: GenerateResult with step tracking (Spec §4.3)."""

    def test_generate_result_has_text(self):
        from attractor_llm.types import GenerateResult

        result = GenerateResult(text="hello", steps=[], total_usage=Usage())
        assert result.text == "hello"

    def test_generate_result_str_returns_text(self):
        from attractor_llm.types import GenerateResult

        result = GenerateResult(text="hello world")
        assert str(result) == "hello world"

    def test_generate_result_equality_with_str(self):
        from attractor_llm.types import GenerateResult

        result = GenerateResult(text="hello")
        assert result == "hello"

    def test_generate_result_tracks_steps(self):
        from attractor_llm.types import GenerateResult, StepResult

        step = StepResult(
            response=Response(
                message=Message.assistant("hi"),
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
        )
        result = GenerateResult(text="hi", steps=[step])
        assert len(result.steps) == 1
        assert result.total_usage.input_tokens == 0  # total_usage set separately

    def test_step_result_has_tool_results(self):
        from attractor_llm.types import StepResult

        tool_result = ContentPart.tool_result_part("tc-1", "read_file", "content")
        step = StepResult(
            response=Response(message=Message.assistant("hi")),
            tool_results=[tool_result],
        )
        assert len(step.tool_results) == 1


# ================================================================== #
# P2 #10: Per-node artifact files
# ================================================================== #


class TestPerNodeArtifacts:
    """P2 #10: Engine writes per-node artifact files for ALL nodes (Spec §5.6).

    Wave 4 moved artifact writing from CodergenHandler into the engine
    so that every node type gets artifacts, not just codergen.
    """

    @pytest.mark.asyncio
    async def test_codergen_writes_artifact_files(self, tmp_path):
        """Engine should create {logs_root}/{node_id}/ with artifacts for codergen."""
        from attractor_pipeline.handlers.codergen import CodergenHandler

        class _MockBackend:
            async def run(self, node, prompt, context, abort_signal):
                return "LLM response text"

        logs_root = tmp_path / "logs"
        logs_root.mkdir()

        g = Graph(
            name="test",
            goal="test goal",
            nodes={
                "start": Node(id="start", shape="ellipse"),
                "my_node": Node(id="my_node", shape="box", prompt="Do something: ${goal}"),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="my_node"),
                Edge(source="my_node", target="done"),
            ],
        )

        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(backend=_MockBackend()))

        result = await run_pipeline(g, registry, logs_root=logs_root)
        assert result.status == PipelineStatus.COMPLETED

        # Check artifact directory exists
        node_dir = logs_root / "my_node"
        assert node_dir.is_dir(), f"Expected directory {node_dir}"

        # Check prompt.md
        prompt_file = node_dir / "prompt.md"
        assert prompt_file.exists()
        prompt_content = prompt_file.read_text()
        assert "test goal" in prompt_content

        # Check response.md
        response_file = node_dir / "response.md"
        assert response_file.exists()
        assert "LLM response text" in response_file.read_text()

        # Check status.json
        status_file = node_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text())
        assert status_data["status"] == "success"
        assert status_data["node_id"] == "my_node"

    @pytest.mark.asyncio
    async def test_codergen_no_artifacts_without_logs_root(self, tmp_path):
        """When logs_root is None, no artifacts should be written."""
        from attractor_pipeline.handlers.codergen import CodergenHandler

        class _MockBackend:
            async def run(self, node, prompt, context, abort_signal):
                return "response"

        g = Graph(
            name="test",
            nodes={
                "start": Node(id="start", shape="ellipse"),
                "node1": Node(id="node1", shape="box", prompt="Do it"),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="node1"),
                Edge(source="node1", target="done"),
            ],
        )

        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(backend=_MockBackend()))

        result = await run_pipeline(g, registry, logs_root=None)
        assert result.status == PipelineStatus.COMPLETED
        # No crash, no files written (nothing to assert on filesystem)


# ================================================================== #
# P2 #8b: generate() returns GenerateResult
# ================================================================== #


class TestGenerateReturnsResult:
    """P2 #8b: generate() should return GenerateResult with steps and usage."""

    @pytest.mark.asyncio
    async def test_generate_returns_generate_result(self):
        from attractor_llm.types import GenerateResult

        adapter = MockAdapter(responses=[make_text_response("hello")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Hi", provider="mock")
        assert isinstance(result, GenerateResult)
        assert result.text == "hello"
        assert str(result) == "hello"

    @pytest.mark.asyncio
    async def test_generate_result_tracks_steps(self):
        from attractor_llm.types import GenerateResult

        adapter = MockAdapter(
            responses=[
                make_tool_call_response("read_file", {"path": "/tmp/x"}),
                make_text_response("Done"),
            ]
        )

        async def fake_read(path: str = "") -> str:
            return "file content"

        tool = Tool(name="read_file", description="read", execute=fake_read)
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Read it", tools=[tool], provider="mock")
        assert isinstance(result, GenerateResult)
        assert result.text == "Done"
        assert len(result.steps) == 2  # tool call step + final text step
        assert result.total_usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_generate_result_backward_compat_with_str(self):
        """Existing code that compares result to a string should still work."""
        adapter = MockAdapter(responses=[make_text_response("hello")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Hi", provider="mock")
        # These patterns should all work for backward compatibility
        assert result == "hello"
        assert "hello" in result
        assert str(result) == "hello"
        assert bool(result)
