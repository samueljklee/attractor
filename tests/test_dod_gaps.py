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
    g.nodes["start"] = Node(id="start", shape="Mdiamond")
    g.nodes["work"] = Node(id="work", shape="box")
    g.nodes["exit"] = Node(id="exit", shape="Msquare")
    g.edges.append(Edge(source="start", target="work"))
    g.edges.append(Edge(source="work", target="exit"))
    return g


class TestPipelineRetryBackoff:
    """P1 #1: Pipeline retry uses anyio.sleep; default preset is 'none' (0-delay)."""

    @pytest.mark.asyncio
    async def test_retry_calls_sleep(self):
        """When a node fails and retries, anyio.sleep must be called.

        The default _PIPELINE_RETRY preset is 'none' (delay=0.0), so delays
        are non-negative.  Callers opt into exponential back-off via
        node-level config or by specifying a different preset.
        """
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
        # Delays must be non-negative (0.0 is valid for the 'none' preset)
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay >= 0, f"Retry delay must be non-negative, got {delay}"

    @pytest.mark.asyncio
    async def test_retry_delays_non_negative(self):
        """Successive retry delays must be non-negative (default 'none' preset = 0.0)."""
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
        for d in delays:
            assert 0 <= d <= 60.0, f"Delay {d} out of range [0, 60]"


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
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
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
                "start": Node(id="start", shape="Mdiamond"),
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
                "start": Node(id="start", shape="Mdiamond"),
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


# ================================================================== #
# D3: goal_gate=true boolean semantic (Spec §3.2 / §11 DoD 11.4.1-2)
# ================================================================== #


class TestGoalGateBooleanTrue:
    """Spec §3.2: goal_gate is a Boolean attribute.

    goal_gate=true  → node must reach SUCCESS before pipeline exits.
    goal_gate=false → no gate (same as omitting the attribute).

    Previously the engine treated goal_gate as a condition-expression
    string, so 'true' was a bare key-lookup that always resolved to ''
    (falsy), meaning goal_gate=true nodes NEVER satisfied their gate.
    """

    @pytest.mark.asyncio
    async def test_goal_gate_true_satisfied_on_success(self):
        """goal_gate=true: node that succeeds lets the pipeline exit."""
        g = Graph(name="test", max_goal_gate_redirects=3)
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["task"] = Node(
            id="task",
            shape="box",
            goal_gate="true",
            retry_target="start",
        )
        g.nodes["done"] = Node(id="done", shape="Msquare")
        g.edges.append(Edge(source="start", target="task"))
        g.edges.append(Edge(source="task", target="done"))

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        registry.register("codergen", _OutcomeHandler(Outcome.SUCCESS))
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        result = await run_pipeline(g, registry)

        # Gate satisfied (SUCCESS) → pipeline should complete, not redirect
        assert result.status == PipelineStatus.COMPLETED
        # task should appear exactly once (no redirects)
        assert result.completed_nodes.count("task") == 1

    @pytest.mark.asyncio
    async def test_goal_gate_true_not_satisfied_on_fail(self):
        """goal_gate=true: node that fails blocks the pipeline exit."""
        g = Graph(name="test", max_goal_gate_redirects=2)
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["task"] = Node(
            id="task",
            shape="box",
            goal_gate="true",
            retry_target="start",
        )
        g.nodes["done"] = Node(id="done", shape="Msquare")
        g.edges.append(Edge(source="start", target="task"))
        g.edges.append(Edge(source="task", target="done"))

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        # Task always fails → gate never satisfied → redirects until circuit breaker
        registry.register("codergen", _OutcomeHandler(Outcome.FAIL))
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        with patch("attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock):
            result = await run_pipeline(g, registry)

        # Gate never satisfied → pipeline should FAIL (circuit breaker)
        assert result.status == PipelineStatus.FAILED

    @pytest.mark.asyncio
    async def test_goal_gate_false_is_no_op(self):
        """goal_gate=false: node is never a gate regardless of outcome."""
        g = Graph(name="test", max_goal_gate_redirects=3)
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["task"] = Node(
            id="task",
            shape="box",
            goal_gate="false",
            retry_target="start",
        )
        g.nodes["done"] = Node(id="done", shape="Msquare")
        g.edges.append(Edge(source="start", target="task"))
        g.edges.append(Edge(source="task", target="done"))

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        # Even though task fails, goal_gate=false means no gate → pipeline exits
        registry.register("codergen", _OutcomeHandler(Outcome.FAIL))
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        with patch("attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock):
            result = await run_pipeline(g, registry)

        # No gate → pipeline completes regardless of task outcome
        assert result.status == PipelineStatus.COMPLETED


# ================================================================== #
# D2: context.* condition variable resolution (Spec §10.4)
# ================================================================== #


class TestContextDotVariableResolution:
    """Spec §10.4: context.key resolves to the pipeline context value for key.

    Previously _resolve('context.foo', flat_vars) returned '' because
    variables were spread as {**context} (flat keys like 'foo'), but
    the resolver looked for 'context.foo' as a flat key (not found) then
    tried variables['context']['foo'] which also didn't exist.

    The fix: when the key starts with 'context.', strip the prefix and
    look up the bare key in the flat variables dict as a fallback.
    """

    def test_context_dot_key_resolves_in_flat_dict(self):
        """context.foo resolves when flat_vars contains 'foo' (select_edge pattern)."""
        from attractor_pipeline.conditions import _resolve

        # select_edge() spreads context as {**context, 'outcome': ..., ...}
        flat_vars = {"foo": "bar", "outcome": "success"}
        assert _resolve("context.foo", flat_vars) == "bar"

    def test_context_dot_missing_key_returns_empty(self):
        """context.missing returns '' when key is absent."""
        from attractor_pipeline.conditions import _resolve

        flat_vars = {"foo": "bar"}
        assert _resolve("context.missing", flat_vars) == ""

    @pytest.mark.asyncio
    async def test_pipeline_edge_condition_with_context_dot_prefix(self):
        """End-to-end: edge condition 'context.quality = high' routes correctly."""
        from attractor_pipeline.graph import Edge as _Edge

        g = Graph(name="test", max_goal_gate_redirects=3)
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["check"] = Node(id="check", shape="box")
        g.nodes["good"] = Node(id="good", shape="Msquare")
        g.nodes["bad"] = Node(id="bad", shape="Msquare")
        g.edges.append(_Edge(source="start", target="check"))
        g.edges.append(_Edge(source="check", target="good", condition="context.quality = high"))
        g.edges.append(_Edge(source="check", target="bad"))

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        registry.register(
            "codergen",
            _OutcomeHandlerWithContext(Outcome.SUCCESS, {"quality": "high"}),
        )
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        result = await run_pipeline(g, registry)

        # Edge condition 'context.quality = high' should route to 'good'
        assert result.status == PipelineStatus.COMPLETED
        assert "good" in result.completed_nodes
        assert "bad" not in result.completed_nodes


# ================================================================== #
# §11.3.8: PipelineStatus values
# ================================================================== #


class TestPipelineStatusValues:
    """§11.3.8: PipelineStatus strings must be 'success'/'fail' per spec."""

    def test_completed_value_is_success(self):
        from attractor_pipeline.engine.runner import PipelineStatus

        assert PipelineStatus.COMPLETED.value == "success", (
            f"PipelineStatus.COMPLETED must be 'success', got '{PipelineStatus.COMPLETED.value}'"
        )

    def test_failed_value_is_fail(self):
        from attractor_pipeline.engine.runner import PipelineStatus

        assert PipelineStatus.FAILED.value == "fail", (
            f"PipelineStatus.FAILED must be 'fail', got '{PipelineStatus.FAILED.value}'"
        )


# ================================================================== #
# §11.6.3: response.md written for codergen nodes even when output empty
# ================================================================== #


class TestResponseMdWritten:
    """§11.6.3: response.md must be written for codergen nodes."""

    @pytest.mark.asyncio
    async def test_response_md_written_even_when_output_empty(self, tmp_path):
        """Codergen node with HandlerResult(output=None) still gets response.md."""
        from attractor_pipeline import HandlerRegistry, parse_dot, run_pipeline
        from attractor_pipeline.handlers import register_default_handlers

        class _EmptyOutputBackend:
            async def run(self, node, prompt, context, abort_signal):
                # Returns HandlerResult with NO output
                return HandlerResult(status=Outcome.SUCCESS, output="", notes="done")

        g = parse_dot("""
        digraph test {
            graph [goal="test"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Do something"]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)

        registry = HandlerRegistry()
        register_default_handlers(registry, codergen_backend=_EmptyOutputBackend())

        logs_root = tmp_path / "logs"
        result = await run_pipeline(g, registry, logs_root=logs_root)

        # response.md must exist for the codergen node
        response_file = logs_root / "task" / "response.md"
        assert response_file.exists(), (
            "response.md must be written for codergen nodes even when backend returns empty output"
        )


class _OutcomeHandlerWithContext:
    """Handler that returns specified outcome and context_updates."""

    def __init__(self, status: Outcome, updates: dict) -> None:
        self._status = status
        self._updates = updates

    async def execute(self, node, context, graph, logs_root, abort_signal) -> HandlerResult:
        return HandlerResult(status=self._status, output="done", context_updates=self._updates)


# ================================================================== #
# Fix 1: §9.10.1 — Emit text events alongside tool calls
# ================================================================== #


class TestInterleavedTextWithToolCalls:
    """§9.10.1: ASSISTANT_TEXT_* events must be emitted even when the response
    also contains tool calls (interleaved text + tool call in one response)."""

    @pytest.mark.asyncio
    async def test_interleaved_text_emitted_alongside_tool_calls(self):
        """When the model returns text AND tool calls in the same response,
        ASSISTANT_TEXT_START / DELTA / END must all be emitted before the
        tool round executes."""
        from attractor_agent.events import EventKind
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import ContentPartKind, FinishReason

        # Build a response that carries BOTH a text part and a tool-call part
        interleaved_response = Response(
            id="mock-interleaved",
            model="mock-model",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart(kind=ContentPartKind.TEXT, text="I'll check that for you"),
                    ContentPart.tool_call_part("tc-1", "echo_tool", {"msg": "hello"}),
                ],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input_tokens=10, output_tokens=15),
        )
        final_response = make_text_response("Done!")

        adapter = MockAdapter(responses=[interleaved_response, final_response])
        client = Client()
        client.register_adapter("mock", adapter)

        async def echo_tool(msg: str = "") -> str:
            return f"echo: {msg}"

        tool = Tool(name="echo_tool", description="echo", execute=echo_tool)
        config = SessionConfig(model="mock-model", provider="mock")

        emitted_kinds: list[str] = []
        captured_events: list[Any] = []

        def _capture(evt: Any) -> None:
            emitted_kinds.append(str(evt.kind))
            captured_events.append(evt)

        async with Session(client=client, config=config, tools=[tool]) as session:
            session.events.on(_capture)
            await session.submit("do it")

        # 1. All three event kinds must be emitted
        assert str(EventKind.ASSISTANT_TEXT_START) in emitted_kinds, (
            f"ASSISTANT_TEXT_START not emitted. Got: {emitted_kinds}"
        )
        assert str(EventKind.ASSISTANT_TEXT_DELTA) in emitted_kinds, (
            f"ASSISTANT_TEXT_DELTA not emitted. Got: {emitted_kinds}"
        )
        assert str(EventKind.ASSISTANT_TEXT_END) in emitted_kinds, (
            f"ASSISTANT_TEXT_END not emitted. Got: {emitted_kinds}"
        )

        # 2. Ordering: text events must appear BEFORE the first TOOL_CALL_OUTPUT_DELTA.
        #    This proves the interleaved branch fired (not just the final-text response).
        text_start_idx = emitted_kinds.index(str(EventKind.ASSISTANT_TEXT_START))
        tool_output_idx = emitted_kinds.index(str(EventKind.TOOL_CALL_OUTPUT_DELTA))
        assert text_start_idx < tool_output_idx, (
            f"ASSISTANT_TEXT_START ({text_start_idx}) must come before "
            f"TOOL_CALL_OUTPUT_DELTA ({tool_output_idx}). Got: {emitted_kinds}"
        )

        # 3. Delta content must match the model's interleaved text
        delta_events = [e for e in captured_events if str(e.kind) == str(EventKind.ASSISTANT_TEXT_DELTA)]
        assert delta_events, "No ASSISTANT_TEXT_DELTA events captured"
        assert delta_events[0].data.get("delta") == "I'll check that for you", (
            f"Unexpected delta content: {delta_events[0].data}"
        )

    @pytest.mark.asyncio
    async def test_no_text_events_when_response_has_no_text(self):
        """Tool-call-only response (no TEXT part) must NOT emit ASSISTANT_TEXT_* events."""
        from attractor_agent.events import EventKind
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import ContentPartKind, FinishReason

        # Response with tool call only — no TEXT content part
        tool_only_response = Response(
            id="mock-tool-only",
            model="mock-model",
            provider="mock",
            message=Message(
                role=Role.ASSISTANT,
                content=[
                    ContentPart.tool_call_part("tc-1", "echo_tool", {"msg": "hello"}),
                ],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input_tokens=5, output_tokens=5),
        )
        final_response = make_text_response("Done!")

        adapter = MockAdapter(responses=[tool_only_response, final_response])
        client = Client()
        client.register_adapter("mock", adapter)

        async def echo_tool(msg: str = "") -> str:
            return f"echo: {msg}"

        tool = Tool(name="echo_tool", description="echo", execute=echo_tool)
        config = SessionConfig(model="mock-model", provider="mock")

        emitted_kinds: list[str] = []

        def _capture(evt: Any) -> None:
            emitted_kinds.append(str(evt.kind))

        async with Session(client=client, config=config, tools=[tool]) as session:
            session.events.on(_capture)
            await session.submit("do it")

        # TOOL_CALL_OUTPUT_DELTA must be present (tool ran)
        assert str(EventKind.TOOL_CALL_OUTPUT_DELTA) in emitted_kinds

        # Before the first TOOL_CALL_OUTPUT_DELTA, no text events should appear
        tool_output_idx = emitted_kinds.index(str(EventKind.TOOL_CALL_OUTPUT_DELTA))
        text_events_before_tool = [
            k for k in emitted_kinds[:tool_output_idx]
            if k in (
                str(EventKind.ASSISTANT_TEXT_START),
                str(EventKind.ASSISTANT_TEXT_DELTA),
                str(EventKind.ASSISTANT_TEXT_END),
            )
        ]
        assert not text_events_before_tool, (
            f"No text events should appear before tool execution when "
            f"response has no text. Got: {text_events_before_tool}"
        )


# ================================================================== #
# Fix 2: §8.1.6 — Wire self._middleware in Client.complete() / stream()
# ================================================================== #


class _CallCountMiddleware:
    """Middleware that tracks both before_request and after_response calls."""

    def __init__(self) -> None:
        self.before_count = 0
        self.after_count = 0

    async def before_request(self, request: Any) -> Any:
        self.before_count += 1
        return request

    async def after_response(self, request: Any, response: Any) -> Any:
        self.after_count += 1
        return response


class _OrderRecordMiddleware:
    """Records call order for verifying middleware execution sequence."""

    def __init__(self, name: str, log: list) -> None:
        self.name = name
        self.log = log

    async def before_request(self, request: Any) -> Any:
        self.log.append(f"{self.name}.before_request")
        return request

    async def after_response(self, request: Any, response: Any) -> Any:
        self.log.append(f"{self.name}.after_response")
        return response


class TestClientConstructorMiddleware:
    """§8.1.6: Client(middleware=[...]) should apply middleware on complete()
    and stream(), even though the constructor pattern is deprecated."""

    @pytest.mark.asyncio
    async def test_middleware_applied_on_complete(self):
        """complete() must call both before_request and after_response."""
        import warnings

        mw = _CallCountMiddleware()
        adapter = MockAdapter(responses=[make_text_response("hello")])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client = Client(middleware=[mw])

        client.register_adapter("mock", adapter)

        request = Request(model="mock-model", provider="mock", messages=[Message.user("hi")])
        await client.complete(request)

        assert mw.before_count == 1, (
            f"before_request should be called once, got {mw.before_count}"
        )
        assert mw.after_count == 1, (
            f"after_response should be called once, got {mw.after_count}"
        )

    @pytest.mark.asyncio
    async def test_middleware_applied_on_stream(self):
        """stream() must call before_request and after_response (deferred until consumed)."""
        import warnings

        mw = _CallCountMiddleware()
        adapter = MockAdapter(responses=[make_text_response("streamed")])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client = Client(middleware=[mw])

        client.register_adapter("mock", adapter)

        request = Request(model="mock-model", provider="mock", messages=[Message.user("hi")])
        stream = await client.stream(request)

        # before_request fires at stream() call time
        assert mw.before_count == 1, f"before_request expected 1 call before consume, got {mw.before_count}"
        # after_response fires only after stream is fully consumed
        assert mw.after_count == 0, f"after_response must NOT fire before stream is consumed, got {mw.after_count}"

        async for _ in stream:
            pass

        assert mw.after_count == 1, (
            f"after_response should be called once after full stream consumption, got {mw.after_count}"
        )

    @pytest.mark.asyncio
    async def test_middleware_ordering_forward_before_reverse_after(self):
        """Two middlewares: before_request fires A→B, after_response fires B→A."""
        import warnings

        call_log: list[str] = []
        mw_a = _OrderRecordMiddleware("A", call_log)
        mw_b = _OrderRecordMiddleware("B", call_log)
        adapter = MockAdapter(responses=[make_text_response("ok")])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client = Client(middleware=[mw_a, mw_b])

        client.register_adapter("mock", adapter)
        request = Request(model="mock-model", provider="mock", messages=[Message.user("hi")])
        await client.complete(request)

        assert call_log == [
            "A.before_request",
            "B.before_request",
            "B.after_response",
            "A.after_response",
        ], f"Unexpected middleware call order: {call_log}"

    def test_constructor_emits_deprecation_warning(self):
        """Client(middleware=[...]) must emit DeprecationWarning."""
        import warnings

        mw = _CallCountMiddleware()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Client(middleware=[mw])

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deprecation_warnings, "Expected DeprecationWarning when passing middleware= to Client()"
        assert "apply_middleware" in str(deprecation_warnings[0].message).lower() or \
               "deprecated" in str(deprecation_warnings[0].message).lower()


# ================================================================== #
# Fix 3: §8.1.7 — client optional in generate / stream / generate_object
# ================================================================== #


class TestGenerateWithDefaultClient:
    """§8.1.7: generate() / stream() / generate_object() must accept omitting
    client and fall back to the module-level default client."""

    @pytest.mark.asyncio
    async def test_generate_uses_default_client_when_omitted(self):
        """generate(model=..., prompt=...) without client uses set_default_client()."""
        import attractor_llm.client as client_mod
        from attractor_llm.client import set_default_client
        from attractor_llm.generate import generate as llm_generate

        adapter = MockAdapter(responses=[make_text_response("from default client")])
        default_client = Client()
        default_client.register_adapter("mock", adapter)

        saved = client_mod._default_client
        try:
            set_default_client(default_client)
            # No 'client' positional argument — must not raise TypeError
            result = await llm_generate(model="mock-model", prompt="hello", provider="mock")
            assert result == "from default client", f"Unexpected result: {result}"
        finally:
            client_mod._default_client = saved

    @pytest.mark.asyncio
    async def test_generate_object_uses_default_client_when_omitted(self):
        """generate_object(model=..., prompt=...) without client uses default."""
        import attractor_llm.client as client_mod
        from attractor_llm.client import set_default_client
        from attractor_llm.generate import generate_object as llm_generate_object

        adapter = MockAdapter(responses=[make_text_response('{"key": "value"}')])
        default_client = Client()
        default_client.register_adapter("mock", adapter)

        saved = client_mod._default_client
        try:
            set_default_client(default_client)
            result = await llm_generate_object(
                model="mock-model",
                prompt="extract entities",
                provider="mock",
            )
            assert result.parsed_object == {"key": "value"}
        finally:
            client_mod._default_client = saved

    @pytest.mark.asyncio
    async def test_stream_uses_default_client_when_omitted(self):
        """stream(model=..., prompt=...) without client uses the default client."""
        import attractor_llm.client as client_mod
        from attractor_llm.client import set_default_client
        from attractor_llm.generate import stream as llm_stream

        adapter = MockAdapter(responses=[make_text_response("streamed text")])
        default_client = Client()
        default_client.register_adapter("mock", adapter)

        saved = client_mod._default_client
        try:
            set_default_client(default_client)
            result = await llm_stream(model="mock-model", prompt="hello", provider="mock")
            # Consume the stream
            text = ""
            async for chunk in result:
                text += chunk
            assert text == "streamed text", f"Unexpected stream text: {text}"
        finally:
            client_mod._default_client = saved

    @pytest.mark.asyncio
    async def test_stream_with_tools_uses_default_client_when_omitted(self):
        """stream_with_tools(model=..., prompt=...) without client uses default."""
        import attractor_llm.client as client_mod
        from attractor_llm.client import set_default_client
        from attractor_llm.generate import stream_with_tools as llm_stream_with_tools

        adapter = MockAdapter(responses=[make_text_response("streamed with tools")])
        default_client = Client()
        default_client.register_adapter("mock", adapter)

        saved = client_mod._default_client
        try:
            set_default_client(default_client)
            # stream_with_tools() is an async function that returns StreamResult
            result = await llm_stream_with_tools(
                model="mock-model", prompt="hello", provider="mock"
            )
            # Consume as text chunks
            text = ""
            async for chunk in result:
                text += chunk
            assert text == "streamed with tools", (
                f"Unexpected stream_with_tools output: {text}"
            )
        finally:
            client_mod._default_client = saved


# §8.4.7/8.4.8: generate_object() schema validation
# ================================================================== #


class TestGenerateObjectSchemaValidation:
    """§8.4.7/8.4.8: generate_object() must validate parsed JSON against schema."""

    @pytest.mark.asyncio
    async def test_valid_object_passes(self):
        """A response that matches the schema returns a GenerateObjectResult."""
        from attractor_llm.errors import NoObjectGeneratedError  # noqa: F401
        from attractor_llm.generate import generate_object as llm_generate_object

        # Schema: {"type": "object", "required": ["name", "age"], "properties": {...}}
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        adapter = MockAdapter(responses=[make_text_response('{"name": "Alice", "age": 30}')])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await llm_generate_object(
            client, model="mock-model", prompt="extract", schema=schema, provider="mock"
        )
        assert result.parsed_object == {"name": "Alice", "age": 30}

    @pytest.mark.asyncio
    async def test_wrong_type_raises_no_object_generated(self):
        """A response with wrong field types raises NoObjectGeneratedError. §8.4.8"""
        from attractor_llm.errors import NoObjectGeneratedError
        from attractor_llm.generate import generate_object as llm_generate_object

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        # age is a string instead of integer
        adapter = MockAdapter(responses=[make_text_response('{"name": "Alice", "age": "thirty"}')])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(NoObjectGeneratedError, match="schema"):
            await llm_generate_object(
                client, model="mock-model", prompt="extract", schema=schema, provider="mock"
            )

    @pytest.mark.asyncio
    async def test_missing_required_field_raises(self):
        """A response missing required fields raises NoObjectGeneratedError. §8.4.8"""
        from attractor_llm.errors import NoObjectGeneratedError
        from attractor_llm.generate import generate_object as llm_generate_object

        schema = {
            "type": "object",
            "required": ["name", "email"],
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        # 'email' is missing
        adapter = MockAdapter(responses=[make_text_response('{"name": "Bob"}')])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(NoObjectGeneratedError, match="schema"):
            await llm_generate_object(
                client, model="mock-model", prompt="extract", schema=schema, provider="mock"
            )

    @pytest.mark.asyncio
    async def test_invalid_json_still_raises(self):
        """Pure JSON parse failure still raises NoObjectGeneratedError. §8.4.8"""
        from attractor_llm.errors import NoObjectGeneratedError
        from attractor_llm.generate import generate_object as llm_generate_object

        schema = {"type": "object"}
        adapter = MockAdapter(responses=[make_text_response("not valid json {")])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(NoObjectGeneratedError):
            await llm_generate_object(
                client, model="mock-model", prompt="extract", schema=schema, provider="mock"
            )

    @pytest.mark.asyncio
    async def test_no_schema_skips_validation(self):
        """Without a schema, any valid JSON is returned without validation."""
        from attractor_llm.generate import generate_object as llm_generate_object

        # This would fail if schema were {"type": "string"}, but no schema is given
        adapter = MockAdapter(responses=[make_text_response('{"arbitrary": true}')])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await llm_generate_object(
            client, model="mock-model", prompt="extract", provider="mock"
        )
        assert result.parsed_object == {"arbitrary": True}

    @pytest.mark.asyncio
    async def test_fallback_required_without_type_key(self):
        """Fallback: required fields checked even when schema omits 'type' key.

        A schema like {"required": ["name"]} (no "type" key) is valid JSON Schema.
        The fallback must still check required fields for dict responses.
        """
        import sys
        from unittest.mock import patch
        from attractor_llm.generate import generate_object as llm_generate_object
        from attractor_llm.errors import NoObjectGeneratedError

        schema = {"required": ["name"], "properties": {"name": {"type": "string"}}}
        # Response is missing "name"
        adapter = MockAdapter(responses=[make_text_response("{}")])
        client = Client()
        client.register_adapter("mock", adapter)

        # Force the fallback path by making jsonschema unavailable
        with patch.dict(sys.modules, {"jsonschema": None}):
            with pytest.raises(NoObjectGeneratedError, match="Missing required"):
                await llm_generate_object(
                    client, model="mock-model", prompt="extract",
                    schema=schema, provider="mock"
                )

    @pytest.mark.asyncio
    async def test_fallback_bool_rejected_for_integer_field(self):
        """Fallback: boolean True/False must not pass as integer field."""
        import sys
        from unittest.mock import patch
        from attractor_llm.generate import generate_object as llm_generate_object
        from attractor_llm.errors import NoObjectGeneratedError

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        adapter = MockAdapter(responses=[make_text_response('{"count": true}')])
        client = Client()
        client.register_adapter("mock", adapter)

        with patch.dict(sys.modules, {"jsonschema": None}):
            with pytest.raises(NoObjectGeneratedError, match="schema"):
                await llm_generate_object(
                    client, model="mock-model", prompt="extract",
                    schema=schema, provider="mock"
                )

    @pytest.mark.asyncio
    async def test_fallback_valid_object_passes(self):
        """Fallback path: a valid object still passes when jsonschema is absent."""
        import sys
        from unittest.mock import patch
        from attractor_llm.generate import generate_object as llm_generate_object

        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        adapter = MockAdapter(responses=[make_text_response('{"name": "Alice"}')])
        client = Client()
        client.register_adapter("mock", adapter)

        with patch.dict(sys.modules, {"jsonschema": None}):
            result = await llm_generate_object(
                client, model="mock-model", prompt="extract",
                schema=schema, provider="mock"
            )
        assert result.parsed_object == {"name": "Alice"}

# §8.4.5: Adapters must emit STREAM_START, not START
# ================================================================== #


class TestStreamStartEvent:
    """§8.4.5: Adapters must emit STREAM_START, not the legacy START event."""

    @pytest.mark.asyncio
    async def test_openai_adapter_emits_stream_start(self):
        """OpenAI adapter's _handle_sse_event emits STREAM_START for response.created."""
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter
        from attractor_llm.types import Request, StreamEventKind

        adapter = OpenAIAdapter(ProviderConfig(api_key="test"))
        request = Request(model="gpt-test", messages=[Message.user("hi")])
        data = {"response": {"model": "gpt-test", "id": "resp-123"}}

        events = []
        async for ev in adapter._handle_sse_event(
            "response.created", data, request, None, None
        ):
            events.append(ev)

        assert events, "Expected at least one event from response.created"
        assert events[0].kind == StreamEventKind.STREAM_START, (
            f"Expected STREAM_START, got {events[0].kind}"
        )
        assert events[0].model == "gpt-test"
        assert events[0].response_id == "resp-123"

    @pytest.mark.asyncio
    async def test_anthropic_adapter_emits_stream_start(self):
        """Anthropic adapter's _handle_sse_event emits STREAM_START for message_start."""
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.types import StreamEventKind

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        data = {"message": {"model": "claude-test", "id": "msg-123", "usage": {}}}

        events = []
        async for ev in adapter._handle_sse_event(
            "message_start", data, None, None, None, "claude-test", ""
        ):
            events.append(ev)

        assert events, "Expected at least one event from message_start"
        assert events[0].kind == StreamEventKind.STREAM_START, (
            f"Expected STREAM_START, got {events[0].kind}"
        )
        assert events[0].model == "claude-test"
        assert events[0].response_id == "msg-123"

    @pytest.mark.asyncio
    async def test_gemini_adapter_emits_stream_start(self):
        """Gemini adapter's _parse_stream emits STREAM_START on first data chunk."""
        import asyncio
        import json

        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter
        from attractor_llm.types import Request, StreamEventKind

        adapter = GeminiAdapter(ProviderConfig(api_key="test"))
        request = Request(model="gemini-test", messages=[Message.user("hi")])

        # Minimal valid Gemini chunk with a text candidate
        chunk = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {},
            "modelVersion": "gemini-test",
            "responseId": "gemini-resp-1",
        }

        async def _fake_aiter_lines():
            yield f"data: {json.dumps(chunk)}"

        class _FakeResponse:
            def aiter_lines(self):
                return _fake_aiter_lines()

        events = []
        async for ev in adapter._parse_stream(_FakeResponse(), request, True):
            events.append(ev)
            break  # only need the first event

        assert events, "Expected at least one event from Gemini stream"
        assert events[0].kind == StreamEventKind.STREAM_START, (
            f"Expected STREAM_START, got {events[0].kind}"
        )

    def test_stream_accumulator_accepts_stream_start(self):
        """StreamAccumulator accepts STREAM_START events (new canonical form)."""
        from attractor_llm.streaming import StreamAccumulator
        from attractor_llm.types import StreamEvent, StreamEventKind

        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.STREAM_START, model="test-model", provider="mock"))
        assert acc.started, "StreamAccumulator must mark started=True on STREAM_START event"

    def test_stream_accumulator_accepts_legacy_start(self):
        """StreamAccumulator also accepts legacy START events for backward compat."""
        from attractor_llm.streaming import StreamAccumulator
        from attractor_llm.types import StreamEvent, StreamEventKind

        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=StreamEventKind.START, model="test-model", provider="mock"))
        assert acc.started, "StreamAccumulator must also accept legacy START event"


# ================================================================== #
# §9.5.3: Truncated tool output must have exactly ONE truncation marker
# ================================================================== #


class TestTruncationMarker:
    """§9.5.3: Truncated tool output must have exactly ONE truncation marker."""

    @pytest.mark.asyncio
    async def test_exactly_one_truncation_marker_in_llm_output(self):
        """When tool output is truncated, the LLM sees exactly one WARNING marker.

        Previously tools/registry.py appended '\\n[output was truncated]' ON TOP
        OF the spec-compliant WARNING already embedded by truncate_output(), giving
        the LLM two distinct truncation signals.
        """
        from attractor_llm.types import Tool

        # Create a registry with tight per-character limits so truncation fires
        registry = ToolRegistry(
            supports_parallel_tool_calls=False,
            tool_output_limits={"big_tool": 50},  # 50-char limit
        )

        big_output = "A" * 200  # 200-char output > 50-char limit

        async def big_tool_func() -> str:
            return big_output

        registry.register(Tool(
            name="big_tool",
            description="returns big output",
            execute=big_tool_func,
        ))

        tc = ContentPart.tool_call_part("tc-1", "big_tool", {})
        results = await registry.execute_tool_calls([tc])

        assert len(results) == 1
        result_text = results[0].output or ""

        # The spec-compliant WARNING from truncate_output() should appear exactly once
        assert result_text.count("[WARNING: Tool output was truncated") == 1, (
            f"Expected exactly 1 truncation WARNING, got multiple. Output: {result_text[:200]}"
        )
        # The legacy marker must NOT appear as a second signal
        assert "[output was truncated]" not in result_text.split("WARNING")[1] if "WARNING" in result_text else True, (
            f"Legacy truncation marker found after spec marker. Output: {result_text[:200]}"
        )

class TestMaxToolRoundsPerInput:
    """§9.1.4: SessionConfig field name matches spec."""

    def test_field_named_max_tool_rounds_per_input(self):
        """SessionConfig must have max_tool_rounds_per_input attribute."""
        from attractor_agent.session import SessionConfig
        config = SessionConfig(model="test", max_tool_rounds_per_input=5)
        assert config.max_tool_rounds_per_input == 5

    def test_deprecated_alias_still_works(self):
        """max_tool_rounds_per_turn alias must still work with DeprecationWarning."""
        import warnings
        from attractor_agent.session import SessionConfig
        config = SessionConfig(model="test", max_tool_rounds_per_input=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = config.max_tool_rounds_per_turn
        assert val == 3
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

class TestDiagnosticEdgeId:
    """§11.2.10: Diagnostic must carry a stable edge_id string."""

    def test_edge_diagnostic_has_edge_id(self):
        """Validate: invalid edge reference → Diagnostic has non-empty edge_id."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.validation import validate

        g = Graph(name="test")
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["exit"] = Node(id="exit", shape="Msquare")
        g.edges.append(Edge(source="start", target="nonexistent_node"))

        diags = validate(g)
        edge_diags = [d for d in diags if d.edge_id]
        assert edge_diags, "Expected at least one diagnostic with edge_id populated"
        assert any("nonexistent_node" in d.edge_id or "start" in d.edge_id for d in edge_diags)

class TestToolHandlerToolCommand:
    """§11.6.8: ToolHandler must read tool_command attribute per spec."""

    @pytest.mark.asyncio
    async def test_tool_command_attribute_used(self):
        """ToolHandler executes node.attrs.get('tool_command') when set."""
        from attractor_pipeline.graph import Graph, Node
        from attractor_pipeline.handlers.basic import ToolHandler
        from attractor_pipeline.engine.runner import Outcome
        from unittest.mock import AsyncMock, patch

        handler = ToolHandler()
        node = Node(id="t1", shape="parallelogram", attrs={"tool_command": "echo hello"})
        ctx = {}
        graph = Graph(name="test")

        # Patch subprocess.run to avoid real shell calls
        fake_result = type("R", (), {"returncode": 0, "stdout": "hello\n", "stderr": ""})()
        with patch("attractor_pipeline.handlers.basic.subprocess.run", return_value=fake_result):
            result = await handler.execute(node, ctx, graph, None, None)

        assert result.status == Outcome.SUCCESS

# §11.11.5: DoD alias routes -- GET /status/{id}
# ================================================================== #


class TestDoDRouteAliases:
    """§11.11.5: HTTP server must expose /run, /status/{id}, /answer/{id} aliases."""

    def test_status_alias_returns_404_for_unknown_pipeline(self):
        """GET /status/{id} returns 404 when the pipeline does not exist."""
        from starlette.testclient import TestClient

        from attractor_server.app import create_app
        from attractor_server.pipeline_manager import PipelineManager

        app = create_app(manager=PipelineManager())
        client = TestClient(app, raise_server_exceptions=True)

        response = client.get("/status/nonexistent-pipeline-id")
        assert response.status_code == 404
        assert "not found" in response.json().get("error", "").lower()

    def test_run_alias_exists(self):
        """POST /run is wired; a bad-body request returns 400 (not 404/405)."""
        from starlette.testclient import TestClient

        from attractor_server.app import create_app
        from attractor_server.pipeline_manager import PipelineManager

        app = create_app(manager=PipelineManager())
        client = TestClient(app, raise_server_exceptions=True)

        response = client.post("/run", json={})  # missing dot_source
        assert response.status_code == 400

    def test_answer_alias_returns_404_for_unknown_pipeline(self):
        """POST /answer/{id} returns 404 when the pipeline does not exist."""
        from starlette.testclient import TestClient

        from attractor_server.app import create_app
        from attractor_server.pipeline_manager import PipelineManager

        app = create_app(manager=PipelineManager())
        client = TestClient(app, raise_server_exceptions=True)

        response = client.post("/answer/no-such-id", json={"answer": "yes"})
        assert response.status_code == 404


# ================================================================== #
# §9.4.1: ExecutionEnvironment.platform() / working_directory()
# ================================================================== #


class TestExecutionEnvironmentIntrospection:
    """§9.4.1: LocalEnvironment must expose platform(), os_version(), working_directory()."""

    def test_platform_returns_non_empty_string(self):
        """LocalEnvironment.platform() returns a non-empty string."""
        from attractor_agent.environment import LocalEnvironment

        env = LocalEnvironment()
        result = env.platform()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_os_version_returns_non_empty_string(self):
        """LocalEnvironment.os_version() returns a non-empty string."""
        from attractor_agent.environment import LocalEnvironment

        env = LocalEnvironment()
        result = env.os_version()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_working_directory_returns_current_directory(self):
        """LocalEnvironment.working_directory() returns the current working directory."""
        import os

        from attractor_agent.environment import LocalEnvironment

        env = LocalEnvironment()
        result = await env.working_directory()
        assert isinstance(result, str)
        assert len(result) > 0
        # Must match os.getcwd() when no working_dir override is set
        assert result == os.getcwd()

    @pytest.mark.asyncio
    async def test_working_directory_honours_override(self):
        """When working_dir is supplied, working_directory() returns that path."""
        import tempfile

        from attractor_agent.environment import LocalEnvironment

        with tempfile.TemporaryDirectory() as tmp:
            env = LocalEnvironment(working_dir=tmp)
            result = await env.working_directory()
            assert result == tmp


# ================================================================== #
# §9.8.2: knowledge_cutoff wired from ModelInfo through env block
# ================================================================== #


class TestKnowledgeCutoffWiring:
    """§9.8.2: ModelInfo.knowledge_cutoff is populated and surfaced in the env block."""

    def test_model_info_has_knowledge_cutoff_field(self):
        """ModelInfo dataclass must have a knowledge_cutoff attribute."""
        from attractor_llm.catalog import ModelInfo

        info = ModelInfo(
            id="test-model",
            provider="test",
            display_name="Test",
            context_window=8192,
            knowledge_cutoff="2024-08",
        )
        assert info.knowledge_cutoff == "2024-08"

    def test_model_info_knowledge_cutoff_defaults_to_none(self):
        """knowledge_cutoff defaults to None for models where it is not set."""
        from attractor_llm.catalog import ModelInfo

        info = ModelInfo(
            id="test-model",
            provider="test",
            display_name="Test",
            context_window=8192,
        )
        assert info.knowledge_cutoff is None

    def test_catalog_models_have_knowledge_cutoff(self):
        """All catalog models with known cutoffs should have the field populated."""
        from attractor_llm.catalog import get_model_info

        for model_id, expected_cutoff in [
            ("claude-opus-4-6", "2024-08"),
            ("claude-sonnet-4-5", "2024-08"),
            ("gpt-5.2", "2024-04"),
            ("gemini-3-pro-preview", "2024-12"),
            ("gemini-3-flash-preview", "2024-12"),
        ]:
            info = get_model_info(model_id)
            assert info is not None, f"Model {model_id} not found in catalog"
            assert info.knowledge_cutoff == expected_cutoff, (
                f"{model_id}: expected {expected_cutoff!r}, got {info.knowledge_cutoff!r}"
            )

    def test_build_environment_context_includes_knowledge_cutoff(self):
        """build_environment_context() emits 'Knowledge cutoff:' when provided."""
        from attractor_agent.env_context import build_environment_context

        result = build_environment_context(
            working_dir="/tmp",
            model="claude-sonnet-4-5",
            knowledge_cutoff="2024-08",
            git_info={"is_git": False, "branch": "", "modified_count": 0,
                      "untracked_count": 0, "recent_commits": []},
        )
        assert "Knowledge cutoff: 2024-08" in result

    def test_build_environment_context_omits_cutoff_when_none(self):
        """build_environment_context() must NOT emit 'Knowledge cutoff:' when None."""
        from attractor_agent.env_context import build_environment_context

        result = build_environment_context(
            working_dir="/tmp",
            model="claude-sonnet-4-5",
            knowledge_cutoff=None,
            git_info={"is_git": False, "branch": "", "modified_count": 0,
                      "untracked_count": 0, "recent_commits": []},
        )
        assert "Knowledge cutoff" not in result


# ================================================================== #
# §4.10 ToolHandler — context_updates only on success
# ================================================================== #


class TestToolHandlerContextSpec:
    """§4.10: tool.<id>.output is set in context ONLY on exit code 0.

    On failure the handler returns Outcome(status=FAIL) with NO context
    update — downstream nodes on the failure edge cannot reference the
    tool's stdout via $tool.<id>.output.
    """

    @pytest.mark.asyncio
    async def test_output_stored_in_context_on_success(self):
        """Exit 0: tool.<id>.output written to context."""
        from attractor_pipeline import parse_dot, run_pipeline, HandlerRegistry
        from attractor_pipeline.handlers import register_default_handlers
        from attractor_pipeline.engine.runner import PipelineStatus

        g = parse_dot("""
        digraph T {
            start [shape=Mdiamond]
            t [shape=parallelogram, tool_command="echo hello_spec"]
            done [shape=Msquare]
            start -> t -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)
        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "hello_spec" in result.context.get("tool.t.output", "")

    @pytest.mark.asyncio
    async def test_output_not_stored_in_context_on_failure(self):
        """Exit non-zero: tool.<id>.output NOT written to context (spec §4.10)."""
        from attractor_pipeline import parse_dot, run_pipeline, HandlerRegistry
        from attractor_pipeline.handlers import register_default_handlers
        from attractor_pipeline.engine.runner import PipelineStatus

        g = parse_dot("""
        digraph T {
            start [shape=Mdiamond]
            t [shape=parallelogram, tool_command="exit 1"]
            done [shape=Msquare]
            start -> t [condition="outcome = fail"]
            t -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)
        result = await run_pipeline(g, registry)
        # tool.t.output must NOT be set — spec §4.10 context_updates on success only
        assert "tool.t.output" not in result.context
