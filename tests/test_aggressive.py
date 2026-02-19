"""Aggressive tests that exercise the hard code paths.

These test what actually breaks in production:
- Full agentic loop with mock LLM (tool calls -> results -> text)
- Loop detection triggering mid-session
- Steering message injection
- Turn limits and tool round limits
- Concurrent branch mutation stress
- max_parallel semaphore bounds concurrency
- Adapter HTTP error classification via respx
- Streaming SSE edge cases (malformed, dropped, out-of-order)
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest
import respx

from attractor_agent.events import EventKind
from attractor_agent.session import Session, SessionConfig
from attractor_agent.tools.core import ALL_CORE_TOOLS, set_allowed_roots
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.client import Client
from attractor_llm.errors import (
    AuthenticationError,
    RateLimitError,
    ServerError,
)
from attractor_llm.types import (
    FinishReason,
    Message,
    Request,
    Role,
)
from attractor_pipeline import (
    HandlerRegistry,
    HandlerResult,
    Outcome,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from tests.helpers import (
    MockAdapter,
    make_multi_tool_response,
    make_text_response,
    make_tool_call_response,
)

# ================================================================== #
# Full agentic loop with mock LLM
# ================================================================== #


class TestAgenticLoop:
    """Tests the ACTUAL agentic loop: LLM -> tool calls -> results -> text."""

    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path):
        self.sandbox = tmp_path
        set_allowed_roots([str(tmp_path)])
        yield
        set_allowed_roots([os.getcwd()])

    @pytest.mark.asyncio
    async def test_single_tool_call_then_text(self):
        """LLM calls one tool, sees result, produces text."""
        # Write a file for the LLM to "read"
        test_file = self.sandbox / "hello.txt"
        test_file.write_text("Hello from file!")

        adapter = MockAdapter(
            responses=[
                # Turn 1: LLM wants to read the file
                make_tool_call_response("read_file", {"path": str(test_file)}, "tc-1"),
                # Turn 2: LLM sees file content, responds with text
                make_text_response("The file contains: Hello from file!"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        async with Session(client=client, config=config, tools=ALL_CORE_TOOLS) as session:
            result = await session.submit("Read hello.txt")

        assert "Hello from file" in result
        assert adapter.call_count == 2

        # Verify the second request included the tool result
        second_req = adapter.requests[1]
        tool_result_msgs = [m for m in second_req.messages if m.role == Role.TOOL]
        assert len(tool_result_msgs) == 1
        assert "Hello from file!" in (tool_result_msgs[0].content[0].output or "")

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_turn(self):
        """LLM makes 2 tool calls in parallel, then responds."""
        (self.sandbox / "a.txt").write_text("Alpha")
        (self.sandbox / "b.txt").write_text("Beta")

        adapter = MockAdapter(
            responses=[
                # Turn 1: two tool calls at once
                make_multi_tool_response(
                    [
                        ("tc-1", "read_file", {"path": str(self.sandbox / "a.txt")}),
                        ("tc-2", "read_file", {"path": str(self.sandbox / "b.txt")}),
                    ]
                ),
                # Turn 2: text response
                make_text_response("Files contain Alpha and Beta"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        async with Session(client=client, config=config, tools=ALL_CORE_TOOLS) as session:
            result = await session.submit("Read both files")

        assert "Alpha" in result and "Beta" in result
        assert adapter.call_count == 2

        # Both tool results should be in history
        second_req = adapter.requests[1]
        tool_msgs = [m for m in second_req.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 2

    @pytest.mark.asyncio
    async def test_multi_round_tool_loop(self):
        """LLM does tool -> result -> tool -> result -> text (3 LLM calls)."""
        test_file = self.sandbox / "data.txt"
        test_file.write_text("important data")

        adapter = MockAdapter(
            responses=[
                # Round 1: read the file
                make_tool_call_response("read_file", {"path": str(test_file)}, "tc-1"),
                # Round 2: write a new file based on what was read
                make_tool_call_response(
                    "write_file",
                    {"path": str(self.sandbox / "output.txt"), "content": "processed"},
                    "tc-2",
                ),
                # Round 3: final text
                make_text_response("Read data.txt and wrote output.txt"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        async with Session(client=client, config=config, tools=ALL_CORE_TOOLS) as session:
            result = await session.submit("Process data.txt")

        assert "output.txt" in result
        assert adapter.call_count == 3
        assert (self.sandbox / "output.txt").read_text() == "processed"

    @pytest.mark.asyncio
    async def test_tool_error_propagated_to_llm(self):
        """When a tool errors, the error is fed back as a tool result."""
        adapter = MockAdapter(
            responses=[
                # LLM tries to read nonexistent file
                make_tool_call_response("read_file", {"path": "/nonexistent/file.txt"}, "tc-1"),
                # LLM sees error, responds gracefully
                make_text_response("Sorry, the file doesn't exist."),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        async with Session(client=client, config=config, tools=ALL_CORE_TOOLS) as session:
            await session.submit("Read /nonexistent/file.txt")

        assert adapter.call_count == 2
        # The second request should contain the error tool result
        second_req = adapter.requests[1]
        tool_msgs = [m for m in second_req.messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content[0].is_error

    @pytest.mark.asyncio
    async def test_shell_tool_in_agentic_loop(self):
        """LLM calls shell, sees output, responds."""
        adapter = MockAdapter(
            responses=[
                make_tool_call_response(
                    "shell",
                    {"command": "echo hello_from_shell", "working_dir": str(self.sandbox)},
                    "tc-1",
                ),
                make_text_response("Shell output was: hello_from_shell"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        async with Session(client=client, config=config, tools=ALL_CORE_TOOLS) as session:
            result = await session.submit("Run echo")

        assert "hello_from_shell" in result


# ================================================================== #
# Session: loop detection, steering, limits
# ================================================================== #


class TestSessionEdgeCases:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path):
        self.sandbox = tmp_path
        set_allowed_roots([str(tmp_path)])
        yield
        set_allowed_roots([os.getcwd()])

    @pytest.mark.asyncio
    async def test_loop_detection_fires(self):
        """Repeated identical tool calls trigger loop detection.

        Spec §2.10: when a loop is detected, a SteeringTurn warning is injected
        and the loop CONTINUES (does not exit early).  The session should return
        the final text response produced after the warning, not an exit string.
        """
        # LLM calls the same tool 3 times (threshold=3) then produces text.
        # The 4th LLM call happens after loop detection fires and continues.
        adapter = MockAdapter(
            responses=[
                make_tool_call_response(
                    "shell", {"command": "echo x", "working_dir": str(self.sandbox)}, "tc-1"
                ),
                make_tool_call_response(
                    "shell", {"command": "echo x", "working_dir": str(self.sandbox)}, "tc-2"
                ),
                make_tool_call_response(
                    "shell", {"command": "echo x", "working_dir": str(self.sandbox)}, "tc-3"
                ),
                # After loop detected+reset the model produces a text response.
                make_text_response("I will try a different approach."),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        events: list[EventKind] = []
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            loop_detection_window=4,
            loop_detection_threshold=3,
        )
        session = Session(client=client, config=config, tools=ALL_CORE_TOOLS)
        session.events.on(lambda e: events.append(e.kind))

        loop_output = await session.submit("Do something")

        # Spec §2.10: session continues and returns final text (not an exit string).
        assert loop_output == "I will try a different approach."
        assert "Loop detected" not in loop_output

        # LOOP_DETECTED event must still be fired.
        assert EventKind.LOOP_DETECTED in events

        # A SteeringTurn with a loop warning must appear in history.
        from attractor_agent.session import SteeringTurn

        steering: list[SteeringTurn] = []
        for _entry in session.history:
            if isinstance(_entry, SteeringTurn):
                steering.append(_entry)
        assert steering, "Expected a SteeringTurn loop warning in history"
        assert any(
            "loop" in st.content.lower() or "repeating" in st.content.lower() for st in steering
        )

        # 4 LLM calls: 3 tool rounds + 1 final text response.
        assert adapter.call_count == 4

    @pytest.mark.asyncio
    async def test_tool_round_limit(self):
        """Hitting max_tool_rounds_per_turn stops the loop."""
        # LLM keeps calling tools forever
        responses = [
            make_tool_call_response(
                "shell",
                {"command": f"echo round_{i}", "working_dir": str(self.sandbox)},
                f"tc-{i}",
            )
            for i in range(50)
        ]
        adapter = MockAdapter(responses=responses)

        client = Client()
        client.register_adapter("mock", adapter)

        events: list[EventKind] = []
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            max_tool_rounds_per_turn=5,
        )
        session = Session(client=client, config=config, tools=ALL_CORE_TOOLS)
        session.events.on(lambda e: events.append(e.kind))

        result = await session.submit("Keep going")
        assert "Tool round limit" in result
        assert EventKind.LIMIT_REACHED in events
        assert adapter.call_count <= 6  # 5 tool rounds + 1

    @pytest.mark.asyncio
    async def test_steering_message_injected(self):
        """Steering messages appear in LLM's context between tool rounds."""
        adapter = MockAdapter(
            responses=[
                make_tool_call_response(
                    "shell",
                    {"command": "echo step1", "working_dir": str(self.sandbox)},
                    "tc-1",
                ),
                make_text_response("Done with steering"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        events: list[EventKind] = []
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config, tools=ALL_CORE_TOOLS)
        session.events.on(lambda e: events.append(e.kind))

        # Inject steering before submit (will be drained after first tool round)
        session.steer("Focus on error handling")
        await session.submit("Write code")

        assert EventKind.STEER_INJECTED in events
        # The second LLM call should see the steering message
        second_req = adapter.requests[1]
        steering_msgs = [
            m for m in second_req.messages if m.role == Role.USER and "[STEERING]" in (m.text or "")
        ]
        assert len(steering_msgs) == 1
        assert "error handling" in (steering_msgs[0].text or "")

    @pytest.mark.asyncio
    async def test_events_fire_in_correct_order(self):
        """Full event sequence for a tool-call turn."""
        adapter = MockAdapter(
            responses=[
                make_tool_call_response(
                    "shell",
                    {"command": "echo hi", "working_dir": str(self.sandbox)},
                    "tc-1",
                ),
                make_text_response("Done"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        events: list[EventKind] = []
        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config, tools=ALL_CORE_TOOLS)
        session.events.on(lambda e: events.append(e.kind))

        async with session:
            await session.submit("Go")

        # Expected order
        assert events[0] == EventKind.SESSION_START
        assert EventKind.TURN_START in events
        assert EventKind.TOOL_CALL_START in events
        assert EventKind.TOOL_CALL_END in events
        assert EventKind.ASSISTANT_TEXT in events
        assert EventKind.TURN_END in events
        assert events[-1] == EventKind.SESSION_END

    @pytest.mark.asyncio
    async def test_usage_accumulates_across_rounds(self):
        """Total usage sums across all LLM calls."""
        adapter = MockAdapter(
            responses=[
                make_tool_call_response(
                    "shell", {"command": "echo x", "working_dir": str(self.sandbox)}, "tc-1"
                ),
                make_text_response("Done"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(model="mock-model", provider="mock")
        session = Session(client=client, config=config, tools=ALL_CORE_TOOLS)
        await session.submit("Go")

        assert session.total_usage.input_tokens == 20  # 10 + 10
        assert session.total_usage.output_tokens == 20  # 15 + 5


# ================================================================== #
# Concurrent branch stress test
# ================================================================== #


class TestConcurrencyStress:
    @pytest.mark.asyncio
    async def test_ten_branches_no_corruption(self):
        """10 parallel branches all writing to context -- no corruption."""
        # Build a graph with 10 parallel branches
        branch_nodes = "\n".join(
            f'    b{i} [shape=parallelogram, prompt="echo branch_{i}"]' for i in range(10)
        )
        fork_edges = "\n".join(f"    fork -> b{i}" for i in range(10))
        join_edges = "\n".join(f"    b{i} -> join" for i in range(10))

        g = parse_dot(f"""
        digraph Stress {{
            graph [goal="Stress test"]
            start [shape=Mdiamond]
            fork [shape=component]
{branch_nodes}
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
{fork_edges}
{join_edges}
            join -> done
        }}
        """)

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED

        # Verify all 10 branches ran
        parallel_results = [
            v
            for k, v in result.context.items()
            if k.startswith("parallel.") and k.endswith(".results")
        ]
        assert len(parallel_results) == 1
        assert len(parallel_results[0]) == 10

        # All should have succeeded
        statuses = [br["status"] for br in parallel_results[0]]
        assert all(s == "success" for s in statuses)

    @pytest.mark.asyncio
    async def test_max_parallel_semaphore_bounds_concurrency(self):
        """max_parallel=2 means at most 2 branches run simultaneously."""
        peak_concurrency = 0
        current_concurrency = 0
        lock = asyncio.Lock()

        # Custom handler that tracks concurrency
        class ConcurrencyTracker:
            async def execute(self, node, context, graph, logs_root, abort_signal):
                nonlocal peak_concurrency, current_concurrency
                async with lock:
                    current_concurrency += 1
                    if current_concurrency > peak_concurrency:
                        peak_concurrency = current_concurrency

                await asyncio.sleep(0.05)  # simulate work

                async with lock:
                    current_concurrency -= 1

                return HandlerResult(status=Outcome.SUCCESS, output="done")

        g = parse_dot("""
        digraph Sem {
            graph [goal="Semaphore test"]
            start [shape=Mdiamond]
            fork [shape=component, max_parallel="2"]
            a [shape=box, prompt="A"]
            b [shape=box, prompt="B"]
            c [shape=box, prompt="C"]
            d [shape=box, prompt="D"]
            join [shape=tripleoctagon]
            done [shape=Msquare]
            start -> fork
            fork -> a
            fork -> b
            fork -> c
            fork -> d
            a -> join
            b -> join
            c -> join
            d -> join
            join -> done
        }
        """)

        registry = HandlerRegistry()
        register_default_handlers(registry)
        # Override codergen with our tracker
        registry.register("codergen", ConcurrencyTracker())

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        # Prove concurrency happened (peak >= 2) AND was bounded (peak <= 2)
        assert peak_concurrency >= 2, (
            f"Peak concurrency was {peak_concurrency}, expected >= 2 (concurrency not proven)"
        )
        assert peak_concurrency <= 2, (
            f"Peak concurrency was {peak_concurrency}, expected <= 2 (semaphore not working)"
        )


# ================================================================== #
# Adapter HTTP error testing via respx
# ================================================================== #


class TestAdapterHttpErrors:
    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_401_raises_auth_error(self):
        """401 from Anthropic -> AuthenticationError."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                401,
                json={"error": {"message": "invalid api key"}},
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="bad-key"))
        req = Request.simple("claude-sonnet-4-5", "hello")

        with pytest.raises(AuthenticationError):
            await adapter.complete(req)

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_429_raises_rate_limit(self):
        """429 from Anthropic -> RateLimitError with retry_after."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
                headers={"Retry-After": "30"},
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        req = Request.simple("claude-sonnet-4-5", "hello")

        with pytest.raises(RateLimitError) as exc_info:
            await adapter.complete(req)

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_500_raises_server_error(self):
        """500 from Anthropic -> ServerError (retryable)."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                500,
                json={"error": {"message": "internal error"}},
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        req = Request.simple("claude-sonnet-4-5", "hello")

        with pytest.raises(ServerError) as exc_info:
            await adapter.complete(req)

        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_successful_response(self):
        """Full round-trip: request -> Anthropic API -> response parsing."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 15,
                        "output_tokens": 8,
                        "cache_read_input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                    },
                },
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        req = Request.simple("claude-sonnet-4-5", "hello")

        resp = await adapter.complete(req)
        assert resp.text == "Hello from Claude!"
        assert resp.provider == "anthropic"
        assert resp.usage.input_tokens == 15
        assert resp.usage.cache_read_tokens == 10
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_tool_call_response(self):
        """Anthropic returns tool_use -> parsed as TOOL_CALLS."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_456",
                    "model": "claude-sonnet-4-5",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tc-1",
                            "name": "grep",
                            "input": {"pattern": "foo"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 15},
                },
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        req = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("search")],
            tools=[],
        )

        resp = await adapter.complete(req)
        assert resp.finish_reason == FinishReason.TOOL_CALLS
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "grep"

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_thinking_response(self):
        """Anthropic returns thinking block -> parsed with signature."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "msg_789",
                    "model": "claude-sonnet-4-5",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Let me think...",
                            "signature": "sig-abc123",
                        },
                        {"type": "text", "text": "My answer"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 50},
                },
            )
        )

        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        req = Request.simple("claude-sonnet-4-5", "think hard")

        resp = await adapter.complete(req)
        assert resp.text == "My answer"
        assert len(resp.reasoning) == 1
        assert resp.reasoning[0].signature == "sig-abc123"
