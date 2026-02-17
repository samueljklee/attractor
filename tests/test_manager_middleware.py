"""Tests for manager loop handler and middleware/interceptor chain.

Covers:
- Manager: file-based child, success condition, max iterations,
  abort, context propagation, missing child_graph
- Middleware: logging, token counting, caching, rate limiting, chain composition,
  MiddlewareClient integration
"""

from __future__ import annotations

from typing import Any

import pytest

from attractor_agent.abort import AbortSignal
from attractor_llm.client import Client
from attractor_llm.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MiddlewareClient,
    RateLimitMiddleware,
    TokenCountingMiddleware,
    apply_middleware,
)
from attractor_llm.types import (
    Message,
    Request,
    Usage,
)
from attractor_pipeline import (
    HandlerRegistry,
    Outcome,
    PipelineStatus,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.graph import Edge, Graph, Node
from attractor_pipeline.handlers.manager import ManagerHandler
from tests.helpers import MockAdapter, make_text_response

# ================================================================== #
# Manager Handler
# ================================================================== #


class TestManagerHandler:
    @pytest.mark.asyncio
    async def test_file_based_child_pipeline(self, tmp_path):
        """Manager loads child pipeline from a .dot file."""
        child_file = tmp_path / "child.dot"
        child_file.write_text(
            "digraph Child {\n"
            "  start [shape=Mdiamond]\n"
            "  task [shape=box]\n"
            "  done [shape=Msquare]\n"
            "  start -> task -> done\n"
            "}\n"
        )

        g = Graph(
            name="Supervised",
            goal="File test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "mgr": Node(
                    id="mgr",
                    shape="hexagon",
                    attrs={"child_graph": str(child_file)},
                ),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="mgr"),
                Edge(source="mgr", target="done"),
            ],
        )

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "mgr" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_missing_child_graph_fails(self):
        """Manager with no child_graph returns FAIL from handler."""
        # Test directly on the handler since the pipeline engine
        # doesn't stop on handler FAIL -- it falls through to edge selection
        handler = ManagerHandler()
        registry = HandlerRegistry()
        register_default_handlers(registry)
        handler.set_handlers(registry)

        node = Node(id="mgr", shape="hexagon", attrs={})
        result = await handler.execute(node, {"goal": "test"}, Graph(name="g"), None, None)
        assert result.status == Outcome.FAIL
        assert "child_graph" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_abort_cancels_manager(self, tmp_path):
        """Abort signal stops the manager loop."""
        child_file = tmp_path / "child.dot"
        child_file.write_text(
            "digraph C { start [shape=Mdiamond]; done [shape=Msquare]; start -> done }\n"
        )

        g = Graph(
            name="Abort",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "mgr": Node(
                    id="mgr",
                    shape="hexagon",
                    attrs={"child_graph": str(child_file)},
                ),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="mgr"),
                Edge(source="mgr", target="done"),
            ],
        )

        abort = AbortSignal()
        abort.set()

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, abort_signal=abort)
        assert result.status == PipelineStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_manager_stores_iteration_results(self, tmp_path):
        """Manager stores iteration metadata in context."""
        child_file = tmp_path / "child.dot"
        child_file.write_text(
            "digraph C { start [shape=Mdiamond]; done [shape=Msquare]; start -> done }\n"
        )

        g = Graph(
            name="Meta",
            goal="Metadata test",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "mgr": Node(
                    id="mgr",
                    shape="hexagon",
                    attrs={"child_graph": str(child_file)},
                ),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="mgr"),
                Edge(source="mgr", target="done"),
            ],
        )

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        iterations = result.context.get("manager.mgr.iterations")
        assert iterations is not None
        assert len(iterations) >= 1
        assert iterations[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_manager_max_iterations(self, tmp_path):
        """Manager stops after max_iterations if child keeps failing."""
        child_file = tmp_path / "child.dot"
        # Child graph with no start node -- will fail validation
        child_file.write_text(
            "digraph Bad { task [shape=box]; done [shape=Msquare]; task -> done }\n"
        )

        g = Graph(
            name="MaxIter",
            goal="Max iterations",
            nodes={
                "start": Node(id="start", shape="Mdiamond"),
                "mgr": Node(
                    id="mgr",
                    shape="hexagon",
                    attrs={
                        "child_graph": str(child_file),
                        "max_iterations": "2",
                        "success_condition": "status=completed",
                    },
                ),
                "done": Node(id="done", shape="Msquare"),
            },
            edges=[
                Edge(source="start", target="mgr"),
                Edge(source="mgr", target="done"),
            ],
        )

        registry = HandlerRegistry()
        register_default_handlers(registry)

        # Test handler directly -- pipeline engine doesn't stop on FAIL,
        # it falls through to edge selection
        handler = ManagerHandler()
        handler.set_handlers(registry)
        node = g.nodes["mgr"]
        ctx: dict[str, Any] = {"goal": "test"}
        result = await handler.execute(node, ctx, g, None, None)
        assert result.status == Outcome.FAIL
        assert "2 iteration" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_manager_handler_direct(self):
        """Direct test of ManagerHandler without pipeline."""
        handler = ManagerHandler()
        handler.set_handlers(HandlerRegistry())

        # No child_graph attribute
        node = Node(id="test", shape="hexagon", attrs={})
        result = await handler.execute(node, {}, Graph(name="g"), None, None)
        assert result.status == Outcome.FAIL
        assert "child_graph" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_manager_handler_no_registry(self):
        """ManagerHandler without registry fails gracefully."""
        handler = ManagerHandler()  # no set_handlers call

        node = Node(id="test", shape="hexagon", attrs={"child_graph": "x"})
        result = await handler.execute(node, {}, Graph(name="g"), None, None)
        assert result.status == Outcome.FAIL
        assert "registry" in (result.failure_reason or "").lower()


# ================================================================== #
# Middleware: Token Counting
# ================================================================== #


class TestTokenCountingMiddleware:
    @pytest.mark.asyncio
    async def test_counts_tokens_across_calls(self):
        counter = TokenCountingMiddleware()
        adapter = MockAdapter(
            responses=[
                make_text_response("hello"),
                make_text_response("world"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [counter])

        await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("1")])
        )
        await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("2")])
        )

        assert counter.call_count == 2
        assert counter.total_usage.input_tokens == 20  # 10 + 10
        assert counter.total_usage.output_tokens == 10  # 5 + 5

    @pytest.mark.asyncio
    async def test_cost_estimate(self):
        counter = TokenCountingMiddleware()
        counter.total_usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
        cost = counter.total_cost_estimate
        assert cost > 0


# ================================================================== #
# Middleware: Caching
# ================================================================== #


class TestCachingMiddleware:
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_llm_call(self):
        cache = CachingMiddleware(max_size=10)
        adapter = MockAdapter(
            responses=[
                make_text_response("cached response"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [cache])

        req = Request(model="mock", provider="mock", messages=[Message.user("hello")])

        # First call: cache miss
        r1 = await mw_client.complete(req)
        assert r1.text == "cached response"
        assert adapter.call_count == 1
        assert cache.size == 1

        # Second call: cache hit (no adapter call)
        r2 = await mw_client.complete(req)
        assert r2.text == "cached response"
        assert adapter.call_count == 1  # still 1 -- cache hit
        assert cache.hit_rate > 0

    @pytest.mark.asyncio
    async def test_different_requests_not_cached(self):
        cache = CachingMiddleware(max_size=10)
        adapter = MockAdapter(
            responses=[
                make_text_response("response 1"),
                make_text_response("response 2"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [cache])

        await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("hello")])
        )
        await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("goodbye")])
        )

        assert adapter.call_count == 2
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        cache = CachingMiddleware(max_size=2)
        adapter = MockAdapter(
            responses=[
                make_text_response("a"),
                make_text_response("b"),
                make_text_response("c"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [cache])

        for msg in ["a", "b", "c"]:
            await mw_client.complete(
                Request(model="mock", provider="mock", messages=[Message.user(msg)])
            )

        assert cache.size == 2  # oldest evicted


# ================================================================== #
# Middleware: Logging
# ================================================================== #


class TestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_logging_passes_through(self):
        """Logging middleware doesn't modify request or response."""
        log_mw = LoggingMiddleware()
        adapter = MockAdapter(responses=[make_text_response("hello")])
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [log_mw])

        result = await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("test")])
        )
        assert result.text == "hello"


# ================================================================== #
# Middleware: Rate Limiting
# ================================================================== #


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_allows_requests_within_limit(self):
        rate = RateLimitMiddleware(max_requests_per_minute=100)
        adapter = MockAdapter(
            responses=[
                make_text_response("1"),
                make_text_response("2"),
                make_text_response("3"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [rate])

        for i in range(3):
            await mw_client.complete(
                Request(
                    model="mock",
                    provider="mock",
                    messages=[Message.user(str(i))],
                )
            )

        assert adapter.call_count == 3


# ================================================================== #
# Middleware: Chain Composition
# ================================================================== #


class TestMiddlewareChain:
    @pytest.mark.asyncio
    async def test_multiple_middleware_compose(self):
        """Multiple middleware execute in correct order."""
        counter = TokenCountingMiddleware()
        cache = CachingMiddleware(max_size=10)
        log_mw = LoggingMiddleware()

        adapter = MockAdapter(
            responses=[
                make_text_response("composed"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [log_mw, counter, cache])

        result = await mw_client.complete(
            Request(model="mock", provider="mock", messages=[Message.user("test")])
        )

        assert result.text == "composed"
        assert counter.call_count == 1
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_apply_middleware_returns_middleware_client(self):
        client = Client()
        mw_client = apply_middleware(client, [LoggingMiddleware()])
        assert isinstance(mw_client, MiddlewareClient)

    @pytest.mark.asyncio
    async def test_middleware_client_context_manager(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)
        mw_client = apply_middleware(client, [TokenCountingMiddleware()])

        async with mw_client:
            result = await mw_client.complete(
                Request(
                    model="mock",
                    provider="mock",
                    messages=[Message.user("test")],
                )
            )
        assert result.text == "ok"
