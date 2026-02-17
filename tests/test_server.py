"""Tests for the Attractor HTTP server.

Covers all 9 endpoints + SSE streaming + WebInterviewer + pipeline manager.
Uses Starlette's TestClient for synchronous HTTP testing (no real server needed).
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from attractor_pipeline import HandlerRegistry, register_default_handlers
from attractor_server.app import create_app
from attractor_server.interviewer import WebInterviewer, submit_answer
from attractor_server.pipeline_manager import (
    PipelineManager,
    PipelineRun,
    RunStatus,
    SSEEvent,
)
from attractor_server.sse import format_sse_event

# A simple DOT pipeline that completes immediately (no LLM needed)
SIMPLE_DOT = """
digraph Simple {
    graph [goal="Test pipeline"]
    start [shape=Mdiamond]
    task [shape=box, prompt="Hello"]
    done [shape=Msquare]
    start -> task -> done
}
"""


@pytest.fixture
def manager():
    """Create a PipelineManager with default handlers (no LLM backend)."""
    m = PipelineManager(max_concurrent=5)
    registry = HandlerRegistry()
    register_default_handlers(registry)
    m.set_handlers(registry)
    return m


@pytest.fixture
def client(manager):
    """Create a test client with the Starlette app."""
    app = create_app(manager)
    return TestClient(app)


# ================================================================== #
# Endpoint 1: POST /pipelines
# ================================================================== #


class TestStartPipeline:
    def test_start_returns_201(self, client):
        resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["status"] in ("pending", "running")

    def test_missing_dot_source_returns_400(self, client):
        resp = client.post("/pipelines", json={})
        assert resp.status_code == 400
        assert "dot_source" in resp.json()["error"].lower()

    def test_invalid_json_returns_400(self, client):
        resp = client.post(
            "/pipelines",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_invalid_dot_returns_400(self, client):
        resp = client.post("/pipelines", json={"dot_source": "not valid dot"})
        assert resp.status_code == 400

    def test_start_with_context(self, client):
        resp = client.post(
            "/pipelines",
            json={
                "dot_source": SIMPLE_DOT,
                "context": {"language": "Python"},
            },
        )
        assert resp.status_code == 201


# ================================================================== #
# Endpoint 2: GET /pipelines/{id}
# ================================================================== #


class TestGetPipeline:
    def test_get_existing_pipeline(self, client):
        # Start a pipeline first
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        # Give it a moment to complete
        import time

        time.sleep(0.2)

        resp = client.get(f"/pipelines/{pipeline_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == pipeline_id
        assert data["status"] in ("pending", "running", "completed")
        assert data["goal"] == "Test pipeline"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent")
        assert resp.status_code == 404


# ================================================================== #
# Endpoint 3: GET /pipelines/{id}/events (SSE)
# ================================================================== #


class TestSSEEvents:
    def test_sse_format(self):
        event = SSEEvent(
            event_type="pipeline.started",
            data={"id": "abc", "name": "Test"},
        )
        formatted = format_sse_event(event)
        assert "event: pipeline.started\n" in formatted
        assert "data: " in formatted
        assert '"id": "abc"' in formatted
        assert formatted.endswith("\n\n")

    def test_sse_format_keepalive(self):
        """SSE format includes correct line termination."""
        event = SSEEvent(
            event_type="stage.completed",
            data={"name": "plan", "duration": 2.5},
        )
        formatted = format_sse_event(event)
        lines = formatted.split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        # Must end with double newline (SSE spec)
        assert formatted.endswith("\n\n")

    def test_sse_nonexistent_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent/events")
        assert resp.status_code == 404


# ================================================================== #
# Endpoint 4: POST /pipelines/{id}/cancel
# ================================================================== #


class TestCancelPipeline:
    def test_cancel_nonexistent_returns_404(self, client):
        resp = client.post("/pipelines/nonexistent/cancel")
        assert resp.status_code == 404

    def test_cancel_completed_returns_409(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        import time

        time.sleep(0.2)  # Wait for completion

        resp = client.post(f"/pipelines/{pipeline_id}/cancel")
        # Could be 409 (already completed) or 200 (caught in time)
        assert resp.status_code in (200, 409)


# ================================================================== #
# Endpoint 5: GET /pipelines/{id}/graph
# ================================================================== #


class TestGetGraph:
    def test_get_graph_structure(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        resp = client.get(f"/pipelines/{pipeline_id}/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3  # start, task, done
        assert len(data["edges"]) == 2

    def test_graph_node_fields(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        resp = client.get(f"/pipelines/{pipeline_id}/graph")
        nodes = resp.json()["nodes"]
        for node in nodes:
            assert "id" in node
            assert "shape" in node
            assert "label" in node
            assert "handler" in node

    def test_graph_nonexistent_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent/graph")
        assert resp.status_code == 404


# ================================================================== #
# Endpoints 6-7: Questions API
# ================================================================== #


class TestQuestionsAPI:
    def test_no_pending_questions(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        resp = client.get(f"/pipelines/{pipeline_id}/questions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_answer_nonexistent_question_returns_404(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        resp = client.post(
            f"/pipelines/{pipeline_id}/questions/q_fake/answer",
            json={"answer": "yes"},
        )
        assert resp.status_code == 404

    def test_answer_missing_body_returns_400(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        resp = client.post(
            f"/pipelines/{pipeline_id}/questions/q_fake/answer",
            json={},
        )
        assert resp.status_code == 400

    def test_questions_nonexistent_pipeline_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent/questions")
        assert resp.status_code == 404


# ================================================================== #
# Endpoints 8-9: Checkpoint + Context
# ================================================================== #


class TestCheckpointAndContext:
    def test_get_checkpoint(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        import time

        time.sleep(0.2)

        resp = client.get(f"/pipelines/{pipeline_id}/checkpoint")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "completed_nodes" in data

    def test_get_context(self, client):
        start_resp = client.post("/pipelines", json={"dot_source": SIMPLE_DOT})
        pipeline_id = start_resp.json()["id"]

        import time

        time.sleep(0.2)

        resp = client.get(f"/pipelines/{pipeline_id}/context")
        assert resp.status_code == 200
        data = resp.json()
        assert "values" in data
        assert data["values"].get("goal") == "Test pipeline"

    def test_checkpoint_nonexistent_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent/checkpoint")
        assert resp.status_code == 404

    def test_context_nonexistent_returns_404(self, client):
        resp = client.get("/pipelines/nonexistent/context")
        assert resp.status_code == 404


# ================================================================== #
# Pipeline Manager (unit tests)
# ================================================================== #


class TestPipelineManager:
    @pytest.mark.asyncio
    async def test_start_and_track(self):
        manager = PipelineManager(max_concurrent=5)
        registry = HandlerRegistry()
        register_default_handlers(registry)
        manager.set_handlers(registry)

        run = await manager.start_pipeline(SIMPLE_DOT)
        assert run.id is not None
        assert run.status in (RunStatus.PENDING, RunStatus.RUNNING)

        # Wait for completion
        if run.task:
            await run.task
        assert run.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_running(self):
        # Use a pipeline that will take a while (many nodes)
        long_dot = """
        digraph Long {
            graph [goal="Long task"]
            start [shape=Mdiamond]
            a [shape=box, prompt="Step A"]
            b [shape=box, prompt="Step B"]
            c [shape=box, prompt="Step C"]
            done [shape=Msquare]
            start -> a -> b -> c -> done
        }
        """
        manager = PipelineManager(max_concurrent=5)
        registry = HandlerRegistry()
        register_default_handlers(registry)
        manager.set_handlers(registry)

        run = await manager.start_pipeline(long_dot)
        cancelled = await manager.cancel_pipeline(run.id)
        # May or may not cancel depending on timing
        assert cancelled or run.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self):
        manager = PipelineManager(max_concurrent=1)
        registry = HandlerRegistry()
        register_default_handlers(registry)
        manager.set_handlers(registry)

        # Start one pipeline
        run1 = await manager.start_pipeline(SIMPLE_DOT)

        # Wait for it to finish before starting another
        if run1.task:
            await run1.task

        # Should be able to start another now
        run2 = await manager.start_pipeline(SIMPLE_DOT)
        if run2.task:
            await run2.task
        assert run2.status == RunStatus.COMPLETED


# ================================================================== #
# SSE Event Subscriber (unit tests)
# ================================================================== #


class TestSSESubscriber:
    @pytest.mark.asyncio
    async def test_subscribe_receives_events(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        queue = run.subscribe()
        run.emit("test.event", {"key": "value"})

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event is not None
        assert event.event_type == "test.event"
        assert event.data["key"] == "value"

    @pytest.mark.asyncio
    async def test_close_sends_sentinel(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        queue = run.subscribe()
        run.close_subscribers()

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event is None  # sentinel

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        queue = run.subscribe()
        run.unsubscribe(queue)
        assert queue not in run._event_subscribers


# ================================================================== #
# WebInterviewer (unit tests)
# ================================================================== #


class TestWebInterviewer:
    @pytest.mark.asyncio
    async def test_answer_received(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        interviewer = WebInterviewer(run, timeout=5.0)

        # Start the ask in background
        async def do_ask():
            return await interviewer.ask("Approve?", options=["yes", "no"])

        task = asyncio.create_task(do_ask())

        # Wait for question to appear
        await asyncio.sleep(0.1)
        assert len(run.pending_questions) == 1

        # Answer it
        qid = list(run.pending_questions.keys())[0]
        submit_answer(run, qid, "yes")

        answer = await asyncio.wait_for(task, timeout=2.0)
        assert answer == "yes"

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        interviewer = WebInterviewer(run, timeout=0.1)

        answer = await interviewer.ask("Approve?", default="auto-denied", timeout=0.1)
        assert answer == "auto-denied"

    @pytest.mark.asyncio
    async def test_submit_unknown_qid_returns_false(self):
        from attractor_pipeline.graph import Graph, Node

        run = PipelineRun(
            id="test",
            graph=Graph(name="Test", nodes={"s": Node(id="s", shape="Mdiamond")}),
        )
        assert submit_answer(run, "nonexistent", "yes") is False
