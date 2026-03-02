"""Starlette application with all 9 REST endpoints plus DoD §11.11.5 aliases.

Spec reference: attractor-spec §9.5, §11.11.5.
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from attractor_server.interviewer import submit_answer
from attractor_server.pipeline_manager import PipelineManager, RunStatus
from attractor_server.sse import sse_stream

# Global pipeline manager (set during app creation)
_manager: PipelineManager | None = None


def get_manager() -> PipelineManager:
    if _manager is None:
        raise RuntimeError("PipelineManager not initialized")
    return _manager


# ------------------------------------------------------------------ #
# Endpoint 1: POST /pipelines -- start a pipeline
# ------------------------------------------------------------------ #


async def start_pipeline(request: Request) -> JSONResponse:
    """Parse DOT source and start pipeline execution."""
    manager = get_manager()

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    dot_source = body.get("dot_source")
    if not dot_source:
        return JSONResponse({"error": "Missing 'dot_source' field"}, status_code=400)

    context = body.get("context", {})

    try:
        run = await manager.start_pipeline(dot_source, context)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=429)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Failed to start pipeline: {e}"}, status_code=400)

    return JSONResponse(
        {"id": run.id, "status": run.status.value},
        status_code=201,
    )


# ------------------------------------------------------------------ #
# Endpoint 2: GET /pipelines/{id} -- get status
# ------------------------------------------------------------------ #


async def get_pipeline(request: Request) -> JSONResponse:
    """Get pipeline status and progress."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    duration = None
    if run.end_time:
        duration = round(run.end_time - run.start_time, 2)
    elif run.status == RunStatus.RUNNING:
        import time

        duration = round(time.time() - run.start_time, 2)

    return JSONResponse(
        {
            "id": run.id,
            "status": run.status.value,
            "goal": run.graph.goal,
            "current_node": run.current_node,
            "completed_nodes": run.completed_nodes,
            "duration": duration,
            "error": run.error,
        }
    )


# ------------------------------------------------------------------ #
# Endpoint 3: GET /pipelines/{id}/events -- SSE stream
# ------------------------------------------------------------------ #


async def get_events(request: Request) -> StreamingResponse:
    """SSE event stream for real-time pipeline events."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse(  # type: ignore[return-value]
            {"error": f"Pipeline {pipeline_id} not found"}, status_code=404
        )

    return StreamingResponse(
        sse_stream(run),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------ #
# Endpoint 4: POST /pipelines/{id}/cancel -- cancel pipeline
# ------------------------------------------------------------------ #


async def cancel_pipeline(request: Request) -> JSONResponse:
    """Cancel a running pipeline."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]

    cancelled = await manager.cancel_pipeline(pipeline_id)
    if not cancelled:
        run = manager.get_run(pipeline_id)
        if not run:
            return JSONResponse(
                {"error": f"Pipeline {pipeline_id} not found"},
                status_code=404,
            )
        return JSONResponse(
            {"error": f"Pipeline {pipeline_id} is not running (status: {run.status.value})"},
            status_code=409,
        )

    return JSONResponse({"id": pipeline_id, "status": "cancelled"})


# ------------------------------------------------------------------ #
# Endpoint 5: GET /pipelines/{id}/graph -- graph structure
# ------------------------------------------------------------------ #


async def get_graph(request: Request) -> JSONResponse:
    """Get pipeline graph structure as JSON."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    nodes = [
        {
            "id": n.id,
            "shape": n.shape,
            "label": n.label or n.id,
            "prompt": n.prompt or "",
            "handler": n.effective_handler,
        }
        for n in run.graph.nodes.values()
    ]

    edges = [
        {
            "source": e.source,
            "target": e.target,
            "label": e.label,
            "condition": e.condition,
            "weight": e.weight,
        }
        for e in run.graph.edges
    ]

    return JSONResponse({"nodes": nodes, "edges": edges})


# ------------------------------------------------------------------ #
# Endpoint 6: GET /pipelines/{id}/questions -- pending questions
# ------------------------------------------------------------------ #


async def get_questions(request: Request) -> JSONResponse:
    """Get pending human gate questions."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    questions = [
        {
            "qid": q.qid,
            "question": q.question,
            "stage": q.stage,
            "timestamp": q.timestamp,
        }
        for q in run.pending_questions.values()
    ]

    return JSONResponse(questions)


# ------------------------------------------------------------------ #
# Endpoint 7: POST /pipelines/{id}/questions/{qid}/answer
# ------------------------------------------------------------------ #


async def answer_question(request: Request) -> JSONResponse:
    """Submit an answer to a pending human gate question."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    qid = request.path_params["qid"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    answer = body.get("answer")
    if not answer:
        return JSONResponse({"error": "Missing 'answer' field"}, status_code=400)

    accepted = submit_answer(run, qid, answer)
    if not accepted:
        return JSONResponse(
            {"error": f"Question {qid} not found or already answered"},
            status_code=404,
        )

    return JSONResponse({"qid": qid, "accepted": True})


# ------------------------------------------------------------------ #
# Endpoint 8: GET /pipelines/{id}/checkpoint
# ------------------------------------------------------------------ #


async def get_checkpoint(request: Request) -> JSONResponse:
    """Get current checkpoint state."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    return JSONResponse(
        {
            "current_node": run.current_node,
            "completed_nodes": run.completed_nodes,
            "context": {
                k: v
                for k, v in run.context.items()
                if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
            },
            "status": run.status.value,
        }
    )


# ------------------------------------------------------------------ #
# Endpoint 9: GET /pipelines/{id}/context
# ------------------------------------------------------------------ #


async def get_context(request: Request) -> JSONResponse:
    """Get current context key-value store."""
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    # Filter to safe-to-serialize values
    safe_context: dict[str, Any] = {}
    for k, v in run.context.items():
        if isinstance(v, (str, int, float, bool, list)):
            safe_context[k] = v
        elif isinstance(v, dict):
            safe_context[k] = str(v)[:500]

    return JSONResponse({"values": safe_context})


# ------------------------------------------------------------------ #
# DoD §11.11.5 alias: POST /answer/{id}
# ------------------------------------------------------------------ #


async def _answer_first_question(request: Request) -> JSONResponse:
    """Submit an answer to the FIRST pending question in a pipeline.

    Simplified alias for POST /pipelines/{id}/questions/{qid}/answer that
    does not require knowing the question ID in advance.  Accepts
    ``{"answer": "..."}`` and applies it to the first pending question.
    """
    manager = get_manager()
    pipeline_id = request.path_params["id"]
    run = manager.get_run(pipeline_id)

    if not run:
        return JSONResponse({"error": f"Pipeline {pipeline_id} not found"}, status_code=404)

    if not run.pending_questions:
        return JSONResponse({"error": "No pending questions for this pipeline"}, status_code=404)

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    answer = body.get("answer")
    if not answer:
        return JSONResponse({"error": "Missing 'answer' field"}, status_code=400)

    # Take the first pending question (ordered by insertion)
    first_qid = next(iter(run.pending_questions))
    accepted = submit_answer(run, first_qid, answer)
    if not accepted:
        return JSONResponse(
            {"error": f"Question {first_qid} could not be answered"},
            status_code=404,
        )

    return JSONResponse({"qid": first_qid, "accepted": True})


# ------------------------------------------------------------------ #
# App factory
# ------------------------------------------------------------------ #


def create_app(
    manager: PipelineManager | None = None,
) -> Starlette:
    """Create the Starlette application with all 9 endpoints plus DoD §11.11.5 aliases."""
    global _manager  # noqa: PLW0603
    _manager = manager or PipelineManager()

    routes = [
        Route("/pipelines", start_pipeline, methods=["POST"]),
        Route("/pipelines/{id}", get_pipeline, methods=["GET"]),
        Route("/pipelines/{id}/events", get_events, methods=["GET"]),
        Route("/pipelines/{id}/cancel", cancel_pipeline, methods=["POST"]),
        Route("/pipelines/{id}/graph", get_graph, methods=["GET"]),
        Route("/pipelines/{id}/questions", get_questions, methods=["GET"]),
        Route(
            "/pipelines/{id}/questions/{qid}/answer",
            answer_question,
            methods=["POST"],
        ),
        Route("/pipelines/{id}/checkpoint", get_checkpoint, methods=["GET"]),
        Route("/pipelines/{id}/context", get_context, methods=["GET"]),
        # DoD §11.11.5 aliases
        Route("/run", start_pipeline, methods=["POST"]),
        Route("/status/{id}", get_pipeline, methods=["GET"]),
        Route("/answer/{id}", _answer_first_question, methods=["POST"]),
    ]

    app = Starlette(routes=routes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
