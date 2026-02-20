"""HTTP server for the attractor pipeline.

Provides a Starlette-based REST API for triggering and querying pipeline
runs.  Spec §11.11.5 requires the following route aliases alongside the
canonical /pipelines paths for backward compatibility:

    /run              → same handler as POST /pipelines
    /status/{id}      → same handler as GET  /pipelines/{id}
    /answer/{id}      → same handler as POST /pipelines/{id}/answer

Both canonical and alias paths are registered; neither set is removed so
existing integrations continue to work.
"""

from __future__ import annotations

import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# In-memory pipeline run store (replace with persistent store for production)
_runs: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ #
# Handlers
# ------------------------------------------------------------------ #


async def _handle_run_pipeline(request: Request) -> JSONResponse:
    """Start a new pipeline run.

    Accepts an optional JSON body with pipeline parameters.
    Returns the new run ID and initial status.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:  # noqa: BLE001
        body = {}

    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "id": run_id,
        "status": "pending",
        "pipeline": body.get("pipeline", ""),
        "input": body.get("input", {}),
        "output": None,
        "answer": None,
    }
    return JSONResponse({"id": run_id, "status": "pending"}, status_code=202)


async def _handle_get_status(request: Request) -> JSONResponse:
    """Retrieve the status of a pipeline run.

    Returns the run record or 404 if not found.
    """
    run_id = request.path_params["id"]
    run = _runs.get(run_id)
    if run is None:
        return JSONResponse({"error": f"Run '{run_id}' not found"}, status_code=404)
    return JSONResponse(run)


async def _handle_submit_answer(request: Request) -> JSONResponse:
    """Submit a human answer to an approval gate in a pipeline run.

    Expects a JSON body with an ``answer`` field.
    """
    run_id = request.path_params["id"]
    run = _runs.get(run_id)
    if run is None:
        return JSONResponse({"error": f"Run '{run_id}' not found"}, status_code=404)

    try:
        body: dict[str, Any] = await request.json()
    except Exception:  # noqa: BLE001
        body = {}

    run["answer"] = body.get("answer")
    run["status"] = "answered"
    return JSONResponse({"id": run_id, "status": "answered"})


# ------------------------------------------------------------------ #
# Routes -- canonical + alias paths (Spec §11.11.5)
# ------------------------------------------------------------------ #

routes = [
    # Canonical paths
    Route("/pipelines", _handle_run_pipeline, methods=["POST"]),
    Route("/pipelines/{id}", _handle_get_status, methods=["GET"]),
    Route("/pipelines/{id}/answer", _handle_submit_answer, methods=["POST"]),
    # Alias paths (Spec §11.11.5)
    Route("/run", _handle_run_pipeline, methods=["POST"]),
    Route("/status/{id}", _handle_get_status, methods=["GET"]),
    Route("/answer/{id}", _handle_submit_answer, methods=["POST"]),
]

app = Starlette(routes=routes)
