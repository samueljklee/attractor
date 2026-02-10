"""Attractor Pipeline Engine.

DOT-based pipeline runner for orchestrating multi-stage AI workflows.
"""

from attractor_pipeline.conditions import evaluate_condition
from attractor_pipeline.engine.runner import (
    Checkpoint,
    Handler,
    HandlerRegistry,
    HandlerResult,
    Outcome,
    PipelineResult,
    PipelineStatus,
    run_pipeline,
    select_edge,
)
from attractor_pipeline.graph import Edge, Graph, Node, NodeShape
from attractor_pipeline.handlers import (
    CodergenBackend,
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    HumanHandler,
    Interviewer,
    StartHandler,
    ToolHandler,
    register_default_handlers,
)
from attractor_pipeline.parser import parse_dot

__all__ = [
    # Parser
    "parse_dot",
    # Graph model
    "Graph",
    "Node",
    "Edge",
    "NodeShape",
    # Engine
    "run_pipeline",
    "select_edge",
    "PipelineResult",
    "PipelineStatus",
    "HandlerResult",
    "Outcome",
    "Handler",
    "HandlerRegistry",
    "Checkpoint",
    # Handlers
    "StartHandler",
    "ExitHandler",
    "ConditionalHandler",
    "ToolHandler",
    "CodergenHandler",
    "CodergenBackend",
    "HumanHandler",
    "Interviewer",
    "register_default_handlers",
    # Conditions
    "evaluate_condition",
]
