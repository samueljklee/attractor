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
    CallbackInterviewer,
    CodergenBackend,
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    HumanHandler,
    Interviewer,
    QuestionType,
    QueueInterviewer,
    StartHandler,
    ToolHandler,
    register_default_handlers,
)
from attractor_pipeline.parser import parse_dot
from attractor_pipeline.stylesheet import (
    Stylesheet,
    apply_stylesheet,
    parse_stylesheet,
)
from attractor_pipeline.transforms import (
    GraphTransform,
    VariableExpansionTransform,
    apply_transforms,
)

__all__ = [
    # Parser
    "parse_dot",
    # Stylesheet
    "Stylesheet",
    "parse_stylesheet",
    "apply_stylesheet",
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
    "CallbackInterviewer",
    "QueueInterviewer",
    "QuestionType",
    "register_default_handlers",
    # Transforms (Spec §9, §11.11)
    "GraphTransform",
    "VariableExpansionTransform",
    "apply_transforms",
    # Conditions
    "evaluate_condition",
]
