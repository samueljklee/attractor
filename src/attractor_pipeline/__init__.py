"""Attractor Pipeline Engine.

DOT-based pipeline runner for orchestrating multi-stage AI workflows.
"""

from attractor_pipeline.conditions import evaluate_condition
from attractor_pipeline.engine.events import (
    CheckpointSaved,
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
    InterviewTimeout,
    ParallelBranchCompleted,
    ParallelBranchStarted,
    ParallelCompleted,
    ParallelStarted,
    PipelineCompleted,
    PipelineEvent,
    PipelineFailed,
    PipelineStarted,
    StageCompleted,
    StageFailed,
    StageRetrying,
    StageStarted,
)
from attractor_pipeline.engine.runner import (
    RETRY_PRESETS,
    Checkpoint,
    Handler,
    HandlerRegistry,
    HandlerResult,
    Outcome,
    PipelineContext,
    PipelineResult,
    PipelineStatus,
    get_retry_preset,
    run_pipeline,
    select_edge,
)
from attractor_pipeline.graph import Edge, Graph, Node, NodeShape
from attractor_pipeline.handlers import (
    Answer,
    AutoApproveInterviewer,
    CallbackInterviewer,
    CodergenBackend,
    CodergenHandler,
    ConditionalHandler,
    ExitHandler,
    HumanHandler,
    Interviewer,
    Question,
    QuestionType,
    QueueInterviewer,
    StartHandler,
    ToolHandler,
    ask_question_via_ask,
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
    "PipelineContext",
    "HandlerResult",
    "Outcome",
    "Handler",
    "HandlerRegistry",
    "Checkpoint",
    "RETRY_PRESETS",
    "get_retry_preset",
    # Handlers
    "StartHandler",
    "ExitHandler",
    "ConditionalHandler",
    "ToolHandler",
    "CodergenHandler",
    "CodergenBackend",
    "HumanHandler",
    "Interviewer",
    "AutoApproveInterviewer",
    "CallbackInterviewer",
    "QueueInterviewer",
    "Question",
    "Answer",
    "QuestionType",
    "ask_question_via_ask",
    "register_default_handlers",
    # Transforms (Spec §9, §11.11)
    "GraphTransform",
    "VariableExpansionTransform",
    "apply_transforms",
    # Conditions
    "evaluate_condition",
    # Events (Spec §9.6)
    "PipelineEvent",
    "EventEmitter",
    "PipelineStarted",
    "PipelineCompleted",
    "PipelineFailed",
    "StageStarted",
    "StageCompleted",
    "StageFailed",
    "StageRetrying",
    "ParallelStarted",
    "ParallelBranchStarted",
    "ParallelBranchCompleted",
    "ParallelCompleted",
    "InterviewStarted",
    "InterviewCompleted",
    "InterviewTimeout",
    "CheckpointSaved",
]
