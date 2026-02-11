"""Node handlers for the Attractor pipeline engine.

Each handler implements the Handler protocol and executes a specific
node type: start, exit, codergen (LLM), conditional, wait.human, tool,
parallel (fan-out), fan_in (join).
"""

from attractor_pipeline.engine.runner import HandlerRegistry
from attractor_pipeline.handlers.basic import (
    ConditionalHandler,
    ExitHandler,
    StartHandler,
    ToolHandler,
)
from attractor_pipeline.handlers.codergen import CodergenBackend, CodergenHandler
from attractor_pipeline.handlers.human import HumanHandler, Interviewer
from attractor_pipeline.handlers.parallel import FanInHandler, ParallelHandler

__all__ = [
    "StartHandler",
    "ExitHandler",
    "ConditionalHandler",
    "ToolHandler",
    "CodergenHandler",
    "CodergenBackend",
    "HumanHandler",
    "Interviewer",
    "ParallelHandler",
    "FanInHandler",
    "register_default_handlers",
]


def register_default_handlers(
    registry: HandlerRegistry,
    *,
    codergen_backend: CodergenBackend | None = None,
    interviewer: Interviewer | None = None,
) -> None:
    """Register all built-in handlers with a HandlerRegistry.

    Args:
        registry: The handler registry to populate.
        codergen_backend: Backend for LLM-powered nodes. If None,
            codergen nodes will return a placeholder response.
        interviewer: Implementation for human-gate nodes. If None,
            human gates auto-approve.
    """
    registry.register("start", StartHandler())
    registry.register("exit", ExitHandler())
    registry.register("conditional", ConditionalHandler())
    registry.register("tool", ToolHandler())
    registry.register("codergen", CodergenHandler(backend=codergen_backend))
    registry.register("wait.human", HumanHandler(interviewer=interviewer))

    # Parallel handlers need access to the registry for subgraph execution
    parallel = ParallelHandler()
    parallel.set_handlers(registry)
    registry.register("parallel", parallel)
    registry.register("fan_in", FanInHandler())
