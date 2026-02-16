"""Attractor Coding Agent Loop.

Provides an autonomous agentic loop pairing LLMs with developer tools.
"""

from attractor_agent.abort import AbortSignal
from attractor_agent.events import EventEmitter, EventKind, SessionEvent
from attractor_agent.session import Session, SessionConfig, SessionState, SteeringTurn
from attractor_agent.tools.core import ALL_CORE_TOOLS
from attractor_agent.tools.registry import ToolRegistry
from attractor_agent.truncation import TruncationLimits, truncate_output

__all__ = [
    # Session
    "Session",
    "SessionConfig",
    "SessionState",
    "SteeringTurn",
    # Events
    "EventEmitter",
    "EventKind",
    "SessionEvent",
    # Tools
    "ToolRegistry",
    "ALL_CORE_TOOLS",
    # Truncation
    "TruncationLimits",
    "truncate_output",
    # Abort
    "AbortSignal",
]
