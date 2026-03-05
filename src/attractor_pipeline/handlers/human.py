"""Human-in-the-loop handler for wait.human nodes.

Implements the Interviewer pattern from attractor-spec §6, §11.8.
The handler delegates to an Interviewer implementation to ask
the human a question and wait for a response.

Implementations:
- AutoApproveInterviewer: always approves (for testing and CI)
- ConsoleInterviewer: prompts on stdin (for CLI use)
- CallbackInterviewer: delegates to an async callback (for embedding)
- QueueInterviewer: reads from a pre-filled answer queue (for testing)
"""

from __future__ import annotations

import time as _time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.events import (
    EventEmitter,
    InterviewCompleted,
    InterviewStarted,
)
from attractor_pipeline.engine.runner import HandlerResult, Outcome
from attractor_pipeline.graph import Graph, Node

# ------------------------------------------------------------------ #
# Question type enum (Spec §6, §11.8)
# ------------------------------------------------------------------ #


class QuestionType(StrEnum):
    """Type of question for human-in-the-loop interactions. Spec §6.

    Informational hint for interviewer implementations -- the type
    is not enforced by the protocol but allows UIs to render
    appropriate controls (radio buttons, checkboxes, text fields, etc.).
    """

    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    FREE_TEXT = "free_text"
    CONFIRM = "confirm"


# ------------------------------------------------------------------ #
# Structured Question / Answer models (Spec §6.1-6.3, §11.8)
# ------------------------------------------------------------------ #


@dataclass
class Question:
    """Structured question for human-in-the-loop interactions. Spec §6.1.

    Carries type hints, option lists, default answers, timeout, stage
    label, and arbitrary metadata so Interviewer implementations can
    render appropriate UI controls.

    Backward compat: the flat ``ask()`` API on ``Interviewer`` is
    unchanged; ``ask_question()`` converts to/from Question/Answer.
    """

    text: str
    question_type: QuestionType = QuestionType.FREE_TEXT
    options: list[str] | None = None
    default: str | None = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    """Structured answer from human-in-the-loop interactions. Spec §6.2-6.3.

    Contains the raw response value, the matched option (for choice
    questions), and the display text (may differ from *value* when an
    interviewer normalises the selection).
    """

    value: str
    selected_option: str | None = None
    text: str = ""


# ------------------------------------------------------------------ #
# Interviewer protocol
# ------------------------------------------------------------------ #


class Interviewer(Protocol):
    """Interface for human-in-the-loop interactions. Spec §6.

    Implementations handle how questions are presented to and
    answered by humans: console prompts, web UI, Slack, etc.

    Only ``ask()`` is required by the protocol.  Concrete classes
    typically also expose ``ask_question()`` for convenience but
    both methods operate on the same ``Question`` / ``Answer`` types.
    """

    async def ask(self, question: Question) -> Answer:
        """Ask the human a question and return their structured response.

        §11.8.1 / §6.1 — ask() accepts a structured Question and returns
        a structured Answer so that all context (options, type hints,
        metadata) is available to the implementation and callers receive
        a rich Answer object (value, selected_option, text).

        Args:
            question: Structured ``Question`` descriptor carrying the
                question text, type hint, option list, default, and
                any implementation-specific metadata.

        Returns:
            Structured ``Answer`` with value, selected_option, and text.
        """
        ...


# ------------------------------------------------------------------ #
# Module-level bridge helper (Spec §6.1-6.3)
# ------------------------------------------------------------------ #


async def ask_question_via_ask(interviewer: Interviewer, question: Question) -> Answer:
    """Delegate to Interviewer.ask() with a structured Question.

    Kept for backward compatibility with external implementors.
    Since ask() now accepts Question and returns Answer directly,
    this helper is a simple pass-through.

    Args:
        interviewer: Any object satisfying the ``Interviewer`` protocol.
        question: Structured ``Question`` descriptor.

    Returns:
        Structured ``Answer`` from the interviewer.
    """
    return await interviewer.ask(question)


# ------------------------------------------------------------------ #
# Built-in interviewer implementations
# ------------------------------------------------------------------ #


class AutoApproveInterviewer:
    """Auto-approves all human gates. For testing and CI."""

    async def ask(self, question: Question) -> Answer:
        if question.options:
            value = question.options[0]
            return Answer(value=value, selected_option=value, text=value)
        return Answer(value="approved", text="approved")

    async def ask_question(self, question: Question) -> Answer:
        return await self.ask(question)


class ConsoleInterviewer:
    """Prompts on stdin for human input. For CLI use."""

    async def ask(self, question: Question) -> Answer:
        import asyncio

        prompt = f"\n[HUMAN GATE: {question.stage}]\n{question.text}"
        if question.options:
            prompt += f"\nOptions: {', '.join(question.options)}"
        prompt += "\n> "

        # Run input() in a thread to avoid blocking the event loop
        value = await asyncio.to_thread(input, prompt)
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)

    async def ask_question(self, question: Question) -> Answer:
        return await self.ask(question)


class CallbackInterviewer:
    """Delegates to a provided async callback function. Spec §6, §11.8.

    Useful for embedding the pipeline in applications that provide
    their own UI or messaging layer for human interaction.
    """

    def __init__(
        self,
        callback: Callable[
            [str, list[str] | None, str | None],
            Awaitable[str],
        ],
    ) -> None:
        self._callback = callback

    async def ask(self, question: Question) -> Answer:
        value = await self._callback(
            question.text,
            question.options,
            question.stage or None,
        )
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)

    async def ask_question(self, question: Question) -> Answer:
        return await self.ask(question)


class QueueInterviewer:
    """Reads from a pre-filled answer queue. Spec §6, §6.4, §11.8.

    Designed for deterministic testing: supply a list of answers
    up front and they are returned in order for each ``ask()`` call.
    Returns an Answer with value ``"SKIPPED"`` when the queue is
    exhausted (Spec §6.4).
    """

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._index = 0

    async def ask(self, question: Question) -> Answer:
        if self._index >= len(self._answers):
            return Answer(value="SKIPPED")
        value = self._answers[self._index]
        self._index += 1
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)

    async def ask_question(self, question: Question) -> Answer:
        return await self.ask(question)


class HumanHandler:
    """Handler for wait.human nodes (shape=house). Spec §4.6.

    Presents a question to the human via the Interviewer interface
    and waits for a response. The response is stored in context
    and used for downstream edge selection.
    """

    def __init__(self, interviewer: Interviewer | None = None) -> None:
        self._interviewer = interviewer or AutoApproveInterviewer()

    async def execute(
        self,
        node: Node,
        context: dict[str, Any],
        graph: Graph,
        logs_root: Path | None,
        abort_signal: AbortSignal | None,
    ) -> HandlerResult:
        # Build question text from node's prompt/label
        question_text = node.prompt or node.label or f"Approve '{node.id}'?"

        # Extract options from outgoing edge labels
        edges = graph.outgoing_edges(node.id)
        options = [e.label for e in edges if e.label] or None

        # Check abort before blocking on human input
        if abort_signal and abort_signal.is_set:
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason="Aborted while waiting for human input",
            )

        # Infer question type from options (Spec §6, §11.8)
        if (
            options
            and len(options) == 2
            and {o.lower() for o in options}
            <= {
                "yes",
                "no",
                "approve",
                "reject",
            }
        ):
            question_type = QuestionType.CONFIRM
        elif options:
            question_type = QuestionType.SINGLE_SELECT
        else:
            question_type = QuestionType.FREE_TEXT

        # Build structured Question (Spec §6.1)
        question = Question(
            text=question_text,
            question_type=question_type,
            options=options,
            stage=node.id,
        )

        # Emit interview event (Spec 9.6)
        _emitter: EventEmitter | None = context.get("_event_emitter")
        if _emitter:
            _emitter.emit(InterviewStarted(question=question_text, stage=node.id))
        interview_start = _time.monotonic()

        # Ask the human
        try:
            answer = await self._interviewer.ask(question)
            response = answer.value
            if _emitter:
                _emitter.emit(
                    InterviewCompleted(
                        question=question_text,
                        answer=response,
                        duration=_time.monotonic() - interview_start,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"Interviewer error: {exc}",
            )

        # Use response as preferred_label for edge selection.
        # Store in context_updates — engine applies at §3.3 step 4.
        return HandlerResult(
            status=Outcome.SUCCESS,
            preferred_label=response,
            output=response,
            context_updates={f"human.{node.id}.response": response},
            notes=f"Human responded: {response}",
        )
