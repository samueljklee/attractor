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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from attractor_agent.abort import AbortSignal
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

    Two complementary APIs:
    - ``ask()``          -- legacy flat-string API (backward compat).
    - ``ask_question()`` -- structured Question → Answer API (Spec §6.1-6.3).
    """

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        question_type: QuestionType | None = None,
    ) -> str:
        """Ask the human a question and return their response (flat API).

        Args:
            question: The question text to present.
            options: Optional list of valid choices (e.g., ["yes", "no"]).
            node_id: The pipeline node ID (for context).
            question_type: Optional hint for the type of question.
                If None, inferred from options: SINGLE_SELECT when
                options are provided, FREE_TEXT otherwise.

        Returns:
            The human's response string.
        """
        ...

    async def ask_question(self, question: Question) -> Answer:
        """Ask the human a structured question and return a structured answer.

        Spec §6.1-6.3.  Implementations that only override ``ask()`` can
        use the mixin ``_ask_question_via_ask()`` helper (provided by concrete
        classes) to avoid code duplication.

        Args:
            question: Structured ``Question`` descriptor.

        Returns:
            Structured ``Answer`` with value, selected_option, and text.
        """
        ...


# ------------------------------------------------------------------ #
# Built-in interviewer implementations
# ------------------------------------------------------------------ #


class AutoApproveInterviewer:
    """Auto-approves all human gates. For testing and CI."""

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        question_type: QuestionType | None = None,
    ) -> str:
        if options:
            return options[0]
        return "approved"

    async def ask_question(self, question: Question) -> Answer:
        value = await self.ask(
            question=question.text,
            options=question.options,
            question_type=question.question_type,
        )
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)


class ConsoleInterviewer:
    """Prompts on stdin for human input. For CLI use."""

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        question_type: QuestionType | None = None,
    ) -> str:
        import asyncio

        prompt = f"\n[HUMAN GATE: {node_id}]\n{question}"
        if options:
            prompt += f"\nOptions: {', '.join(options)}"
        prompt += "\n> "

        # Run input() in a thread to avoid blocking the event loop
        return await asyncio.to_thread(input, prompt)

    async def ask_question(self, question: Question) -> Answer:
        value = await self.ask(
            question=question.text,
            options=question.options,
            question_type=question.question_type,
        )
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)


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

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        question_type: QuestionType | None = None,
    ) -> str:
        return await self._callback(question, options, node_id or None)

    async def ask_question(self, question: Question) -> Answer:
        value = await self.ask(
            question=question.text,
            options=question.options,
            question_type=question.question_type,
        )
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)


class QueueInterviewer:
    """Reads from a pre-filled answer queue. Spec §6, §6.4, §11.8.

    Designed for deterministic testing: supply a list of answers
    up front and they are returned in order for each ``ask()`` call.
    Returns ``"SKIPPED"`` when the queue is exhausted (Spec §6.4).
    """

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self._index = 0

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        question_type: QuestionType | None = None,
    ) -> str:
        if self._index >= len(self._answers):
            return "SKIPPED"
        answer = self._answers[self._index]
        self._index += 1
        return answer

    async def ask_question(self, question: Question) -> Answer:
        value = await self.ask(
            question=question.text,
            options=question.options,
            question_type=question.question_type,
        )
        selected = value if (question.options and value in question.options) else None
        return Answer(value=value, selected_option=selected, text=value)


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
        # Build question from node's prompt/label
        question = node.prompt or node.label or f"Approve '{node.id}'?"

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
        question_type: QuestionType | None = None
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

        # Ask the human
        try:
            response = await self._interviewer.ask(
                question=question,
                options=options,
                node_id=node.id,
                question_type=question_type,
            )
        except Exception as exc:  # noqa: BLE001
            return HandlerResult(
                status=Outcome.FAIL,
                failure_reason=f"Interviewer error: {exc}",
            )

        # Store response in context
        context[f"human.{node.id}.response"] = response

        # Use response as preferred_label for edge selection
        return HandlerResult(
            status=Outcome.SUCCESS,
            preferred_label=response,
            output=response,
            notes=f"Human responded: {response}",
        )
