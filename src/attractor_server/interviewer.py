"""WebInterviewer -- HTTP-bridged human gate implementation.

Implements the Interviewer protocol (attractor-spec §6.1) using the
REST API instead of a terminal prompt.  When a pipeline reaches a
human gate (house node), the WebInterviewer:

1. Creates a PendingQuestion with a unique qid
2. Emits an ``interview.started`` SSE event
3. Blocks until an answer arrives via ``POST /pipelines/{id}/questions/{qid}/answer``
4. Returns a structured ``Answer`` to the pipeline engine

Spec reference: attractor-spec §6.1, §6.2, §9.5.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from attractor_pipeline.handlers.human import Answer, Question
from attractor_server.pipeline_manager import PendingQuestion, PipelineRun


class WebInterviewer:
    """HTTP-bridged interviewer for human gate nodes.

    Satisfies the ``Interviewer`` protocol: ``ask(Question) -> Answer``.

    Usage::

        interviewer = WebInterviewer(run)
        # Pipeline engine calls interviewer.ask(Question(...))
        # Web client sees interview.started in SSE stream
        # Web client POSTs answer to /questions/{qid}/answer
        # Pipeline continues with structured Answer
    """

    def __init__(
        self,
        run: PipelineRun,
        timeout: float = 300.0,
    ) -> None:
        self._run = run
        self._timeout = timeout

    async def ask(self, question: Question) -> Answer:
        """Present a question and wait for an answer via HTTP.

        Implements ``Interviewer.ask(question: Question) -> Answer``
        from attractor-spec §6.1.

        Blocks until either:
        - An answer arrives via POST /questions/{qid}/answer
        - The timeout expires (returns Question.default or "denied")
        - The pipeline is cancelled

        Args:
            question: Structured Question descriptor with text, options,
                type hint, default, and timeout override.

        Returns:
            Structured Answer with value, selected_option, and text.
        """
        qid = f"q_{uuid.uuid4().hex[:8]}"
        effective_timeout = (
            question.timeout_seconds if question.timeout_seconds is not None else self._timeout
        )
        default = question.default or "denied"
        stage = question.stage or self._run.current_node or "unknown"

        # Create pending question
        pending = PendingQuestion(
            qid=qid,
            question=question.text,
            stage=stage,
            timestamp=time.time(),
        )
        self._run.pending_questions[qid] = pending

        # Emit SSE event so web clients know a question is waiting
        self._run.emit(
            "interview.started",
            {
                "qid": qid,
                "question": question.text,
                "stage": stage,
                "options": question.options or [],
                "question_type": str(question.question_type),
            },
        )

        # Wait for answer
        answer_value: str
        try:
            await asyncio.wait_for(
                pending.answer_event.wait(),
                timeout=effective_timeout,
            )
            answer_value = pending.answer or default
        except TimeoutError:
            answer_value = default
            self._run.emit(
                "interview.timeout",
                {
                    "qid": qid,
                    "question": question.text,
                    "stage": stage,
                    "duration": effective_timeout,
                },
            )
        finally:
            # Clean up
            self._run.pending_questions.pop(qid, None)

        self._run.emit(
            "interview.completed",
            {
                "qid": qid,
                "question": question.text,
                "answer": answer_value,
                "duration": time.time() - pending.timestamp,
            },
        )

        # Match the selected option if it's in the options list
        selected: str | None = None
        if question.options and answer_value in question.options:
            selected = answer_value

        return Answer(value=answer_value, selected_option=selected, text=answer_value)


def submit_answer(run: PipelineRun, qid: str, answer: str) -> bool:
    """Submit an answer to a pending question.

    Called by the POST /questions/{qid}/answer endpoint.

    Returns True if the answer was accepted, False if qid not found.
    """
    pending = run.pending_questions.get(qid)
    if not pending:
        return False

    pending.answer = answer
    pending.answer_event.set()
    return True
