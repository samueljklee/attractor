"""WebInterviewer -- HTTP-bridged human gate implementation.

Implements the Interviewer protocol but instead of prompting on the
terminal, it pushes questions to a pending queue and waits for answers
via the REST API (POST /pipelines/{id}/questions/{qid}/answer).

Spec reference: attractor-spec §6.1, §9.5.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from attractor_server.pipeline_manager import PendingQuestion, PipelineRun


class WebInterviewer:
    """HTTP-bridged interviewer for human gate nodes.

    When a pipeline hits a human gate (house node), the WebInterviewer:
    1. Creates a PendingQuestion with a unique qid
    2. Emits an interview.started SSE event
    3. Blocks until an answer arrives via POST /questions/{qid}/answer
    4. Returns the answer to the pipeline engine

    Usage::

        interviewer = WebInterviewer(run)
        # Pipeline engine calls interviewer.ask() at human gates
        # Web client sees interview.started in SSE stream
        # Web client POSTs answer to /questions/{qid}/answer
        # Pipeline continues
    """

    def __init__(
        self,
        run: PipelineRun,
        timeout: float = 300.0,
    ) -> None:
        self._run = run
        self._timeout = timeout

    async def ask(
        self,
        question: str,
        options: list[str] | None = None,
        node_id: str = "",
        *,
        timeout: float | None = None,
        default: str | None = None,
    ) -> str:
        """Present a question and wait for an answer via HTTP.

        This method blocks until either:
        - An answer arrives via POST /questions/{qid}/answer
        - The timeout expires (returns default or "denied")
        - The pipeline is cancelled

        Args:
            question: The question text to present.
            options: Optional list of valid answers.
            timeout: Override timeout in seconds.
            default: Default answer if timeout expires.

        Returns:
            The answer string.
        """
        qid = f"q_{uuid.uuid4().hex[:8]}"
        effective_timeout = timeout or self._timeout

        # Create pending question
        pending = PendingQuestion(
            qid=qid,
            question=question,
            stage=self._run.current_node or "unknown",
            timestamp=time.time(),
        )
        self._run.pending_questions[qid] = pending

        # Emit SSE event so web clients know a question is waiting
        self._run.emit(
            "interview.started",
            {
                "qid": qid,
                "question": question,
                "stage": pending.stage,
                "options": options or [],
            },
        )

        # Wait for answer
        try:
            await asyncio.wait_for(
                pending.answer_event.wait(),
                timeout=effective_timeout,
            )
            answer = pending.answer or default or "denied"
        except TimeoutError:
            answer = default or "denied"
            self._run.emit(
                "interview.timeout",
                {
                    "qid": qid,
                    "question": question,
                    "stage": pending.stage,
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
                "question": question,
                "answer": answer,
                "duration": time.time() - pending.timestamp,
            },
        )

        return answer


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
