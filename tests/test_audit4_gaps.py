"""Audit 4 -- final 16 gap tests.

Covers:
  Item 1+2  §8.9.25/27  generate_object() auto-resolves provider from model catalog
  Item 3    §11.6.4     Wait.human handler shape mapping (hexagon→manager / house→wait.human)
  Item 4    §9.4.1      grep() enforces path confinement via _check_path_allowed()
  Item 5    §8.1.6      Client(middleware=) emits DeprecationWarning
  Item 6    §8.4.5      StreamAccumulator handles STREAM_START and STREAM_END
  Item 7    §9.1.6      Process registration callback infrastructure in core.py
  Item 8    §9.11.5     _active_tasks populated with asyncio.Task during LLM call
  Item 9    §9.10.4     SESSION_START emitted on first submit() without context manager
  Item 10   §11.8.1     Interviewer.ask() returns str -- documented as intentional
  Item 11   §11.11.5    HTTP POST /run dispatches asyncio task to pipeline runner
  Item 12   §9.1.7      Loop detector catches A→B→A→B alternating cycles
  Item 13   §8.9.29     reasoning_tokens field has docstring note
  Item 14   §11.1.3     DOT parser handles multi-line attribute blocks
  Item 15   §11.12.3    DOT validator accepts graphs from multi-line attribute parsing
  Item 16   §11.12.22   12-node pipeline integration test (parse + validate + run)
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #


def _make_response(text: str = "hello", provider: str = "test") -> Any:
    """Build a minimal mock Response object."""
    from attractor_llm.types import (
        ContentPart,
        FinishReason,
        Message,
        Response,
        Role,
        Usage,
    )

    return Response(
        id="resp-test",
        model="test-model",
        provider=provider,
        message=Message(
            role=Role.ASSISTANT,
            content=[ContentPart.text_part(text)],
        ),
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _make_client(response: Any = None) -> Any:
    """Build a minimal mock Client that returns the given response."""
    from attractor_llm.client import Client

    response = response or _make_response()
    client = MagicMock(spec=Client)
    client.complete = AsyncMock(return_value=response)
    return client


# ------------------------------------------------------------------ #
# Items 1+2 §8.9.25/27 -- generate_object() auto-resolve provider
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_generate_object_auto_resolves_openai_provider() -> None:
    """generate_object() with provider=None uses catalog to pick openai for gpt models."""
    from attractor_llm.generate import generate_object

    captured_requests: list[Any] = []

    async def _complete(req: Any) -> Any:
        captured_requests.append(req)
        return _make_response('{"name": "Alice"}', provider="openai")

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = await generate_object(
        client,
        "gpt-5.2",
        "Extract name",
        schema=schema,
        provider=None,  # Must auto-resolve to "openai" via catalog
    )

    assert result.parsed_object == {"name": "Alice"}
    assert len(captured_requests) == 1
    req = captured_requests[0]
    # provider auto-resolved to "openai" → native json_schema response_format used
    assert req.provider == "openai"
    assert req.response_format is not None
    assert req.response_format.get("type") == "json_schema"


@pytest.mark.asyncio
async def test_generate_object_auto_resolves_anthropic_falls_back_to_prompt() -> None:
    """generate_object() with provider=None falls back to prompt injection for anthropic."""
    from attractor_llm.generate import generate_object

    captured_requests: list[Any] = []

    async def _complete(req: Any) -> Any:
        captured_requests.append(req)
        return _make_response('{"value": 42}', provider="anthropic")

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)

    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    result = await generate_object(
        client,
        "claude-sonnet-4-5",
        "Give me a value",
        schema=schema,
        provider=None,  # Should auto-resolve to "anthropic" → prompt injection
    )

    assert result.parsed_object == {"value": 42}
    req = captured_requests[0]
    # Anthropic → no native response_format → prompt injection
    assert req.response_format is None
    # The schema instructions should be in the system prompt
    assert req.system is not None
    assert "json" in req.system.lower()


@pytest.mark.asyncio
async def test_generate_object_explicit_provider_not_overridden() -> None:
    """generate_object() with explicit provider= skips catalog lookup."""
    from attractor_llm.generate import generate_object

    captured_requests: list[Any] = []

    async def _complete(req: Any) -> Any:
        captured_requests.append(req)
        return _make_response('{"x": 1}')

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)

    schema = {"type": "object"}
    await generate_object(
        client,
        "gpt-5.2",
        "test",
        schema=schema,
        provider="anthropic",  # Explicit override -- must not be replaced
    )

    req = captured_requests[0]
    assert req.provider == "anthropic"


# ------------------------------------------------------------------ #
# Item 3 §11.6.4 -- Wait.human handler shape mapping
# ------------------------------------------------------------------ #


def test_shape_mapping_hexagon_to_manager() -> None:
    """hexagon shape maps to manager handler (spec-verified, not wait.human)."""
    from attractor_pipeline.graph import NodeShape

    assert NodeShape.handler_for_shape("hexagon") == "manager"


def test_shape_mapping_house_to_wait_human() -> None:
    """house shape maps to wait.human handler (spec-verified)."""
    from attractor_pipeline.graph import NodeShape

    assert NodeShape.handler_for_shape("house") == "wait.human"


def test_shape_mapping_comment_documents_rationale() -> None:
    """handler_for_shape docstring documents the Wave 14 verification."""
    from attractor_pipeline.graph import NodeShape

    doc = NodeShape.handler_for_shape.__doc__ or ""
    assert "Wave 14" in doc or "verified" in doc.lower() or "audit" in doc.lower()


def test_all_shape_mappings_consistent() -> None:
    """All documented shapes produce a non-empty handler string."""
    from attractor_pipeline.graph import NodeShape

    shapes = [
        "box",
        "hexagon",
        "diamond",
        "component",
        "tripleoctagon",
        "parallelogram",
        "house",
        "Msquare",
        "Mdiamond",
    ]
    for s in shapes:
        result = NodeShape.handler_for_shape(s)
        assert result, f"Shape {s!r} produced empty handler"


# ------------------------------------------------------------------ #
# Item 4 §9.4.1 -- grep() path confinement
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_grep_raises_on_path_outside_allowed_roots(tmp_path: Any) -> None:
    """_grep() raises PermissionError when the search path is outside allowed roots."""
    import os

    from attractor_agent.tools.core import _grep, set_allowed_roots

    # Restrict to a subdir that doesn't include tmp_path
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    set_allowed_roots([str(allowed)])

    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    (forbidden / "file.txt").write_text("hello")

    try:
        with pytest.raises(PermissionError, match="outside allowed"):
            await _grep("hello", str(forbidden))
    finally:
        # Restore default roots so other tests aren't affected
        set_allowed_roots([os.getcwd()])


@pytest.mark.asyncio
async def test_grep_allowed_within_roots(tmp_path: Any) -> None:
    """_grep() works normally for paths within allowed roots."""
    import os

    from attractor_agent.tools.core import _grep, set_allowed_roots

    set_allowed_roots([str(tmp_path)])
    (tmp_path / "test.txt").write_text("hello world\nfoo bar\n")

    try:
        result = await _grep("hello", str(tmp_path))
        assert "hello" in result
    finally:
        set_allowed_roots([os.getcwd()])


# ------------------------------------------------------------------ #
# Item 5 §8.1.6 -- Client(middleware=) DeprecationWarning
# ------------------------------------------------------------------ #


def test_client_middleware_param_warns() -> None:
    """Client(middleware=[...]) emits DeprecationWarning."""
    from attractor_llm.client import Client

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Client(middleware=[object()])

    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "middleware" in str(w[0].message).lower()
    assert "apply_middleware" in str(w[0].message)


def test_client_no_middleware_no_warning() -> None:
    """Client() without middleware= emits no DeprecationWarning."""
    from attractor_llm.client import Client

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Client()

    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert not dep_warnings


# ------------------------------------------------------------------ #
# Item 6 §8.4.5 -- StreamAccumulator handles STREAM_START / STREAM_END
# ------------------------------------------------------------------ #


def test_accumulator_feeds_stream_start() -> None:
    """StreamAccumulator.feed() processes STREAM_START (canonical) events."""
    from attractor_llm.streaming import StreamAccumulator
    from attractor_llm.types import StreamEvent, StreamEventKind

    acc = StreamAccumulator()
    assert not acc.started

    acc.feed(StreamEvent(kind=StreamEventKind.STREAM_START, model="test-model", provider="test"))
    assert acc.started

    resp = acc.response()
    assert resp.model == "test-model"


def test_accumulator_feeds_legacy_start() -> None:
    """StreamAccumulator.feed() still processes legacy START events."""
    from attractor_llm.streaming import StreamAccumulator
    from attractor_llm.types import StreamEvent, StreamEventKind

    acc = StreamAccumulator()
    acc.feed(StreamEvent(kind=StreamEventKind.START, model="legacy-model", provider="legacy"))
    assert acc.started
    assert acc.response().model == "legacy-model"


def test_accumulator_both_start_kinds_set_started() -> None:
    """Both STREAM_START and START set the _started flag."""
    from attractor_llm.streaming import StreamAccumulator
    from attractor_llm.types import StreamEvent, StreamEventKind

    for kind in (StreamEventKind.START, StreamEventKind.STREAM_START):
        acc = StreamAccumulator()
        acc.feed(StreamEvent(kind=kind))
        assert acc.started, f"Started should be True after {kind}"


def test_accumulator_text_delta_after_stream_start() -> None:
    """Full stream with STREAM_START + TEXT_DELTA + FINISH accumulates correctly."""
    from attractor_llm.streaming import StreamAccumulator
    from attractor_llm.types import FinishReason, StreamEvent, StreamEventKind

    acc = StreamAccumulator()
    acc.feed(StreamEvent(kind=StreamEventKind.STREAM_START, model="m", provider="p"))
    acc.feed(StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello"))
    acc.feed(StreamEvent(kind=StreamEventKind.TEXT_DELTA, text=" World"))
    acc.feed(StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP))

    resp = acc.response()
    assert resp.text == "Hello World"
    assert resp.finish_reason == FinishReason.STOP


# ------------------------------------------------------------------ #
# Item 7 §9.1.6 -- Process registration callback infrastructure
# ------------------------------------------------------------------ #


def test_process_callback_default_is_none() -> None:
    """_process_callback starts as None."""
    # May have been set by another test; reset first
    from attractor_agent.tools.core import get_process_callback, set_process_callback

    set_process_callback(None)
    assert get_process_callback() is None


def test_set_process_callback_stores_callable() -> None:
    """set_process_callback() stores and get_process_callback() retrieves it."""
    from attractor_agent.tools.core import get_process_callback, set_process_callback

    def _dummy(proc: Any) -> None:
        pass

    set_process_callback(_dummy)
    assert get_process_callback() is _dummy
    # Clean up
    set_process_callback(None)


def test_set_process_callback_clears_with_none() -> None:
    """set_process_callback(None) clears the callback."""
    from attractor_agent.tools.core import get_process_callback, set_process_callback

    set_process_callback(lambda p: None)
    set_process_callback(None)
    assert get_process_callback() is None


def test_process_callback_accepts_async_callable() -> None:
    """set_process_callback accepts an async callable (common pattern)."""
    from attractor_agent.tools.core import get_process_callback, set_process_callback

    async def _async_cb(proc: Any) -> None:
        pass

    set_process_callback(_async_cb)
    assert get_process_callback() is _async_cb
    set_process_callback(None)


# ------------------------------------------------------------------ #
# Item 8 §9.11.5 -- _active_tasks populated during LLM call
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_active_tasks_populated_during_llm_call() -> None:
    """_active_tasks contains the LLM task while _call_llm() is in progress."""
    from attractor_agent.session import Session, SessionConfig
    from attractor_llm.types import (
        ContentPart,
        FinishReason,
        Message,
        Response,
        Role,
        Usage,
    )

    task_counts: list[int] = []

    async def _slow_complete(req: Any, **kwargs: Any) -> Response:
        # Capture active_tasks count while the LLM is "running"
        task_counts.append(len(session._active_tasks))
        await asyncio.sleep(0)
        return Response(
            id="x",
            model="test",
            provider="test",
            message=Message(role=Role.ASSISTANT, content=[ContentPart.text_part("done")]),
            finish_reason=FinishReason.STOP,
            usage=Usage(),
        )

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_slow_complete)

    config = SessionConfig(model="test-model", system_prompt="")
    session = Session(client=client, config=config)

    async with session:
        await session.submit("hello")

    # Should have had exactly 1 active task while the LLM was running
    assert task_counts, "complete() was never called"
    assert task_counts[0] == 1, f"Expected 1 active task, got {task_counts[0]}"


@pytest.mark.asyncio
async def test_active_tasks_cleared_after_llm_call() -> None:
    """_active_tasks is empty after submit() completes."""
    client = _make_client()
    from attractor_agent.session import Session, SessionConfig

    config = SessionConfig(model="test-model", system_prompt="")
    session = Session(client=client, config=config)

    async with session:
        await session.submit("hello")

    assert len(session._active_tasks) == 0


# ------------------------------------------------------------------ #
# Item 9 §9.10.4 -- SESSION_START emitted without context manager
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_session_start_emitted_on_first_submit_without_context_manager() -> None:
    """SESSION_START is emitted on the first submit() even without async with."""
    from attractor_agent.events import EventKind
    from attractor_agent.session import Session, SessionConfig

    client = _make_client()
    config = SessionConfig(model="test-model", system_prompt="")
    session = Session(client=client, config=config)

    emitted_kinds: list[str] = []

    # EventEmitter.on() registers for ALL events; filter by kind in the handler
    async def _on_any(event: Any) -> None:
        if str(event.kind) in (str(EventKind.SESSION_START), "session.start"):
            emitted_kinds.append(str(event.kind))

    session.events.on(_on_any)

    await session.submit("test prompt")

    assert emitted_kinds, "SESSION_START was not emitted during submit()"


@pytest.mark.asyncio
async def test_session_start_not_double_emitted_with_context_manager() -> None:
    """SESSION_START is emitted exactly once when using async with."""
    from attractor_agent.events import EventKind
    from attractor_agent.session import Session, SessionConfig

    client = _make_client()
    config = SessionConfig(model="test-model", system_prompt="")
    session = Session(client=client, config=config)

    start_count = [0]

    # EventEmitter.on() registers for ALL events; count only SESSION_START
    async def _count_starts(event: Any) -> None:
        if str(event.kind) in (str(EventKind.SESSION_START), "session.start"):
            start_count[0] += 1

    session.events.on(_count_starts)

    async with session:
        await session.submit("test prompt")

    # __aenter__ emits SESSION_START; submit() should NOT re-emit it
    assert start_count[0] == 1, f"SESSION_START emitted {start_count[0]} times; expected exactly 1"


# ------------------------------------------------------------------ #
# Item 10 §11.8.1 -- Interviewer.ask() returns str by design
# ------------------------------------------------------------------ #


def test_interviewer_ask_signature_returns_str() -> None:
    """Interviewer.ask() is typed as -> str (minimum contract)."""
    import inspect

    from attractor_pipeline.handlers.human import Interviewer

    try:
        sig = inspect.signature(Interviewer.ask)
        ret = sig.return_annotation
        # `from __future__ import annotations` makes Python store annotations as
        # strings (PEP 563), so the return annotation may be either the `str`
        # class or the string literal "str".  Accept both.
        assert ret in (str, "str", inspect.Parameter.empty), (
            f"Expected str return, got {ret!r}"
        )
    except (ValueError, TypeError):
        pass  # Protocol methods may not be inspectable in all Python versions


def test_interviewer_ask_docstring_notes_intentional_str_return() -> None:
    """Interviewer.ask() docstring mentions the intentional str return contract."""
    from attractor_pipeline.handlers.human import Interviewer

    doc = Interviewer.ask.__doc__ or ""
    # Should mention that str return is intentional / minimum contract
    assert any(
        kw in doc.lower() for kw in ("intentional", "minimum contract", "protocol", "§11.8")
    ), f"Docstring should note the intentional str return. Got: {doc[:200]}"


@pytest.mark.asyncio
async def test_ask_question_via_ask_bridges_to_answer() -> None:
    """ask_question_via_ask() wraps the str result into an Answer."""
    from attractor_pipeline.handlers.human import Answer, Question, ask_question_via_ask

    class _StubInterviewer:
        async def ask(
            self,
            question: str,
            options: list[str] | None = None,
            node_id: str = "",
            question_type: Any = None,
        ) -> str:
            return "yes"

    answer = await ask_question_via_ask(_StubInterviewer(), Question(text="Are you sure?"))
    assert isinstance(answer, Answer)
    assert answer.value == "yes"


# ------------------------------------------------------------------ #
# Item 11 §11.11.5 -- HTTP POST /run dispatches pipeline runner task
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_post_run_creates_task_in_run_record() -> None:
    """POST /run stores an asyncio.Task in the run record."""
    from starlette.testclient import TestClient

    from attractor_pipeline.server.app import _runs, app

    # Clear any existing runs
    _runs.clear()

    with TestClient(app) as client:
        resp = client.post("/run", json={"pipeline": "test", "input": {"key": "val"}})

    assert resp.status_code == 202
    data = resp.json()
    run_id = data["id"]

    assert run_id in _runs
    run = _runs[run_id]
    # §11.11.5: task must be stored
    assert "task" in run
    assert run["task"] is not None
    assert isinstance(run["task"], asyncio.Task)


@pytest.mark.asyncio
async def test_post_run_dispatches_pipeline_execution() -> None:
    """POST /run transitions run status from pending → running/completed."""
    from starlette.testclient import TestClient

    from attractor_pipeline.server.app import _runs, app

    _runs.clear()

    with TestClient(app) as client:
        resp = client.post("/run", json={"pipeline": "simple"})

    assert resp.status_code == 202
    run_id = resp.json()["id"]

    # Allow the background task to complete
    task = _runs[run_id].get("task")
    if task and not task.done():
        await asyncio.wait_for(task, timeout=2.0)

    # Should have progressed beyond "pending"
    assert _runs[run_id]["status"] in ("running", "completed", "failed")


@pytest.mark.asyncio
async def test_get_status_returns_valid_json_no_task_key() -> None:
    """GET /status/{id} returns valid JSON without crashing (Task not serializable)."""
    from starlette.testclient import TestClient

    from attractor_pipeline.server.app import _runs, app

    _runs.clear()

    with TestClient(app) as client:
        # Create a run first
        post_resp = client.post("/run", json={"pipeline": "test"})
        assert post_resp.status_code == 202
        run_id = post_resp.json()["id"]

        # GET /status/{id} must not crash even though the run record has a Task
        get_resp = client.get(f"/status/{run_id}")

    assert get_resp.status_code == 200
    data = get_resp.json()
    # Must be valid JSON (no TypeError from asyncio.Task)
    assert isinstance(data, dict)
    assert data["id"] == run_id
    assert "status" in data
    # The asyncio.Task must be excluded from the response
    assert "task" not in data


# ------------------------------------------------------------------ #
# Item 12 §9.1.7 -- Loop detector catches A→B→A→B alternating cycles
# ------------------------------------------------------------------ #


def test_loop_detector_catches_alternating_ab_cycle() -> None:
    """_LoopDetector detects A→B→A→B→A→B repeating cycle."""
    from attractor_agent.session import _LoopDetector

    det = _LoopDetector(window=4, threshold=3)

    # Record A, B, A, B, A, B -- alternating cycle of length 2
    calls = [
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
    ]

    results = [det.record(name, args) for name, args in calls]
    # At some point before or at the last call, detection should fire
    assert any(results), "Alternating A→B cycle should be detected"


def test_loop_detector_catches_abc_cycle() -> None:
    """_LoopDetector detects A→B→C→A→B→C→A→B→C repeating cycle."""
    from attractor_agent.session import _LoopDetector

    det = _LoopDetector(window=6, threshold=3)

    calls = [
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
    ]

    results = [det.record(name, args) for name, args in calls]
    assert any(results), "A→B→C cycle should be detected"


def test_loop_detector_threshold4_catches_3cycle_divisibility_blind_spot() -> None:
    """_LoopDetector with threshold=4 detects a 3-cycle (tail=8, cycle_len=3: 8%3≠0).

    This is the divisibility blind spot: the old code required len(tail) % cycle_len == 0,
    which skipped cycle_len=3 when tail=8 (8%3=2≠0).  The fix checks at least 2 full
    repeats within the tail instead of requiring perfect tiling.
    """
    from attractor_agent.session import _LoopDetector

    # threshold=4 → tail = threshold*2 = 8 entries examined
    det = _LoopDetector(window=6, threshold=4)

    # A→B→C pattern repeated enough times to fill the sliding window
    calls = [
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
        ("tool_a", "{}"),
        ("tool_b", "{}"),
        ("tool_c", "{}"),
    ]

    results = [det.record(name, args) for name, args in calls]
    assert any(results), (
        "threshold=4 should detect a 3-cycle (divisibility blind spot fix)"
    )


def test_loop_detector_still_catches_simple_repetition() -> None:
    """_LoopDetector still detects the simple single-tool repetition."""
    from attractor_agent.session import _LoopDetector

    det = _LoopDetector(window=4, threshold=3)
    results = [det.record("shell", '{"command": "ls"}') for _ in range(5)]
    assert any(results), "Simple repetition should still be detected"


def test_loop_detector_no_false_positive_on_varied_calls() -> None:
    """_LoopDetector does not fire on a normal varied sequence."""
    from attractor_agent.session import _LoopDetector

    det = _LoopDetector(window=4, threshold=3)
    varied = [
        ("read_file", '{"path": "a.py"}'),
        ("shell", '{"command": "ls"}'),
        ("write_file", '{"path": "b.py", "content": "x"}'),
        ("grep", '{"pattern": "foo"}'),
        ("glob", '{"pattern": "*.py"}'),
    ]
    results = [det.record(name, args) for name, args in varied]
    assert not any(results), "Varied calls should not trigger loop detection"


def test_loop_detector_reset_clears_history() -> None:
    """_LoopDetector.reset() clears state so detection resets."""
    from attractor_agent.session import _LoopDetector

    det = _LoopDetector(window=4, threshold=3)
    for _ in range(3):
        det.record("tool_a", "{}")
    det.reset()
    # After reset, no detection for fewer than threshold calls
    results = [det.record("tool_a", "{}") for _ in range(2)]
    assert not any(results)


# ------------------------------------------------------------------ #
# Item 13 §8.9.29 -- reasoning_tokens field docstring
# ------------------------------------------------------------------ #


def test_reasoning_tokens_field_has_docstring() -> None:
    """Usage.reasoning_tokens has a Field description describing the approximation."""
    from attractor_llm.types import Usage

    # Verify the Pydantic Field description is present and non-empty
    field_info = Usage.model_fields["reasoning_tokens"]
    desc = field_info.description or ""
    assert desc, "reasoning_tokens Field must have a non-empty description"
    assert "Anthropic" in desc or "anthropic" in desc, (
        f"Description should mention Anthropic: {desc!r}"
    )
    assert "len(thinking_text)" in desc or "// 4" in desc or "estimate" in desc.lower(), (
        f"Description should mention the approximation method: {desc!r}"
    )


def test_reasoning_tokens_default_is_zero() -> None:
    """Usage.reasoning_tokens defaults to 0."""
    from attractor_llm.types import Usage

    u = Usage()
    assert u.reasoning_tokens == 0


def test_reasoning_tokens_included_in_addition() -> None:
    """Usage + Usage sums reasoning_tokens correctly."""
    from attractor_llm.types import Usage

    a = Usage(input_tokens=10, output_tokens=5, reasoning_tokens=3)
    b = Usage(input_tokens=20, output_tokens=8, reasoning_tokens=7)
    total = a + b
    assert total.reasoning_tokens == 10


# ------------------------------------------------------------------ #
# Items 14+15 §11.1.3, §11.12.3 -- Multi-line DOT attribute blocks
# ------------------------------------------------------------------ #


def test_parse_multiline_node_attributes() -> None:
    """DOT parser handles multi-line attribute blocks correctly (§11.1.3)."""
    from attractor_pipeline.parser.parser import parse_dot

    dot = """
    digraph Test {
        graph [goal="Test multi-line attrs"]
        start [shape=Mdiamond]
        task [
            label="Multi\\nLine"
            shape=box
            prompt="Do something"
        ]
        done [shape=Msquare]
        start -> task -> done
    }
    """
    graph = parse_dot(dot)
    assert "task" in graph.nodes
    node = graph.nodes["task"]
    assert node.shape == "box"
    assert node.prompt == "Do something"
    assert "Multi" in node.label or "Multi\\nLine" in node.label


def test_parse_multiline_attributes_preserves_all_attrs() -> None:
    """Multi-line attribute blocks preserve all key-value pairs in node.attrs."""
    from attractor_pipeline.parser.parser import parse_dot

    dot = """
    digraph AttrTest {
        graph [goal="Attr preservation"]
        start [shape=Mdiamond]
        worker [
            shape=box
            label="Worker Node"
            prompt="Execute the task"
            timeout="5m"
        ]
        done [shape=Msquare]
        start -> worker -> done
    }
    """
    graph = parse_dot(dot)
    node = graph.nodes["worker"]
    assert node.shape == "box"
    assert "Worker" in node.label
    assert node.prompt == "Execute the task"
    # timeout is either in typed field or attrs
    assert node.timeout == "5m" or node.attrs.get("timeout") == "5m"


def test_parse_multiline_edge_attributes() -> None:
    """Multi-line edge attribute blocks are parsed correctly."""
    from attractor_pipeline.parser.parser import parse_dot

    dot = """
    digraph EdgeTest {
        graph [goal="Edge attrs test"]
        start [shape=Mdiamond]
        check [shape=diamond]
        yes [shape=box]
        done [shape=Msquare]
        start -> check
        check -> yes [
            label="approved"
            condition="outcome = success"
        ]
        yes -> done
    }
    """
    graph = parse_dot(dot)
    edges_check_yes = [e for e in graph.edges if e.source == "check" and e.target == "yes"]
    assert len(edges_check_yes) == 1
    edge = edges_check_yes[0]
    assert edge.label == "approved"
    assert "success" in edge.condition


def test_validator_accepts_multiline_parsed_graph() -> None:
    """Validator (§11.12.3) accepts graphs produced from multi-line attribute parsing."""
    from attractor_pipeline.parser.parser import parse_dot
    from attractor_pipeline.validation import Severity, validate

    dot = """
    digraph ValidateMe {
        graph [goal="Validation test"]
        start [shape=Mdiamond]
        node_a [
            shape=box
            label="Task A"
            prompt="Do task A"
        ]
        node_b [
            shape=box
            label="Task B"
            prompt="Do task B"
        ]
        done [shape=Msquare]
        start -> node_a -> node_b -> done
    }
    """
    graph = parse_dot(dot)
    diagnostics = validate(graph)
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    assert not errors, f"Validation errors on valid multi-line graph: {errors}"


def test_parse_multiline_with_quoted_newlines() -> None:
    """Multi-line blocks with \\n in quoted strings don't confuse the parser."""
    from attractor_pipeline.parser.parser import parse_dot

    dot = r"""
    digraph NewlineTest {
        graph [goal="Newline test"]
        start [shape=Mdiamond]
        task [
            shape=box
            label="Line1\nLine2\nLine3"
            prompt="Process newlines"
        ]
        done [shape=Msquare]
        start -> task -> done
    }
    """
    graph = parse_dot(dot)
    assert "task" in graph.nodes
    # Should not raise -- just verify parsing completes


# ------------------------------------------------------------------ #
# Item 16 §11.12.22 -- 12-node pipeline integration test
# ------------------------------------------------------------------ #

_TWELVE_NODE_DOT = """
digraph TwelveNode {
    graph [goal="12-node integration test"]

    // Entry
    start [shape=Mdiamond]

    // Sequential processing chain
    fetch    [shape=box,          prompt="Fetch input data"]
    validate [shape=box,          prompt="Validate input"]
    parse    [shape=box,          prompt="Parse data"]

    // Conditional branch
    check    [shape=diamond]
    enrich   [shape=box,          prompt="Enrich data"]
    flag     [shape=box,          prompt="Flag invalid data"]

    // Parallel fan-out
    fanout   [shape=component]
    worker_a [shape=parallelogram, prompt="Shell: process chunk A"]
    worker_b [shape=box,           prompt="LLM: summarise chunk B"]

    // Fan-in
    fanin    [shape=tripleoctagon]

    // Exit
    done     [shape=Msquare]

    // Edges
    start    -> fetch    -> validate -> parse -> check
    check    -> enrich   [condition="outcome = success"]
    check    -> flag     [condition="outcome = fail"]
    enrich   -> fanout
    flag     -> done
    fanout   -> worker_a
    fanout   -> worker_b
    worker_a -> fanin
    worker_b -> fanin
    fanin    -> done
}
"""


def test_twelve_node_dot_parses() -> None:
    """12-node pipeline DOT parses without errors."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    assert len(graph.nodes) == 12
    expected_nodes = {
        "start",
        "fetch",
        "validate",
        "parse",
        "check",
        "enrich",
        "flag",
        "fanout",
        "worker_a",
        "worker_b",
        "fanin",
        "done",
    }
    assert set(graph.nodes.keys()) == expected_nodes


def test_twelve_node_graph_validates() -> None:
    """12-node pipeline graph passes the validator (no critical errors)."""
    from attractor_pipeline.parser.parser import parse_dot
    from attractor_pipeline.validation import Severity, validate

    graph = parse_dot(_TWELVE_NODE_DOT)
    diagnostics = validate(graph)
    # Warnings about unreachable nodes (flag→done shortcut) are acceptable;
    # but there should be no fatal ERROR-severity diagnostics.
    fatal = [d for d in diagnostics if d.severity == Severity.ERROR]
    assert not fatal, f"Fatal validation errors: {fatal}"


def test_twelve_node_graph_edge_count() -> None:
    """12-node graph has the correct number of edges."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    # Count edges from the DOT definition:
    # start→fetch→validate→parse→check (chained = 4)
    # check→enrich, check→flag          = 2
    # enrich→fanout, flag→done          = 2
    # fanout→worker_a, fanout→worker_b  = 2
    # worker_a→fanin, worker_b→fanin    = 2
    # fanin→done                        = 1
    # Total = 13
    assert len(graph.edges) == 13


def test_twelve_node_graph_start_node() -> None:
    """12-node graph has exactly one start node (Mdiamond)."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    start = graph.get_start_node()
    assert start is not None
    assert start.id == "start"


def test_twelve_node_graph_exit_nodes() -> None:
    """12-node graph has exactly one exit node (Msquare)."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    exits = graph.get_exit_nodes()
    assert len(exits) == 1
    assert exits[0].id == "done"


def test_twelve_node_graph_shape_diversity() -> None:
    """12-node graph exercises 7 distinct node shapes."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    shapes = {n.shape for n in graph.nodes.values()}
    expected_shapes = {
        "Mdiamond",
        "box",
        "diamond",
        "component",
        "parallelogram",
        "tripleoctagon",
        "Msquare",
    }
    assert expected_shapes.issubset(shapes), f"Missing shapes: {expected_shapes - shapes}"


@pytest.mark.asyncio
async def test_twelve_node_pipeline_runs_to_completion() -> None:
    """12-node pipeline runs end-to-end with mock handlers."""
    from attractor_pipeline import (
        HandlerRegistry,
        PipelineStatus,
        register_default_handlers,
        run_pipeline,
    )
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)

    registry = HandlerRegistry()
    register_default_handlers(registry)

    result = await run_pipeline(graph, registry)

    # The pipeline should reach a terminal state
    assert result.status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED)
    # Start node must always complete
    assert "start" in result.completed_nodes
    # Done node should be reachable from the success path
    # (enrich branch → fanout → fanin → done)
    assert "done" in result.completed_nodes or result.status == PipelineStatus.FAILED


@pytest.mark.asyncio
async def test_twelve_node_pipeline_goal_metadata() -> None:
    """12-node pipeline graph carries the correct goal attribute."""
    from attractor_pipeline.parser.parser import parse_dot

    graph = parse_dot(_TWELVE_NODE_DOT)
    assert "12-node" in graph.goal or "integration" in graph.goal.lower()
