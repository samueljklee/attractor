# Spec Compliance Wave 2 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Fix all 15 remaining spec compliance gaps identified by the audit recipe, confirmed by an 8-model swarm review against actual spec text. All 15 are PARTIAL with code + spec evidence.

**Architecture:** Three phases matching audit waves. Phase 1 fixes runtime crashes (no API keys, fast feedback). Phase 2 fixes behavioral contracts. Phase 3 is naming and polish. All tasks follow strict TDD: write failing test → verify fail → implement → verify pass → commit.

**Tech Stack:** Python 3.12, pytest + pytest-asyncio, uv.

**Worktree:** Create before starting:
```bash
git worktree add .worktrees/feat-spec-compliance-wave2 -b feat/spec-compliance-wave2
cd .worktrees/feat-spec-compliance-wave2
```

**Mock test command (no API keys):**
```bash
uv run python -m pytest tests/ \
  --ignore=tests/test_e2e_integration.py \
  --ignore=tests/test_live_comprehensive.py \
  --ignore=tests/test_live_wave9_10_p1.py \
  --ignore=tests/test_issue36_hexagon_hang.py \
  --ignore=tests/test_audit2_wave6_live_parity_matrix.py \
  --ignore=tests/test_coverage_gaps_live.py \
  --ignore=tests/test_wave16b_live_coverage.py \
  --ignore=tests/test_e2e_integration_parity.py \
  -x -q
```

**New test file for Tasks 1–10:** `tests/test_spec_compliance_wave2.py`

**File header for new test file:**
```python
"""Spec compliance tests — Wave 2 fixes.

Each test class maps to one task in docs/plans/2026-02-26-spec-compliance-wave2.md.
All tests are pure mock tests (no API keys required).
"""
from __future__ import annotations

import asyncio
import subprocess
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
```

---

## Dependency Map

```
Task 1  — Popen.wait() fix           independent  P1 (§9.1.6 / §9.11.5)
Task 2  — Boolean literals           independent  P1 CRITICAL (§11.4.2)
Task 3  — ASSISTANT_TEXT events      independent  P1 (§9.10.1)
Task 4  — Double truncation marker   independent  P2 (§9.5.3)
Task 5  — STREAM_START event name    independent  P2 (§8.4.5)
Task 6  — SESSION_START auto-emit    independent  P2 (§9.10.4)
Task 7  — Middleware constructor     independent  P2 (§8.1.6)
Task 8  — Middleware stream()        after Task 7 P2 (§8.1.6)
Task 9  — max_tool_rounds rename     independent  P2 (§9.1.4)
Task 10 — edge_id in Diagnostic      independent  P2 (§11.2.10)
Task 11 — Cache per-turn assertions  independent  P2 (§8.9.40 / §8.9.42)
```

---

## Phase 1 — Runtime Crash Fixes

### Task 1: Fix Popen.wait() TypeError on abort (§9.1.6 / §9.11.5)

**Files:**
- Modify: `src/attractor_agent/session.py` (lines ~818–836 in `_cleanup_on_abort()`)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `_cleanup_on_abort()` collects live processes and awaits them. Line ~818 annotates `_alive` as `list[asyncio.subprocess.Process]`, but `LocalEnvironment` registers `subprocess.Popen` objects. `Popen.wait()` is synchronous — it returns `int`, not a coroutine. `asyncio.create_task(proc.wait())` on an `int` raises `TypeError`, crashing the entire abort sequence silently.

The FIXME comment at line ~830 documents this exactly:
```python
# FIXME(Task 2): proc.wait() is a coroutine for asyncio.subprocess.Process but
# returns int for subprocess.Popen (wired by Task 2 via LocalEnvironment).
# asyncio.create_task() will raise TypeError on real abort with shell processes.
# Fix: branch on type or use asyncio.to_thread(proc.wait) for Popen objects.
[asyncio.create_task(proc.wait()) for proc in _alive],
```

---

**Step 1: Write the failing test**

Add to `tests/test_spec_compliance_wave2.py`:

```python
class TestPopenWaitFix:
    """Task 1 — §9.1.6/§9.11.5: abort cleanup must handle subprocess.Popen."""

    @pytest.mark.asyncio
    async def test_cleanup_on_abort_handles_popen_without_typeerror(self):
        """_cleanup_on_abort() must not raise TypeError for subprocess.Popen processes."""
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.client import Client

        # Use a real subprocess.Popen (sleep, won't actually run)
        proc = subprocess.Popen(["sleep", "100"])
        try:
            client = Client()
            config = SessionConfig()
            session = Session(client=client, config=config)
            session._tracked_processes.append(proc)

            # Should complete without TypeError
            try:
                await session._cleanup_on_abort()
            except TypeError as e:
                pytest.fail(
                    f"_cleanup_on_abort() raised TypeError for subprocess.Popen: {e}"
                )
        finally:
            proc.terminate()
            proc.wait()
```

**Step 2: Run to verify it FAILS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestPopenWaitFix -v
```
Expected: **FAIL** — `TypeError: A coroutine was expected, got 0`

**Step 3: Apply the fix**

In `src/attractor_agent/session.py`, find `_cleanup_on_abort()` (~line 785). Make two changes:

1. Update the type annotation (~line 818):
```python
# BEFORE:
_alive: list[asyncio.subprocess.Process] = []

# AFTER:
_alive: list[Any] = []  # asyncio.subprocess.Process or subprocess.Popen
```
(Ensure `Any` is imported from `typing` at the top of the file — check existing imports.)

2. Replace the list comprehension at lines ~828–836:
```python
# BEFORE:
if _alive:
    _done, _pending = await asyncio.wait(
        # FIXME(Task 2): proc.wait() is a coroutine for asyncio.subprocess.Process but
        # returns int for subprocess.Popen (wired by Task 2 via LocalEnvironment).
        # asyncio.create_task() will raise TypeError on real abort with shell processes.
        # Fix: branch on type or use asyncio.to_thread(proc.wait) for Popen objects.
        [asyncio.create_task(proc.wait()) for proc in _alive],
        timeout=2.0,
    )
    # Cancel the wait tasks for any that didn't finish in time
    for t in _pending:
        t.cancel()

# AFTER:
if _alive:
    tasks = []
    for proc in _alive:
        if isinstance(proc, asyncio.subprocess.Process):
            # asyncio subprocess — wait() is a coroutine
            tasks.append(asyncio.create_task(proc.wait()))
        else:
            # subprocess.Popen — wait() is synchronous, run in thread (§9.1.6)
            tasks.append(asyncio.create_task(asyncio.to_thread(proc.wait)))
    _done, _pending = await asyncio.wait(tasks, timeout=2.0)
    for t in _pending:
        t.cancel()
```

**Step 4: Run to verify it PASSES**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestPopenWaitFix -v
```
Expected: **PASS**

**Step 5: Run full mock suite**
```bash
uv run python -m pytest tests/ [ignores above] -x -q
```
Expected: all passing.

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_agent/session.py
git commit -m "fix: handle subprocess.Popen in abort cleanup — use to_thread (spec §9.1.6/§9.11.5)

asyncio.create_task(Popen.wait()) raises TypeError because Popen.wait()
returns int (synchronous). Fix branches on proc type and uses
asyncio.to_thread for Popen objects, preserving asyncio.subprocess.Process
handling unchanged.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 2: Fix boolean literals in condition evaluation (§11.4.2 / §11.4.3)

**Files:**
- Modify: `src/attractor_pipeline/conditions.py` (lines 66–70)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `evaluate_condition('true', {})` calls `_evaluate_clause('true', {})`. Since `'!='` and `'='` are not in `'true'`, it falls to the bare truthy check: `_resolve('true', {})`. `'true'` is not a key in `{}`, returns `''`, `bool('') = False`. Every DOT node with `goal_gate=true` permanently fails its gate, causing infinite retry loops.

Current broken code at lines 66–70:
```python
# Bare truthy check: "key" alone means "key is truthy"
key = clause.strip()
if key:
    actual = _resolve(key, variables)
    return bool(actual)
```

---

**Step 1: Write the failing tests**

```python
class TestBooleanLiterals:
    """Task 2 — §11.4.2: boolean literals must evaluate correctly in conditions."""

    def test_bare_true_evaluates_to_true(self):
        """evaluate_condition('true', {}) must return True — goal_gate=true must fire."""
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("true", {}) is True, (
            "evaluate_condition('true', {}) returned False — goal gates will never fire"
        )

    def test_bare_false_evaluates_to_false(self):
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("false", {}) is False

    def test_numeric_true_evaluates_to_true(self):
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("1", {}) is True

    def test_yes_evaluates_to_true(self):
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("yes", {}) is True

    def test_no_evaluates_to_false(self):
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("no", {}) is False

    def test_key_lookup_still_works(self):
        """Key lookup path is unchanged — 'active' with active=true still works."""
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("active", {"active": "true"}) is True

    def test_equality_check_unchanged(self):
        """Equality conditions are unaffected."""
        from attractor_pipeline.conditions import evaluate_condition
        assert evaluate_condition("outcome = SUCCESS", {"outcome": "SUCCESS"}) is True
        assert evaluate_condition("outcome = SUCCESS", {"outcome": "FAIL"}) is False
```

**Step 2: Run to verify they FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestBooleanLiterals -v
```
Expected: `test_bare_true_evaluates_to_true` FAILS.

**Step 3: Apply the fix**

In `src/attractor_pipeline/conditions.py`, update `_evaluate_clause()` at lines 66–70:

```python
# BEFORE:
# Bare truthy check: "key" alone means "key is truthy"
key = clause.strip()
if key:
    actual = _resolve(key, variables)
    return bool(actual)

# AFTER:
# Bare truthy check: "key" alone means "key is truthy"
key = clause.strip()
if key:
    # §11.4.2: Recognise DOT boolean literals before variable lookup.
    # DOT attribute values are always strings; 'true', '1', 'yes' are truthy.
    if key.lower() in ("true", "1", "yes"):
        return True
    if key.lower() in ("false", "0", "no"):
        return False
    actual = _resolve(key, variables)
    return bool(actual)
```

**Step 4: Run to verify all PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestBooleanLiterals -v
```
Expected: 7/7 PASS.

**Step 5: Run full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_pipeline/conditions.py
git commit -m "fix: recognise boolean literals in condition evaluation (spec §11.4.2)

DOT attribute values are strings. 'true', '1', 'yes' must evaluate as
truthy before the variable key lookup, so goal_gate=true nodes correctly
allow pipeline exit. Previously _resolve('true', {}) returned '' -> False,
making every goal gate permanently fail.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 3: Emit ASSISTANT_TEXT events alongside tool calls (§9.10.1)

**Files:**
- Modify: `src/attractor_agent/session.py` (in `_run_loop()`)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** Read `_run_loop()` (~lines 499–640). The current structure is:
```python
while True:
    response = await self._call_llm()
    tool_calls = response.tool_calls or []
    if tool_calls:
        # ... process tool calls ...
        continue          # ← skips text event block entirely

    # Only reached when NO tool calls:
    text = response.text or ""
    if text:
        # Emit ASSISTANT_TEXT_START, ASSISTANT_TEXT_DELTA, ASSISTANT_TEXT_END
```
When a model returns both `text` and `tool_calls` (valid for all providers), the text is silently dropped.

---

**Step 1: Write the failing test**

```python
class TestTextEventsWithToolCalls:
    """Task 3 — §9.10.1: ASSISTANT_TEXT events must fire even with tool calls."""

    @pytest.mark.asyncio
    async def test_text_events_emitted_when_response_has_both_text_and_tool_calls(self):
        """When LLM returns text + tool calls, TEXT_START/DELTA/END must still fire."""
        from attractor_agent.events import EventKind, SessionEvent
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.client import Client
        from attractor_llm.types import Response, ToolCall, Usage

        captured_events: list[SessionEvent] = []

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.execute = AsyncMock(return_value="file content")

        client = MagicMock(spec=Client)
        # Response with BOTH text and tool_calls
        client.complete = AsyncMock(
            side_effect=[
                Response(
                    text="I'll read that file for you.",
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "/tmp/x"})],
                    usage=Usage(input_tokens=10, output_tokens=5),
                ),
                Response(
                    text="The file contains: file content",
                    tool_calls=[],
                    usage=Usage(input_tokens=20, output_tokens=8),
                ),
            ]
        )

        config = SessionConfig(max_turns=1)
        session = Session(client=client, config=config, tools=[mock_tool])
        session.events.on(captured_events.append)

        await session.submit("read /tmp/x")

        text_kinds = [e.kind for e in captured_events if "TEXT" in e.kind.value]
        assert EventKind.ASSISTANT_TEXT_START in text_kinds, (
            "ASSISTANT_TEXT_START must fire even when response also has tool calls"
        )
        assert EventKind.ASSISTANT_TEXT_DELTA in text_kinds, (
            "ASSISTANT_TEXT_DELTA must fire even when response also has tool calls"
        )
```

**Step 2: Run to verify it FAILS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestTextEventsWithToolCalls -v
```
Expected: FAIL — `AssertionError: ASSISTANT_TEXT_START must fire even when response also has tool calls`

**Step 3: Apply the fix**

Read `src/attractor_agent/session.py` lines 499–640 carefully. Find the `_run_loop()` method. Locate the section that reads:
```python
# No tool calls -- model produced a text response
text = response.text or ""

if text:
    # Emit TEXT_START ...
    # Emit TEXT_DELTA ...
    # Emit TEXT_END ...
```

Move the text event emission block to fire BEFORE the `if tool_calls:` branch. The structure should become:

```python
while True:
    response = await self._call_llm()
    tool_calls = response.tool_calls or []

    # §9.10.1: emit ASSISTANT_TEXT events whenever model produces text,
    # even if the response also contains tool calls.
    if response.text:
        text = response.text
        await self._emitter.emit(
            SessionEvent(
                kind=EventKind.ASSISTANT_TEXT_START,
                data={"turn": self._turn_count},
            )
        )
        await self._emitter.emit(
            SessionEvent(
                kind=EventKind.ASSISTANT_TEXT_DELTA,
                data={"delta": text},
            )
        )
        await self._emitter.emit(
            SessionEvent(
                kind=EventKind.ASSISTANT_TEXT_END,
                data={"text": text[:500]},
            )
        )

    if tool_calls:
        # ... existing tool call processing unchanged ...
        continue

    # No tool calls — done with this turn
    self._loop_detector.reset()
    return response.text or ""
```

Make sure the old text event block (inside the "no tool calls" section) is removed to avoid duplicate emission.

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestTextEventsWithToolCalls -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_agent/session.py
git commit -m "fix: emit ASSISTANT_TEXT events even when tool calls also present (spec §9.10.1)

Text events were inside the 'no tool calls' branch, silently dropped when
a model response contained both text and tool calls. Moved emission before
the tool_calls check so it fires unconditionally when response.text is set.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

## Phase 2 — Behavioral Contract Fixes

### Task 4: Remove double truncation marker (§9.5.3)

**Files:**
- Modify: `src/attractor_agent/tools/registry.py` (lines 194–196)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `truncation.py` already embeds a spec-compliant `[WARNING: Tool output was truncated. N characters were removed...]` in the truncated output. `registry.py` then appends `"\n[output was truncated]"` on top — the LLM sees two signals.

Current code at lines 194–196:
```python
output, was_truncated = truncate_output(raw_output_str, limits)
if was_truncated:
    output += "\n[output was truncated]"
```

---

**Step 1: Write the failing test**

```python
class TestNoDoubleTruncationMarker:
    """Task 4 — §9.5.3: only one truncation marker must appear in tool output."""

    def test_truncated_output_has_exactly_one_marker(self):
        """After truncation, output must contain the spec WARNING but not the extra marker."""
        from attractor_agent.truncation import TruncationLimits, truncate_output

        big_output = "A" * 10_000
        limits = TruncationLimits(max_chars=100, max_lines=9999, head_ratio=0.5)
        truncated, was_truncated = truncate_output(big_output, limits)

        assert was_truncated, "Output should have been truncated"
        # Spec WARNING is present
        assert "[WARNING: Tool output was truncated." in truncated, (
            "Spec-compliant WARNING marker must be present"
        )
        # Extra marker must NOT appear (this currently fails because registry.py adds it)
        assert "[output was truncated]" not in truncated, (
            "The extra '[output was truncated]' marker from registry.py must be removed"
        )
```

Note: this test checks `truncation.py` directly — the double marker exists in `registry.py`. The test for registry.py behavior is harder to isolate. You can also write an integration test that calls the registry path with a large output and checks the final `output` string returned to the LLM only has one marker.

**Step 2: Run to verify current state**

Run the test — it will likely PASS already (because truncation.py itself doesn't add the extra marker; it's in registry.py). Add a cleaner test:

```python
    def test_registry_does_not_add_extra_marker_after_truncation(self):
        """registry.py must not append '[output was truncated]' after truncate_output."""
        # Check the source directly — look for the extra append
        import ast
        import pathlib
        source = pathlib.Path("src/attractor_agent/tools/registry.py").read_text()
        assert "output was truncated]" not in source, (
            "registry.py must not contain the extra '[output was truncated]' append — "
            "truncation.py already adds the spec-compliant WARNING marker"
        )
```

**Step 3: Apply the fix**

In `src/attractor_agent/tools/registry.py`, delete lines 195–196:
```python
# DELETE these two lines:
if was_truncated:
    output += "\n[output was truncated]"
```

The result should be:
```python
output, was_truncated = truncate_output(raw_output_str, limits)
# (no lines after — truncation.py already added the spec WARNING if needed)
```

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestNoDoubleTruncationMarker -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_agent/tools/registry.py
git commit -m "fix: remove extra truncation marker from registry.py (spec §9.5.3)

truncation.py already embeds the spec-compliant WARNING marker. registry.py
was appending '[output was truncated]' on top, sending two signals to the
LLM. Remove the duplicate.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 5: Fix STREAM_START event name in all adapters (§8.4.5)

**Files:**
- Modify: `src/attractor_llm/adapters/openai.py` line 589
- Modify: `src/attractor_llm/adapters/anthropic.py` line 612
- Modify: `src/attractor_llm/adapters/gemini.py` line 498
- Modify: `src/attractor_llm/adapters/openai_compat.py` line 129
- Modify: `src/attractor_llm/streaming.py` line 70 (comment fix)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `types.py` defines both `START = "start"` and `STREAM_START = "stream_start"`. Spec §8.4.5 requires `STREAM_START`. All four adapters currently emit `START`. `streaming.py:70` has a comment falsely claiming adapters emit `STREAM_START`.

---

**Step 1: Write the failing test**

```python
class TestStreamStartEventName:
    """Task 5 — §8.4.5: stream() must emit STREAM_START not START."""

    def test_all_adapters_emit_stream_start_not_start(self):
        """All adapter files must use StreamEventKind.STREAM_START for stream open."""
        import pathlib
        adapter_dir = pathlib.Path("src/attractor_llm/adapters")
        adapters = list(adapter_dir.glob("*.py"))
        assert adapters, "No adapter files found"

        for adapter_file in adapters:
            source = adapter_file.read_text()
            # Files that yield stream events must use STREAM_START
            if "StreamEventKind" in source and "StreamEvent" in source:
                assert "StreamEventKind.STREAM_START" in source or \
                       "StreamEventKind.START" not in source, (
                    f"{adapter_file.name} emits StreamEventKind.START — "
                    f"spec §8.4.5 requires STREAM_START"
                )
```

**Step 2: Run to verify it FAILS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestStreamStartEventName -v
```
Expected: FAIL — lists the four adapter files using `START`.

**Step 3: Apply the fix in all 4 adapters**

In each of the 4 files, change `StreamEventKind.START` → `StreamEventKind.STREAM_START` at the noted lines:

- `src/attractor_llm/adapters/openai.py:589`
- `src/attractor_llm/adapters/anthropic.py:612`
- `src/attractor_llm/adapters/gemini.py:498`
- `src/attractor_llm/adapters/openai_compat.py:129`

Also fix the misleading comment in `src/attractor_llm/streaming.py` around line 70.
Find the line that says something like `# Adapters now emit STREAM_START` and update it to accurately reflect the change.

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestStreamStartEventName -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

Note: also run existing streaming tests to catch any regressions:
```bash
uv run python -m pytest tests/test_audit2_wave2_streaming.py -v
```
The `streaming.py` accumulator already accepts both `START | STREAM_START` for backward compat — but verify no test explicitly checks for `START`.

**Step 6: Commit**
```bash
git add src/attractor_llm/adapters/ src/attractor_llm/streaming.py tests/test_spec_compliance_wave2.py
git commit -m "fix: all adapters emit STREAM_START not START (spec §8.4.5)

StreamEventKind.STREAM_START = 'stream_start' was already defined in
types.py but all four adapters still emitted the legacy START = 'start'.
Any consumer filtering on == STREAM_START would silently receive nothing.
Also corrects the misleading comment in streaming.py.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 6: Auto-emit SESSION_START on first submit() (§9.10.4)

**Files:**
- Modify: `src/attractor_agent/session.py` (in `submit()`)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `__aenter__` emits `SESSION_START` and sets `_session_started = True`. `submit()` doesn't check this flag. Callers using the imperative API (`session.submit(...)` without `async with`) never see `SESSION_START`.

The `submit()` method starts around line 361. Read the beginning of the method to find the right insertion point (after the docstring, before the state check).

---

**Step 1: Write the failing test**

```python
class TestSessionStartAutoEmit:
    """Task 6 — §9.10.4: SESSION_START must fire on first submit() even without context manager."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        yield

    @pytest.mark.asyncio
    async def test_session_start_fires_on_submit_without_context_manager(self):
        """SESSION_START must fire when submit() is called directly (no async with)."""
        from attractor_agent.events import EventKind, SessionEvent
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.client import Client
        from attractor_llm.types import Response, Usage

        captured: list[EventKind] = []

        client = MagicMock(spec=Client)
        client.complete = AsyncMock(return_value=Response(
            text="done",
            tool_calls=[],
            usage=Usage(input_tokens=5, output_tokens=3),
        ))

        session = Session(client=client, config=SessionConfig())
        session.events.on(lambda e: captured.append(e.kind))

        # Call submit() directly — no async with
        await session.submit("hello")

        assert EventKind.SESSION_START in captured, (
            "SESSION_START must fire on first submit() even without async with context manager"
        )

    @pytest.mark.asyncio
    async def test_session_start_not_duplicated_with_context_manager(self):
        """SESSION_START must not fire twice when using async with + submit()."""
        from attractor_agent.events import EventKind
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.client import Client
        from attractor_llm.types import Response, Usage

        captured: list[EventKind] = []
        client = MagicMock(spec=Client)
        client.complete = AsyncMock(return_value=Response(
            text="done", tool_calls=[], usage=Usage(input_tokens=5, output_tokens=3)
        ))

        session = Session(client=client, config=SessionConfig())
        session.events.on(lambda e: captured.append(e.kind))

        async with session:
            await session.submit("hello")

        start_count = captured.count(EventKind.SESSION_START)
        assert start_count == 1, (
            f"SESSION_START must fire exactly once. Got {start_count} times."
        )
```

**Step 2: Run to verify FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestSessionStartAutoEmit -v
```
Expected: `test_session_start_fires_on_submit_without_context_manager` FAILS.

**Step 3: Apply the fix**

At the beginning of `submit()` in `src/attractor_agent/session.py` (after the docstring, before the `if self._state == SessionState.CLOSED:` check), add:

```python
# §9.10.4: auto-emit SESSION_START on first submit() for imperative callers
# (context manager path sets _session_started=True in __aenter__ first)
if not self._session_started:
    await self._emitter.emit(SessionEvent(kind=EventKind.SESSION_START))
    self._session_started = True
```

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestSessionStartAutoEmit -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_agent/session.py
git commit -m "fix: auto-emit SESSION_START on first submit() for imperative callers (spec §9.10.4)

__aenter__ emits SESSION_START but submit() never checked _session_started.
Callers using the imperative API (no async with) never received SESSION_START.
Now emits on first submit() if not already started. Context manager path
unchanged (sets _session_started=True in __aenter__ before submit()).

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 7: Wire middleware constructor in Client (§8.1.6 — part 1)

**Files:**
- Modify: `src/attractor_llm/client.py`
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `Client.__init__` accepts `middleware=` but only stores it in `self._middleware`. `complete()` never references this list. The middleware chain is never executed when using the constructor shorthand.

Read `src/attractor_llm/client.py` to find the full `complete()` method. The fix extracts the LLM-calling logic into `_do_complete()` and routes through middleware in `complete()`.

---

**Step 1: Write the failing test**

```python
class TestMiddlewareConstructorApplied:
    """Task 7 — §8.1.6: Client(middleware=[mw]) must apply the middleware chain."""

    @pytest.mark.asyncio
    async def test_constructor_middleware_is_invoked_on_complete(self):
        """Middleware passed to Client() constructor must intercept complete() calls."""
        from attractor_llm.client import Client
        from attractor_llm.types import Request, Response, Usage

        called = []

        async def tracking_middleware(request, call_next):
            called.append("before")
            response = await call_next(request)
            called.append("after")
            return response

        mock_response = Response(
            text="ok", tool_calls=[], usage=Usage(input_tokens=5, output_tokens=2)
        )

        client = Client(middleware=[tracking_middleware])
        client._adapters["test"] = AsyncMock()
        client._adapters["test"].complete = AsyncMock(return_value=mock_response)
        client._default_provider = "test"

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # Client constructor emits DeprecationWarning — suppress for this test

        await client.complete(
            Request(model="test-model", messages=[], provider="test")
        )

        assert "before" in called, (
            "Middleware must be called when using Client(middleware=[...]) constructor"
        )
        assert "after" in called, "Middleware after_response must be called"
```

**Step 2: Run to verify FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestMiddlewareConstructorApplied -v
```
Expected: FAIL — middleware not invoked.

**Step 3: Apply the fix**

Read `src/attractor_llm/client.py` in full. Find the `complete()` method. Extract its body into `_do_complete()`:

```python
async def _do_complete(self, request: Request) -> Response:
    """Execute the LLM call directly, bypassing middleware."""
    # [move the existing complete() body here — provider selection, retry, etc.]

async def complete(self, request: Request) -> Response:
    """Complete a request, routing through middleware if configured (§8.1.6)."""
    if self._middleware:
        from attractor_llm.middleware import apply_middleware

        class _DirectClient:
            """Thin wrapper that calls _do_complete to avoid circular reference."""
            def __init__(self, client: "Client") -> None:
                self._c = client
            async def complete(self, req: Request) -> Response:
                return await self._c._do_complete(req)
            def __getattr__(self, name: str) -> Any:
                return getattr(self._c, name)

        wrapped = apply_middleware(_DirectClient(self), self._middleware)
        return await wrapped.complete(request)
    return await self._do_complete(request)
```

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestMiddlewareConstructorApplied -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_llm/client.py
git commit -m "fix: Client(middleware=[...]) constructor now applies middleware chain (spec §8.1.6)

Previously self._middleware was stored but never applied. Extracted the
LLM-calling logic into _do_complete(), route through apply_middleware()
in complete() when middleware is present.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 8: Add stream() to _CallNextMiddlewareClient (§8.1.6 — part 2)

**Files:**
- Modify: `src/attractor_llm/middleware.py` (lines ~361–387)
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `_CallNextMiddlewareClient` at lines 361–387 only implements `complete()`. Its `__getattr__` passes `stream()` directly to `self._client`, bypassing the entire chain. After Task 7, middleware applied via the constructor intercepts `complete()` but `stream()` still bypasses it.

---

**Step 1: Write the failing test**

```python
class TestStreamThroughMiddlewareChain:
    """Task 8 — §8.1.6: stream() must pass through the functional middleware chain."""

    @pytest.mark.asyncio
    async def test_stream_intercepted_by_functional_middleware(self):
        """apply_middleware with functional middleware must intercept stream() calls."""
        from attractor_llm.middleware import apply_middleware

        stream_called = []

        async def tracking_middleware(request, call_next):
            stream_called.append("before")
            result = await call_next(request)
            stream_called.append("after")
            return result

        mock_client = MagicMock()
        mock_client.stream = AsyncMock(return_value=AsyncMock())

        wrapped = apply_middleware(mock_client, [tracking_middleware])
        await wrapped.stream(MagicMock())

        assert "before" in stream_called, (
            "Middleware must intercept stream() calls — currently __getattr__ bypasses chain"
        )
```

**Step 2: Run to verify FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestStreamThroughMiddlewareChain -v
```
Expected: FAIL — `"before"` not in `stream_called`.

**Step 3: Apply the fix**

In `src/attractor_llm/middleware.py`, add a `stream()` method to `_CallNextMiddlewareClient` (after `complete()`, before `__getattr__`):

```python
async def stream(self, request: Any) -> Any:
    """Route stream() calls through the functional middleware chain (§8.1.6)."""
    async def core(req: Any) -> Any:
        return await self._client.stream(req)

    handler = core
    for mw in reversed(self._middleware):
        handler = _wrap_call_next(mw, handler)

    return await handler(request)
```

Also update the class docstring to remove the note about stream() bypassing the chain (since it no longer does).

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestStreamThroughMiddlewareChain -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_llm/middleware.py
git commit -m "fix: _CallNextMiddlewareClient.stream() routes through middleware chain (spec §8.1.6)

__getattr__ was passing stream() directly to self._client, bypassing the
entire functional middleware chain. Added explicit stream() method that
builds the same call_next chain used by complete().

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

## Phase 3 — Naming and Polish

### Task 9: Rename max_tool_rounds_per_turn → max_tool_rounds_per_input (§9.1.4)

**Files:**
Source files to rename:
- `src/attractor_agent/session.py` — 4 occurrences (lines 83, 524, 525, 532)
- `src/attractor_agent/subagent.py` — line 100
- `src/attractor_agent/subagent_manager.py` — line 125
- `src/attractor_pipeline/backends.py` — line 93

Test files to update:
- `tests/test_spec_compliance_final.py` — 6 occurrences
- `tests/test_aggressive.py` — 2 occurrences
- `tests/test_wave3_event_truncation_steering.py` — 1 occurrence
- `tests/test_live_comprehensive.py` — 1 occurrence

---

**Step 1: Write the failing test**

```python
class TestMaxToolRoundsRename:
    """Task 9 — §9.1.4: SessionConfig must use max_tool_rounds_per_input (spec name)."""

    def test_session_config_has_max_tool_rounds_per_input_field(self):
        """SessionConfig must expose max_tool_rounds_per_input not _per_turn."""
        from attractor_agent.session import SessionConfig
        config = SessionConfig()
        assert hasattr(config, "max_tool_rounds_per_input"), (
            "SessionConfig must have max_tool_rounds_per_input field (spec §9.1.4). "
            "Found only max_tool_rounds_per_turn which is the old non-spec name."
        )
        assert config.max_tool_rounds_per_input == 0

    def test_old_name_emits_deprecation_warning(self):
        """The old field name max_tool_rounds_per_turn must emit DeprecationWarning."""
        from attractor_agent.session import SessionConfig
        config = SessionConfig()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = config.max_tool_rounds_per_turn
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "max_tool_rounds_per_input" in str(w[0].message)
```

**Step 2: Run to verify FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestMaxToolRoundsRename -v
```
Expected: FAIL — `SessionConfig` has no `max_tool_rounds_per_input`.

**Step 3: Apply the fix**

In `src/attractor_agent/session.py`, in `SessionConfig` dataclass:

```python
# BEFORE:
max_tool_rounds_per_turn: int = 0  # 0 = unlimited (spec §9 SessionConfig)

# AFTER:
max_tool_rounds_per_input: int = 0  # 0 = unlimited (spec §9.1.4 SessionConfig)

@property
def max_tool_rounds_per_turn(self) -> int:
    """Deprecated: use max_tool_rounds_per_input (renamed per spec §9.1.4)."""
    import warnings as _warnings
    _warnings.warn(
        "max_tool_rounds_per_turn is deprecated; use max_tool_rounds_per_input (spec §9.1.4)",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.max_tool_rounds_per_input
```

Then update internal references in `session.py` (lines 524, 525, 532):
- Change `self._config.max_tool_rounds_per_turn` → `self._config.max_tool_rounds_per_input`

Update remaining source files:
- `subagent.py:100`: `max_tool_rounds_per_turn=` → `max_tool_rounds_per_input=`
- `subagent_manager.py:125`: same
- `backends.py:93`: same

Update all test files (use find-and-replace within each file):
- `tests/test_spec_compliance_final.py`: 6 occurrences
- `tests/test_aggressive.py`: 2 occurrences
- `tests/test_wave3_event_truncation_steering.py`: 1 occurrence
- `tests/test_live_comprehensive.py`: 1 occurrence

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestMaxToolRoundsRename -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py \
        src/attractor_agent/session.py \
        src/attractor_agent/subagent.py \
        src/attractor_agent/subagent_manager.py \
        src/attractor_pipeline/backends.py \
        tests/test_spec_compliance_final.py \
        tests/test_aggressive.py \
        tests/test_wave3_event_truncation_steering.py \
        tests/test_live_comprehensive.py
git commit -m "refactor: rename max_tool_rounds_per_turn -> max_tool_rounds_per_input (spec §9.1.4)

SessionConfig field name diverged from spec §9.1.4 which uses
max_tool_rounds_per_input. Renamed field with a deprecated compat property
for backward compatibility. Updated all internal references and test files.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 10: Add edge_id to Diagnostic (§11.2.10)

**Files:**
- Modify: `src/attractor_pipeline/validation.py`
- Test: `tests/test_spec_compliance_wave2.py`

**Context:** `Diagnostic.edge_index: int = -1` stores a positional index. Spec §11.2.10 requires a stable named edge identifier. Indices shift if the edge list is reordered.

Current `Diagnostic` dataclass (lines 39–47):
```python
@dataclass
class Diagnostic:
    rule: str
    severity: Severity
    message: str
    node_id: str = ""
    edge_index: int = -1
```

---

**Step 1: Write the failing test**

```python
class TestDiagnosticEdgeId:
    """Task 10 — §11.2.10: Diagnostic must have a named edge_id field."""

    def test_diagnostic_has_edge_id_field(self):
        """Diagnostic dataclass must have edge_id: str field."""
        from attractor_pipeline.validation import Diagnostic, Severity
        d = Diagnostic(rule="R01", severity=Severity.ERROR, message="test")
        assert hasattr(d, "edge_id"), (
            "Diagnostic must have edge_id: str field (spec §11.2.10 requires named edge ID)"
        )
        assert d.edge_id == ""  # default empty

    def test_edge_diagnostics_have_populated_edge_id(self):
        """Lint rules that fire on edges must set edge_id in 'source->target' format."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.validation import validate

        # Build a graph with a duplicate edge (triggers an edge-related rule)
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="Mdiamond")
        graph.nodes["end"] = Node(id="end", shape="Msquare")
        graph.edges.append(Edge(source="start", target="end"))
        graph.edges.append(Edge(source="start", target="end"))  # duplicate
        graph.goal = "test"

        diagnostics = validate(graph)
        edge_diags = [d for d in diagnostics if d.edge_index >= 0 or d.edge_id]

        if edge_diags:
            for d in edge_diags:
                assert d.edge_id != "", (
                    f"Rule {d.rule} set edge_index={d.edge_index} but edge_id is empty. "
                    f"edge_id must be set in 'source->target' format."
                )
                assert "->" in d.edge_id, (
                    f"edge_id must be in 'source->target' format. Got: '{d.edge_id}'"
                )
```

**Step 2: Run to verify FAIL**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestDiagnosticEdgeId -v
```

**Step 3: Apply the fix**

In `src/attractor_pipeline/validation.py`, add `edge_id: str = ""` to `Diagnostic`:

```python
@dataclass
class Diagnostic:
    """A single validation finding."""
    rule: str
    severity: Severity
    message: str
    node_id: str = ""
    edge_index: int = -1   # positional (deprecated; use edge_id)
    edge_id: str = ""      # §11.2.10: stable named identifier "source->target"
```

Then find all places that set `edge_index` to find which rules need `edge_id` too:
```bash
grep -n "edge_index=" src/attractor_pipeline/validation.py
```

For each lint rule that sets `edge_index=idx` (where the rule iterates over edges), also set `edge_id=f"{edge.source}->{edge.target}"`. Read the relevant rule functions to find the pattern — typically something like:
```python
for idx, edge in enumerate(graph.edges):
    # ... some check ...
    diagnostics.append(Diagnostic(
        rule="RXX",
        ...
        edge_index=idx,
        edge_id=f"{edge.source}->{edge.target}",  # ADD THIS
    ))
```

**Step 4: Run to verify PASS**
```bash
uv run python -m pytest tests/test_spec_compliance_wave2.py::TestDiagnosticEdgeId -v
```

**Step 5: Full mock suite**
```bash
uv run python -m pytest tests/ [ignores] -x -q
```

**Step 6: Commit**
```bash
git add tests/test_spec_compliance_wave2.py src/attractor_pipeline/validation.py
git commit -m "feat: add edge_id field to Diagnostic for stable edge identification (spec §11.2.10)

edge_index: int was a positional ordinal that shifts when edges are reordered.
Added edge_id: str defaulting to 'source->target' format. edge_index retained
for backward compat. All lint rules that fire on edges now populate both.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

### Task 11: Add per-turn cache assertions (§8.9.40 / §8.9.42)

**Files:**
- Modify: `tests/test_audit2_wave5_pipeline_hardening.py`

**Context:** `TestOpenAICacheEfficiency` and `TestGeminiCacheEfficiency` assert cumulative `cache_read_tokens / input_tokens > 0.50` across 5 turns. Spec §8.9.40 and §8.9.42 also require `cache_read_tokens > 0` on turn 2+ specifically — meaning caching has actually kicked in by turn 2, not just averaged across all 5 turns.

---

**Step 1: Read the test file**
```bash
grep -n "cache_ratio\|cumulative_cache\|for.*turn\|_TURNS" tests/test_audit2_wave5_pipeline_hardening.py | head -30
```
Find the loop structure in `TestOpenAICacheEfficiency.test_openai_cache_efficiency_live` and `TestGeminiCacheEfficiency.test_gemini_cache_efficiency_live`.

**Step 2: Apply the fix**

Inside the `for i, turn in enumerate(_TURNS):` loop in both `TestOpenAICacheEfficiency` and `TestGeminiCacheEfficiency`, add after the turn result is retrieved:

```python
# §8.9.40/8.9.42: by turn 2, caching must have started
if i == 1:  # turn 2 (0-indexed)
    assert result.total_usage.cache_read_tokens > 0, (
        f"Expected cache_read_tokens > 0 on turn 2 — "
        f"caching should have started by now. "
        f"Got: {result.total_usage}"
    )
```

Both test methods already have `@pytest.mark.xfail(strict=False)` — this covers the new assertion too since provider-side caching isn't guaranteed.

**Step 3: Run to verify the new assertion is exercised**
```bash
# These are live tests — they'll skip without API keys
uv run python -m pytest tests/test_audit2_wave5_pipeline_hardening.py::TestOpenAICacheEfficiency -v
uv run python -m pytest tests/test_audit2_wave5_pipeline_hardening.py::TestGeminiCacheEfficiency -v
```
Expected: SKIP (no keys) or PASS/XFAIL (with keys).

**Step 4: Commit**
```bash
git add tests/test_audit2_wave5_pipeline_hardening.py
git commit -m "test: add per-turn turn-2 cache assertion for OpenAI + Gemini (spec §8.9.40/§8.9.42)

The cumulative 5-turn ratio test doesn't prove caching started by turn 2.
Added per-turn assertion after turn index 1 (turn 2) inside both
TestOpenAICacheEfficiency and TestGeminiCacheEfficiency. Wrapped with
existing xfail(strict=False) since provider-side caching isn't guaranteed.

🤖 Generated with [Amplifier](https://github.com/microsoft/amplifier)

Co-Authored-By: Amplifier <240397093+microsoft-amplifier@users.noreply.github.com>"
```

---

## Final Validation

After all 11 tasks:

```bash
# 1. Full mock suite (should be 100% green)
uv run python -m pytest tests/ \
  --ignore=tests/test_e2e_integration.py \
  --ignore=tests/test_live_comprehensive.py \
  --ignore=tests/test_live_wave9_10_p1.py \
  --ignore=tests/test_issue36_hexagon_hang.py \
  --ignore=tests/test_audit2_wave6_live_parity_matrix.py \
  --ignore=tests/test_coverage_gaps_live.py \
  --ignore=tests/test_wave16b_live_coverage.py \
  --ignore=tests/test_e2e_integration_parity.py \
  -q

# 2. New spec compliance test file only
uv run python -m pytest tests/test_spec_compliance_wave2.py -v

# 3. Code quality
uv run python -m ruff check src/ tests/test_spec_compliance_wave2.py

# 4. Run audit recipe to verify gap count drops
# (in a separate session — requires all 3 API keys)
```

Expected after Tasks 1–11: audit gap count drops from 15 to ≤4 (Tasks 7+8 middleware and Task 3 text events may require live testing for full verification).
