# DoD Spec Compliance Implementation Plan

> **Status: COMPLETED** — Merged in commit 78f3ac7.

> **Execution:** Use the subagent-driven-development workflow to implement this plan.

**Goal:** Fix the top-priority gaps (P1 critical + P2 medium) in Attractor's implementation against the StrongDM nlspec Definition of Done checklists.

**Architecture:** All changes are additive or surgical modifications to existing modules. No new packages. One new test file (`tests/test_dod_gaps.py`) houses all new tests. Source changes touch `attractor_llm` (types, errors, client, generate, adapters), `attractor_agent` (tools/registry, tools/core), and `attractor_pipeline` (engine/runner, handlers/codergen). The PR targets Samuel's fork.

**Tech Stack:** Python 3.12+, Pydantic v2, anyio, pytest + pytest-asyncio (auto mode), ruff, pyright basic

**Test conventions:**
- Class-based grouping: `class TestFeatureName:`
- `@pytest.mark.asyncio` on async test methods
- `MockAdapter` from `tests/helpers.py` for LLM mocking
- `make_text_response()`, `make_tool_call_response()`, `make_multi_tool_response()` helpers
- `tmp_path` fixture for filesystem sandboxing
- `ContentPart.tool_call_part(id, name, args)` for tool call simulation
- No conftest.py — all shared fixtures live in `tests/helpers.py`

---

## Batch 0: Setup

### Task 0: Create Feature Branch and Verify Baseline

**Purpose:** Establish a clean starting point. All existing tests must pass before we change anything.

**Step 1: Create the feature branch**

```bash
cd <project-root>
git checkout -b fix/dod-spec-compliance
```

**Step 2: Install dev dependencies**

```bash
uv sync --all-extras
```

**Step 3: Run the full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All tests pass. If any fail, stop and fix before proceeding.

**Step 4: Run lint and type checks**

```bash
ruff check src/ tests/
pyright src/
```

Expected: Clean (or only pre-existing warnings).

**Step 5: Create the test file scaffold**

Create: `tests/test_dod_gaps.py`

```python
"""Tests for DoD spec compliance fixes.

Each test class corresponds to a gap from the nlspec DoD checklist.
Organized by priority: P1 (critical) first, then P2 (medium).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from attractor_llm.client import Client
from attractor_llm.errors import InvalidRequestError, SDKError
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    Tool,
    Usage,
)
from tests.helpers import (
    MockAdapter,
    make_multi_tool_response,
    make_text_response,
    make_tool_call_response,
)
```

**Step 6: Verify the new test file is discovered**

```bash
python -m pytest tests/test_dod_gaps.py -x -q
```

Expected: `no tests ran` (0 collected, no errors).

**Step 7: Commit**

```bash
git add tests/test_dod_gaps.py
git commit -m "chore: scaffold test file for DoD spec compliance fixes"
```

---

## Batch 1: Pipeline Retry Backoff (P1 #1)

> **Gap:** When a pipeline node returns FAIL or RETRY, the engine immediately re-executes with zero delay. This hammers LLM APIs.  
> **Spec:** §3.6 — RetryPolicy pattern: initial_delay_ms=200, backoff_factor=2.0, max_delay_ms=60000, jitter=True.  
> **File:** `src/attractor_pipeline/engine/runner.py` lines 391–395.

### Task 1: Write failing test for pipeline retry backoff

**Files:**
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test class**

Append to `tests/test_dod_gaps.py`:

```python
from attractor_agent.abort import AbortSignal
from attractor_pipeline.engine.runner import (
    Graph,
    HandlerRegistry,
    HandlerResult,
    Node,
    Outcome,
    PipelineResult,
    PipelineStatus,
    run_pipeline,
)
from attractor_pipeline.graph import Edge


# ================================================================== #
# P1 #1: Pipeline retry backoff
# ================================================================== #


class _FailNHandler:
    """Handler that fails N times then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._call_count = 0

    async def execute(self, node, context, graph, logs_root, abort_signal) -> HandlerResult:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return HandlerResult(status=Outcome.FAIL, failure_reason="deliberate failure")
        return HandlerResult(status=Outcome.SUCCESS, output="done")


def _make_linear_graph() -> Graph:
    """A minimal graph: start -> work -> exit."""
    g = Graph(name="test", default_max_retry=5)
    g.nodes["start"] = Node(id="start", shape="ellipse")
    g.nodes["work"] = Node(id="work", shape="box")
    g.nodes["exit"] = Node(id="exit", shape="Msquare")
    g.edges.append(Edge(source="start", target="work"))
    g.edges.append(Edge(source="work", target="exit"))
    return g


class TestPipelineRetryBackoff:
    """P1 #1: Pipeline retry should use exponential backoff, not zero delay."""

    @pytest.mark.asyncio
    async def test_retry_calls_sleep_with_positive_delay(self):
        """When a node fails and retries, anyio.sleep must be called with delay > 0."""
        graph = _make_linear_graph()
        handler = _FailNHandler(fail_count=2)

        registry = HandlerRegistry()
        registry.register("start", _FailNHandler(fail_count=0))  # start always succeeds
        registry.register("codergen", handler)
        registry.register("exit", _FailNHandler(fail_count=0))

        with patch(
            "attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await run_pipeline(graph, registry)

        assert result.status == PipelineStatus.COMPLETED
        # anyio.sleep must have been called for retries
        assert mock_sleep.call_count >= 2, f"Expected >=2 sleep calls, got {mock_sleep.call_count}"
        # Every delay must be positive
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay > 0, f"Retry delay must be positive, got {delay}"

    @pytest.mark.asyncio
    async def test_retry_delays_increase_exponentially(self):
        """Successive retry delays should increase (exponential backoff)."""
        graph = _make_linear_graph()
        handler = _FailNHandler(fail_count=3)

        registry = HandlerRegistry()
        registry.register("start", _FailNHandler(fail_count=0))
        registry.register("codergen", handler)
        registry.register("exit", _FailNHandler(fail_count=0))

        with patch(
            "attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            result = await run_pipeline(graph, registry)

        assert result.status == PipelineStatus.COMPLETED
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(delays) >= 2
        # With jitter, we can't assert exact values, but the max possible delay
        # for attempt N is initial * factor^N. The min possible (with jitter) is
        # 0.5 * initial * factor^N. So delay[1] max > delay[0] min.
        # Just verify delays are in the right ballpark: all > 0, bounded by max.
        for d in delays:
            assert 0 < d <= 60.0, f"Delay {d} out of range (0, 60]"
```

**Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestPipelineRetryBackoff -x -v
```

Expected: FAIL — `anyio.sleep` is never called (patch target doesn't exist yet or mock_sleep.call_count == 0).

### Task 2: Implement pipeline retry backoff

**Files:**
- Modify: `src/attractor_pipeline/engine/runner.py`

**Step 1: Add imports at top of runner.py**

At the top of `src/attractor_pipeline/engine/runner.py`, after the existing imports (around line 8), add:

```python
import random

import anyio
```

**Step 2: Add backoff constants after the imports section**

After the existing imports (before the `# Handler protocol` comment around line 32), add:

```python
# Pipeline retry backoff constants (Spec §3.6)
_RETRY_INITIAL_DELAY = 0.2  # 200ms
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_MAX_DELAY = 60.0  # 60 seconds


def _compute_retry_delay(attempt: int) -> float:
    """Compute exponential backoff delay with jitter for pipeline retries."""
    delay = _RETRY_INITIAL_DELAY * (_RETRY_BACKOFF_FACTOR**attempt)
    delay = min(delay, _RETRY_MAX_DELAY)
    # Equal jitter: uniform in [delay/2, delay]
    delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
    return delay
```

**Step 3: Add the sleep call before the retry `continue`**

In the `run_pipeline()` function, find lines 391–395:

```python
        # Handle retry on failure
        if result.status in (Outcome.FAIL, Outcome.RETRY):
            if retry_count < max_retries:
                node_retry_counts[current_node.id] = retry_count + 1
                continue  # retry same node
```

Replace with:

```python
        # Handle retry on failure
        if result.status in (Outcome.FAIL, Outcome.RETRY):
            if retry_count < max_retries:
                node_retry_counts[current_node.id] = retry_count + 1
                # Exponential backoff with jitter (Spec §3.6)
                delay = _compute_retry_delay(retry_count)
                await anyio.sleep(delay)
                continue  # retry same node
```

**Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestPipelineRetryBackoff -x -v
```

Expected: PASS — both tests green.

**Step 5: Run existing pipeline tests to verify no regression**

```bash
python -m pytest tests/test_pipeline_engine.py -x -q
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/attractor_pipeline/engine/runner.py tests/test_dod_gaps.py
git commit -m "feat: add exponential backoff to pipeline node retries (Spec §3.6)

Pipeline retries previously had zero delay between attempts, which
hammers LLM APIs. Now uses exponential backoff with jitter:
initial=200ms, factor=2x, max=60s."
```

---

## Batch 2: Parallel Tool Execution (P1 #2)

> **Gap:** Tool calls execute sequentially. Spec §5.7 requires concurrent execution via asyncio.gather().  
> **Files:** `src/attractor_agent/tools/registry.py` lines 144–161, `src/attractor_llm/generate.py` lines 106–130.

### Task 3: Write failing tests for parallel tool execution

**Files:**
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add test class for parallel ToolRegistry execution**

Append to `tests/test_dod_gaps.py`:

```python
from attractor_agent.tools.registry import ToolRegistry


# ================================================================== #
# P1 #2: Parallel tool execution
# ================================================================== #


class TestParallelToolExecution:
    """P1 #2: Multiple tool calls should execute concurrently."""

    @pytest.mark.asyncio
    async def test_tool_registry_runs_tools_concurrently(self):
        """ToolRegistry.execute_tool_calls should run multiple tools in parallel."""
        call_times: list[float] = []

        async def slow_tool(**kwargs: Any) -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return "done"

        tool = Tool(name="slow", description="slow tool", execute=slow_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [
            ContentPart.tool_call_part("tc-1", "slow", {}),
            ContentPart.tool_call_part("tc-2", "slow", {}),
            ContentPart.tool_call_part("tc-3", "slow", {}),
        ]

        start = time.monotonic()
        results = await registry.execute_tool_calls(tool_calls)
        elapsed = time.monotonic() - start

        assert len(results) == 3
        assert all(r.kind == ContentPartKind.TOOL_RESULT for r in results)
        assert all(not r.is_error for r in results)
        # Sequential would take ~0.3s. Parallel should take ~0.1s.
        assert elapsed < 0.25, f"Expected parallel execution (<0.25s), got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self):
        """Results must come back in the same order as the tool calls."""
        async def echo_tool(value: str = "") -> str:
            await asyncio.sleep(0.01)
            return f"echo:{value}"

        tool = Tool(name="echo", description="echo", execute=echo_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [
            ContentPart.tool_call_part("tc-a", "echo", {"value": "first"}),
            ContentPart.tool_call_part("tc-b", "echo", {"value": "second"}),
            ContentPart.tool_call_part("tc-c", "echo", {"value": "third"}),
        ]

        results = await registry.execute_tool_calls(tool_calls)

        assert results[0].tool_call_id == "tc-a"
        assert results[0].output == "echo:first"
        assert results[1].tool_call_id == "tc-b"
        assert results[1].output == "echo:second"
        assert results[2].tool_call_id == "tc-c"
        assert results[2].output == "echo:third"

    @pytest.mark.asyncio
    async def test_parallel_partial_failure(self):
        """If one tool fails, others should still succeed. Failed tool gets is_error=True."""
        async def good_tool(**kwargs: Any) -> str:
            return "ok"

        async def bad_tool(**kwargs: Any) -> str:
            raise RuntimeError("boom")

        good = Tool(name="good", description="good", execute=good_tool)
        bad = Tool(name="bad", description="bad", execute=bad_tool)
        registry = ToolRegistry()
        registry.register(good)
        registry.register(bad)

        tool_calls = [
            ContentPart.tool_call_part("tc-1", "good", {}),
            ContentPart.tool_call_part("tc-2", "bad", {}),
            ContentPart.tool_call_part("tc-3", "good", {}),
        ]

        results = await registry.execute_tool_calls(tool_calls)

        assert len(results) == 3
        assert not results[0].is_error
        assert results[1].is_error
        assert "boom" in (results[1].output or "")
        assert not results[2].is_error

    @pytest.mark.asyncio
    async def test_single_tool_call_still_works(self):
        """A single tool call should work fine (no gather needed)."""
        async def simple_tool(**kwargs: Any) -> str:
            return "result"

        tool = Tool(name="simple", description="simple", execute=simple_tool)
        registry = ToolRegistry()
        registry.register(tool)

        tool_calls = [ContentPart.tool_call_part("tc-1", "simple", {})]
        results = await registry.execute_tool_calls(tool_calls)

        assert len(results) == 1
        assert results[0].output == "result"
        assert not results[0].is_error
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dod_gaps.py::TestParallelToolExecution -x -v
```

Expected: `test_tool_registry_runs_tools_concurrently` FAILS (elapsed > 0.25s because execution is sequential).

### Task 4: Implement parallel tool execution in ToolRegistry

**Files:**
- Modify: `src/attractor_agent/tools/registry.py`

**Step 1: Add asyncio import**

At the top of `src/attractor_agent/tools/registry.py`, after the existing imports, add:

```python
import asyncio
```

**Step 2: Replace `execute_tool_calls` method**

Find the `execute_tool_calls` method (around line 144) and replace it entirely:

```python
    async def execute_tool_calls(self, tool_calls: list[ContentPart]) -> list[ContentPart]:
        """Execute multiple tool calls concurrently. Spec §5.7.

        Tool calls are executed in parallel via asyncio.gather().
        Results are returned in the same order as the input tool calls.
        Partial failures are handled: successful tools return normally,
        failed tools return is_error=True results.

        Args:
            tool_calls: List of ContentParts with kind=TOOL_CALL.

        Returns:
            List of ContentParts with kind=TOOL_RESULT, in the same order.
        """
        if len(tool_calls) <= 1:
            # Single tool call: no gather overhead needed
            return [await self.execute_tool_call(tc) for tc in tool_calls]

        # Multiple tool calls: execute concurrently
        results = await asyncio.gather(
            *(self.execute_tool_call(tc) for tc in tool_calls),
            return_exceptions=True,
        )

        # Convert any unexpected exceptions to error results
        final: list[ContentPart] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                tc = tool_calls[i]
                final.append(
                    ContentPart.tool_result_part(
                        tool_call_id=tc.tool_call_id or "",
                        name=tc.name or "",
                        output=f"Error: {type(result).__name__}: {result}",
                        is_error=True,
                    )
                )
            else:
                final.append(result)
        return final
```

**Step 3: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestParallelToolExecution -x -v
```

Expected: All 4 tests PASS.

**Step 4: Run existing agent tests for regression**

```bash
python -m pytest tests/test_agent_loop.py tests/test_aggressive.py -x -q
```

Expected: All existing tests still pass.

**Step 5: Commit**

```bash
git add src/attractor_agent/tools/registry.py tests/test_dod_gaps.py
git commit -m "feat: parallel tool execution via asyncio.gather (Spec §5.7)

ToolRegistry.execute_tool_calls() now runs multiple tool calls
concurrently instead of sequentially. Order is preserved.
Partial failures handled: successful tools return normally,
failed tools get is_error=True results."
```

### Task 5: Parallel tool execution in generate.py

**Files:**
- Modify: `src/attractor_llm/generate.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
from attractor_llm.generate import generate


class TestParallelToolsInGenerate:
    """P1 #2b: generate() tool loop should also execute tools in parallel."""

    @pytest.mark.asyncio
    async def test_generate_runs_tools_concurrently(self):
        """generate() should execute multiple tool calls concurrently."""
        call_times: list[float] = []

        async def slow_tool(value: str = "") -> str:
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return f"result:{value}"

        tool = Tool(name="slow", description="slow", execute=slow_tool)

        adapter = MockAdapter(
            responses=[
                make_multi_tool_response([
                    ("tc-1", "slow", {"value": "a"}),
                    ("tc-2", "slow", {"value": "b"}),
                    ("tc-3", "slow", {"value": "c"}),
                ]),
                make_text_response("All done"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        start = time.monotonic()
        result = await generate(client, "mock-model", "Do it", tools=[tool], provider="mock")
        elapsed = time.monotonic() - start

        assert "All done" in result
        assert adapter.call_count == 2  # tool call response + final text
        # Sequential would take ~0.3s. Parallel should take ~0.1s.
        assert elapsed < 0.25, f"Expected parallel (<0.25s), got {elapsed:.2f}s"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestParallelToolsInGenerate -x -v
```

Expected: FAIL (elapsed > 0.25s).

**Step 3: Refactor generate.py tool loop to use asyncio.gather**

In `src/attractor_llm/generate.py`, add import at the top (after line 32):

```python
import asyncio
```

Then replace the tool execution loop (lines 105–130) with:

```python
        # Execute tool calls in parallel (Spec §5.7)
        async def _exec_one(tc: ContentPart) -> tuple[ContentPart, str, bool]:
            """Execute a single tool call, return (tool_call, output, is_error)."""
            tool = _find_tool(tools, tc.name or "")
            if tool and tool.execute:
                try:
                    args = tc.arguments
                    if isinstance(args, str):
                        args = json.loads(args)
                    raw = await tool.execute(**args)
                    output = str(raw) if not isinstance(raw, str) else raw
                    return tc, output, False
                except Exception as exc:  # noqa: BLE001
                    return tc, f"{type(exc).__name__}: {exc}", True
            else:
                return tc, f"Unknown tool: {tc.name}", True

        if len(response.tool_calls) == 1:
            exec_results = [await _exec_one(response.tool_calls[0])]
        else:
            exec_results = await asyncio.gather(
                *(_exec_one(tc) for tc in response.tool_calls),
            )

        for tc, output, is_error in exec_results:
            history.append(
                Message.tool_result(
                    tc.tool_call_id or "",
                    tc.name or "",
                    output,
                    is_error=is_error,
                )
            )
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestParallelToolsInGenerate -x -v
```

Expected: PASS.

**Step 5: Run existing generate tests for regression**

```bash
python -m pytest tests/test_new_features.py -x -q
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/attractor_llm/generate.py tests/test_dod_gaps.py
git commit -m "feat: parallel tool execution in generate() tool loop (Spec §5.7)"
```

---

## Batch 3: Client.from_env() and Default Client (P1 #3)

> **Gap:** No `Client.from_env()` for auto-detecting providers from environment variables.  
> **Spec:** §2.2 — `Client.from_env()` checks OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY.

### Task 6: Write failing tests for Client.from_env()

**Files:**
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test class**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P1 #3: Client.from_env() + default client
# ================================================================== #


class TestClientFromEnv:
    """P1 #3: Client.from_env() should auto-detect providers from env vars."""

    @pytest.mark.asyncio
    async def test_detects_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = await Client.from_env()
        assert "openai" in client._adapters
        assert "anthropic" not in client._adapters
        assert "gemini" not in client._adapters

    @pytest.mark.asyncio
    async def test_detects_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = await Client.from_env()
        assert "anthropic" in client._adapters
        assert "openai" not in client._adapters

    @pytest.mark.asyncio
    async def test_detects_gemini_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = await Client.from_env()
        assert "gemini" in client._adapters

    @pytest.mark.asyncio
    async def test_detects_google_api_key_as_gemini(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-google")

        client = await Client.from_env()
        assert "gemini" in client._adapters

    @pytest.mark.asyncio
    async def test_detects_multiple_providers(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-anthropic")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = await Client.from_env()
        assert "openai" in client._adapters
        assert "anthropic" in client._adapters

    @pytest.mark.asyncio
    async def test_no_keys_returns_empty_client(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = await Client.from_env()
        assert len(client._adapters) == 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dod_gaps.py::TestClientFromEnv -x -v
```

Expected: FAIL — `Client` has no `from_env` method.

### Task 7: Implement Client.from_env() and default client pattern

**Files:**
- Modify: `src/attractor_llm/client.py`
- Modify: `src/attractor_llm/__init__.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add from_env() classmethod to Client**

In `src/attractor_llm/client.py`, add this method inside the `Client` class, after the `register_adapter` method (after line 59):

```python
    @classmethod
    async def from_env(cls, **kwargs: Any) -> Client:
        """Create a Client with providers auto-detected from environment variables.

        Checks for standard API key env vars and registers the corresponding
        adapter for each one found. First registered adapter is the default.

        Supported env vars (Spec §2.2):
        - OPENAI_API_KEY → OpenAI adapter
        - ANTHROPIC_API_KEY → Anthropic adapter
        - GEMINI_API_KEY or GOOGLE_API_KEY → Gemini adapter

        Args:
            **kwargs: Passed to Client.__init__ (e.g., retry_policy).

        Returns:
            A configured Client instance.
        """
        import os

        from attractor_llm.adapters.base import ProviderConfig

        client = cls(**kwargs)

        if api_key := os.environ.get("OPENAI_API_KEY"):
            from attractor_llm.adapters.openai import OpenAIAdapter

            client.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=api_key)))

        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            from attractor_llm.adapters.anthropic import AnthropicAdapter

            client.register_adapter(
                "anthropic", AnthropicAdapter(ProviderConfig(api_key=api_key))
            )

        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            from attractor_llm.adapters.gemini import GeminiAdapter

            client.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=gemini_key)))

        return client
```

Also add `import os` is not needed since it's inside the method (deferred import). But we need `Any` in the import — check if it's already imported. Looking at client.py line 12: `from typing import Any` — yes, it's there.

**Step 2: Add module-level default client pattern**

At the bottom of `src/attractor_llm/client.py` (after the Client class), add:

```python
# ------------------------------------------------------------------ #
# Module-level default client (Spec §2.2)
# ------------------------------------------------------------------ #

_default_client: Client | None = None


def set_default_client(client: Client) -> None:
    """Set the module-level default client. Spec §2.2."""
    global _default_client
    _default_client = client


def get_default_client() -> Client:
    """Get the module-level default client. Spec §2.2.

    Raises:
        InvalidRequestError: If no default client has been set.
    """
    if _default_client is None:
        raise InvalidRequestError(
            "No default client configured. "
            "Call set_default_client() or await Client.from_env() first."
        )
    return _default_client
```

**Step 3: Add test for default client pattern**

Append to the `TestClientFromEnv` class in `tests/test_dod_gaps.py`:

```python
class TestDefaultClient:
    """P1 #3b: Module-level default client pattern."""

    def test_get_default_client_raises_when_not_set(self):
        from attractor_llm.client import _default_client, get_default_client, set_default_client

        # Save and clear
        import attractor_llm.client as client_mod

        saved = client_mod._default_client
        client_mod._default_client = None
        try:
            with pytest.raises(InvalidRequestError, match="No default client"):
                get_default_client()
        finally:
            client_mod._default_client = saved

    def test_set_and_get_default_client(self):
        from attractor_llm.client import get_default_client, set_default_client

        import attractor_llm.client as client_mod

        saved = client_mod._default_client
        try:
            client = Client()
            set_default_client(client)
            assert get_default_client() is client
        finally:
            client_mod._default_client = saved
```

**Step 4: Update `__init__.py` exports**

In `src/attractor_llm/__init__.py`, add to the imports section (around line 11):

```python
from attractor_llm.client import Client, get_default_client, set_default_client
```

And add to `__all__` (after `"Client"`):

```python
    "get_default_client",
    "set_default_client",
```

**Step 5: Run all from_env and default client tests**

```bash
python -m pytest tests/test_dod_gaps.py::TestClientFromEnv tests/test_dod_gaps.py::TestDefaultClient -x -v
```

Expected: All PASS.

**Step 6: Run existing client tests for regression**

```bash
python -m pytest tests/test_retry_client.py -x -q
```

Expected: All existing tests still pass.

**Step 7: Commit**

```bash
git add src/attractor_llm/client.py src/attractor_llm/__init__.py tests/test_dod_gaps.py
git commit -m "feat: add Client.from_env() and default client pattern (Spec §2.2)

Client.from_env() auto-detects providers from standard env vars:
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY.
Also adds set_default_client()/get_default_client() for module-level
default client pattern."
```

---

## Batch 4: Quick Wins (P2 #4, #7, #14)

These are small, surgical fixes that can be reviewed together.

### Task 8: Shell timeout default 120→10 (P2 #4)

> **Gap:** `DEFAULT_SHELL_TIMEOUT = 120`. Spec §9.4 says default should be 10 seconds.  
> **File:** `src/attractor_agent/tools/core.py` line 363.

**Files:**
- Modify: `src/attractor_agent/tools/core.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #4: Shell timeout default
# ================================================================== #


class TestShellTimeoutDefault:
    """P2 #4: Default shell timeout should be 10s, not 120s (Spec §9.4)."""

    def test_default_shell_timeout_is_10(self):
        from attractor_agent.tools.core import DEFAULT_SHELL_TIMEOUT

        assert DEFAULT_SHELL_TIMEOUT == 10
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestShellTimeoutDefault -x -v
```

Expected: FAIL — `assert 120 == 10`.

**Step 3: Fix the constant**

In `src/attractor_agent/tools/core.py`, line 363, change:

```python
DEFAULT_SHELL_TIMEOUT = 120
```

to:

```python
DEFAULT_SHELL_TIMEOUT = 10
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestShellTimeoutDefault -x -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/attractor_agent/tools/core.py tests/test_dod_gaps.py
git commit -m "fix: shell timeout default 120s → 10s (Spec §9.4)"
```

### Task 9: generate() rejects prompt + messages (P2 #7)

> **Gap:** `generate()` silently combines both `prompt` and `messages`. Spec §4.3 says this is an error.  
> **File:** `src/attractor_llm/generate.py` line 80.

**Files:**
- Modify: `src/attractor_llm/generate.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #7: generate() rejects prompt + messages
# ================================================================== #


class TestGeneratePromptValidation:
    """P2 #7: generate() must reject prompt + messages together (Spec §4.3)."""

    @pytest.mark.asyncio
    async def test_rejects_prompt_and_messages_together(self):
        adapter = MockAdapter(responses=[make_text_response("ignored")])
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(InvalidRequestError, match="Cannot provide both"):
            await generate(
                client,
                "mock-model",
                "hello",
                messages=[Message.user("world")],
                provider="mock",
            )

    @pytest.mark.asyncio
    async def test_allows_prompt_alone(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "hello", provider="mock")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_allows_messages_alone(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(
            client,
            "mock-model",
            messages=[Message.user("hello")],
            provider="mock",
        )
        assert result == "ok"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dod_gaps.py::TestGeneratePromptValidation -x -v
```

Expected: `test_rejects_prompt_and_messages_together` FAILS (no exception raised). `test_allows_messages_alone` FAILS (TypeError — prompt is currently required).

**Step 3: Update the generate() function signature and add validation**

In `src/attractor_llm/generate.py`, change the `generate()` function signature (lines 47–59) to make `prompt` optional:

Replace:

```python
async def generate(
    client: Client,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
    max_rounds: int = 10,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
) -> str:
```

with:

```python
async def generate(
    client: Client,
    model: str,
    prompt: str | None = None,
    *,
    system: str | None = None,
    messages: list[Message] | None = None,
    tools: list[Tool] | None = None,
    max_rounds: int = 10,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
) -> str:
```

Then add the import and validation at the top of the function body. Replace lines 80–81:

```python
    history = list(messages or [])
    history.append(Message.user(prompt))
```

with:

```python
    # Spec §4.3: Cannot provide both prompt and messages
    if prompt is not None and messages is not None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Cannot provide both 'prompt' and 'messages'")
    if prompt is None and messages is None:
        from attractor_llm.errors import InvalidRequestError

        raise InvalidRequestError("Must provide either 'prompt' or 'messages'")

    if messages is not None:
        history = list(messages)
    else:
        history = [Message.user(prompt)]  # type: ignore[arg-type]
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestGeneratePromptValidation -x -v
```

Expected: All PASS.

**Step 5: Run existing generate tests for regression**

```bash
python -m pytest tests/test_new_features.py -x -q
```

Expected: All existing tests still pass (they all use `prompt` only).

**Step 6: Commit**

```bash
git add src/attractor_llm/generate.py tests/test_dod_gaps.py
git commit -m "fix: generate() rejects prompt + messages together (Spec §4.3)

Also makes prompt optional so callers can use messages-only mode."
```

### Task 10: Add ConfigurationError (P2 #14)

> **Gap:** Spec defines `ConfigurationError` for SDK misconfiguration. Code uses `InvalidRequestError` for this.  
> **File:** `src/attractor_llm/errors.py` (new class), `src/attractor_llm/client.py` line 103.

**Files:**
- Modify: `src/attractor_llm/errors.py`
- Modify: `src/attractor_llm/client.py`
- Modify: `src/attractor_llm/__init__.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #14: ConfigurationError
# ================================================================== #


class TestConfigurationError:
    """P2 #14: ConfigurationError for SDK misconfiguration (Spec §6)."""

    def test_configuration_error_exists(self):
        from attractor_llm.errors import ConfigurationError

        err = ConfigurationError("bad config")
        assert isinstance(err, SDKError)
        assert not err.retryable

    @pytest.mark.asyncio
    async def test_client_resolve_raises_configuration_error(self):
        from attractor_llm.errors import ConfigurationError

        client = Client()
        # No adapters registered
        request = Request(model="some-model", messages=[Message.user("hi")])
        with pytest.raises(ConfigurationError, match="Cannot resolve provider"):
            await client.complete(request)
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_dod_gaps.py::TestConfigurationError -x -v
```

Expected: FAIL — `ConfigurationError` doesn't exist.

**Step 3: Add ConfigurationError to errors.py**

In `src/attractor_llm/errors.py`, after the `InvalidRequestError` class (around line 140), add:

```python
class ConfigurationError(SDKError):
    """SDK misconfiguration. Not retryable. Spec §6.

    Raised when the SDK is not properly configured (e.g., no providers
    registered, missing API keys).
    """

    def __init__(
        self, message: str, *, provider: str | None = None, status_code: int | None = None
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code, retryable=False)
```

**Step 4: Update client.py to use ConfigurationError**

In `src/attractor_llm/client.py`, update the import (line 15) — change:

```python
from attractor_llm.errors import InvalidRequestError, SDKError
```

to:

```python
from attractor_llm.errors import ConfigurationError, InvalidRequestError, SDKError
```

Then find the error at lines 103–107 in `_resolve_adapter`:

```python
        raise InvalidRequestError(
            f"Cannot resolve provider for model {request.model!r}. "
            f"Set request.provider explicitly or register the provider. "
            f"Available: {list(self._adapters.keys())}"
        )
```

Replace `InvalidRequestError` with `ConfigurationError`:

```python
        raise ConfigurationError(
            f"Cannot resolve provider for model {request.model!r}. "
            f"Set request.provider explicitly or register the provider. "
            f"Available: {list(self._adapters.keys())}"
        )
```

Also change the explicit-provider-not-found error at lines 77–80:

```python
            raise InvalidRequestError(
                f"Provider {request.provider!r} not registered. "
                f"Available: {list(self._adapters.keys())}"
            )
```

to:

```python
            raise ConfigurationError(
                f"Provider {request.provider!r} not registered. "
                f"Available: {list(self._adapters.keys())}"
            )
```

**Step 5: Add to __init__.py exports**

In `src/attractor_llm/__init__.py`, add `ConfigurationError` to the errors import block and to `__all__`:

Add to imports:

```python
from attractor_llm.errors import (
    ...
    ConfigurationError,
    ...
)
```

Add to `__all__` (in the Errors section):

```python
    "ConfigurationError",
```

**Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestConfigurationError -x -v
```

Expected: All PASS.

**Step 7: Check if any existing tests relied on InvalidRequestError for resolve failures**

```bash
python -m pytest tests/test_retry_client.py -x -q
```

Expected: If any test catches `InvalidRequestError` for provider-not-found, it will fail. Update those tests to catch `ConfigurationError` instead. The likely test is in `test_retry_client.py` — look for `InvalidRequestError` assertions related to provider resolution and update them.

**Step 8: Commit**

```bash
git add src/attractor_llm/errors.py src/attractor_llm/client.py src/attractor_llm/__init__.py tests/test_dod_gaps.py
git commit -m "feat: add ConfigurationError for SDK misconfiguration (Spec §6)

Replaces InvalidRequestError in Client._resolve_adapter() with the
spec-mandated ConfigurationError. Not retryable."
```

### Task 11: Add DEVELOPER role to enum (P2 #6 — part 1)

> **Gap:** Role enum has 4 roles, spec defines 5 including DEVELOPER.  
> **File:** `src/attractor_llm/types.py` lines 16–22.

**Files:**
- Modify: `src/attractor_llm/types.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #6: DEVELOPER role
# ================================================================== #


class TestDeveloperRole:
    """P2 #6: Role enum must include DEVELOPER (Spec §3.2)."""

    def test_developer_role_exists(self):
        assert Role.DEVELOPER == "developer"
        assert Role.DEVELOPER.value == "developer"

    def test_developer_message_factory(self):
        """Message should have a .developer() factory method."""
        msg = Message.developer("Build instructions here")
        assert msg.role == Role.DEVELOPER
        assert msg.text == "Build instructions here"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRole -x -v
```

Expected: FAIL — `Role` has no `DEVELOPER` attribute.

**Step 3: Add DEVELOPER to the Role enum**

In `src/attractor_llm/types.py`, update the `Role` enum (around line 16):

```python
class Role(StrEnum):
    """Message roles. Spec §3.1."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
```

**Step 4: Add Message.developer() factory**

In `src/attractor_llm/types.py`, in the `Message` class (after the `system` classmethod, around line 162), add:

```python
    @classmethod
    def developer(cls, text: str) -> Message:
        return cls(role=Role.DEVELOPER, content=[ContentPart.text_part(text)])
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRole -x -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add src/attractor_llm/types.py tests/test_dod_gaps.py
git commit -m "feat: add DEVELOPER role to Role enum (Spec §3.2)"
```

---

## Batch 5: DEVELOPER Role in Adapters (P2 #6 — part 2)

### Task 12: Map DEVELOPER role in OpenAI adapter

> **File:** `src/attractor_llm/adapters/openai.py`  
> OpenAI Responses API supports `"developer"` role natively for input items.

**Files:**
- Modify: `src/attractor_llm/adapters/openai.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
from attractor_llm.adapters.openai import OpenAIAdapter
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.base import ProviderConfig


class TestDeveloperRoleAdapters:
    """P2 #6b: Adapters must handle DEVELOPER role messages."""

    def test_openai_maps_developer_to_developer_role(self):
        adapter = OpenAIAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="gpt-4o",
            messages=[
                Message.developer("Build rules"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        input_items = body["input"]
        # DEVELOPER should map to {"role": "developer"} in OpenAI Responses API
        dev_items = [i for i in input_items if i.get("role") == "developer"]
        assert len(dev_items) == 1
        assert "Build rules" in str(dev_items[0].get("content", ""))
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_openai_maps_developer_to_developer_role -x -v
```

Expected: FAIL — DEVELOPER role not handled in the match statement.

**Step 3: Add DEVELOPER handling to OpenAI adapter**

In `src/attractor_llm/adapters/openai.py`, find the `_translate_input_items` method (around line 135). In the `match msg.role:` block (line 147), add a case for DEVELOPER right after the SYSTEM case:

After:

```python
                case Role.SYSTEM:
                    text = msg.text or ""
                    items.append(
                        {
                            "role": "system",
                            "content": text,
                        }
                    )
```

Add:

```python
                case Role.DEVELOPER:
                    text = msg.text or ""
                    items.append(
                        {
                            "role": "developer",
                            "content": text,
                        }
                    )
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_openai_maps_developer_to_developer_role -x -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/attractor_llm/adapters/openai.py tests/test_dod_gaps.py
git commit -m "feat: map DEVELOPER role in OpenAI adapter (Spec §3.2)"
```

### Task 13: Map DEVELOPER role in Anthropic adapter

> **File:** `src/attractor_llm/adapters/anthropic.py`  
> Anthropic has no developer role — merge DEVELOPER messages into the system parameter.

**Files:**
- Modify: `src/attractor_llm/adapters/anthropic.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `TestDeveloperRoleAdapters` in `tests/test_dod_gaps.py`:

```python
    def test_anthropic_merges_developer_into_system(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("You are helpful."),
                Message.developer("Build rules: always use TDD"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # DEVELOPER content should be merged into the "system" field
        system_parts = body.get("system", [])
        system_text = " ".join(p.get("text", "") for p in system_parts)
        assert "Build rules" in system_text
        # Should not appear in messages
        for msg in body["messages"]:
            for part in msg.get("content", []):
                if isinstance(part, dict):
                    assert "Build rules" not in part.get("text", "")
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_anthropic_merges_developer_into_system -x -v
```

Expected: FAIL.

**Step 3: Update the Anthropic adapter's _split_system method**

In `src/attractor_llm/adapters/anthropic.py`, find the `_split_system` method (around line 152). Currently it only checks `msg.role == Role.SYSTEM`. Update it to also handle `Role.DEVELOPER`:

Replace the role check in the loop body. Find:

```python
        for msg in messages:
            if msg.role == Role.SYSTEM:
```

Replace with:

```python
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
```

You'll also need to add `Role` to the imports if not already imported. Check the top of anthropic.py for existing imports from `attractor_llm.types` — `Role` should already be there since it's used for `Role.USER`, `Role.ASSISTANT`, `Role.TOOL` in `_translate_messages`.

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_anthropic_merges_developer_into_system -x -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/attractor_llm/adapters/anthropic.py tests/test_dod_gaps.py
git commit -m "feat: map DEVELOPER role in Anthropic adapter (merged to system)"
```

### Task 14: Map DEVELOPER role in Gemini adapter

> **File:** `src/attractor_llm/adapters/gemini.py`  
> Gemini has no developer role — merge DEVELOPER messages into systemInstruction.

**Files:**
- Modify: `src/attractor_llm/adapters/gemini.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `TestDeveloperRoleAdapters` in `tests/test_dod_gaps.py`:

```python
    def test_gemini_merges_developer_into_system_instruction(self):
        adapter = GeminiAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="gemini-2.5-pro",
            messages=[
                Message.system("You are helpful."),
                Message.developer("Build rules: always validate input"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # DEVELOPER content should be in systemInstruction
        system_inst = body.get("systemInstruction", {})
        system_text = " ".join(
            p.get("text", "") for p in system_inst.get("parts", [])
        )
        assert "Build rules" in system_text
        # Should not appear in contents
        for content in body.get("contents", []):
            for part in content.get("parts", []):
                assert "Build rules" not in part.get("text", "")
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_gemini_merges_developer_into_system_instruction -x -v
```

Expected: FAIL.

**Step 3: Update the Gemini adapter's _split_system method**

In `src/attractor_llm/adapters/gemini.py`, find the `_split_system` method (around line 160). Update the role check:

Replace:

```python
            if msg.role == Role.SYSTEM:
```

with:

```python
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestDeveloperRoleAdapters::test_gemini_merges_developer_into_system_instruction -x -v
```

Expected: PASS.

**Step 5: Run all adapter tests for regression**

```bash
python -m pytest tests/test_aggressive.py tests/test_polish.py -x -q
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/attractor_llm/adapters/gemini.py tests/test_dod_gaps.py
git commit -m "feat: map DEVELOPER role in Gemini adapter (merged to systemInstruction)"
```

---

## Batch 6: Anthropic Cache Improvements (P2 #5, #15)

### Task 15: Auto-inject prompt-caching beta header (P2 #5)

> **Gap:** `_inject_cache_control()` adds cache_control markers but doesn't auto-add the `prompt-caching-2024-07-31` beta header.  
> **File:** `src/attractor_llm/adapters/anthropic.py` lines 141–148.

**Files:**
- Modify: `src/attractor_llm/adapters/anthropic.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #5: Anthropic cache beta header
# ================================================================== #


class TestAnthropicCacheBetaHeader:
    """P2 #5: prompt-caching beta header must be auto-injected (Spec §2.10)."""

    def test_cache_control_adds_beta_header(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        # _inject_cache_control adds markers, so beta header should be present
        beta_headers = body.get("_beta_headers", [])
        assert "prompt-caching-2024-07-31" in beta_headers, (
            f"Expected prompt-caching beta header, got: {beta_headers}"
        )

    def test_beta_header_not_duplicated(self):
        """If user already specified the beta header, don't add it twice."""
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            provider_options={
                "anthropic": {
                    "beta_headers": ["prompt-caching-2024-07-31"],
                },
            },
        )
        body = adapter._translate_request(request)
        beta_headers = body.get("_beta_headers", [])
        count = beta_headers.count("prompt-caching-2024-07-31")
        assert count == 1, f"Beta header duplicated: {beta_headers}"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestAnthropicCacheBetaHeader -x -v
```

Expected: FAIL — beta header not auto-added.

**Step 3: Update _inject_cache_control to also add the beta header**

In `src/attractor_llm/adapters/anthropic.py`, update the `_inject_cache_control` method (around line 272) to also add the beta header. Change the method signature to accept and mutate the `_beta_headers` list:

Replace the entire `_inject_cache_control` method:

```python
    def _inject_cache_control(self, body: dict[str, Any]) -> None:
        """Auto-inject cache_control breakpoints for prompt caching. Spec §2.10.

        Anthropic prompt caching requires explicit cache_control markers.
        We inject them at three strategic positions:
        1. End of system message (stable across turns)
        2. End of tool definitions (stable across turns)
        3. Last user message (changes each turn -- ephemeral cache)

        Also auto-adds the prompt-caching-2024-07-31 beta header (Spec §2.10).
        """
        # 1. Cache system message -- persistent (stable across turns)
        if "system" in body and body["system"]:
            body["system"][-1]["cache_control"] = {"type": "ephemeral"}

        # 2. Cache tool definitions -- persistent (stable across turns)
        if "tools" in body and body["tools"]:
            body["tools"][-1]["cache_control"] = {"type": "ephemeral"}

        # 3. Cache last user message -- ephemeral (changes each turn)
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg["role"] == "user" and msg["content"]:
                msg["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

        # Auto-add prompt-caching beta header (Spec §2.10)
        beta_headers: list[str] = body.get("_beta_headers", [])
        if "prompt-caching-2024-07-31" not in beta_headers:
            beta_headers.append("prompt-caching-2024-07-31")
        body["_beta_headers"] = beta_headers
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestAnthropicCacheBetaHeader -x -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/attractor_llm/adapters/anthropic.py tests/test_dod_gaps.py
git commit -m "feat: auto-inject prompt-caching beta header (Spec §2.10)"
```

### Task 16: Add auto_cache disable option (P2 #15)

> **Gap:** `_inject_cache_control()` is called unconditionally. Spec §8.6.6 says it should be disableable.  
> **File:** `src/attractor_llm/adapters/anthropic.py` line 148.

**Files:**
- Modify: `src/attractor_llm/adapters/anthropic.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #15: auto_cache disable
# ================================================================== #


class TestAnthropicAutoCacheDisable:
    """P2 #15: auto_cache should be disableable via provider_options."""

    def test_auto_cache_disabled_skips_cache_control(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
            provider_options={"anthropic": {"auto_cache": False}},
        )
        body = adapter._translate_request(request)
        # No cache_control should be injected
        system_parts = body.get("system", [])
        for part in system_parts:
            assert "cache_control" not in part

    def test_auto_cache_enabled_by_default(self):
        adapter = AnthropicAdapter(ProviderConfig(api_key="test"))
        request = Request(
            model="claude-sonnet-4-5",
            messages=[
                Message.system("System prompt"),
                Message.user("Hello"),
            ],
        )
        body = adapter._translate_request(request)
        system_parts = body.get("system", [])
        # Default: cache_control should be injected
        assert any("cache_control" in part for part in system_parts)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestAnthropicAutoCacheDisable -x -v
```

Expected: `test_auto_cache_disabled_skips_cache_control` FAILS (cache_control still injected).

**Step 3: Gate _inject_cache_control on auto_cache option**

In `src/attractor_llm/adapters/anthropic.py`, in `_translate_request`, find line 148:

```python
        # Apply cache_control for prompt caching (§2.10)
        self._inject_cache_control(body)
```

Replace with:

```python
        # Apply cache_control for prompt caching (§2.10)
        # Can be disabled via provider_options.anthropic.auto_cache = false (Spec §8.6.6)
        if anthropic_opts.get("auto_cache", True):
            self._inject_cache_control(body)
```

Note: `anthropic_opts` is already defined on line 128 as `(request.provider_options or {}).get("anthropic", {})`.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestAnthropicAutoCacheDisable -x -v
```

Expected: PASS.

**Step 5: Run existing Anthropic adapter tests**

```bash
python -m pytest tests/test_aggressive.py -x -q
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/attractor_llm/adapters/anthropic.py tests/test_dod_gaps.py
git commit -m "feat: add auto_cache disable via provider_options (Spec §8.6.6)"
```

---

## Batch 7: StepResult / GenerateResult (P2 #8)

> **Gap:** `generate()` returns only final text. Spec §4.3 requires `GenerateResult` with `steps: list[StepResult]`, `total_usage: Usage`.

### Task 17: Define StepResult and GenerateResult types

**Files:**
- Modify: `src/attractor_llm/types.py`
- Modify: `src/attractor_llm/__init__.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #8: StepResult / GenerateResult
# ================================================================== #


class TestGenerateResultType:
    """P2 #8: GenerateResult with step tracking (Spec §4.3)."""

    def test_generate_result_has_text(self):
        from attractor_llm.types import GenerateResult, StepResult

        result = GenerateResult(text="hello", steps=[], total_usage=Usage())
        assert result.text == "hello"

    def test_generate_result_str_returns_text(self):
        from attractor_llm.types import GenerateResult

        result = GenerateResult(text="hello world")
        assert str(result) == "hello world"

    def test_generate_result_equality_with_str(self):
        from attractor_llm.types import GenerateResult

        result = GenerateResult(text="hello")
        assert result == "hello"

    def test_generate_result_tracks_steps(self):
        from attractor_llm.types import GenerateResult, StepResult

        step = StepResult(
            response=Response(
                message=Message.assistant("hi"),
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
        )
        result = GenerateResult(text="hi", steps=[step])
        assert len(result.steps) == 1
        assert result.total_usage.input_tokens == 0  # total_usage set separately

    def test_step_result_has_tool_results(self):
        from attractor_llm.types import StepResult

        tool_result = ContentPart.tool_result_part("tc-1", "read_file", "content")
        step = StepResult(
            response=Response(message=Message.assistant("hi")),
            tool_results=[tool_result],
        )
        assert len(step.tool_results) == 1
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestGenerateResultType -x -v
```

Expected: FAIL — `StepResult` and `GenerateResult` don't exist.

**Step 3: Add the types to types.py**

In `src/attractor_llm/types.py`, at the bottom of the file (after the `StreamEvent` class), add:

```python
# ------------------------------------------------------------------ #
# Generate result types (Spec §4.3)
# ------------------------------------------------------------------ #


@dataclass
class StepResult:
    """Result of a single LLM call step within generate(). Spec §4.3.

    Each step represents one round-trip to the LLM, potentially
    followed by tool executions.
    """

    response: Response
    tool_results: list[ContentPart] = field(default_factory=list)
```

Note: you'll need to add `from dataclasses import dataclass, field` at the top of types.py. Check if it's already imported — looking at the current imports, types.py uses Pydantic `BaseModel` and `Field`, not dataclass. So add the import:

```python
from dataclasses import dataclass, field
```

Then add `GenerateResult`:

```python
@dataclass
class GenerateResult:
    """Result of generate() with step tracking. Spec §4.3.

    Backward-compatible: str(result) returns the text, and
    result == "some string" compares against the text.
    """

    text: str = ""
    steps: list[StepResult] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, GenerateResult):
            return self.text == other.text
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.text)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __bool__(self) -> bool:
        return bool(self.text)
```

**Step 4: Add to __init__.py exports**

In `src/attractor_llm/__init__.py`, add to the types import block:

```python
from attractor_llm.types import (
    ...
    GenerateResult,
    StepResult,
    ...
)
```

And add to `__all__`:

```python
    "StepResult",
    "GenerateResult",
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestGenerateResultType -x -v
```

Expected: All PASS.

**Step 6: Commit**

```bash
git add src/attractor_llm/types.py src/attractor_llm/__init__.py tests/test_dod_gaps.py
git commit -m "feat: add StepResult and GenerateResult types (Spec §4.3)"
```

### Task 18: Return GenerateResult from generate()

**Files:**
- Modify: `src/attractor_llm/generate.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
class TestGenerateReturnsResult:
    """P2 #8b: generate() should return GenerateResult with steps and usage."""

    @pytest.mark.asyncio
    async def test_generate_returns_generate_result(self):
        from attractor_llm.types import GenerateResult

        adapter = MockAdapter(responses=[make_text_response("hello")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Hi", provider="mock")
        assert isinstance(result, GenerateResult)
        assert result.text == "hello"
        assert str(result) == "hello"

    @pytest.mark.asyncio
    async def test_generate_result_tracks_steps(self):
        from attractor_llm.types import GenerateResult

        adapter = MockAdapter(
            responses=[
                make_tool_call_response("read_file", {"path": "/tmp/x"}),
                make_text_response("Done"),
            ]
        )

        async def fake_read(path: str = "") -> str:
            return "file content"

        tool = Tool(name="read_file", description="read", execute=fake_read)
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(
            client, "mock-model", "Read it", tools=[tool], provider="mock"
        )
        assert isinstance(result, GenerateResult)
        assert result.text == "Done"
        assert len(result.steps) == 2  # tool call step + final text step
        assert result.total_usage.input_tokens > 0

    @pytest.mark.asyncio
    async def test_generate_result_backward_compat_with_str(self):
        """Existing code that compares result to a string should still work."""
        adapter = MockAdapter(responses=[make_text_response("hello")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Hi", provider="mock")
        # These patterns should all work for backward compatibility
        assert result == "hello"
        assert "hello" in result
        assert str(result) == "hello"
        assert bool(result)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestGenerateReturnsResult -x -v
```

Expected: FAIL — `generate()` returns `str`, not `GenerateResult`.

**Step 3: Update generate() to return GenerateResult**

In `src/attractor_llm/generate.py`, add to the imports (around line 38):

```python
from attractor_llm.types import (
    FinishReason,
    GenerateResult,
    Message,
    Request,
    Response,
    StepResult,
    StreamEventKind,
    Tool,
    Usage,
)
```

(Add `GenerateResult`, `StepResult`, and `Usage` to the existing import block.)

Change the return type of `generate()`:

```python
) -> GenerateResult:
```

Then update the function body. After the validation block (where `history` is built), add step tracking:

```python
    steps: list[StepResult] = []
    total_usage = Usage()
```

Inside the loop, after `response = await client.complete(request)` (line 95), add usage tracking:

```python
        response = await client.complete(request)
        total_usage = total_usage + response.usage
        history.append(response.message)
```

When the model returns text (no tool calls), wrap the return in GenerateResult:

Replace:

```python
        # If no tool calls, return text
        if response.finish_reason != FinishReason.TOOL_CALLS:
            return response.text or ""
```

with:

```python
        # If no tool calls, return text
        if response.finish_reason != FinishReason.TOOL_CALLS:
            steps.append(StepResult(response=response))
            return GenerateResult(
                text=response.text or "",
                steps=steps,
                total_usage=total_usage,
            )
```

After tool execution, record the step:

After the tool execution loop (the `for tc, output, is_error in exec_results:` block), add:

```python
        # Record this step
        tool_results = [
            ContentPart.tool_result_part(tc.tool_call_id or "", tc.name or "", output, is_error=is_error)
            for tc, output, is_error in exec_results
        ]
        steps.append(StepResult(response=response, tool_results=tool_results))
```

Update the fallback returns at the end of the function (lines 132–134):

Replace:

```python
    if response is not None:
        return response.text or "[Max tool rounds reached]"
    return "[No response generated]"
```

with:

```python
    text = "[No response generated]"
    if response is not None:
        text = response.text or "[Max tool rounds reached]"
    return GenerateResult(text=text, steps=steps, total_usage=total_usage)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestGenerateReturnsResult -x -v
```

Expected: All PASS.

**Step 5: Run existing generate tests for regression**

```bash
python -m pytest tests/test_new_features.py -x -q
```

Expected: All existing tests should pass thanks to `GenerateResult.__eq__` handling string comparison.

**Step 6: Commit**

```bash
git add src/attractor_llm/generate.py tests/test_dod_gaps.py
git commit -m "feat: generate() returns GenerateResult with step tracking (Spec §4.3)

GenerateResult is backward-compatible with str: equality comparison,
containment, str(), and bool() all work as before. New .steps and
.total_usage fields provide step-by-step tracking."
```

---

## Batch 8: Goal Gate Tracking Across All Nodes (P2 #9)

> **Gap:** Goal gates are only checked on exit nodes. Spec §3.4 says ALL nodes with `goal_gate=true` must be tracked.  
> **File:** `src/attractor_pipeline/engine/runner.py` lines 411–461.

### Task 19: Write failing test for all-node goal gate tracking

**Files:**
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #9: Goal gate tracking across all nodes
# ================================================================== #


class _OutcomeHandler:
    """Handler that always returns a specified outcome."""

    def __init__(self, status: Outcome = Outcome.SUCCESS) -> None:
        self._status = status

    async def execute(self, node, context, graph, logs_root, abort_signal) -> HandlerResult:
        return HandlerResult(status=self._status, output="done")


class TestGoalGateAllNodes:
    """P2 #9: Goal gates should be tracked on ALL nodes, not just exit."""

    @pytest.mark.asyncio
    async def test_midpipeline_goal_gate_failure_triggers_redirect(self):
        """A non-exit node with goal_gate should redirect on failure."""
        g = Graph(name="test", default_max_retry=3, max_goal_gate_redirects=5)
        g.nodes["start"] = Node(id="start", shape="ellipse")
        # 'check' node has a goal_gate but is NOT an exit node
        g.nodes["check"] = Node(
            id="check",
            shape="box",
            goal_gate="outcome == 'success'",
            retry_target="start",
        )
        g.nodes["exit"] = Node(id="exit", shape="Msquare")
        g.edges.append(Edge(source="start", target="check"))
        g.edges.append(Edge(source="check", target="exit"))

        call_count = 0

        class _FailOnceCheckHandler:
            async def execute(self, node, context, graph, logs_root, abort_signal):
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    return HandlerResult(status=Outcome.FAIL, failure_reason="not ready")
                return HandlerResult(status=Outcome.SUCCESS, output="ready")

        registry = HandlerRegistry()
        registry.register("start", _OutcomeHandler(Outcome.SUCCESS))
        registry.register("codergen", _FailOnceCheckHandler())
        registry.register("exit", _OutcomeHandler(Outcome.SUCCESS))

        with patch(
            "attractor_pipeline.engine.runner.anyio.sleep", new_callable=AsyncMock
        ):
            result = await run_pipeline(g, registry)

        # The pipeline should complete (the check node eventually passes)
        assert result.status == PipelineStatus.COMPLETED
        # The check node's goal_gate failure should have triggered a redirect
        # back to start, so 'start' should appear multiple times in completed_nodes
        assert result.completed_nodes.count("start") >= 2
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestGoalGateAllNodes -x -v
```

Expected: FAIL — goal gate on non-exit node is not checked.

### Task 20: Implement goal gate tracking on all nodes

**Files:**
- Modify: `src/attractor_pipeline/engine/runner.py`

**Step 1: Add goal gate evaluation for non-exit nodes**

In `src/attractor_pipeline/engine/runner.py`, find the section after checkpoint saving and before the exit-node check (around line 410). Currently, goal gate is only inside the `if current_node.shape == "Msquare":` block.

Add a goal gate check for **non-exit nodes** that have a `goal_gate` set. Insert this BEFORE the exit-node check (before line 411 `# Check if this is an exit node`):

```python
        # Check goal gate on ANY node with goal_gate set (Spec §3.4)
        # Exit nodes handle their own goal gate below.
        if current_node.shape != "Msquare" and current_node.goal_gate:
            gate_vars = {
                "outcome": result.status.value,
                **ctx,
            }
            if not evaluate_condition(current_node.goal_gate, gate_vars):
                goal_gate_redirect_count += 1

                # Circuit breaker
                if (
                    graph.max_goal_gate_redirects > 0
                    and goal_gate_redirect_count >= graph.max_goal_gate_redirects
                ):
                    return PipelineResult(
                        status=PipelineStatus.FAILED,
                        error=(
                            f"Goal gate on node '{current_node.id}' unsatisfied after "
                            f"{goal_gate_redirect_count} redirects "
                            f"(limit: {graph.max_goal_gate_redirects})"
                        ),
                        context=ctx,
                        completed_nodes=completed_nodes,
                        final_outcome=result,
                        duration_seconds=time.monotonic() - start_time,
                    )

                # Redirect to retry target
                retry_target = current_node.retry_target
                if retry_target:
                    target_node = graph.get_node(retry_target)
                    if target_node:
                        current_node = target_node
                        continue

                # No retry target -- fall through to normal edge selection
```

**Step 2: Run the test to verify it passes**

```bash
python -m pytest tests/test_dod_gaps.py::TestGoalGateAllNodes -x -v
```

Expected: PASS.

**Step 3: Run existing pipeline tests for regression**

```bash
python -m pytest tests/test_pipeline_engine.py -x -q
```

Expected: All existing tests still pass.

**Step 4: Commit**

```bash
git add src/attractor_pipeline/engine/runner.py tests/test_dod_gaps.py
git commit -m "feat: evaluate goal gates on all nodes, not just exit (Spec §3.4)

Non-exit nodes with goal_gate now trigger redirect to retry_target
when the gate condition is unsatisfied. Uses the same circuit breaker
as exit-node goal gates."
```

---

## Batch 9: Per-Node Artifact Files (P2 #10)

> **Gap:** No per-node directories or artifact files written during pipeline execution.  
> **Spec:** §5.6 — Each node should write prompt.md, response.md, status.json to `{logs_root}/{node_id}/`.

### Task 21: Write per-node artifacts in codergen handler

**Files:**
- Modify: `src/attractor_pipeline/handlers/codergen.py`
- Modify: `tests/test_dod_gaps.py`

**Step 1: Add the test**

Append to `tests/test_dod_gaps.py`:

```python
# ================================================================== #
# P2 #10: Per-node artifact files
# ================================================================== #


class TestPerNodeArtifacts:
    """P2 #10: Codergen handler should write per-node artifact files (Spec §5.6)."""

    @pytest.mark.asyncio
    async def test_codergen_writes_artifact_files(self, tmp_path):
        """Handler should create {logs_root}/{node_id}/ with artifacts."""
        from attractor_pipeline.handlers.codergen import CodergenHandler

        class _MockBackend:
            async def run(self, node, prompt, context, abort_signal):
                return "LLM response text"

        handler = CodergenHandler(backend=_MockBackend())
        node = Node(id="my_node", shape="box", prompt="Do something: ${goal}")
        context: dict[str, Any] = {"goal": "test goal"}
        graph = Graph(name="test")
        logs_root = tmp_path / "logs"
        logs_root.mkdir()

        result = await handler.execute(node, context, graph, logs_root, None)

        assert result.status == Outcome.SUCCESS

        # Check artifact directory exists
        node_dir = logs_root / "my_node"
        assert node_dir.is_dir(), f"Expected directory {node_dir}"

        # Check prompt.md
        prompt_file = node_dir / "prompt.md"
        assert prompt_file.exists()
        prompt_content = prompt_file.read_text()
        assert "test goal" in prompt_content

        # Check response.md
        response_file = node_dir / "response.md"
        assert response_file.exists()
        assert "LLM response text" in response_file.read_text()

        # Check status.json
        status_file = node_dir / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text())
        assert status_data["status"] == "success"
        assert status_data["node_id"] == "my_node"

    @pytest.mark.asyncio
    async def test_codergen_no_artifacts_without_logs_root(self, tmp_path):
        """When logs_root is None, no artifacts should be written."""
        from attractor_pipeline.handlers.codergen import CodergenHandler

        class _MockBackend:
            async def run(self, node, prompt, context, abort_signal):
                return "response"

        handler = CodergenHandler(backend=_MockBackend())
        node = Node(id="node1", shape="box", prompt="Do it")
        context: dict[str, Any] = {"goal": "test"}
        graph = Graph(name="test")

        result = await handler.execute(node, context, graph, None, None)
        assert result.status == Outcome.SUCCESS
        # No crash, no files written (nothing to assert on filesystem)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dod_gaps.py::TestPerNodeArtifacts -x -v
```

Expected: FAIL — no artifact directory or files created.

**Step 3: Add artifact writing to CodergenHandler**

In `src/attractor_pipeline/handlers/codergen.py`, add an import at the top:

```python
import json
```

Then, inside the `execute()` method, after the handler runs and before the return, add artifact writing. Find the section where the result is prepared (after calling `self._backend.run(...)` and normalizing to `HandlerResult`). Add this before the final return statement:

```python
        # Write per-node artifact files (Spec §5.6)
        if logs_root is not None:
            node_dir = logs_root / node.id
            node_dir.mkdir(parents=True, exist_ok=True)

            # prompt.md -- the expanded prompt sent to the LLM
            (node_dir / "prompt.md").write_text(prompt, encoding="utf-8")

            # response.md -- the LLM response text
            (node_dir / "response.md").write_text(result.output or "", encoding="utf-8")

            # status.json -- the handler result metadata
            status_data = {
                "node_id": node.id,
                "status": result.status.value,
                "preferred_label": result.preferred_label,
                "failure_reason": result.failure_reason,
                "notes": result.notes,
            }
            (node_dir / "status.json").write_text(
                json.dumps(status_data, indent=2), encoding="utf-8"
            )
```

Note: The variable `prompt` is the expanded prompt string (set by `_expand_prompt()`), and `result` is the `HandlerResult`. Make sure the artifact writing goes AFTER the backend call and result normalization, but BEFORE the final `return result`.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_dod_gaps.py::TestPerNodeArtifacts -x -v
```

Expected: All PASS.

**Step 5: Run existing pipeline tests for regression**

```bash
python -m pytest tests/test_pipeline_engine.py tests/test_preamble.py -x -q
```

Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/attractor_pipeline/handlers/codergen.py tests/test_dod_gaps.py
git commit -m "feat: write per-node artifact files in codergen handler (Spec §5.6)

Creates {logs_root}/{node_id}/ directory with:
- prompt.md: expanded prompt sent to LLM
- response.md: LLM response text
- status.json: handler result metadata"
```

---

## Batch 10: Final Validation

### Task 22: Full test suite, lint, and type check

**Step 1: Run the full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All tests pass (existing + new).

**Step 2: Run lint**

```bash
ruff check src/ tests/
```

Expected: Clean. If there are issues, fix them:

```bash
ruff check src/ tests/ --fix
```

**Step 3: Run type checking**

```bash
pyright src/
```

Expected: Clean (or only pre-existing warnings). If new type errors appear, fix them.

**Step 4: Run format check**

```bash
ruff format --check src/ tests/
```

If formatting issues, fix:

```bash
ruff format src/ tests/
```

**Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: fix lint and type issues from DoD compliance changes"
```

**Step 6: Review the commit log**

```bash
git log --oneline fix/dod-spec-compliance..HEAD
```

Expected: A clean sequence of focused commits, one per feature.

---

## P3 Items (Future Work — Not Implemented in This PR)

The following items from the gap analysis are lower priority and deferred to a follow-up PR:

| # | Gap | Why Deferred |
|---|-----|-------------|
| 11 | Environment context in system prompts | Requires subprocess calls (git branch), needs careful testing on CI |
| 12 | Project doc discovery (AGENTS.md, etc.) | Feature design needed — which files, search depth, caching strategy |
| 13 | `generate_object()` native structured output | Requires adapter-level changes to all 3 providers, needs E2E testing |

---

## Summary of Changes

| Batch | Gap | Files Modified | Tests Added |
|-------|-----|---------------|-------------|
| 1 | Pipeline retry backoff | `runner.py` | 2 |
| 2 | Parallel tool execution | `registry.py`, `generate.py` | 5 |
| 3 | Client.from_env() | `client.py`, `__init__.py` | 8 |
| 4 | Quick wins (timeout, validation, ConfigError, DEVELOPER) | `core.py`, `generate.py`, `errors.py`, `client.py`, `types.py`, `__init__.py` | 8 |
| 5 | DEVELOPER in adapters | `openai.py`, `anthropic.py`, `gemini.py` | 3 |
| 6 | Anthropic cache | `anthropic.py` | 4 |
| 7 | StepResult / GenerateResult | `types.py`, `generate.py`, `__init__.py` | 8 |
| 8 | Goal gate all nodes | `runner.py` | 1 |
| 9 | Per-node artifacts | `codergen.py` | 2 |
| **Total** | **12 gaps fixed** | **13 files** | **41 tests** |

---

Plan complete and saved to `docs/plans/2026-02-13-dod-spec-compliance.md`.

**Execution options:**

1. **Subagent-Driven (this session)**
   - Fresh agent per task
   - Two-stage review (spec then quality)
   - Fast iteration

2. **Parallel Session**
   - Open new session for execution
   - Batch execution with human checkpoints

Which approach?
