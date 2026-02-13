"""Tests for apply_patch, generate() API, and subagent spawning.

Covers:
- apply_patch: parse, apply, create, delete, multi-hunk, errors
- generate(): text, tool loop, structured output, streaming
- subagent: spawn, depth limiting, abort, profile integration
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from attractor_agent.abort import AbortSignal
from attractor_agent.subagent import (
    MaxDepthError,
    SubagentResult,
    spawn_subagent,
)
from attractor_agent.tools.apply_patch import (
    FilePatch,
    Hunk,
    apply_patch_to_file,
    parse_patch,
)
from attractor_agent.tools.core import ALL_CORE_TOOLS, ALL_TOOLS_WITH_PATCH, set_allowed_roots
from attractor_llm.client import Client
from attractor_llm.generate import generate, generate_object
from attractor_llm.types import (
    Tool,
)
from tests.helpers import (
    MockAdapter,
    make_text_response,
    make_tool_call_response,
)

# ================================================================== #
# apply_patch: Parser
# ================================================================== #


class TestPatchParser:
    def test_simple_modification(self):
        patch = """\
--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,3 @@
 line 1
-old line 2
+new line 2
 line 3
"""
        ps = parse_patch(patch)
        assert len(ps.patches) == 1
        assert ps.patches[0].target_path == "hello.py"
        assert len(ps.patches[0].hunks) == 1
        assert ps.patches[0].hunks[0].old_start == 1
        assert ps.patches[0].hunks[0].old_count == 3

    def test_file_creation(self):
        patch = """\
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,3 @@
+line 1
+line 2
+line 3
"""
        ps = parse_patch(patch)
        assert len(ps.patches) == 1
        assert ps.patches[0].is_creation
        assert ps.patches[0].target_path == "new_file.py"

    def test_file_deletion(self):
        patch = """\
--- a/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-line 1
-line 2
"""
        ps = parse_patch(patch)
        assert len(ps.patches) == 1
        assert ps.patches[0].is_deletion
        assert ps.patches[0].target_path == "old_file.py"

    def test_multi_file_patch(self):
        patch = """\
--- a/file1.py
+++ b/file1.py
@@ -1,1 +1,1 @@
-old
+new
--- a/file2.py
+++ b/file2.py
@@ -1,1 +1,1 @@
-foo
+bar
"""
        ps = parse_patch(patch)
        assert len(ps.patches) == 2
        assert ps.patches[0].target_path == "file1.py"
        assert ps.patches[1].target_path == "file2.py"

    def test_multi_hunk_patch(self):
        patch = """\
--- a/code.py
+++ b/code.py
@@ -1,3 +1,3 @@
 line 1
-old line 2
+new line 2
 line 3
@@ -10,3 +10,3 @@
 line 10
-old line 11
+new line 11
 line 12
"""
        ps = parse_patch(patch)
        assert len(ps.patches) == 1
        assert len(ps.patches[0].hunks) == 2
        assert ps.patches[0].hunks[0].old_start == 1
        assert ps.patches[0].hunks[1].old_start == 10

    def test_empty_patch(self):
        ps = parse_patch("")
        assert len(ps.patches) == 0

    def test_no_prefix_paths(self):
        patch = """\
--- hello.py
+++ hello.py
@@ -1,1 +1,1 @@
-old
+new
"""
        ps = parse_patch(patch)
        assert ps.patches[0].target_path == "hello.py"


# ================================================================== #
# apply_patch: Applicator
# ================================================================== #


class TestPatchApplicator:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path):
        self.sandbox = tmp_path
        set_allowed_roots([str(tmp_path)])
        yield
        set_allowed_roots([os.getcwd()])

    @pytest.mark.asyncio
    async def test_apply_simple_modification(self):
        target = self.sandbox / "hello.py"
        target.write_text("line 1\nold line 2\nline 3\n")

        patch = FilePatch(
            old_path="a/hello.py",
            new_path="b/hello.py",
            hunks=[
                Hunk(
                    old_start=1,
                    old_count=3,
                    new_start=1,
                    new_count=3,
                    lines=[" line 1", "-old line 2", "+new line 2", " line 3"],
                )
            ],
        )
        result = await apply_patch_to_file(self.sandbox, patch)
        assert "Patched" in result
        assert "new line 2" in target.read_text()
        assert "old line 2" not in target.read_text()

    @pytest.mark.asyncio
    async def test_apply_file_creation(self):
        patch = FilePatch(
            old_path="/dev/null",
            new_path="b/new_file.py",
            hunks=[
                Hunk(
                    old_start=0,
                    old_count=0,
                    new_start=1,
                    new_count=2,
                    lines=["+print('hello')", "+print('world')"],
                )
            ],
        )
        result = await apply_patch_to_file(self.sandbox, patch)
        assert "Created" in result
        assert (self.sandbox / "new_file.py").exists()
        content = (self.sandbox / "new_file.py").read_text()
        assert "hello" in content

    @pytest.mark.asyncio
    async def test_apply_file_deletion(self):
        target = self.sandbox / "delete_me.py"
        target.write_text("content")

        patch = FilePatch(
            old_path="a/delete_me.py",
            new_path="/dev/null",
            hunks=[],
        )
        result = await apply_patch_to_file(self.sandbox, patch)
        assert "Deleted" in result
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises(self):
        patch = FilePatch(
            old_path="a/missing.py",
            new_path="b/missing.py",
            hunks=[Hunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=["-x", "+y"])],
        )
        with pytest.raises(FileNotFoundError):
            await apply_patch_to_file(self.sandbox, patch)

    @pytest.mark.asyncio
    async def test_path_confinement(self):
        patch = FilePatch(
            old_path="/dev/null",
            new_path="b/../../../etc/evil.py",
            hunks=[Hunk(old_start=0, old_count=0, new_start=1, new_count=1, lines=["+evil"])],
        )
        with pytest.raises(PermissionError):
            await apply_patch_to_file(self.sandbox, patch)

    @pytest.mark.asyncio
    async def test_nested_directory_creation(self):
        patch = FilePatch(
            old_path="/dev/null",
            new_path="b/deep/nested/dir/file.py",
            hunks=[
                Hunk(
                    old_start=0,
                    old_count=0,
                    new_start=1,
                    new_count=1,
                    lines=["+content"],
                )
            ],
        )
        await apply_patch_to_file(self.sandbox, patch)
        assert (self.sandbox / "deep" / "nested" / "dir" / "file.py").exists()


# ================================================================== #
# apply_patch: Tool integration
# ================================================================== #


class TestApplyPatchTool:
    def test_apply_patch_in_tool_list(self):
        assert any(t.name == "apply_patch" for t in ALL_TOOLS_WITH_PATCH)
        assert len(ALL_TOOLS_WITH_PATCH) == 7
        assert len(ALL_CORE_TOOLS) == 6  # original list unchanged


# ================================================================== #
# generate() API
# ================================================================== #


class TestGenerateAPI:
    @pytest.mark.asyncio
    async def test_simple_text_generation(self):
        adapter = MockAdapter(responses=[make_text_response("Hello world")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(client, "mock-model", "Say hello", provider="mock")
        assert result == "Hello world"
        assert adapter.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        await generate(
            client,
            "mock-model",
            "test",
            system="Be helpful",
            provider="mock",
        )
        assert adapter.requests[0].system == "Be helpful"

    @pytest.mark.asyncio
    async def test_generate_with_tool_loop(self):
        """generate() auto-executes tool calls and feeds results back."""
        call_count = 0

        async def mock_execute(input: str = "", **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return f"tool result {call_count}"

        tool = Tool(
            name="my_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            execute=mock_execute,
        )

        adapter = MockAdapter(
            responses=[
                # Round 1: model calls tool
                make_tool_call_response("my_tool", {"input": "test"}, "tc-1"),
                # Round 2: model responds with text
                make_text_response("Done with tool result"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate(
            client,
            "mock-model",
            "Use the tool",
            tools=[tool],
            provider="mock",
        )
        assert "Done with tool result" in result
        assert call_count == 1
        assert adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_max_rounds(self):
        """generate() stops after max_rounds even if model keeps calling tools."""

        async def noop(**kwargs: Any) -> str:
            return "ok"

        tool = Tool(
            name="t",
            description="t",
            parameters={"type": "object", "properties": {}},
            execute=noop,
        )

        # Model always calls tools, never text
        responses = [make_tool_call_response("t", {}, f"tc-{i}") for i in range(20)]
        adapter = MockAdapter(responses=responses)
        client = Client()
        client.register_adapter("mock", adapter)

        await generate(
            client,
            "mock-model",
            "loop",
            tools=[tool],
            max_rounds=3,
            provider="mock",
        )
        assert adapter.call_count <= 4  # max_rounds + 1

    @pytest.mark.asyncio
    async def test_generate_object(self):
        """generate_object() returns parsed JSON."""
        adapter = MockAdapter(
            responses=[
                make_text_response('{"name": "Alice", "age": 30}'),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate_object(
            client,
            "mock-model",
            "Extract info",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            provider="mock",
        )
        assert result["name"] == "Alice"
        assert result["age"] == 30

    @pytest.mark.asyncio
    async def test_generate_object_strips_markdown(self):
        """generate_object() handles JSON wrapped in markdown code fences."""
        adapter = MockAdapter(
            responses=[
                make_text_response('```json\n{"key": "value"}\n```'),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        result = await generate_object(
            client,
            "mock-model",
            "Extract",
            provider="mock",
        )
        assert result["key"] == "value"

    @pytest.mark.asyncio
    async def test_generate_object_invalid_json_raises(self):
        adapter = MockAdapter(
            responses=[
                make_text_response("not json at all"),
            ]
        )
        client = Client()
        client.register_adapter("mock", adapter)

        with pytest.raises(ValueError, match="not valid JSON"):
            await generate_object(
                client,
                "mock-model",
                "Extract",
                provider="mock",
            )


# ================================================================== #
# Subagent spawning
# ================================================================== #


class TestSubagentSpawning:
    @pytest.mark.asyncio
    async def test_basic_spawn(self):
        adapter = MockAdapter(responses=[make_text_response("Task done")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await spawn_subagent(
            client=client,
            prompt="Do the thing",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            include_tools=False,
        )
        assert isinstance(result, SubagentResult)
        assert "Task done" in result.text
        assert result.depth == 1
        assert result.model == "mock-model"

    @pytest.mark.asyncio
    async def test_depth_limit_raises(self):
        client = Client()

        with pytest.raises(MaxDepthError, match="depth limit"):
            await spawn_subagent(
                client=client,
                prompt="test",
                parent_depth=3,
                max_depth=3,
            )

    @pytest.mark.asyncio
    async def test_depth_exactly_at_limit(self):
        """depth == max_depth should raise (child would be depth+1)."""
        client = Client()

        with pytest.raises(MaxDepthError):
            await spawn_subagent(
                client=client,
                prompt="test",
                parent_depth=5,
                max_depth=5,
            )

    @pytest.mark.asyncio
    async def test_abort_propagated_to_child(self):
        abort = AbortSignal()
        abort.set()

        adapter = MockAdapter(responses=[make_text_response("ignored")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            abort_signal=abort,
            include_tools=False,
        )
        assert "aborted" in result.text.lower()

    @pytest.mark.asyncio
    async def test_profile_applied_to_subagent(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            include_tools=False,
        )

        # The system prompt should include the base profile + subagent marker
        req = adapter.requests[0]
        assert "[SUBAGENT]" in (req.system or "")
        assert "depth 1/3" in (req.system or "")

    @pytest.mark.asyncio
    async def test_context_injected_into_system_prompt(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            context={"project": "attractor", "task": "refactor"},
            include_tools=False,
        )

        req = adapter.requests[0]
        assert "attractor" in (req.system or "")
        assert "refactor" in (req.system or "")

    @pytest.mark.asyncio
    async def test_subagent_gets_delegate_tool_when_not_at_max_depth(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            include_tools=True,
        )

        req = adapter.requests[0]
        tool_names = [t.name for t in (req.tools or [])]
        assert "delegate" in tool_names

    @pytest.mark.asyncio
    async def test_subagent_no_delegate_tool_at_max_depth_minus_one(self):
        """At depth max_depth-1, the child is at max_depth, so no delegate."""
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=2,
            max_depth=3,
            model="mock-model",
            provider="mock",
            include_tools=True,
        )

        req = adapter.requests[0]
        tool_names = [t.name for t in (req.tools or [])]
        # Child depth is 3, max_depth is 3 -> can't delegate further
        assert "delegate" not in tool_names

    @pytest.mark.asyncio
    async def test_usage_tracked(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        result = await spawn_subagent(
            client=client,
            prompt="test",
            parent_depth=0,
            max_depth=3,
            model="mock-model",
            provider="mock",
            include_tools=False,
        )
        assert result.usage.input_tokens > 0 or result.usage.output_tokens > 0
