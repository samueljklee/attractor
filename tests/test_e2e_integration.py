"""Comprehensive end-to-end integration tests with real LLM APIs.

These tests simulate ACTUAL usage patterns -- not mocks. They call
real provider APIs, use real tools (read/write/edit files), and
verify real outputs.

Each test creates a temp workspace, runs a pipeline or agent session,
and validates that the output is actually correct.

Requirements: ANTHROPIC_API_KEY must be set.

Run: uv run python -m pytest tests/test_e2e_integration.py -v -s
(Use -s to see LLM output in real time)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from attractor_agent.profiles import get_profile
from attractor_agent.session import Session, SessionConfig
from attractor_agent.subagent import spawn_subagent
from attractor_agent.tools.core import ALL_CORE_TOOLS, set_allowed_roots
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.client import Client
from attractor_llm.generate import generate, generate_object
from attractor_pipeline import (
    HandlerRegistry,
    PipelineStatus,
    parse_dot,
    register_default_handlers,
    run_pipeline,
)
from attractor_pipeline.backends import AgentLoopBackend, DirectLLMBackend

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace and confine tools to it."""
    set_allowed_roots([str(tmp_path)])
    yield tmp_path
    set_allowed_roots([os.getcwd()])


@pytest.fixture
def anthropic_client():
    """Create a Client with Anthropic adapter."""
    client = Client()
    adapter = AnthropicAdapter(
        ProviderConfig(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            timeout=120.0,
        )
    )
    client.register_adapter("anthropic", adapter)
    return client


# ================================================================== #
# TEST 1: Agent session with real tools writing real files
# ================================================================== #


class TestAgentWithRealTools:
    """Agent loop calling real tools against real LLM."""

    @pytest.mark.asyncio
    async def test_agent_writes_and_reads_file(self, workspace, anthropic_client):
        """Agent creates a file with write_file, then reads it back."""
        profile = get_profile("anthropic")
        config = SessionConfig(
            model="claude-sonnet-4-5",
            provider="anthropic",
            max_turns=10,
        )
        config = profile.apply_to_config(config)
        tools = profile.get_tools(list(ALL_CORE_TOOLS))

        async with anthropic_client:
            session = Session(client=anthropic_client, config=config, tools=tools)
            result = await session.submit(
                f"Write a file called 'hello.py' in {workspace} with a "
                f"function called greet() that returns the string "
                f"'Hello from Attractor'. Then read the file back and "
                f"tell me what it contains."
            )

        # Verify the file was actually created
        hello_file = workspace / "hello.py"
        assert hello_file.exists(), "Agent should have created hello.py"

        content = hello_file.read_text()
        assert "def greet" in content, "File should contain greet function"
        assert "Hello from Attractor" in content, "File should contain the greeting string"

        # Verify the agent's response mentions reading the file
        assert "greet" in result.lower() or "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_agent_edits_existing_file(self, workspace, anthropic_client):
        """Agent uses edit_file to modify an existing file."""
        # Pre-create a file
        target = workspace / "config.py"
        target.write_text('DB_HOST = "localhost"\nDB_PORT = 5432\nDB_NAME = "mydb"\n')

        profile = get_profile("anthropic")
        config = SessionConfig(
            model="claude-sonnet-4-5",
            provider="anthropic",
            max_turns=10,
        )
        config = profile.apply_to_config(config)
        tools = profile.get_tools(list(ALL_CORE_TOOLS))

        async with anthropic_client:
            session = Session(client=anthropic_client, config=config, tools=tools)
            await session.submit(
                f"Read the file {target} and change the DB_PORT from "
                f"5432 to 3306. Use edit_file, not write_file."
            )

        # Verify the edit was applied
        content = target.read_text()
        assert "3306" in content, "Port should be changed to 3306"
        assert "5432" not in content, "Old port should be gone"
        # Other lines should be preserved
        assert "DB_HOST" in content
        assert "DB_NAME" in content


# ================================================================== #
# TEST 2: Pipeline with AgentLoopBackend (real LLM + real tools)
# ================================================================== #


class TestPipelineWithAgentLoop:
    """Pipeline running through AgentLoopBackend with real LLM."""

    @pytest.mark.asyncio
    async def test_pipeline_agent_writes_code(self, workspace, anthropic_client):
        """Pipeline node uses agent loop to write actual code to disk."""
        g = parse_dot(f"""
        digraph WriteCode {{
            graph [goal="Write a Python function"]
            start [shape=Mdiamond]
            code [
                shape=box,
                prompt="Write a function called add that takes two numbers and returns their sum. Save it to {workspace}/add.py"
            ]
            done [shape=Msquare]
            start -> code -> done
        }}
        """)

        backend = AgentLoopBackend(
            anthropic_client,
            default_model="claude-sonnet-4-5",
            default_provider="anthropic",
        )

        registry = HandlerRegistry()
        register_default_handlers(registry, codergen_backend=backend)

        async with anthropic_client:
            result = await run_pipeline(g, registry)

        assert result.status == PipelineStatus.COMPLETED
        # Verify actual file was created by the agent
        add_file = workspace / "add.py"
        assert add_file.exists(), "Pipeline agent should have created add.py"
        content = add_file.read_text()
        assert "def add" in content, "File should contain add function"


# ================================================================== #
# TEST 3: Multi-stage pipeline where outputs chain
# ================================================================== #


class TestMultiStagePipeline:
    """Pipeline where stage 2 uses stage 1's output."""

    @pytest.mark.asyncio
    async def test_two_stage_chained_output(self, anthropic_client):
        """Stage 1 plans, stage 2 implements based on the plan."""
        g = parse_dot("""
        digraph Chained {
            graph [goal="Build a temperature converter"]
            start [shape=Mdiamond]
            plan [
                shape=box,
                prompt="Create a brief 3-bullet plan for: $goal. Be very concise."
            ]
            implement [
                shape=box,
                prompt="Based on the plan, write Python code for: $goal. Output only the code."
            ]
            done [shape=Msquare]
            start -> plan -> implement -> done
        }
        """)

        backend = DirectLLMBackend(
            anthropic_client,
            default_model="claude-sonnet-4-5",
            default_provider="anthropic",
        )
        registry = HandlerRegistry()
        register_default_handlers(registry, codergen_backend=backend)

        async with anthropic_client:
            result = await run_pipeline(g, registry)

        assert result.status == PipelineStatus.COMPLETED

        # Verify both stages produced output
        plan_output = result.context.get("codergen.plan.output", "")
        impl_output = result.context.get("codergen.implement.output", "")
        assert len(plan_output) > 20, "Plan should have content"
        assert len(impl_output) > 20, "Implementation should have content"
        assert "def" in impl_output or "def" in impl_output.lower(), (
            "Implementation should contain a function"
        )


# ================================================================== #
# TEST 4: generate() high-level API with real tools
# ================================================================== #


class TestGenerateAPIReal:
    """High-level generate() API against real provider."""

    @pytest.mark.asyncio
    async def test_generate_with_real_tools(self, workspace, anthropic_client):
        """generate() drives a real tool loop."""
        # Create a file for the LLM to read
        data_file = workspace / "data.txt"
        data_file.write_text("The secret code is: XRAY42")

        from attractor_llm.types import Tool

        # Wrap read_file as a Tool compatible with generate()
        async def read_execute(path: str, **kwargs: Any) -> str:
            return Path(path).read_text()

        read_tool = Tool(
            name="read_file",
            description="Read a file. Args: path (string).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
            execute=read_execute,
        )

        async with anthropic_client:
            result = await generate(
                anthropic_client,
                "claude-sonnet-4-5",
                f"Read the file at {data_file} and tell me what the secret code is.",
                tools=[read_tool],
                provider="anthropic",
            )

        assert "XRAY42" in result, f"Agent should have found the secret code. Got: {result[:200]}"

    @pytest.mark.asyncio
    async def test_generate_object_real(self, anthropic_client):
        """generate_object() returns structured data from real LLM."""
        async with anthropic_client:
            obj = await generate_object(
                anthropic_client,
                "claude-sonnet-4-5",
                "Extract: 'The Eiffel Tower is 330 meters tall and located in Paris.'",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "height_meters": {"type": "integer"},
                        "city": {"type": "string"},
                    },
                },
                provider="anthropic",
            )

        assert obj["name"] == "Eiffel Tower" or "eiffel" in obj.get("name", "").lower()
        assert obj["height_meters"] == 330
        assert obj["city"] == "Paris"


# ================================================================== #
# TEST 5: Subagent with real delegation
# ================================================================== #


class TestSubagentReal:
    """Subagent spawning with real LLM calls."""

    @pytest.mark.asyncio
    async def test_subagent_completes_task(self, anthropic_client):
        """Subagent handles a delegated coding question."""
        result = await spawn_subagent(
            client=anthropic_client,
            prompt=(
                "Write a Python one-liner that reverses a string. "
                "Just output the code, nothing else."
            ),
            parent_depth=0,
            max_depth=3,
            model="claude-sonnet-4-5",
            provider="anthropic",
            include_tools=False,
        )

        assert result.depth == 1
        assert len(result.text) > 5
        # Should contain some form of string reversal
        assert "[::-1]" in result.text or "reverse" in result.text.lower()

    @pytest.mark.asyncio
    async def test_subagent_with_tools(self, workspace, anthropic_client):
        """Subagent uses tools to write a file."""
        result = await spawn_subagent(
            client=anthropic_client,
            prompt=(
                f"Write a file called 'answer.txt' in {workspace} containing just the number 42."
            ),
            parent_depth=0,
            max_depth=3,
            model="claude-sonnet-4-5",
            provider="anthropic",
            include_tools=True,
        )

        assert result.depth == 1
        answer_file = workspace / "answer.txt"
        assert answer_file.exists(), "Subagent should have created answer.txt"
        content = answer_file.read_text().strip()
        assert "42" in content


# ================================================================== #
# TEST 6: Full software factory simulation
# ================================================================== #


class TestSoftwareFactory:
    """End-to-end: pipeline that plans, implements, and reviews code."""

    @pytest.mark.asyncio
    async def test_plan_implement_pipeline(self, anthropic_client):
        """3-stage pipeline: plan -> implement -> done (with real LLM)."""
        g = parse_dot("""
        digraph Factory {
            graph [goal="Write a Python function that checks if a string is a palindrome"]
            start [shape=Mdiamond]

            plan [
                shape=box,
                prompt="Create a 2-bullet plan for: $goal. Be concise."
            ]

            implement [
                shape=box,
                prompt="Write the Python code for: $goal. Include the function and test assertions. Code only."
            ]

            done [shape=Msquare]
            start -> plan -> implement -> done
        }
        """)

        backend = DirectLLMBackend(
            anthropic_client,
            default_model="claude-sonnet-4-5",
            default_provider="anthropic",
        )
        registry = HandlerRegistry()
        register_default_handlers(registry, codergen_backend=backend)

        async with anthropic_client:
            result = await run_pipeline(g, registry)

        assert result.status == PipelineStatus.COMPLETED

        plan = result.context.get("codergen.plan.output", "")
        code = result.context.get("codergen.implement.output", "")

        assert len(plan) > 10, "Plan should have content"
        assert "def" in code, "Implementation should contain a function"
        assert "palindrome" in code.lower() or "[::-1]" in code, "Code should be about palindromes"
