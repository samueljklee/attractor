"""Live parity tests — OpenAI and Gemini coverage for §9.12 scenarios.

Requires API keys. Each test is individually guarded by the appropriate
skip marker so tests run whenever the key is available, independent of
other providers.

Run all:   uv run python -m pytest tests/test_e2e_integration_parity.py -v -x
OpenAI:    uv run python -m pytest tests/test_e2e_integration_parity.py -k "OpenAI" -v
Gemini:    uv run python -m pytest tests/test_e2e_integration_parity.py -k "Gemini" -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from attractor_agent.session import Session, SessionConfig
from attractor_agent.subagent import spawn_subagent
from attractor_agent.tools.core import ALL_CORE_TOOLS, set_allowed_roots

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-2.0-flash"
ANTHROPIC_MODEL = "claude-sonnet-4-5"

skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not GEMINI_KEY, reason="GOOGLE_API_KEY/GEMINI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace and confine tools to it."""
    set_allowed_roots([str(tmp_path)])
    yield tmp_path
    set_allowed_roots([os.getcwd()])


@pytest.fixture
def openai_client():
    """Client with OpenAI adapter."""
    from attractor_llm import ProviderConfig
    from attractor_llm.adapters.openai import OpenAIAdapter
    from attractor_llm.client import Client

    c = Client()
    c.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=120.0)))
    return c


@pytest.fixture
def gemini_client():
    """Client with Gemini adapter."""
    from attractor_llm import ProviderConfig
    from attractor_llm.adapters.gemini import GeminiAdapter
    from attractor_llm.client import Client

    c = Client()
    c.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=120.0)))
    return c


@pytest.fixture
def anthropic_client():
    """Client with Anthropic adapter."""
    from attractor_llm import ProviderConfig
    from attractor_llm.adapters.anthropic import AnthropicAdapter
    from attractor_llm.client import Client

    c = Client()
    c.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=120.0))
    )
    return c


def _get_profile_and_tools(provider: str):
    """Return (profile, tools) for the given provider."""
    from attractor_agent.profiles import get_profile

    profile = get_profile(provider)
    tools = profile.get_tools(list(ALL_CORE_TOOLS))
    return profile, tools


# ================================================================== #
# Task 11: File creation — §9.12.1 (OpenAI), §9.12.3 (Gemini)
# ================================================================== #


class TestOpenAIFileCreation:
    """§9.12.1: OpenAI agent writes and reads back a file."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_agent_writes_and_reads_file(self, workspace, openai_client):
        """Agent creates hello.py via write_file, then reads it back."""
        profile, tools = _get_profile_and_tools("openai")
        config = SessionConfig(model=OPENAI_MODEL, provider="openai", max_turns=10)
        config = profile.apply_to_config(config)

        async with openai_client:
            session = Session(client=openai_client, config=config, tools=tools)
            await session.submit(
                f"Write a file called 'hello.py' in {workspace} with a "
                f"function called greet() that returns the string "
                f"'Hello from Attractor'. Then read the file back and "
                f"tell me what it contains."
            )

        hello_file = workspace / "hello.py"
        assert hello_file.exists(), "Agent should have created hello.py"
        content = hello_file.read_text()
        assert "def greet" in content, f"File should define greet(). Got:\n{content[:300]}"
        assert "Hello from Attractor" in content, (
            f"File should contain greeting. Got:\n{content[:300]}"
        )


class TestGeminiFileCreation:
    """§9.12.3: Gemini agent writes and reads back a file."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_agent_writes_and_reads_file(self, workspace, gemini_client):
        """Agent creates hello.py via write_file, then reads it back."""
        profile, tools = _get_profile_and_tools("gemini")
        config = SessionConfig(model=GEMINI_MODEL, provider="gemini", max_turns=10)
        config = profile.apply_to_config(config)

        async with gemini_client:
            session = Session(client=gemini_client, config=config, tools=tools)
            await session.submit(
                f"Write a file called 'hello.py' in {workspace} with a "
                f"function called greet() that returns the string "
                f"'Hello from Attractor'. Then read the file back and "
                f"tell me what it contains."
            )

        hello_file = workspace / "hello.py"
        assert hello_file.exists(), "Agent should have created hello.py"
        content = hello_file.read_text()
        assert "def greet" in content, f"File should define greet(). Got:\n{content[:300]}"
        assert "Hello from Attractor" in content, (
            f"File should contain greeting. Got:\n{content[:300]}"
        )


# ================================================================== #
# Task 12: Read + edit — §9.12.4 (OpenAI), §9.12.6 (Gemini)
# ================================================================== #


class TestOpenAIReadAndEdit:
    """§9.12.4: OpenAI agent reads existing file and edits it."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_agent_edits_existing_file(self, workspace, openai_client):
        """Agent uses edit_file to modify a pre-seeded file."""
        target = workspace / "config.py"
        target.write_text('DB_HOST = "localhost"\nDB_PORT = 5432\nDB_NAME = "mydb"\n')

        profile, tools = _get_profile_and_tools("openai")
        config = SessionConfig(model=OPENAI_MODEL, provider="openai", max_turns=10)
        config = profile.apply_to_config(config)

        async with openai_client:
            session = Session(client=openai_client, config=config, tools=tools)
            await session.submit(
                f"Read the file {target} and change the DB_PORT from "
                f"5432 to 3306. Use edit_file, not write_file."
            )

        content = target.read_text()
        assert "3306" in content, f"Port should be changed to 3306. Got:\n{content}"
        assert "5432" not in content, f"Old port should be gone. Got:\n{content}"
        assert "DB_HOST" in content, f"DB_HOST should be preserved. Got:\n{content}"
        assert "DB_NAME" in content, f"DB_NAME should be preserved. Got:\n{content}"


class TestGeminiReadAndEdit:
    """§9.12.6: Gemini agent reads existing file and edits it."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_agent_edits_existing_file(self, workspace, gemini_client):
        """Agent uses edit_file to modify a pre-seeded file."""
        target = workspace / "config.py"
        target.write_text('DB_HOST = "localhost"\nDB_PORT = 5432\nDB_NAME = "mydb"\n')

        profile, tools = _get_profile_and_tools("gemini")
        config = SessionConfig(model=GEMINI_MODEL, provider="gemini", max_turns=10)
        config = profile.apply_to_config(config)

        async with gemini_client:
            session = Session(client=gemini_client, config=config, tools=tools)
            await session.submit(
                f"Read the file {target} and change the DB_PORT from "
                f"5432 to 3306. Use edit_file, not write_file."
            )

        content = target.read_text()
        assert "3306" in content, f"Port should be changed to 3306. Got:\n{content}"
        assert "5432" not in content, f"Old port should be gone. Got:\n{content}"
        assert "DB_HOST" in content, f"DB_HOST should be preserved. Got:\n{content}"
        assert "DB_NAME" in content, f"DB_NAME should be preserved. Got:\n{content}"


# ================================================================== #
# Task 13: Shell execution — §9.12.13-15 (OpenAI + Gemini)
# ================================================================== #


class TestOpenAIShellExecution:
    """§9.12.13: OpenAI agent executes shell commands."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_agent_runs_shell_command(self, workspace, openai_client):
        """Agent uses shell tool to run a command and reports the output."""
        profile, tools = _get_profile_and_tools("openai")
        config = SessionConfig(model=OPENAI_MODEL, provider="openai", max_turns=5)
        config = profile.apply_to_config(config)

        async with openai_client:
            session = Session(client=openai_client, config=config, tools=tools)
            result = await session.submit(
                f"Use the shell tool to run 'echo SHELL_OK' in {workspace}. "
                f"Tell me the exact output you got."
            )

        assert "SHELL_OK" in result, f"Agent must report shell output. Got: {result[:200]}"


class TestGeminiShellExecution:
    """§9.12.15: Gemini agent executes shell commands."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_agent_runs_shell_command(self, workspace, gemini_client):
        """Agent uses shell tool to run a command and reports the output."""
        profile, tools = _get_profile_and_tools("gemini")
        config = SessionConfig(model=GEMINI_MODEL, provider="gemini", max_turns=5)
        config = profile.apply_to_config(config)

        async with gemini_client:
            session = Session(client=gemini_client, config=config, tools=tools)
            result = await session.submit(
                f"Use the shell tool to run 'echo SHELL_OK' in {workspace}. "
                f"Tell me the exact output you got."
            )

        assert "SHELL_OK" in result, f"Agent must report shell output. Got: {result[:200]}"


# ================================================================== #
# Task 14: Parallel tool calls — §9.12.25-27
# ================================================================== #


class TestParallelToolCallsOpenAI:
    """§9.12.25-26: OpenAI issues parallel tool calls when appropriate."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_parallel_file_reads(self, workspace, openai_client):
        """Agent reads two files in parallel (both tool calls in single turn)."""
        (workspace / "file_a.txt").write_text("Content of file A")
        (workspace / "file_b.txt").write_text("Content of file B")

        profile, tools = _get_profile_and_tools("openai")
        config = SessionConfig(model=OPENAI_MODEL, provider="openai", max_turns=5)
        config = profile.apply_to_config(config)

        async with openai_client:
            session = Session(client=openai_client, config=config, tools=tools)
            result = await session.submit(
                f"Read BOTH files {workspace}/file_a.txt AND {workspace}/file_b.txt "
                f"and tell me the content of each."
            )

        assert "Content of file A" in result, (
            f"Agent must report file A content. Got: {result[:300]}"
        )
        assert "Content of file B" in result, (
            f"Agent must report file B content. Got: {result[:300]}"
        )


class TestParallelToolCallsGemini:
    """§9.12.27: Gemini issues parallel tool calls when appropriate."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_parallel_file_reads(self, workspace, gemini_client):
        """Agent reads two files in parallel."""
        (workspace / "file_a.txt").write_text("Content of file A")
        (workspace / "file_b.txt").write_text("Content of file B")

        profile, tools = _get_profile_and_tools("gemini")
        config = SessionConfig(model=GEMINI_MODEL, provider="gemini", max_turns=5)
        config = profile.apply_to_config(config)

        async with gemini_client:
            session = Session(client=gemini_client, config=config, tools=tools)
            result = await session.submit(
                f"Read BOTH files {workspace}/file_a.txt AND {workspace}/file_b.txt "
                f"and tell me the content of each."
            )

        assert "Content of file A" in result, (
            f"Agent must report file A content. Got: {result[:300]}"
        )
        assert "Content of file B" in result, (
            f"Agent must report file B content. Got: {result[:300]}"
        )


# ================================================================== #
# Task 16: Subagent spawn — §9.12.34 (OpenAI), §9.12.36 (Gemini)
# ================================================================== #


class TestSubagentOpenAI:
    """§9.12.34: OpenAI subagent spawning."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_subagent_completes_task(self, openai_client):
        """Subagent handles a delegated coding question (no tools)."""
        async with openai_client:
            result = await spawn_subagent(
                client=openai_client,
                prompt=(
                    "Write a Python one-liner that reverses a string. "
                    "Just output the code, nothing else."
                ),
                parent_depth=0,
                max_depth=3,
                model=OPENAI_MODEL,
                provider="openai",
                include_tools=False,
            )
        assert result.depth == 1, f"Expected depth=1, got {result.depth}"
        assert len(result.text) > 5, f"Expected non-trivial response, got: {result.text!r}"
        assert "[::-1]" in result.text or "reverse" in result.text.lower(), (
            f"Expected reversal one-liner, got: {result.text[:200]}"
        )

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_subagent_with_tools(self, workspace, openai_client):
        """Subagent uses write_file to create a file."""
        async with openai_client:
            result = await spawn_subagent(
                client=openai_client,
                prompt=(
                    f"Write a file called 'answer.txt' in {workspace} "
                    f"containing just the number 42."
                ),
                parent_depth=0,
                max_depth=3,
                model=OPENAI_MODEL,
                provider="openai",
                include_tools=True,
            )
        assert result.depth == 1, f"Expected depth=1, got {result.depth}"
        answer_file = workspace / "answer.txt"
        assert answer_file.exists(), "Subagent should have created answer.txt"
        assert "42" in answer_file.read_text().strip(), (
            f"Expected '42' in answer.txt, got: {answer_file.read_text()!r}"
        )


class TestSubagentGemini:
    """§9.12.36: Gemini subagent spawning."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_subagent_completes_task(self, gemini_client):
        """Subagent handles a delegated coding question (no tools)."""
        async with gemini_client:
            result = await spawn_subagent(
                client=gemini_client,
                prompt=(
                    "Write a Python one-liner that reverses a string. "
                    "Just output the code, nothing else."
                ),
                parent_depth=0,
                max_depth=3,
                model=GEMINI_MODEL,
                provider="gemini",
                include_tools=False,
            )
        assert result.depth == 1, f"Expected depth=1, got {result.depth}"
        assert len(result.text) > 5, f"Expected non-trivial response, got: {result.text!r}"
        assert "[::-1]" in result.text or "reverse" in result.text.lower(), (
            f"Expected reversal one-liner, got: {result.text[:200]}"
        )

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_subagent_with_tools(self, workspace, gemini_client):
        """Subagent uses write_file to create a file."""
        async with gemini_client:
            result = await spawn_subagent(
                client=gemini_client,
                prompt=(
                    f"Write a file called 'answer.txt' in {workspace} "
                    f"containing just the number 42."
                ),
                parent_depth=0,
                max_depth=3,
                model=GEMINI_MODEL,
                provider="gemini",
                include_tools=True,
            )
        assert result.depth == 1, f"Expected depth=1, got {result.depth}"
        answer_file = workspace / "answer.txt"
        assert answer_file.exists(), "Subagent should have created answer.txt"
        assert "42" in answer_file.read_text().strip(), (
            f"Expected '42' in answer.txt, got: {answer_file.read_text()!r}"
        )


# ================================================================== #
# Task 17: Multi-file edit — §9.12.7-9
# ================================================================== #


def _seed_multi_file_edit(workspace: Path) -> None:
    """Seed two files sharing a constant name."""
    (workspace / "module_a.py").write_text('OLD_VALUE = "alpha"\n')
    (workspace / "module_b.py").write_text(
        'from module_a import OLD_VALUE\nresult = OLD_VALUE + "_b"\n'
    )


def _assert_multi_file_edit(workspace: Path) -> None:
    """Assert both files were updated."""
    a_content = (workspace / "module_a.py").read_text()
    b_content = (workspace / "module_b.py").read_text()
    assert "NEW_VALUE" in a_content, f"module_a.py must contain NEW_VALUE. Got:\n{a_content}"
    assert "OLD_VALUE" not in a_content, (
        f"OLD_VALUE must be gone from module_a.py. Got:\n{a_content}"
    )
    assert "NEW_VALUE" in b_content, f"module_b.py must contain NEW_VALUE. Got:\n{b_content}"


class TestMultiFileEditAnthropic:
    """§9.12.7: Anthropic coordinated multi-file edit."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_coordinated_edit_across_two_files(self, workspace, anthropic_client):
        _seed_multi_file_edit(workspace)
        profile, tools = _get_profile_and_tools("anthropic")
        config = SessionConfig(model=ANTHROPIC_MODEL, provider="anthropic", max_turns=10)
        config = profile.apply_to_config(config)
        async with anthropic_client:
            session = Session(client=anthropic_client, config=config, tools=tools)
            await session.submit(
                f"In BOTH {workspace}/module_a.py AND {workspace}/module_b.py, "
                f"rename the constant OLD_VALUE to NEW_VALUE. Edit both files."
            )
        _assert_multi_file_edit(workspace)


class TestMultiFileEditOpenAI:
    """§9.12.8: OpenAI coordinated multi-file edit."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_coordinated_edit_across_two_files(self, workspace, openai_client):
        _seed_multi_file_edit(workspace)
        profile, tools = _get_profile_and_tools("openai")
        config = SessionConfig(model=OPENAI_MODEL, provider="openai", max_turns=10)
        config = profile.apply_to_config(config)
        async with openai_client:
            session = Session(client=openai_client, config=config, tools=tools)
            await session.submit(
                f"In BOTH {workspace}/module_a.py AND {workspace}/module_b.py, "
                f"rename the constant OLD_VALUE to NEW_VALUE. Edit both files."
            )
        _assert_multi_file_edit(workspace)


class TestMultiFileEditGemini:
    """§9.12.9: Gemini coordinated multi-file edit."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_coordinated_edit_across_two_files(self, workspace, gemini_client):
        _seed_multi_file_edit(workspace)
        profile, tools = _get_profile_and_tools("gemini")
        config = SessionConfig(model=GEMINI_MODEL, provider="gemini", max_turns=10)
        config = profile.apply_to_config(config)
        async with gemini_client:
            session = Session(client=gemini_client, config=config, tools=tools)
            await session.submit(
                f"In BOTH {workspace}/module_a.py AND {workspace}/module_b.py, "
                f"rename the constant OLD_VALUE to NEW_VALUE. Edit both files."
            )
        _assert_multi_file_edit(workspace)


# ================================================================== #
# Task 18: Grep + Glob — §9.12.16-18
# ================================================================== #


def _seed_grep_glob(workspace: Path) -> None:
    """Seed files for grep/glob tests."""
    (workspace / "alpha.py").write_text('SECRET_TOKEN = "abc123"\n')
    (workspace / "beta.py").write_text('SECRET_TOKEN = "def456"\nOTHER = 1\n')
    (workspace / "gamma.txt").write_text("not python\n")


async def _run_grep_glob(workspace: Path, client: Any, model: str, provider: str) -> str:
    _seed_grep_glob(workspace)
    profile, tools = _get_profile_and_tools(provider)
    config = SessionConfig(model=model, provider=provider, max_turns=5)
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config, tools=tools)
        result = await session.submit(
            f"In directory {workspace}: "
            f"(1) Use glob to find all .py files. "
            f"(2) Use grep to find which .py files contain 'SECRET_TOKEN'. "
            f"Tell me the filenames that contain SECRET_TOKEN."
        )
    return result


class TestGrepGlobAnthropic:
    """§9.12.16: Anthropic grep and glob search."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_grep_and_glob(self, workspace, anthropic_client):
        result = await _run_grep_glob(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")
        assert "alpha" in result.lower() or "alpha.py" in result, (
            f"Expected alpha.py in result. Got:\n{result[:300]}"
        )
        assert "beta" in result.lower() or "beta.py" in result, (
            f"Expected beta.py in result. Got:\n{result[:300]}"
        )


class TestGrepGlobOpenAI:
    """§9.12.17: OpenAI grep and glob search."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_grep_and_glob(self, workspace, openai_client):
        result = await _run_grep_glob(workspace, openai_client, OPENAI_MODEL, "openai")
        assert "alpha" in result.lower() or "alpha.py" in result, (
            f"Expected alpha.py in result. Got:\n{result[:300]}"
        )
        assert "beta" in result.lower() or "beta.py" in result, (
            f"Expected beta.py in result. Got:\n{result[:300]}"
        )


class TestGrepGlobGemini:
    """§9.12.18: Gemini grep and glob search."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_grep_and_glob(self, workspace, gemini_client):
        result = await _run_grep_glob(workspace, gemini_client, GEMINI_MODEL, "gemini")
        assert "alpha" in result.lower() or "alpha.py" in result, (
            f"Expected alpha.py in result. Got:\n{result[:300]}"
        )
        assert "beta" in result.lower() or "beta.py" in result, (
            f"Expected beta.py in result. Got:\n{result[:300]}"
        )


# ================================================================== #
# Task 19: Multi-step read→analyze→edit — §9.12.19-21
# ================================================================== #


async def _run_read_analyze_edit(workspace: Path, client: Any, model: str, provider: str) -> None:
    target = workspace / "scores.py"
    target.write_text(
        "PASSING_SCORE = 60\n"
        "FAILING_SCORE = 40\n"
        "# scores above PASSING_SCORE are considered passing\n"
    )
    profile, tools = _get_profile_and_tools(provider)
    config = SessionConfig(model=model, provider=provider, max_turns=8)
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config, tools=tools)
        await session.submit(
            f"Read {target}. "
            f"If the PASSING_SCORE is below 70, raise it to 70. "
            f"Use edit_file to make the change."
        )
    content = target.read_text()
    assert "70" in content, f"PASSING_SCORE should be updated to 70. Got:\n{content}"
    assert "60" not in content, f"Old value 60 should be replaced. Got:\n{content}"


class TestReadAnalyzeEditAnthropic:
    """§9.12.19: Anthropic multi-step read→analyze→edit."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_read_analyze_edit(self, workspace, anthropic_client):
        await _run_read_analyze_edit(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestReadAnalyzeEditOpenAI:
    """§9.12.20: OpenAI multi-step read→analyze→edit."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_read_analyze_edit(self, workspace, openai_client):
        await _run_read_analyze_edit(workspace, openai_client, OPENAI_MODEL, "openai")


class TestReadAnalyzeEditGemini:
    """§9.12.21: Gemini multi-step read→analyze→edit."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_read_analyze_edit(self, workspace, gemini_client):
        await _run_read_analyze_edit(workspace, gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 20: Tool output truncation — §9.12.22-24
# ================================================================== #


async def _run_truncation_test(workspace: Path, client: Any, model: str, provider: str) -> None:
    large_file = workspace / "large.txt"
    large_file.write_text("\n".join(f"Line {i}: {'x' * 100}" for i in range(5000)))
    config = SessionConfig(
        model=model,
        provider=provider,
        max_turns=5,
        tool_output_limits={"read_file": 500},
    )
    profile, tools = _get_profile_and_tools(provider)
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config, tools=tools)
        result = await session.submit(
            f"Read the file {large_file} and tell me what the first line says. "
            f"Note: the file may be truncated in your view."
        )
    assert result is not None, "Session must return a result on truncated output"
    assert len(result) > 0, "Result must be non-empty"
    assert "[Error:" not in result, (
        f"Agent should handle truncation gracefully. Got: {result[:200]}"
    )


class TestTruncationAnthropic:
    """§9.12.22: Anthropic tool output truncation."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_tool_output_truncation(self, workspace, anthropic_client):
        await _run_truncation_test(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestTruncationOpenAI:
    """§9.12.23: OpenAI tool output truncation."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_tool_output_truncation(self, workspace, openai_client):
        await _run_truncation_test(workspace, openai_client, OPENAI_MODEL, "openai")


class TestTruncationGemini:
    """§9.12.24: Gemini tool output truncation."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_tool_output_truncation(self, workspace, gemini_client):
        await _run_truncation_test(workspace, gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 21: Steering mid-task — §9.12.28-30
# ================================================================== #


async def _run_steering_test(workspace: Path, client: Any, model: str, provider: str) -> None:
    (workspace / "target.txt").write_text("original content\n")
    profile, tools = _get_profile_and_tools(provider)
    config = SessionConfig(model=model, provider=provider, max_turns=10)
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config, tools=tools)
        await session.submit(f"Read the file {workspace}/target.txt and summarize it.")
        session.steer("Actually, instead of summarizing, overwrite the file with 'STEERED CONTENT'")
        result = await session.submit("Please proceed with the updated instruction.")
    content = (workspace / "target.txt").read_text()
    assert "STEERED" in content or "steered" in content.lower(), (
        f"Agent should incorporate steering. File: {content!r}, result: {result[:100]}"
    )


class TestSteeringAnthropic:
    """§9.12.28: Anthropic steering mid-task."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_steering_mid_task(self, workspace, anthropic_client):
        await _run_steering_test(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestSteeringOpenAI:
    """§9.12.29: OpenAI steering mid-task."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_steering_mid_task(self, workspace, openai_client):
        await _run_steering_test(workspace, openai_client, OPENAI_MODEL, "openai")


class TestSteeringGemini:
    """§9.12.30: Gemini steering mid-task."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_steering_mid_task(self, workspace, gemini_client):
        await _run_steering_test(workspace, gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 22: Reasoning effort change — §9.12.31-33
# ================================================================== #


async def _run_reasoning_effort_test(client: Any, model: str, provider: str) -> None:
    config = SessionConfig(model=model, provider=provider, max_turns=3, reasoning_effort=None)
    profile = _get_profile_and_tools(provider)[0]
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config)
        await session.submit("What is 2+2?")
        session._config.reasoning_effort = "low"
        result = await session.submit("What is 3+3?")
    assert result is not None, "Session must return result after reasoning_effort change"
    assert len(result) > 0, f"Result must be non-empty. Got: {result!r}"


class TestReasoningEffortAnthropic:
    """§9.12.31: Anthropic reasoning effort change mid-session."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_reasoning_effort_change(self, anthropic_client):
        await _run_reasoning_effort_test(anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestReasoningEffortOpenAI:
    """§9.12.32: OpenAI reasoning effort change mid-session."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_reasoning_effort_change(self, openai_client):
        await _run_reasoning_effort_test(openai_client, OPENAI_MODEL, "openai")


class TestReasoningEffortGemini:
    """§9.12.33: Gemini reasoning effort change mid-session."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_reasoning_effort_change(self, gemini_client):
        await _run_reasoning_effort_test(gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 23: Loop detection — §9.12.37-39
# ================================================================== #


async def _run_loop_detection_test(workspace: Path, client: Any, model: str, provider: str) -> None:
    from attractor_agent.events import EventKind, SessionEvent

    (workspace / "stuck.txt").write_text("loop bait\n")
    profile, tools = _get_profile_and_tools(provider)
    config = SessionConfig(
        model=model,
        provider=provider,
        max_turns=20,
        loop_detection_window=4,
        loop_detection_threshold=3,
    )
    config = profile.apply_to_config(config)
    events: list[EventKind] = []

    async with client:
        session = Session(client=client, config=config, tools=tools)

        async def capture(e: SessionEvent) -> None:
            events.append(e.kind)

        session._emitter.on(capture)
        result = await session.submit(
            f"Read the file {workspace}/stuck.txt over and over, at least 10 times, "
            f"and tell me the content each time. Do nothing else."
        )

    assert EventKind.LOOP_DETECTION in events, (
        f"Expected LOOP_DETECTION event on repetitive tool use "
        f"(window=4, threshold=3). Got events: {events}"
    )
    assert result is not None, "Session must return a result"


class TestLoopDetectionAnthropic:
    """§9.12.37: Anthropic loop detection."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_loop_detection(self, workspace, anthropic_client):
        await _run_loop_detection_test(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestLoopDetectionOpenAI:
    """§9.12.38: OpenAI loop detection."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_loop_detection(self, workspace, openai_client):
        await _run_loop_detection_test(workspace, openai_client, OPENAI_MODEL, "openai")


class TestLoopDetectionGemini:
    """§9.12.39: Gemini loop detection."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_loop_detection(self, workspace, gemini_client):
        await _run_loop_detection_test(workspace, gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 24: Error recovery — §9.12.40-42
# ================================================================== #


async def _run_error_recovery_test(client: Any, model: str, provider: str) -> None:
    call_count = [0]

    async def flaky_tool(**kwargs: Any) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Simulated tool failure on first call")
        return "Tool succeeded on retry"

    from attractor_llm.types import Tool

    flaky = Tool(
        name="flaky_tool",
        description="A tool that fails on first call. Args: message (string).",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        execute=flaky_tool,
    )
    config = SessionConfig(model=model, provider=provider, max_turns=5)
    profile = _get_profile_and_tools(provider)[0]
    config = profile.apply_to_config(config)
    async with client:
        session = Session(client=client, config=config, tools=[flaky])
        result = await session.submit(
            "Call the flaky_tool with message='test'. If it fails, try calling it again."
        )
    assert result is not None, "Session must return result after tool error"
    assert len(result) > 0, f"Result must be non-empty. Got: {result!r}"
    assert "Authentication" not in result, "Should not see auth errors during error recovery"


class TestErrorRecoveryAnthropic:
    """§9.12.40: Anthropic error recovery."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_error_recovery(self, anthropic_client):
        await _run_error_recovery_test(anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestErrorRecoveryOpenAI:
    """§9.12.41: OpenAI error recovery."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_error_recovery(self, openai_client):
        await _run_error_recovery_test(openai_client, OPENAI_MODEL, "openai")


class TestErrorRecoveryGemini:
    """§9.12.42: Gemini error recovery."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_error_recovery(self, gemini_client):
        await _run_error_recovery_test(gemini_client, GEMINI_MODEL, "gemini")


# ================================================================== #
# Task 25: Provider-specific tool format validation — §9.12.43-45
# ================================================================== #


async def _run_format_validation_test(
    workspace: Path, client: Any, model: str, provider: str
) -> None:
    from attractor_agent.events import EventKind, SessionEvent

    profile, tools = _get_profile_and_tools(provider)
    config = SessionConfig(model=model, provider=provider, max_turns=5)
    config = profile.apply_to_config(config)
    captured_tool_names: list[str] = []

    async with client:
        session = Session(client=client, config=config, tools=tools)

        async def capture(e: SessionEvent) -> None:
            if e.kind == EventKind.TOOL_CALL and e.data:
                captured_tool_names.append(e.data.get("tool", ""))

        session._emitter.on(capture)
        await session.submit(
            f"Write a file called 'validate.txt' in {workspace} containing the word 'validated'."
        )

    assert any(name in ("write_file", "edit_file") for name in captured_tool_names), (
        f"Provider {provider} must accept and call write_file or edit_file. "
        f"Captured tools: {captured_tool_names}"
    )
    assert (workspace / "validate.txt").exists(), "write_file must have created validate.txt"


class TestToolFormatAnthropic:
    """§9.12.43: Anthropic provider-specific tool format validation."""

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_tool_schema_accepted(self, workspace, anthropic_client):
        """Anthropic receives canned edit_file description override."""
        await _run_format_validation_test(workspace, anthropic_client, ANTHROPIC_MODEL, "anthropic")


class TestToolFormatOpenAI:
    """§9.12.44: OpenAI provider-specific tool format validation."""

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_tool_schema_accepted(self, workspace, openai_client):
        """OpenAI receives function-calling JSON schema for write_file."""
        await _run_format_validation_test(workspace, openai_client, OPENAI_MODEL, "openai")


class TestToolFormatGemini:
    """§9.12.45: Gemini provider-specific tool format validation."""

    @skip_no_gemini
    @pytest.mark.asyncio
    async def test_tool_schema_accepted(self, workspace, gemini_client):
        """Gemini receives FunctionDeclaration format for write_file."""
        await _run_format_validation_test(workspace, gemini_client, GEMINI_MODEL, "gemini")
