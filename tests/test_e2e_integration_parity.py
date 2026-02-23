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


