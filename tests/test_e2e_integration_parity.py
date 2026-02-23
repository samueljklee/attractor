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
        assert "def greet" in content
        assert "Hello from Attractor" in content


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
        assert "def greet" in content
        assert "Hello from Attractor" in content


