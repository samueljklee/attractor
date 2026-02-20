"""Shared fixtures and markers for live API tests.

Centralises API-key detection, skip markers, model constants, and
per-provider Client fixtures so that every live-test file can import
them from one place instead of re-declaring ~80 lines of boilerplate.
"""

from __future__ import annotations

import os

import pytest

from attractor_llm import Client, ProviderConfig, RetryPolicy
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.gemini import GeminiAdapter
from attractor_llm.adapters.openai import OpenAIAdapter

# ------------------------------------------------------------------ #
# API key detection
# ------------------------------------------------------------------ #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

HAS_OPENAI = bool(OPENAI_KEY)
HAS_ANTHROPIC = bool(ANTHROPIC_KEY)
HAS_GEMINI = bool(GEMINI_KEY)

# ------------------------------------------------------------------ #
# Skip markers
# ------------------------------------------------------------------ #

skip_no_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GOOGLE_API_KEY not set")

# ------------------------------------------------------------------ #
# Model constants
# ------------------------------------------------------------------ #

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.0-flash"

OPENAI_REASONING_MODEL = "o4-mini"
ANTHROPIC_REASONING_MODEL = "claude-sonnet-4-5"  # same as base (extended-thinking mode)
GEMINI_REASONING_MODEL = "gemini-2.5-flash"

# ------------------------------------------------------------------ #
# Client fixtures
# ------------------------------------------------------------------ #

_TIMEOUT = 60.0
_RETRY = RetryPolicy(max_retries=1)


@pytest.fixture
def openai_client() -> Client:
    """Client with only OpenAI registered."""
    c = Client(retry_policy=_RETRY)
    c.register_adapter(
        "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=_TIMEOUT))
    )
    return c


@pytest.fixture
def anthropic_client() -> Client:
    """Client with only Anthropic registered."""
    c = Client(retry_policy=_RETRY)
    c.register_adapter(
        "anthropic", AnthropicAdapter(ProviderConfig(api_key=ANTHROPIC_KEY, timeout=_TIMEOUT))
    )
    return c


@pytest.fixture
def gemini_client() -> Client:
    """Client with only Gemini registered."""
    c = Client(retry_policy=_RETRY)
    c.register_adapter(
        "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=_TIMEOUT))
    )
    return c
