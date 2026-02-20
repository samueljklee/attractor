"""Audit 2 Wave 5 - Pipeline Hardening regression tests.

Covers 3 items:
  Item 1  S11.5.3  - Pipeline retry preset wiring per-node
  Item 2  S11.5.4  - Jitter activates automatically for non-none presets (cascade)
  Item 3  S8.6.9   - Live cache-efficiency tests for OpenAI and Gemini
"""

from __future__ import annotations

import os

import pytest

from attractor_llm.retry import RetryPolicy
from attractor_pipeline.engine.runner import (
    _PIPELINE_RETRY,
    RETRY_PRESETS,
    _get_retry_policy,
)
from attractor_pipeline.graph import Node

# ================================================================== #
# Helpers
# ================================================================== #


def _node(node_id: str = "n1", **attrs: object) -> Node:
    """Build a minimal Node, optionally injecting attrs entries."""
    return Node(id=node_id, attrs=dict(attrs))


# ================================================================== #
# Item 1  S11.5.3 -- Per-node retry preset resolution
# ================================================================== #


class TestNodeRetryPresetResolution:
    """_get_retry_policy() resolves the correct policy for a node."""

    def test_node_retry_preset_from_attrs(self) -> None:
        """Node with retry_preset='standard' returns the standard RetryPolicy."""
        node = _node(retry_preset="standard")
        policy = _get_retry_policy(node)

        expected = RETRY_PRESETS["standard"]
        assert policy is expected, f"Expected RETRY_PRESETS['standard'] but got {policy!r}"
        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.backoff_factor == 2.0
        assert policy.max_delay == 30.0
        assert policy.jitter is True

    def test_node_retry_preset_aggressive(self) -> None:
        """Node with retry_preset='aggressive' returns the aggressive RetryPolicy."""
        node = _node(retry_preset="aggressive")
        policy = _get_retry_policy(node)

        assert policy is RETRY_PRESETS["aggressive"]
        assert policy.max_retries == 5
        assert policy.jitter is True

    def test_node_retry_preset_linear(self) -> None:
        """Node with retry_preset='linear' returns the linear RetryPolicy."""
        node = _node(retry_preset="linear")
        policy = _get_retry_policy(node)

        assert policy is RETRY_PRESETS["linear"]
        assert policy.jitter is False

    def test_node_retry_preset_patient(self) -> None:
        """Node with retry_preset='patient' returns the patient RetryPolicy."""
        node = _node(retry_preset="patient")
        policy = _get_retry_policy(node)

        assert policy is RETRY_PRESETS["patient"]
        assert policy.max_retries == 10

    def test_node_retry_preset_none_explicit(self) -> None:
        """Node with retry_preset='none' returns the none RetryPolicy (0 retries)."""
        node = _node(retry_preset="none")
        policy = _get_retry_policy(node)

        assert policy is RETRY_PRESETS["none"]
        assert policy.max_retries == 0

    def test_node_retry_preset_default_none(self) -> None:
        """Node without retry_preset falls back to the pipeline default ('none')."""
        node = _node()  # no retry_preset in attrs
        policy = _get_retry_policy(node)

        # Pipeline default is the 'none' preset
        assert policy is _PIPELINE_RETRY
        assert policy.max_retries == 0

    def test_node_retry_preset_unknown_falls_back(self) -> None:
        """Unknown preset name falls back gracefully to the pipeline default."""
        node = _node(retry_preset="does_not_exist")
        policy = _get_retry_policy(node)

        # Unknown name must NOT raise; must fall back to pipeline default
        assert policy is _PIPELINE_RETRY
        assert policy.max_retries == 0

    def test_node_retry_preset_empty_string_falls_back(self) -> None:
        """An empty string for retry_preset falls back to the pipeline default."""
        node = _node(retry_preset="")
        policy = _get_retry_policy(node)

        assert policy is _PIPELINE_RETRY


# ================================================================== #
# Item 2  S11.5.4 -- Jitter activates with non-none presets (cascade)
# ================================================================== #


class TestJitterActivationCascade:
    """Jitter activates automatically once a non-none preset is wired in."""

    def test_jitter_activates_with_standard_preset(self) -> None:
        """Standard preset has jitter=True; compute_delay varies across calls."""
        policy = RETRY_PRESETS["standard"]
        assert policy.jitter is True, "standard preset must enable jitter"

        # With jitter, repeated calls for the same attempt must not all
        # return the same value (with overwhelming probability over 50 draws).
        delays = [policy.compute_delay(0) for _ in range(50)]
        assert len(set(delays)) > 1, "All 50 jittered delays were identical -- jitter is not active"

    def test_jitter_activates_with_aggressive_preset(self) -> None:
        """Aggressive preset has jitter=True; delays vary."""
        policy = RETRY_PRESETS["aggressive"]
        assert policy.jitter is True

        delays = [policy.compute_delay(0) for _ in range(50)]
        assert len(set(delays)) > 1

    def test_jitter_activates_with_patient_preset(self) -> None:
        """Patient preset has jitter=True; delays vary."""
        policy = RETRY_PRESETS["patient"]
        assert policy.jitter is True

        delays = [policy.compute_delay(0) for _ in range(50)]
        assert len(set(delays)) > 1

    def test_no_jitter_with_none_preset(self) -> None:
        """'none' preset has jitter=False and returns exact 0.0 delay."""
        policy = RETRY_PRESETS["none"]
        assert policy.jitter is False
        assert policy.compute_delay(0) == 0.0

    def test_no_jitter_with_linear_preset(self) -> None:
        """'linear' preset has jitter=False; delays are deterministic."""
        policy = RETRY_PRESETS["linear"]
        assert policy.jitter is False

        # Same attempt always yields the same delay
        delays = {policy.compute_delay(0) for _ in range(20)}
        assert len(delays) == 1, "linear preset (no jitter) must be deterministic"

    def test_jitter_delay_within_bounds(self) -> None:
        """Standard preset jitter stays within [0.5x, 1.5x] of base delay."""
        policy = RETRY_PRESETS["standard"]
        # attempt=0: base = 1.0 * 2.0^0 = 1.0; jitter range [0.5, 1.5]
        samples = [policy.compute_delay(0) for _ in range(500)]
        assert min(samples) >= 0.5 - 1e-9
        assert max(samples) <= 1.5 + 1e-9

    def test_per_node_policy_with_jitter_via_get_retry_policy(self) -> None:
        """_get_retry_policy with 'standard' returns a jitter-enabled policy."""
        node = _node(retry_preset="standard")
        policy = _get_retry_policy(node)

        assert policy.jitter is True
        delays = [policy.compute_delay(0) for _ in range(50)]
        assert len(set(delays)) > 1


# ================================================================== #
# Pipeline default unchanged
# ================================================================== #


class TestPipelineDefaultUnchanged:
    """_PIPELINE_RETRY must remain the 'none' preset (Spec S3.6)."""

    def test_pipeline_default_unchanged(self) -> None:
        """_PIPELINE_RETRY is RETRY_PRESETS['none'] -- no-retry by default."""
        assert _PIPELINE_RETRY is RETRY_PRESETS["none"], (
            "_PIPELINE_RETRY must remain the 'none' preset to preserve backward compatibility"
        )
        assert _PIPELINE_RETRY.max_retries == 0
        assert _PIPELINE_RETRY.jitter is False
        assert _PIPELINE_RETRY.initial_delay == 0.0

    def test_all_preset_names_present(self) -> None:
        """The required 5 preset names are all registered in RETRY_PRESETS."""
        required = {"none", "standard", "aggressive", "linear", "patient"}
        assert required.issubset(RETRY_PRESETS.keys()), (
            f"Missing presets: {required - RETRY_PRESETS.keys()}"
        )

    def test_presets_are_retry_policy_instances(self) -> None:
        """Every entry in RETRY_PRESETS is a RetryPolicy instance."""
        for name, policy in RETRY_PRESETS.items():
            assert isinstance(policy, RetryPolicy), f"RETRY_PRESETS[{name!r}] is not a RetryPolicy"


# ================================================================== #
# Item 3  S8.6.9 -- Live cache-efficiency tests (OpenAI + Gemini)
# ================================================================== #

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

HAS_OPENAI = bool(OPENAI_KEY)
HAS_GEMINI = bool(GEMINI_KEY)

skip_no_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
skip_no_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="GOOGLE_API_KEY not set")

OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-2.0-flash"

# Long enough system prompt to cross the OpenAI 1024-token prefix-caching threshold
# and encourage Gemini implicit caching.  Repeated verbatim across turns so the
# provider-side cache key matches.
_LONG_SYSTEM = (
    "You are a concise, factual assistant helping with general knowledge questions. " * 150
    + "Always answer in one short sentence."
)

_TURNS = [
    "What is the capital of France?",
    "What is 7 times 8?",
    "Name the largest planet in the solar system.",
    "What colour is the sky on a clear day?",
    "How many sides does a hexagon have?",
]


class TestOpenAICacheEfficiency:
    """Live cache-efficiency check for OpenAI prefix caching (S8.6.9)."""

    @skip_no_openai
    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "OpenAI prefix caching is provider-side and may not trigger for "
            "every model/region/account; xfail permits the test to be informative "
            "without being a hard gate."
        ),
    )
    async def test_openai_cache_efficiency_live(self) -> None:
        """Multi-turn conversation; cumulative cache_read_tokens should be > 0 by turn 5."""
        from attractor_llm import Client, ProviderConfig, RetryPolicy, generate
        from attractor_llm.adapters.openai import OpenAIAdapter

        client = Client(retry_policy=RetryPolicy(max_retries=1))
        client.register_adapter(
            "openai", OpenAIAdapter(ProviderConfig(api_key=OPENAI_KEY, timeout=60.0))
        )

        # OpenAI automatic prefix caching: identical long prompt prefix is cached
        # after the 1st request.  Subsequent calls with the same long system prompt
        # should report cached_tokens in input_tokens_details.
        cumulative_cache_reads = 0

        for turn, user_msg in enumerate(_TURNS, start=1):
            result = await generate(
                client,
                OPENAI_MODEL,
                user_msg,
                system=_LONG_SYSTEM,
                provider="openai",
            )
            assert result.total_usage.input_tokens > 0, f"Turn {turn}: expected input_tokens > 0"
            cumulative_cache_reads += result.total_usage.cache_read_tokens

        # By turn 3-5 at least one response should show cached tokens
        assert cumulative_cache_reads > 0, (
            f"Expected cache_read_tokens > 0 across {len(_TURNS)} turns with a "
            f"{len(_LONG_SYSTEM)}-char system prompt, got 0 total cache reads. "
            "OpenAI prefix caching may not be active for this account/region."
        )


class TestGeminiCacheEfficiency:
    """Live cache-efficiency check for Gemini implicit caching (S8.6.9)."""

    @skip_no_gemini
    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Gemini implicit caching requires long-enough prompts and is "
            "provider-side; xfail permits the test to be informative without "
            "being a hard gate."
        ),
    )
    async def test_gemini_cache_efficiency_live(self) -> None:
        """Multi-turn conversation; cumulative cache_read_tokens should be > 0 by turn 5."""
        from attractor_llm import Client, ProviderConfig, RetryPolicy, generate
        from attractor_llm.adapters.gemini import GeminiAdapter

        client = Client(retry_policy=RetryPolicy(max_retries=1))
        client.register_adapter(
            "gemini", GeminiAdapter(ProviderConfig(api_key=GEMINI_KEY, timeout=60.0))
        )

        # Gemini implicit caching: identical context prefix is served from cache
        # for repeated requests (cachedContentTokenCount in usage metadata).
        cumulative_cache_reads = 0

        for turn, user_msg in enumerate(_TURNS, start=1):
            result = await generate(
                client,
                GEMINI_MODEL,
                user_msg,
                system=_LONG_SYSTEM,
                provider="gemini",
            )
            assert result.total_usage.input_tokens > 0, f"Turn {turn}: expected input_tokens > 0"
            cumulative_cache_reads += result.total_usage.cache_read_tokens

        # By turn 3-5 at least one response should show cached tokens
        assert cumulative_cache_reads > 0, (
            f"Expected cache_read_tokens > 0 across {len(_TURNS)} turns with a "
            f"{len(_LONG_SYSTEM)}-char system prompt, got 0 total cache reads. "
            "Gemini implicit caching may not be active for this prompt length/model."
        )
