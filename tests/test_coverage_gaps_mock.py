"""Mock tests covering coverage gaps across Waves 9-13.

P1: Provider identity in system prompts (3 tests)
P4: Profile apply_to_config end-to-end (2 tests)
"""

from __future__ import annotations

import pytest

from attractor_agent.profiles.anthropic import AnthropicProfile
from attractor_agent.profiles.gemini import GeminiProfile
from attractor_agent.profiles.openai import OpenAIProfile
from attractor_agent.session import Session, SessionConfig
from attractor_llm.client import Client
from tests.helpers import MockAdapter, make_text_response

# ================================================================== #
# P1: Provider identity in system prompts
# ================================================================== #


class TestProviderIdentityInSystemPrompts:
    """Each profile's system_prompt mentions its own provider by name.

    Verifies the profile wires the correct, provider-specific instructions
    into the session -- not a generic or accidentally swapped prompt.
    """

    def test_anthropic_profile_system_prompt_contains_claude(self) -> None:
        """AnthropicProfile.system_prompt explicitly names 'Claude'."""
        profile = AnthropicProfile()
        assert "Claude" in profile.system_prompt, (
            f"Expected 'Claude' in AnthropicProfile.system_prompt. "
            f"Got: {profile.system_prompt[:200]!r}"
        )

    def test_openai_profile_system_prompt_contains_openai(self) -> None:
        """OpenAIProfile.system_prompt explicitly names 'OpenAI'."""
        profile = OpenAIProfile()
        assert "OpenAI" in profile.system_prompt, (
            f"Expected 'OpenAI' in OpenAIProfile.system_prompt. "
            f"Got: {profile.system_prompt[:200]!r}"
        )

    def test_gemini_profile_system_prompt_contains_gemini(self) -> None:
        """GeminiProfile.system_prompt explicitly names 'Gemini'."""
        profile = GeminiProfile()
        assert "Gemini" in profile.system_prompt, (
            f"Expected 'Gemini' in GeminiProfile.system_prompt. "
            f"Got: {profile.system_prompt[:200]!r}"
        )


# ================================================================== #
# P4: Profile apply_to_config end-to-end
# ================================================================== #


class TestProfileApplyToConfigEndToEnd:
    """When Session is created with a profile, its system_prompt and
    default_model actually arrive in the LLM request.

    Uses a MockAdapter that records every Request it receives so we can
    inspect the exact payload sent to the provider.
    """

    @pytest.mark.asyncio
    async def test_profile_system_prompt_reaches_llm_request(self) -> None:
        """AnthropicProfile.system_prompt appears verbatim in request.system.

        _build_enriched_system_prompt() prepends the profile's system_prompt
        as the first layer, so the full request.system should contain it as a
        substring (additional env/project context is appended after).
        """
        profile = AnthropicProfile()

        # Register MockAdapter as "anthropic" -- matches the provider that
        # AnthropicProfile.apply_to_config() sets on the config.
        adapter = MockAdapter(responses=[make_text_response("done")])
        client = Client()
        client.register_adapter("anthropic", adapter)

        # Default SessionConfig has system_prompt="" (falsy) so the profile
        # will fill it in. model defaults to "claude-sonnet-4-5" (same as
        # profile.default_model) so it stays unchanged.
        session = Session(client=client, profile=profile)
        await session.submit("Hello")

        assert adapter.call_count == 1, "Expected exactly one LLM call"
        req = adapter.requests[0]

        assert req.system is not None, (
            "Expected request.system to be set; profile should have injected it"
        )
        assert profile.system_prompt in req.system, (
            f"Profile system_prompt should be a prefix of request.system.\n"
            f"Profile start: {profile.system_prompt[:80]!r}\n"
            f"Request system start: {req.system[:80]!r}"
        )

    @pytest.mark.asyncio
    async def test_profile_default_model_reaches_llm_request(self) -> None:
        """GeminiProfile.default_model is used when SessionConfig.model is empty.

        Passing SessionConfig(model="") makes the condition `not config.model`
        True, so apply_to_config() fills in the profile's default_model.
        The captured request should then carry that model string.
        """
        profile = GeminiProfile()

        # Register MockAdapter as "gemini" -- matches what GeminiProfile sets.
        adapter = MockAdapter(responses=[make_text_response("done")])
        client = Client()
        client.register_adapter("gemini", adapter)

        # model="" is falsy → profile.apply_to_config() overwrites it.
        # system_prompt="" is falsy → profile fills that in too.
        config = SessionConfig(model="", system_prompt="")
        session = Session(client=client, config=config, profile=profile)
        await session.submit("Hello")

        assert adapter.call_count == 1, "Expected exactly one LLM call"
        req = adapter.requests[0]

        assert req.model == profile.default_model, (
            f"Expected request.model == {profile.default_model!r} "
            f"(GeminiProfile.default_model), got {req.model!r}"
        )
