"""Tests for provider profiles: protocol, registry, 3 provider profiles, integration.

Level 1: Unit tests (no API calls)
Level 2: Integration tests (MockAdapter)
Level 3: Aggressive edge cases
"""

from __future__ import annotations

import os

import pytest

from attractor_agent.profiles import get_profile, list_profiles
from attractor_agent.profiles.anthropic import AnthropicProfile
from attractor_agent.profiles.base import BaseProfile
from attractor_agent.profiles.gemini import GeminiProfile
from attractor_agent.profiles.openai import OpenAIProfile
from attractor_agent.session import Session, SessionConfig
from attractor_agent.tools.core import ALL_CORE_TOOLS, set_allowed_roots
from attractor_llm.client import Client
from attractor_llm.types import (
    Tool,
)
from tests.helpers import MockAdapter, make_text_response, make_tool_call_response

# ================================================================== #
# Level 1: Profile protocol and construction
# ================================================================== #


class TestProfileProtocol:
    def test_anthropic_profile_has_required_fields(self):
        p = AnthropicProfile()
        assert p.name == "anthropic"
        assert len(p.system_prompt) > 100
        assert p.default_model == "claude-sonnet-4-5"

    def test_openai_profile_has_required_fields(self):
        p = OpenAIProfile()
        assert p.name == "openai"
        assert len(p.system_prompt) > 100
        assert p.default_model == "gpt-5.2"

    def test_gemini_profile_has_required_fields(self):
        p = GeminiProfile()
        assert p.name == "gemini"
        assert len(p.system_prompt) > 100
        assert p.default_model == "gemini-3-flash-preview"

    def test_base_profile_has_required_fields(self):
        p = BaseProfile()
        assert p.name == "base"
        assert len(p.system_prompt) > 100
        assert p.default_model == "claude-sonnet-4-5"


class TestProfileRegistry:
    def test_get_anthropic_profile(self):
        p = get_profile("anthropic")
        assert p.name == "anthropic"

    def test_get_openai_profile(self):
        p = get_profile("openai")
        assert p.name == "openai"

    def test_get_gemini_profile(self):
        p = get_profile("gemini")
        assert p.name == "gemini"

    def test_unknown_provider_returns_base(self):
        p = get_profile("unknown_provider")
        assert p.name == "base"

    def test_list_profiles(self):
        profiles = list_profiles()
        assert "anthropic" in profiles
        assert "openai" in profiles
        assert "gemini" in profiles
        assert len(profiles) == 3


# ================================================================== #
# Level 1: System prompt content verification
# ================================================================== #


class TestSystemPromptContent:
    """Verify system prompts contain critical instructions."""

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_prompt_mentions_edit_file(self, provider):
        p = get_profile(provider)
        assert "edit_file" in p.system_prompt

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_prompt_mentions_read_before_edit(self, provider):
        p = get_profile(provider)
        prompt_lower = p.system_prompt.lower()
        assert "read" in prompt_lower and "edit" in prompt_lower

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_prompt_discourages_shell_for_file_ops(self, provider):
        p = get_profile(provider)
        prompt_lower = p.system_prompt.lower()
        assert "shell" in prompt_lower

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_prompt_mentions_error_handling(self, provider):
        p = get_profile(provider)
        prompt_lower = p.system_prompt.lower()
        assert "error" in prompt_lower or "fail" in prompt_lower

    def test_anthropic_emphasizes_exact_matching(self):
        p = AnthropicProfile()
        assert "EXACTLY" in p.system_prompt or "exactly" in p.system_prompt.lower()

    def test_openai_emphasizes_conciseness(self):
        p = OpenAIProfile()
        prompt_lower = p.system_prompt.lower()
        assert "concise" in prompt_lower or "short" in prompt_lower or "decisive" in prompt_lower

    def test_gemini_discourages_guessing(self):
        p = GeminiProfile()
        assert "guess" in p.system_prompt.lower() or "assume" in p.system_prompt.lower()


# ================================================================== #
# Level 1: Tool customization
# ================================================================== #


class TestToolCustomization:
    def test_all_profiles_return_6_tools(self):
        for provider in ["anthropic", "openai", "gemini"]:
            p = get_profile(provider)
            tools = p.get_tools(list(ALL_CORE_TOOLS))
            assert len(tools) == 6, f"{provider} returned {len(tools)} tools"

    def test_base_profile_returns_tools_unmodified(self):
        p = BaseProfile()
        tools = p.get_tools(list(ALL_CORE_TOOLS))
        for orig, custom in zip(ALL_CORE_TOOLS, tools, strict=True):
            assert orig.name == custom.name
            assert orig.description == custom.description

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_edit_file_description_enhanced(self, provider):
        p = get_profile(provider)
        tools = p.get_tools(list(ALL_CORE_TOOLS))
        edit_tool = next(t for t in tools if t.name == "edit_file")
        # Enhanced descriptions should be longer than the base
        base_edit = next(t for t in ALL_CORE_TOOLS if t.name == "edit_file")
        assert len(edit_tool.description) >= len(base_edit.description)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_tool_execute_handlers_preserved(self, provider):
        """Profile tool customization must NOT change execute handlers."""
        p = get_profile(provider)
        tools = p.get_tools(list(ALL_CORE_TOOLS))
        for orig, custom in zip(ALL_CORE_TOOLS, tools, strict=True):
            assert custom.execute is orig.execute, (
                f"{provider}: {custom.name} execute handler changed"
            )

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
    def test_tool_parameters_preserved(self, provider):
        """Profile must NOT change tool parameter schemas."""
        p = get_profile(provider)
        tools = p.get_tools(list(ALL_CORE_TOOLS))
        for orig, custom in zip(ALL_CORE_TOOLS, tools, strict=True):
            assert custom.parameters == orig.parameters, (
                f"{provider}: {custom.name} parameters changed"
            )


# ================================================================== #
# Level 1: Config application
# ================================================================== #


class TestConfigApplication:
    def test_anthropic_sets_provider(self):
        config = SessionConfig()
        p = AnthropicProfile()
        config = p.apply_to_config(config)
        assert config.provider == "anthropic"
        assert config.system_prompt == p.system_prompt

    def test_openai_sets_temperature(self):
        config = SessionConfig()
        p = OpenAIProfile()
        config = p.apply_to_config(config)
        assert config.temperature == 0.2
        # reasoning_effort only set for o-series models
        assert config.reasoning_effort is None  # default model isn't o-series

    def test_openai_sets_reasoning_for_o_series(self):
        config = SessionConfig(model="o3")
        p = OpenAIProfile()
        config = p.apply_to_config(config)
        assert config.reasoning_effort == "medium"

    def test_gemini_sets_temperature_zero(self):
        config = SessionConfig()
        p = GeminiProfile()
        config = p.apply_to_config(config)
        assert config.temperature == 0.0
        # reasoning_effort only set for models that support thinkingConfig
        assert config.reasoning_effort is None  # default model doesn't support it

    def test_gemini_tightens_loop_detection(self):
        config = SessionConfig()
        p = GeminiProfile()
        config = p.apply_to_config(config)
        assert config.loop_detection_threshold == 2

    def test_explicit_config_overrides_profile(self):
        """User-set values must NOT be overwritten by profile."""
        config = SessionConfig(
            system_prompt="My custom prompt",
            model="my-model",
            provider="my-provider",
            temperature=0.5,
            reasoning_effort="high",
        )
        p = AnthropicProfile()
        config = p.apply_to_config(config)
        assert config.system_prompt == "My custom prompt"
        assert config.model == "my-model"
        assert config.provider == "my-provider"

    def test_openai_explicit_temperature_preserved(self):
        config = SessionConfig(temperature=0.8)
        p = OpenAIProfile()
        config = p.apply_to_config(config)
        assert config.temperature == 0.8  # not overwritten

    def test_anthropic_leaves_temperature_none(self):
        config = SessionConfig()
        p = AnthropicProfile()
        config = p.apply_to_config(config)
        assert config.temperature is None  # Anthropic doesn't set temperature


# ================================================================== #
# Level 2: Integration with Session (MockAdapter)
# ================================================================== #


class TestProfileSessionIntegration:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path):
        self.sandbox = tmp_path
        set_allowed_roots([str(tmp_path)])
        yield
        set_allowed_roots([os.getcwd()])

    @pytest.mark.asyncio
    async def test_anthropic_profile_system_prompt_in_request(self):
        """Anthropic profile system prompt reaches the LLM."""
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        profile = AnthropicProfile()
        config = SessionConfig(model="mock-model", provider="mock")
        config = profile.apply_to_config(config)

        tools = profile.get_tools(list(ALL_CORE_TOOLS))
        session = Session(client=client, config=config, tools=tools)
        await session.submit("test")

        req = adapter.requests[0]
        assert req.system == profile.system_prompt

    @pytest.mark.asyncio
    async def test_openai_profile_system_prompt_in_request(self):
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        profile = OpenAIProfile()
        config = SessionConfig(model="mock-model", provider="mock")
        config = profile.apply_to_config(config)

        session = Session(client=client, config=config)
        await session.submit("test")

        req = adapter.requests[0]
        assert req.system == profile.system_prompt

    @pytest.mark.asyncio
    async def test_profile_tools_used_in_session(self):
        """Profile-customized tool descriptions reach the LLM."""
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        profile = AnthropicProfile()
        config = SessionConfig(model="mock-model", provider="mock")
        config = profile.apply_to_config(config)

        tools = profile.get_tools(list(ALL_CORE_TOOLS))
        session = Session(client=client, config=config, tools=tools)
        await session.submit("test")

        req = adapter.requests[0]
        assert req.tools is not None
        edit_tool = next((t for t in req.tools if t.name == "edit_file"), None)
        assert edit_tool is not None
        assert "EXACTLY" in edit_tool.description or "exactly" in edit_tool.description.lower()

    @pytest.mark.asyncio
    async def test_profile_with_tool_call_flow(self):
        """Full flow: profile system prompt + tool call + tool result."""
        test_file = self.sandbox / "test.txt"
        test_file.write_text("hello world")

        adapter = MockAdapter(
            responses=[
                make_tool_call_response("read_file", {"path": str(test_file)}, "tc-1"),
                make_text_response("File contains: hello world"),
            ]
        )

        client = Client()
        client.register_adapter("mock", adapter)

        profile = GeminiProfile()
        config = SessionConfig(model="mock-model", provider="mock")
        config = profile.apply_to_config(config)

        tools = profile.get_tools(list(ALL_CORE_TOOLS))
        session = Session(client=client, config=config, tools=tools)
        result = await session.submit("Read the file")

        assert "hello world" in result
        # Verify Gemini profile's system prompt was used
        assert adapter.requests[0].system == profile.system_prompt


# ================================================================== #
# Level 3: Aggressive edge cases
# ================================================================== #


class TestProfileEdgeCases:
    def test_empty_base_tools_returns_empty(self):
        """Profile with empty tool list returns empty."""
        for provider in ["anthropic", "openai", "gemini", "base"]:
            p = get_profile(provider) if provider != "base" else BaseProfile()
            tools = p.get_tools([])
            assert len(tools) == 0

    def test_profile_with_unknown_tool_passes_through(self):
        """Unknown tools in base list pass through unmodified."""
        custom_tool = Tool(
            name="custom_tool",
            description="A custom tool",
            parameters={"type": "object", "properties": {}},
        )
        p = AnthropicProfile()
        tools = p.get_tools([custom_tool])
        assert len(tools) == 1
        assert tools[0].name == "custom_tool"
        # Description unchanged (not in override dict)
        assert tools[0].description == "A custom tool"

    def test_multiple_apply_to_config_is_idempotent(self):
        """Applying the same profile twice doesn't change values."""
        config = SessionConfig()
        p = OpenAIProfile()
        config = p.apply_to_config(config)
        temp1 = config.temperature
        reasoning1 = config.reasoning_effort

        config = p.apply_to_config(config)
        assert config.temperature == temp1
        assert config.reasoning_effort == reasoning1

    def test_different_profiles_on_same_config(self):
        """Applying different profiles: first one wins for set fields."""
        config = SessionConfig()
        anthropic = AnthropicProfile()
        config = anthropic.apply_to_config(config)
        assert config.system_prompt == anthropic.system_prompt

        # Now apply OpenAI -- system_prompt already set, shouldn't change
        openai = OpenAIProfile()
        config = openai.apply_to_config(config)
        assert config.system_prompt == anthropic.system_prompt  # first wins

    @pytest.mark.asyncio
    async def test_profile_in_pipeline_backend(self):
        """Profile is applied when running through pipeline backend."""
        from attractor_pipeline.backends import AgentLoopBackend
        from attractor_pipeline.graph import Node

        adapter = MockAdapter(responses=[make_text_response("coded")])
        client = Client()
        # Register as "mock" -- backend uses default_provider="mock"
        client.register_adapter("mock", adapter)

        backend = AgentLoopBackend(
            client,
            default_model="mock-model",
            default_provider="mock",
        )

        # Node uses provider "mock" (matching the registered adapter)
        node = Node(id="test", shape="box", llm_provider="mock")
        result = await backend.run(node, "Write code", {"goal": "test"})

        assert result == "coded" or (hasattr(result, "output") and result.output)
        # Verify the profile was applied (falls back to base since "mock" is unknown)
        req = adapter.requests[0]
        base_profile = BaseProfile()
        assert req.system == base_profile.system_prompt
