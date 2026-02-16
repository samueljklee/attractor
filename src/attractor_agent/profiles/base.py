"""Provider profile protocol and base implementation.

Defines the ProviderProfile protocol that all profiles implement,
plus BaseProfile as a generic fallback for unknown providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from attractor_agent.session import SessionConfig
from attractor_llm.types import Tool


class ProviderProfile(Protocol):
    """Protocol for provider-specific agent profiles.

    Each profile configures how the coding agent behaves with a
    specific LLM provider: system prompt, tool descriptions,
    and behavioral defaults.
    """

    @property
    def name(self) -> str:
        """Profile name (matches provider name)."""
        ...

    @property
    def system_prompt(self) -> str:
        """The full system prompt for this provider."""
        ...

    @property
    def default_model(self) -> str:
        """Default model ID for this provider."""
        ...

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Whether the provider supports parallel tool execution. Spec §2.5."""
        ...

    def get_tools(self, base_tools: list[Tool]) -> list[Tool]:
        """Customize tool descriptions for this provider.

        Takes the base tools and returns a (possibly modified) list
        with provider-optimized descriptions. The tool execute handlers
        remain the same -- only descriptions change.
        """
        ...

    def apply_to_config(self, config: SessionConfig) -> SessionConfig:
        """Apply profile defaults to a session config.

        Only sets values that aren't already explicitly configured.
        Explicit config values always win over profile defaults.
        """
        ...


@dataclass
class BaseProfile:
    """Generic fallback profile for unknown providers.

    Provides sensible defaults that work with any LLM provider.
    Used when no provider-specific profile is registered.
    """

    @property
    def name(self) -> str:
        return "base"

    @property
    def system_prompt(self) -> str:
        return _BASE_SYSTEM_PROMPT

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-5"

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Base profile defaults to sequential execution."""
        return False

    def get_tools(self, base_tools: list[Tool]) -> list[Tool]:
        """Return base tools unmodified."""
        return list(base_tools)

    def apply_to_config(self, config: SessionConfig) -> SessionConfig:
        """Apply base defaults. Only fills in unset values."""
        if not config.system_prompt:
            config.system_prompt = self.system_prompt
        return config


_BASE_SYSTEM_PROMPT = """\
You are an expert software engineer acting as an autonomous coding agent.

WORKFLOW
1. Understand first. Read relevant files and search the codebase before making changes.
2. Plan, then act. For non-trivial tasks, state your approach before executing.
3. Verify after changing. Run tests or read back files to confirm correctness.

EDITING FILES
- Use edit_file for targeted modifications. The old_string must match exactly.
- Use write_file only for new files or complete rewrites.
- Always read_file before edit_file to get exact current content.

SEARCHING
- Use grep to find patterns, definitions, and usages across the codebase.
- Use glob to discover file structure and find files by name.

SHELL
- Use shell for running tests, builds, git operations, and installations.
- Do NOT use shell for reading files (use read_file) or editing files (use edit_file).

ERROR HANDLING
- If edit_file fails, re-read the file and retry with correct content.
- If a command fails, read the error carefully and fix the root cause.
- Do not retry identical failing tool calls.

PRINCIPLES
- Make the smallest change that solves the problem correctly.
- Preserve existing code style and conventions.
- Process files one at a time: read, edit, verify.\
"""
