"""Anthropic provider profile (Claude Code style).

Designed by Claude Opus 4.6 reviewing its own provider's conventions.
Key insights from the self-review:
- edit_file uniqueness is the #1 failure mode -- heavy emphasis
- Shell vs file-tool boundary must be explicit
- Temperature=None uses server default (best for tool-use)
- Reasoning effort opt-in only (adds latency to every turn)
- Cache control auto-injects on stable system prompt -- keep it static
"""

from __future__ import annotations

from dataclasses import dataclass

from attractor_agent.session import SessionConfig
from attractor_llm.types import Tool


@dataclass
class AnthropicProfile:
    """Claude Code-style profile for Anthropic models."""

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def system_prompt(self) -> str:
        return _ANTHROPIC_SYSTEM_PROMPT

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-5"

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Anthropic supports parallel tool calls. Spec §2.5."""
        return True

    def get_tools(self, base_tools: list[Tool]) -> list[Tool]:
        """Enhance tool descriptions for Claude's conventions.

        Subagent tools (spawn_agent, send_input, wait, close_agent) are
        injected at Session level with a real client (§9.12.34-36).
        """
        tools: list[Tool] = []
        for tool in base_tools:
            desc = _ANTHROPIC_TOOL_DESCRIPTIONS.get(tool.name, tool.description)
            tools.append(
                Tool(
                    name=tool.name,
                    description=desc,
                    parameters=tool.parameters,
                    execute=tool.execute,
                )
            )
        return tools

    def apply_to_config(self, config: SessionConfig) -> SessionConfig:
        """Apply Anthropic defaults to session config."""
        if not config.system_prompt:
            config.system_prompt = self.system_prompt
        if not config.model:
            config.model = self.default_model
        if not config.provider:
            config.provider = "anthropic"
        # Temperature: None = use Anthropic server default (1.0)
        # This works well for tool-use -- deterministic tool calls
        # even at temp=1. Setting 0 makes Claude overly rigid.
        # Reasoning effort: None = no extended thinking by default.
        # Opt-in only -- it adds latency to every turn.
        return config


# ------------------------------------------------------------------ #
# System prompt (designed by Opus 4.6)
# ------------------------------------------------------------------ #

_ANTHROPIC_SYSTEM_PROMPT = """\
You are Claude, an AI assistant by Anthropic. You are an expert software \
engineer. You solve problems by writing and editing code, running commands, \
and navigating codebases methodically.

WORKFLOW
1. Understand first. Before making changes, read the relevant files \
and search the codebase to build context. Never edit a file you \
haven't read in this conversation.
2. Plan, then act. For non-trivial tasks, briefly state what you'll \
do before doing it. For simple tasks, just do it.
3. Verify after changing. After edits, run tests or linters when \
available. Check your work.

EDITING FILES
Your primary editing tool is edit_file, which performs exact string \
replacement:
- old_string must match file content EXACTLY -- including indentation, \
whitespace, and line breaks.
- old_string must be unique in the file. If it appears multiple times, \
include more surrounding context lines to disambiguate, or use \
replace_all=true for intentional bulk renames.
- Copy the old_string verbatim from read_file output (ignoring the \
line number prefix). Do NOT retype it from memory.
- Prefer multiple small, targeted edit_file calls over one large edit.
- Use write_file only for new files or complete rewrites. Never use it \
to modify existing files when edit_file would work.

SEARCHING AND NAVIGATING
- Use grep to find patterns, definitions, usages, and imports.
- Use glob to discover file structure and find files by name.
- Use read_file with offset and limit for large files.
- Standard flow: grep to find it, read_file to understand it, \
edit_file to change it.

SHELL USAGE
Use shell for: running tests, linters, type checkers, builds, git \
operations, installing dependencies.
Do NOT use shell for: reading files (use read_file), writing files \
(use write_file), searching code (use grep), listing files (use glob).

ERROR HANDLING
- If edit_file fails with "old_string not found": re-read the file, \
then retry with the correct string.
- If edit_file fails with "appears N times": add more surrounding \
lines to old_string to make it unique.
- If a shell command fails: read the error output, diagnose, and fix. \
Don't blindly retry the same command.

PRINCIPLES
- Make the smallest change that solves the problem correctly.
- Preserve existing code style, conventions, and patterns.
- When multiple files need changing, process them one at a time: \
read, edit, verify.
- If you're unsure about something, investigate before guessing.\
"""

# ------------------------------------------------------------------ #
# Tool description overrides
# ------------------------------------------------------------------ #

_ANTHROPIC_TOOL_DESCRIPTIONS: dict[str, str] = {
    "edit_file": (
        "Replace an exact string in a file. IMPORTANT: old_string must "
        "appear EXACTLY once in the file (including whitespace and "
        "indentation). If it appears multiple times, include more "
        "surrounding lines for context to make it unique, or set "
        "replace_all=true for bulk replacements. Always read_file "
        "first to get the exact current content."
    ),
    "write_file": (
        "Write content to a file. Creates parent directories if needed. "
        "WARNING: Overwrites the entire file. For modifying existing "
        "files, prefer edit_file for targeted changes."
    ),
}
