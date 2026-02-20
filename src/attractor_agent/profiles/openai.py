"""OpenAI provider profile (codex-rs style).

Designed by O3 reviewing its own provider's conventions.
Key insights from the self-review:
- GPT/o-series tend to over-explain -- prompt must force action-first
- Temperature=0.2 for deterministic tool use without rigidity
- reasoning_effort=medium balances quality vs verbosity
- max_tokens=1400 encourages concise output
- Anti-patterns: asking for confirmation, long preambles before acting
"""

from __future__ import annotations

from dataclasses import dataclass

from attractor_agent.session import SessionConfig
from attractor_llm.types import Tool


@dataclass
class OpenAIProfile:
    """codex-rs-style profile for OpenAI models."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def system_prompt(self) -> str:
        return _OPENAI_SYSTEM_PROMPT

    @property
    def default_model(self) -> str:
        return "gpt-5.2"

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """OpenAI supports parallel tool calls. Spec §2.5."""
        return True

    def get_tools(self, base_tools: list[Tool]) -> list[Tool]:
        """Enhance tool descriptions for OpenAI conventions.

        Also injects apply_patch if not already present -- OpenAI models
        prefer the v4a unified-diff format for code changes (§9.2).
        Also injects subagent tools (spawn_agent, send_input, wait,
        close_agent) for interactive multi-agent workflows (§9.12.34-36).
        """
        from attractor_agent.subagent_manager import SubagentManager, create_interactive_tools
        from attractor_agent.tools.core import APPLY_PATCH

        tools: list[Tool] = []
        has_apply_patch = any(t.name == "apply_patch" for t in base_tools)
        for tool in base_tools:
            desc = _OPENAI_TOOL_DESCRIPTIONS.get(tool.name, tool.description)
            tools.append(
                Tool(
                    name=tool.name,
                    description=desc,
                    parameters=tool.parameters,
                    execute=tool.execute,
                )
            )
        if base_tools and not has_apply_patch:
            tools.append(APPLY_PATCH)
        # Inject subagent tools if not already present (§9.12.34-36)
        if base_tools:
            existing_names = {t.name for t in tools}
            subagent_tools = create_interactive_tools(SubagentManager())
            for t in subagent_tools:
                if t.name not in existing_names:
                    tools.append(t)
        return tools

    def apply_to_config(self, config: SessionConfig) -> SessionConfig:
        """Apply OpenAI defaults to session config."""
        if not config.system_prompt:
            config.system_prompt = self.system_prompt
        if not config.model:
            config.model = self.default_model
        if not config.provider:
            config.provider = "openai"
        if config.temperature is None:
            config.temperature = 0.2
        # Only set reasoning_effort for models that support it (o-series).
        # GPT models don't support reasoning.effort and will error.
        if config.reasoning_effort is None:
            model = config.model or ""
            if model.startswith(("o1", "o3", "o4")):
                config.reasoning_effort = "medium"
        return config


# ------------------------------------------------------------------ #
# System prompt (designed by O3)
# ------------------------------------------------------------------ #

_OPENAI_SYSTEM_PROMPT = """\
You are an AI assistant powered by OpenAI. You are an expert coding \
agent. Work in short, action-oriented steps. Be decisive: read files, \
make edits, and report results without excessive explanation.

RULES
- Always inspect files before editing. Use read_file to get exact context.
- Prefer apply_patch as the primary editing tool for code changes. \
apply_patch uses the v4a unified-diff format and handles multi-hunk edits \
atomically. Use edit_file only for single-string replacements when \
apply_patch is unavailable. Use write_file only for new files or full rewrites.
- Use shell only for running tests, builds, or inspection commands.
- Use grep and glob to locate code before reading large files.
- After tool use, summarize what changed and what remains.

WORKFLOW PATTERN
discover (glob/grep) -> inspect (read_file) -> modify (apply_patch) -> \
verify (shell)

EDITING
- apply_patch applies unified diffs: include 3 lines of context around \
each change. Multiple hunks in one call are atomic.
- edit_file requires old_string to match exactly and be unique in the file.
- Always read_file first to see the exact content before editing.
- For multiple changes to one file, prefer a single apply_patch call \
over multiple edit_file calls.

ERROR HANDLING
- If apply_patch or edit_file fails, re-read the file and retry with \
correct content.
- If a command fails, read the error output and fix the root cause.
- Do not retry identical failing tool calls.

AVOID
- Long explanations or speculative plans.
- Asking for confirmation if the task is clear.
- Rewriting entire files when a targeted edit works.
- Using shell for file reading (use read_file), file writing \
(use write_file), or code search (use grep).\
"""

# ------------------------------------------------------------------ #
# Tool description overrides
# ------------------------------------------------------------------ #

_OPENAI_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read a file's contents with line numbers. "
        "Must read before edit. Use offset and limit for large files."
    ),
    "edit_file": (
        "Replace a specific string in a file. Use only after reading "
        "the file. Ensure old_string is unique unless replace_all "
        "is true."
    ),
    "write_file": (
        "Write content to a file. Creates directories if needed. "
        "Overwrites existing files. Only for new files or full rewrites "
        "-- use edit_file for modifications."
    ),
}
