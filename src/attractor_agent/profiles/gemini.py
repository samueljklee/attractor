"""Gemini provider profile (gemini-cli style).

Designed by Gemini reviewing its own provider's conventions.
Key insights from the self-review:
- Temperature=0.0 is critical -- Gemini at 1.0 produces inconsistent tool args
- reasoning_effort=medium maps to thinkingBudget=4096 (sweet spot)
- Tighter loop detection (threshold=2) -- Gemini retries failed edits identically
- Tool descriptions need examples (Gemini parses output format better with them)
- Anti-pattern: guessing file contents from training data instead of reading
"""

from __future__ import annotations

from dataclasses import dataclass

from attractor_agent.session import SessionConfig
from attractor_llm.types import Tool


@dataclass
class GeminiProfile:
    """gemini-cli-style profile for Google Gemini models."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def system_prompt(self) -> str:
        return _GEMINI_SYSTEM_PROMPT

    @property
    def default_model(self) -> str:
        return "gemini-3-flash-preview"

    @property
    def supports_parallel_tool_calls(self) -> bool:
        """Gemini supports parallel tool calls. Spec §2.5."""
        return True

    def get_tools(self, base_tools: list[Tool]) -> list[Tool]:
        """Enhance tool descriptions with examples for Gemini.

        Also injects list_dir and read_many_files if not already present
        (Spec §3.6: Gemini profile tool list).
        Subagent tools (spawn_agent, send_input, wait, close_agent) are
        injected at Session level with a real client (§9.12.34-36).
        """
        from attractor_agent.tools.core import LIST_DIR, READ_MANY_FILES

        tools: list[Tool] = []
        for tool in base_tools:
            desc = _GEMINI_TOOL_DESCRIPTIONS.get(tool.name, tool.description)
            tools.append(
                Tool(
                    name=tool.name,
                    description=desc,
                    parameters=tool.parameters,
                    execute=tool.execute,
                )
            )
        if base_tools:
            if not any(t.name == "list_dir" for t in tools):
                tools.append(
                    Tool(
                        name=LIST_DIR.name,
                        description=_GEMINI_TOOL_DESCRIPTIONS["list_dir"],
                        parameters=LIST_DIR.parameters,
                        execute=LIST_DIR.execute,
                    )
                )
            if not any(t.name == "read_many_files" for t in tools):
                tools.append(
                    Tool(
                        name=READ_MANY_FILES.name,
                        description=_GEMINI_TOOL_DESCRIPTIONS["read_many_files"],
                        parameters=READ_MANY_FILES.parameters,
                        execute=READ_MANY_FILES.execute,
                    )
                )
        return tools

    def apply_to_config(self, config: SessionConfig) -> SessionConfig:
        """Apply Gemini defaults to session config."""
        if not config.system_prompt:
            config.system_prompt = self.system_prompt
        if not config.model:
            config.model = self.default_model
        if not config.provider:
            config.provider = "gemini"
        if config.temperature is None:
            config.temperature = 0.0
        # Only set reasoning_effort for models that support thinkingConfig.
        # gemini-2.5-flash and older models don't support it.
        if config.reasoning_effort is None:
            model = config.model or ""
            if "2.5-pro" in model or "2.5-flash-preview" in model:
                config.reasoning_effort = "medium"
        # Tighter loop detection -- Gemini retries failed edits identically
        if config.loop_detection_threshold is None:
            config.loop_detection_threshold = 2
        return config


# ------------------------------------------------------------------ #
# System prompt (designed by Gemini)
# ------------------------------------------------------------------ #

_GEMINI_SYSTEM_PROMPT = """\
You are Gemini, an AI assistant by Google. You are an expert software \
engineer acting as an autonomous coding agent.

You have access to tools for reading, writing, editing files, running \
shell commands, and searching codebases. Use them proactively.

CORE WORKFLOW
1. Understand first. Before making changes, read the relevant files \
to understand existing code structure, conventions, and context. Use \
glob to discover project layout and grep to find relevant code.
2. Plan explicitly. State your approach before executing. Break \
complex tasks into steps.
3. Make targeted changes. Use edit_file for surgical modifications. \
Use write_file only for new files. Never rewrite an entire file when \
a targeted edit suffices.
4. Verify your work. After changes, use shell to run tests or linters. \
Read back edited files to confirm correctness.

TOOL USE RULES
- Call multiple independent tools in a single response when possible.
- When a tool returns an error, read the error message carefully, \
diagnose the root cause, and fix your approach. Do not retry the \
identical call.
- For edit_file: old_string must match file content exactly, including \
whitespace and indentation. If ambiguous, include more surrounding \
context to make it unique.
- For shell: prefer short, focused commands. Chain with && for \
dependent steps.
- For grep: use regex patterns. Combine with include to filter by \
file type.
- For read_file: use offset and limit for large files.

WHAT NOT TO DO
- Do NOT guess file contents. Always read_file before edit_file. \
Never assume you know what a file contains.
- Do NOT use write_file to modify existing files. This destroys \
content outside the section you're changing.
- Do NOT retry a failed tool call with identical arguments.
- Do NOT generate placeholder or stub code. Implement completely.
- Do NOT explain at length before acting. State the plan briefly, \
then execute with tools.

RESPONSE STYLE
- Be direct and concise. Lead with the action or answer.
- When explaining code changes, show what changed and why.
- If a task is ambiguous, state your assumptions and proceed.\
"""

# ------------------------------------------------------------------ #
# Tool description overrides (with examples for Gemini)
# ------------------------------------------------------------------ #

_GEMINI_TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read a file's contents with line numbers. "
        "Returns numbered lines in 'LINENO\\tCONTENT' format. "
        "For large files, use offset and limit to paginate "
        "(e.g., offset=100, limit=50 reads lines 101-150). "
        "Always read a file before attempting to edit it."
    ),
    "write_file": (
        "Create a new file or completely overwrite an existing file. "
        "Creates parent directories automatically. "
        "Use edit_file instead when modifying existing files -- "
        "write_file should only be used for new files or full rewrites."
    ),
    "edit_file": (
        "Replace a specific string in an existing file. "
        "old_string must appear EXACTLY as it exists in the file "
        "(including indentation and whitespace). "
        "If old_string appears multiple times, include more "
        "surrounding lines to make it unique, or set replace_all=true. "
        "You MUST read_file first to see the exact content."
    ),
    "shell": (
        "Execute a shell command via bash and return stdout, stderr, "
        "and exit code. Subject to timeout (default 120s) and safety "
        "filtering. Use for: running tests, installing packages, git "
        "operations, build commands. Prefer short focused commands."
    ),
    "grep": (
        "Search file contents recursively using regex patterns. "
        "Returns matching lines as 'FILEPATH:LINE:CONTENT'. "
        "Use include parameter to filter by extension "
        "(e.g., include='*.py'). "
        "Use to find definitions, usages, imports, and patterns."
    ),
    "glob": (
        "Find files matching a glob pattern. "
        "Returns paths relative to the search directory. "
        "Common patterns: '**/*.py' (all Python files), "
        "'src/**/*.ts' (TypeScript in src). "
        "Use to understand project structure before diving in."
    ),
    "list_dir": (
        "List the contents of a directory. Returns files and subdirectories "
        "with optional depth control. Use depth=0 for immediate children only, "
        "depth=1 (default) for one level of subdirectories. Directories are "
        "marked with trailing /. Use this to understand project structure "
        "before reading specific files."
    ),
    "read_many_files": (
        "Read multiple files in a single call. More efficient than multiple "
        "read_file calls when you need to examine several files at once. "
        "Returns concatenated content with file headers and line numbers. "
        "Missing files are reported per-file without aborting the batch."
    ),
}
