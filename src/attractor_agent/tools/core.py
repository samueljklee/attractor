"""Core developer tools for the Coding Agent Loop.

Implements the shared toolset used across all provider profiles:
read_file, write_file, edit_file, shell, grep, glob.

Security model:
- File tools enforce path confinement via `_allowed_roots`. By default,
  the working directory is the only allowed root. Callers can expand this
  by modifying `_allowed_roots` before registering tools.
- Shell tool checks a configurable deny-list of dangerous command patterns
  before execution.
- Environment variables are filtered using suffix-based patterns to avoid
  leaking secrets to child processes.

Spec reference: coding-agent-loop §3.3.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from attractor_agent.environment import (
    ExecutionEnvironment,
    LocalEnvironment,
)
from attractor_llm.types import Tool

# ------------------------------------------------------------------ #
# Execution Environment
# ------------------------------------------------------------------ #

# Module-level environment used by all tools. Default: LocalEnvironment
# (direct host access, identical to pre-abstraction behavior).
# Swap to DockerEnvironment or KubernetesEnvironment for sandboxing.
_environment: ExecutionEnvironment = LocalEnvironment()


def set_environment(env: ExecutionEnvironment) -> None:
    """Set the execution environment for all tools."""
    global _environment  # noqa: PLW0603
    _environment = env


def get_environment() -> ExecutionEnvironment:
    """Get the current execution environment."""
    return _environment


# ------------------------------------------------------------------ #
# Process registration callback (§9.1.6)
# ------------------------------------------------------------------ #

# Optional callback invoked when the shell tool spawns a subprocess.
# Session.register_process can be wired here so that LLM-initiated
# shell commands are automatically tracked for graceful abort/cleanup.
# Once LocalEnvironment exposes subprocess objects through its protocol,
# the _shell() function below will call this with the live process.
_process_callback: Any | None = None


def set_process_callback(callback: Any | None) -> None:
    """Set a callback to be invoked when a subprocess is created.

    The callback signature is: ``callback(proc: asyncio.subprocess.Process)``

    Session sets this to ``session.register_process`` so that any shell
    command spawned inside the agent loop is automatically tracked for
    graceful SIGTERM/SIGKILL on abort.  Pass ``None`` to clear.

    Note: The callback is called only when the execution environment
    surfaces the subprocess object.  Currently LocalEnvironment wraps
    the process internally; full auto-registration requires an
    environment protocol extension (env.last_process or similar).
    """
    global _process_callback  # noqa: PLW0603
    _process_callback = callback


def get_process_callback() -> Any | None:
    """Return the current process registration callback, or None."""
    return _process_callback


# ------------------------------------------------------------------ #
# Shell command timeout ceiling (Spec §2.2)
# ------------------------------------------------------------------ #

# max_command_timeout_ms = 600_000 (10 minutes) expressed in seconds.
_max_command_timeout_s: int = 600


def set_max_command_timeout(ms: int) -> None:
    """Override the maximum shell command timeout (in milliseconds)."""
    global _max_command_timeout_s  # noqa: PLW0603
    _max_command_timeout_s = max(1, ms // 1000)


# ------------------------------------------------------------------ #
# Security: Path confinement
# ------------------------------------------------------------------ #

# Allowed root directories for file operations. Resolved paths must
# start with one of these. Default: current working directory.
# Callers can modify this set before registering tools.
_allowed_roots: list[Path] = [Path.cwd().resolve()]


def set_allowed_roots(roots: list[str | Path]) -> None:
    """Configure allowed root directories for file tools."""
    _allowed_roots.clear()
    _allowed_roots.extend(Path(r).resolve() for r in roots)


def _check_path_allowed(file_path: Path) -> str | None:
    """Validate that a resolved path is within allowed roots.

    Returns None if allowed, or an error message if not.
    """
    resolved = file_path.resolve()
    for root in _allowed_roots:
        try:
            resolved.relative_to(root)
            return None  # Path is within this root
        except ValueError:
            continue
    roots_str = ", ".join(str(r) for r in _allowed_roots)
    return f"Error: Path '{resolved}' is outside allowed directories. Allowed roots: {roots_str}"


# ------------------------------------------------------------------ #
# Security: Shell command deny-list
# ------------------------------------------------------------------ #

# Patterns that should be blocked before shell execution.
# These are checked against the full command string.
SHELL_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-[^\s]*r[^\s]*f\s+/\s*$"),  # rm -rf /
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+.*of=/dev/"),
    re.compile(r"\b:()\s*\{"),  # fork bomb
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\binit\s+0\b"),
    re.compile(r"\bsudo\s+rm\b"),
]

# Env var suffixes that indicate secrets. More precise than substring matching.
_SECRET_SUFFIXES = (
    "_KEY",
    "_SECRET",
    "_TOKEN",
    "_PASSWORD",
    "_CREDENTIAL",
    "_API_KEY",
    "_ACCESS_KEY",
    "_SECRET_KEY",
)
# Exact env var names that are secrets
_SECRET_EXACT = {
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "PGPASSWORD",
    "MYSQL_PWD",
}


def _filter_env() -> dict[str, str]:
    """Build a filtered environment for child processes."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in _SECRET_EXACT
        and not any(k.upper().endswith(suffix) for suffix in _SECRET_SUFFIXES)
    }


def _check_shell_command(command: str) -> str | None:
    """Check if a shell command matches the deny-list.

    Returns None if allowed, or an error message if blocked.
    """
    for pattern in SHELL_DENY_PATTERNS:
        if pattern.search(command):
            return f"Error: Command blocked by safety filter. Pattern matched: {pattern.pattern}"
    return None


def _make_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Any,
) -> Tool:
    """Helper to create a Tool with an execute handler."""
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        execute=handler,
    )


# ------------------------------------------------------------------ #
# read_file
# ------------------------------------------------------------------ #


async def _read_file(
    path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read a file with optional line offset and limit."""
    file_path = Path(path).expanduser().resolve()

    # Security: path confinement check (skip for non-local environments
    # where the container IS the sandbox)
    if isinstance(_environment, LocalEnvironment):
        error = _check_path_allowed(file_path)
        if error:
            raise PermissionError(error)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

    text = await _environment.read_file(str(file_path))

    lines = text.split("\n")
    total = len(lines)

    # Apply offset and limit
    start = max(0, offset)
    end = start + limit
    selected = lines[start:end]

    # Format with line numbers (1-indexed)
    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:6d}\t{line}")

    result = "\n".join(numbered)
    if end < total:
        result += f"\n\n[{total - end} more lines not shown. Use offset={end} to continue.]"

    return result


READ_FILE = _make_tool(
    name="read_file",
    description=(
        "Read a file's contents with line numbers. Supports offset and limit for large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line offset (0-indexed). Default: 0",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read. Default: 2000",
                "default": 2000,
            },
        },
        "required": ["path"],
    },
    handler=_read_file,
)


# ------------------------------------------------------------------ #
# write_file
# ------------------------------------------------------------------ #


async def _write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories as needed."""
    file_path = Path(path).expanduser().resolve()

    # Security: path confinement (skip for non-local -- container IS sandbox)
    if isinstance(_environment, LocalEnvironment):
        error = _check_path_allowed(file_path)
        if error:
            raise PermissionError(error)

    await _environment.write_file(str(file_path), content)

    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return f"Wrote {len(content)} bytes ({line_count} lines) to {path}"


WRITE_FILE = _make_tool(
    name="write_file",
    description=(
        "Write content to a file. Creates parent directories if needed. Overwrites existing files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to write to",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    },
    handler=_write_file,
)


# ------------------------------------------------------------------ #
# edit_file
# ------------------------------------------------------------------ #


async def _edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Replace a string in a file."""
    file_path = Path(path).expanduser().resolve()

    # Security: path confinement (skip for non-local -- container IS sandbox)
    if isinstance(_environment, LocalEnvironment):
        error = _check_path_allowed(file_path)
        if error:
            raise PermissionError(error)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    text = await _environment.read_file(str(file_path))

    if old_string not in text:
        raise ValueError(f"old_string not found in {path}")

    if not replace_all:
        count = text.count(old_string)
        if count > 1:
            raise ValueError(
                f"old_string appears {count} times in {path}. "
                f"Provide more context to make it unique, "
                f"or set replace_all=true."
            )
        text = text.replace(old_string, new_string, 1)
    else:
        text = text.replace(old_string, new_string)

    await _environment.write_file(str(file_path), text)

    label = "all occurrences" if replace_all else "1 occurrence"
    return f"Edited {path}: replaced {label}"


EDIT_FILE = _make_tool(
    name="edit_file",
    description=(
        "Replace a specific string in a file. The old_string "
        "must be unique unless replace_all is true."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences. Default: false",
                "default": False,
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
    handler=_edit_file,
)


# ------------------------------------------------------------------ #
# shell
# ------------------------------------------------------------------ #

DEFAULT_SHELL_TIMEOUT = 10


async def _shell(
    command: str,
    timeout: int = DEFAULT_SHELL_TIMEOUT,
    timeout_ms: int | None = None,
    working_dir: str | None = None,
) -> str:
    """Execute a shell command and return stdout + stderr.

    Delegates to the execution environment (local or Docker).

    Args:
        command: Shell command to execute.
        timeout: Timeout in seconds (default 10). Kept for backward compat.
        timeout_ms: Timeout in milliseconds (Spec §4.2). Takes precedence
            over ``timeout`` when provided; converted to seconds internally.
        working_dir: Working directory for the command.
    """
    # timeout_ms takes precedence (Spec §4.2), converted ms→s with ceiling
    effective_timeout: int = max(1, -(-timeout_ms // 1000)) if timeout_ms is not None else timeout

    # Clamp to max_command_timeout (Spec §2.2)
    effective_timeout = min(effective_timeout, _max_command_timeout_s)

    # Security: check deny-list
    blocked = _check_shell_command(command)
    if blocked:
        raise PermissionError(blocked)

    # Security: validate working_dir is within allowed roots
    cwd = working_dir or os.getcwd()
    cwd_path = Path(cwd).expanduser().resolve()
    path_error = _check_path_allowed(cwd_path)
    if path_error:
        raise PermissionError(f"Shell working_dir outside allowed roots: {path_error}")
    filtered_env = _filter_env()

    result = await _environment.exec_shell(
        command,
        timeout=effective_timeout,
        working_dir=cwd,
        env=filtered_env,
    )

    if result.returncode == -1 and "timed out" in result.stderr:
        raise RuntimeError(f"Command timed out after {effective_timeout}s: {command}")

    return result.output


SHELL = _make_tool(
    name="shell",
    description=(
        "Execute a shell command via bash. Returns stdout, stderr, "
        "and exit code. Subject to timeout (default 10s) and "
        "command safety filtering."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 10",
                "default": DEFAULT_SHELL_TIMEOUT,
            },
            "timeout_ms": {
                "type": "integer",
                "description": (
                    "Timeout in milliseconds (Spec §4.2). Takes precedence over timeout."
                ),
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory. Default: cwd",
            },
        },
        "required": ["command"],
    },
    handler=_shell,
)


# ------------------------------------------------------------------ #
# grep
# ------------------------------------------------------------------ #

# Directories to skip in grep and glob
_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
    }
)


async def _grep(
    pattern: str,
    path: str = ".",
    include: str | None = None,
    max_results: int = 200,
) -> str:
    """Search file contents with regex patterns.

    Routes through _environment for Docker compatibility (spec S4.1).
    Local mode uses fast host-side Python implementation.
    Non-local mode uses grep/egrep via exec_shell inside the container.
    """
    # For non-local environments, delegate to exec_shell with grep
    if not isinstance(_environment, LocalEnvironment):
        import shlex

        cmd = f"grep -rn -E {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd += f" --include={shlex.quote(include)}"
        cmd += f" | head -{max_results}"
        result = await _environment.exec_shell(cmd, timeout=30)
        if result.returncode == 1:
            return f"No matches for '{pattern}'"
        if result.returncode != 0:
            return f"grep error: {result.stderr}"
        return result.stdout.strip() or f"No matches for '{pattern}'"

    # Local mode: fast host-side implementation.
    # Intentional design (§9.4.1): grep/glob/list_dir run directly on the
    # host filesystem for performance, bypassing ExecutionEnvironment.exec_shell.
    # The same is true for _glob() and _list_dir().  Path confinement via
    # _check_path_allowed() is enforced below as the security boundary.
    search_path = Path(path).expanduser().resolve()

    # Security: path confinement (mirrors _read_file / _list_dir)
    path_error = _check_path_allowed(search_path)
    if path_error:
        raise PermissionError(path_error)

    if not search_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}") from e

    results: list[str] = []
    files_searched = 0

    if search_path.is_file():
        files_to_search = [search_path]
    else:
        files_to_search = sorted(search_path.rglob("*"))

    for file_path in files_to_search:
        if not file_path.is_file():
            continue

        if _SKIP_DIRS & set(file_path.parts):
            continue

        if include and not fnmatch.fnmatch(file_path.name, include):
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        files_searched += 1
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if regex.search(line):
                rel = file_path.relative_to(search_path) if search_path.is_dir() else file_path.name
                results.append(f"{rel}:{i + 1}:{line.rstrip()}")

                if len(results) >= max_results:
                    results.append(f"\n[... truncated at {max_results}]")
                    return "\n".join(results)

    if not results:
        return f"No matches for '{pattern}' in {files_searched} files"

    return "\n".join(results)


GREP = _make_tool(
    name="grep",
    description=("Search file contents with regex patterns. Searches recursively in directories."),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search. Default: cwd",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Glob filter (e.g., '*.py')",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results. Default: 200",
                "default": 200,
            },
        },
        "required": ["pattern"],
    },
    handler=_grep,
)


# ------------------------------------------------------------------ #
# glob
# ------------------------------------------------------------------ #


async def _glob(
    pattern: str,
    path: str = ".",
    max_results: int = 500,
) -> str:
    """Find files matching a glob pattern.

    Routes through _environment for Docker compatibility (spec S4.1).
    Local mode uses fast host-side Path.glob().
    Non-local mode delegates to _environment.glob().
    """
    # For non-local environments, delegate to _environment.glob()
    if not isinstance(_environment, LocalEnvironment):
        results = await _environment.glob(pattern, path)
        if not results:
            return f"No files matching '{pattern}' in {path}"
        output = "\n".join(results[:max_results])
        if len(results) > max_results:
            output += f"\n\n[... truncated at {max_results}]"
        return output

    # Local mode: fast host-side implementation
    search_path = Path(path).expanduser().resolve()

    if not search_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    results_local: list[str] = []

    for match in sorted(search_path.glob(pattern)):
        if _SKIP_DIRS & set(match.parts):
            continue

        rel = match.relative_to(search_path)
        suffix = "/" if match.is_dir() else ""
        results_local.append(f"{rel}{suffix}")

        if len(results_local) >= max_results:
            results_local.append(f"\n[... truncated at {max_results}]")
            break

    if not results_local:
        return f"No files matching '{pattern}' in {path}"

    return "\n".join(results_local)


GLOB = _make_tool(
    name="glob",
    description=(
        "Find files matching a glob pattern. Returns paths relative to the search directory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py')",
            },
            "path": {
                "type": "string",
                "description": "Base directory. Default: cwd",
                "default": ".",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results. Default: 500",
                "default": 500,
            },
        },
        "required": ["pattern"],
    },
    handler=_glob,
)


# ------------------------------------------------------------------ #
# list_dir
# ------------------------------------------------------------------ #


async def _list_dir(path: str = ".", depth: int = 1) -> str:
    """List directory contents with optional depth control.

    Depth semantics:
        0 = immediate children only (no recursion into subdirectories)
        1 = one level of subdirectories expanded (default)
        N = N levels deep (capped at 5)
    """
    depth = min(depth, 5)  # Cap to prevent enormous output
    dir_path = Path(path).expanduser().resolve()

    # Security: path confinement (skip for non-local -- container IS sandbox)
    if isinstance(_environment, LocalEnvironment):
        error = _check_path_allowed(dir_path)
        if error:
            return error
        if not dir_path.exists():
            return f"Error: Path not found: {path}"
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"
    else:
        # Non-local: delegate to environment abstraction (flat listing only)
        entries = await _environment.list_dir(str(dir_path))
        if not entries:
            return f"{path}/\n  (empty)"
        formatted = "\n".join(f"  {e}" for e in entries)
        return f"{path}/\n{formatted}"

    lines: list[str] = [f"{path}/"]

    def _walk(p: Path, current_depth: int, indent: str) -> None:
        try:
            # Directories first, then files, each group sorted by name
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            lines.append(f"{indent}[permission denied]")
            return
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                if current_depth < depth:
                    _walk(entry, current_depth + 1, indent + "  ")
            else:
                lines.append(f"{indent}{entry.name}")

    _walk(dir_path, 0, "  ")

    if len(lines) == 1:
        lines.append("  (empty)")

    return "\n".join(lines)


LIST_DIR = _make_tool(
    name="list_dir",
    description="List directory contents with optional depth control.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory.",
            },
            "depth": {
                "type": "integer",
                "description": (
                    "Recursion depth. 0=immediate children only, "
                    "1=one level deep (default). Max: 5."
                ),
                "default": 1,
            },
        },
    },
    handler=_list_dir,
)


# ------------------------------------------------------------------ #
# read_many_files
# ------------------------------------------------------------------ #


async def _read_many_files(paths: list[str]) -> str:
    """Read multiple files in a single call, returning concatenated content."""
    if not paths:
        return "(no files requested)"

    parts: list[str] = []
    for path in paths:
        header = f"=== file: {path} ==="
        file_path = Path(path).expanduser().resolve()

        # Security + existence checks for local environments
        if isinstance(_environment, LocalEnvironment):
            error = _check_path_allowed(file_path)
            if error:
                parts.append(f"{header}\n{error}\n")
                continue
            if not file_path.exists():
                parts.append(f"{header}\nError: File not found: {path}\n")
                continue
            if not file_path.is_file():
                parts.append(f"{header}\nError: Not a file: {path}\n")
                continue

        try:
            text = await _environment.read_file(str(file_path))
        except (FileNotFoundError, OSError) as e:
            parts.append(f"{header}\nError: {e}\n")
            continue

        lines = text.split("\n")
        numbered = [f"{i:6d}\t{line}" for i, line in enumerate(lines, start=1)]
        content = "\n".join(numbered)
        parts.append(f"{header}\n{content}\n")

    return "\n".join(parts)


READ_MANY_FILES = _make_tool(
    name="read_many_files",
    description=(
        "Read multiple files in a single call. Returns concatenated content with file headers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to read.",
            }
        },
        "required": ["paths"],
    },
    handler=_read_many_files,
)


# ------------------------------------------------------------------ #
# apply_patch (unified diff format)
# ------------------------------------------------------------------ #


async def _apply_patch(patch: str, working_dir: str | None = None) -> str:
    """Apply a unified diff patch."""
    from attractor_agent.tools.apply_patch import _apply_patch_execute

    return await _apply_patch_execute(patch, working_dir)


APPLY_PATCH = Tool(
    name="apply_patch",
    description=(
        "Apply a unified diff patch to files. Accepts standard unified "
        "diff format (--- a/file, +++ b/file, @@ hunks). Supports "
        "file creation, deletion, and modification. Use for making "
        "code changes via patches instead of edit_file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "Unified diff text to apply",
            },
            "working_dir": {
                "type": "string",
                "description": "Base directory for paths. Default: cwd",
            },
        },
        "required": ["patch"],
    },
    handler=_apply_patch,
)


# ------------------------------------------------------------------ #
# All core tools as a list
# ------------------------------------------------------------------ #

ALL_CORE_TOOLS: list[Tool] = [
    READ_FILE,
    WRITE_FILE,
    EDIT_FILE,
    SHELL,
    GREP,
    GLOB,
]

# Extended tools list including apply_patch (used by OpenAI profile)
ALL_TOOLS_WITH_PATCH: list[Tool] = [
    READ_FILE,
    WRITE_FILE,
    EDIT_FILE,
    APPLY_PATCH,
    SHELL,
    GREP,
    GLOB,
]
