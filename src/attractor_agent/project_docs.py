"""Project documentation discovery for system prompts.

Walks from the git root (or *cwd* when not inside a repository) to the
current working directory, collecting recognised instruction files and
filtering them by the active provider profile.

Recognised files
~~~~~~~~~~~~~~~~

* ``AGENTS.md`` -- universal, always loaded regardless of provider.
* ``CLAUDE.md`` -- loaded when provider is ``anthropic``.
* ``GEMINI.md`` -- loaded when provider is ``gemini``.
* ``.codex/instructions.md`` -- loaded when provider is ``openai``.

Loading rules
~~~~~~~~~~~~~

* Root-level files are loaded first.
* Subdirectory files are appended (deeper = higher precedence).
* Total byte budget: **32 KB**.  If exceeded the content is truncated and
  a ``[Project instructions truncated at 32KB]`` marker is appended.

Spec reference: coding-agent-loop-spec S6.5.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_MAX_BYTES: int = 32 * 1024  # 32 KB
_TRUNCATION_MARKER: str = "[Project instructions truncated at 32KB]"

# AGENTS.md is loaded for every provider.
_UNIVERSAL_FILES: list[str] = ["AGENTS.md"]

# Provider-specific instruction files.
_PROVIDER_FILES: dict[str, list[str]] = {
    "anthropic": ["CLAUDE.md"],
    "openai": [".codex/instructions.md"],
    "gemini": ["GEMINI.md"],
}


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def discover_project_docs(
    working_dir: str,
    provider_id: str | None = None,
    git_root: str | None = None,
) -> str:
    """Discover and load project documentation files.

    Args:
        working_dir: Current working directory.
        provider_id: Provider name (``"anthropic"``, ``"openai"``,
            ``"gemini"``).  When *None* only ``AGENTS.md`` files are
            collected.
        git_root: Pre-resolved git root directory.  When provided the
            function skips calling ``_find_git_root()`` (avoids a
            redundant subprocess).

    Returns:
        Concatenated content wrapped in ``<project_instructions>`` tags,
        or an empty string when no files are found.
    """
    root = git_root or _find_git_root(working_dir) or working_dir
    root_path = Path(root).resolve()
    cwd_path = Path(working_dir).resolve()

    dirs_to_scan = _walk_path(root_path, cwd_path)

    # Which filenames to look for
    target_files = list(_UNIVERSAL_FILES)
    if provider_id and provider_id in _PROVIDER_FILES:
        target_files.extend(_PROVIDER_FILES[provider_id])

    # Collect in order: root first, deeper dirs appended
    collected: list[tuple[str, str]] = []  # (display_path, content)
    total_bytes = 0
    truncated = False

    for scan_dir in dirs_to_scan:
        for filename in target_files:
            filepath = scan_dir / filename
            if not filepath.is_file():
                continue
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            content_bytes = len(content.encode("utf-8"))

            if total_bytes + content_bytes > _MAX_BYTES:
                remaining = _MAX_BYTES - total_bytes
                if remaining > 0:
                    content = content.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
                    try:
                        display = str(filepath.relative_to(root_path))
                    except ValueError:
                        display = str(filepath)
                    collected.append((display, content))
                truncated = True
                break

            try:
                display = str(filepath.relative_to(root_path))
            except ValueError:
                display = str(filepath)
            collected.append((display, content))
            total_bytes += content_bytes

        if truncated:
            break

    if not collected:
        return ""

    # Format
    parts: list[str] = ["<project_instructions>"]
    for display_path, content in collected:
        parts.append(f"# From {display_path}")
        parts.append(content.strip())
        parts.append("")

    if truncated:
        parts.append(_TRUNCATION_MARKER)

    parts.append("</project_instructions>")
    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _find_git_root(working_dir: str) -> str | None:
    """Return the git repository root directory, or *None*."""
    try:
        cp = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _walk_path(root: Path, target: Path) -> list[Path]:
    """Build the list of directories from *root* to *target* (inclusive).

    Returns ``[root, ..., target]``.  If *target* is not a descendant of
    *root*, returns ``[target]`` as a fallback.
    """
    dirs: list[Path] = [root]

    if root == target:
        return dirs

    try:
        relative = target.relative_to(root)
    except ValueError:
        return [target]

    current = root
    for part in relative.parts:
        current = current / part
        dirs.append(current)

    return dirs
