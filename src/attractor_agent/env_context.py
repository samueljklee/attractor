"""Dynamic environment context for system prompts.

Builds the ``<environment>`` block and git context snapshot that gets
injected into every system prompt.  Gives the LLM awareness of its
runtime context: working directory, platform, git state, date, model.

Spec reference: coding-agent-loop-spec S6.3-6.4.
"""

from __future__ import annotations

import datetime
import platform as platform_mod
import subprocess
import sys


def build_environment_context(
    *,
    working_dir: str,
    model: str,
    platform: str | None = None,
    os_version: str | None = None,
    today: str | None = None,
    knowledge_cutoff: str | None = None,
    git_info: dict[str, object] | None = None,
) -> str:
    """Build the ``<environment>`` block for the system prompt.

    Args:
        working_dir: Current working directory path.
        model: Model display name (e.g., ``"claude-sonnet-4-5"``).
        platform: Platform string.  Auto-detected from ``sys.platform`` if *None*.
        os_version: OS version string.  Auto-detected if *None*.
        today: Today's date as ``YYYY-MM-DD``.  Auto-generated if *None*.
        knowledge_cutoff: Model knowledge cutoff date.  Omitted if *None*.
        git_info: Git context from :func:`get_git_context`.  Auto-fetched if *None*.

    Returns:
        Formatted ``<environment>`` block followed by an optional
        ``<git_status>`` snapshot.
    """
    if platform is None:
        platform = sys.platform
    if os_version is None:
        os_version = platform_mod.platform()
    if today is None:
        today = datetime.date.today().isoformat()
    if git_info is None:
        git_info = get_git_context(working_dir)

    is_git = bool(git_info.get("is_git", False))
    branch = str(git_info.get("branch", ""))

    lines: list[str] = [
        "<environment>",
        f"Working directory: {working_dir}",
        f"Is git repository: {'true' if is_git else 'false'}",
    ]
    if is_git and branch:
        lines.append(f"Git branch: {branch}")
    lines.extend(
        [
            f"Platform: {platform}",
            f"OS version: {os_version}",
            f"Today's date: {today}",
            f"Model: {model}",
        ]
    )
    if knowledge_cutoff:
        lines.append(f"Knowledge cutoff: {knowledge_cutoff}")
    lines.append("</environment>")

    result = "\n".join(lines)

    # Append git context snapshot if inside a repository
    git_snapshot = _build_git_snapshot(git_info)
    if git_snapshot:
        result += "\n\n" + git_snapshot

    return result


# ------------------------------------------------------------------ #
# Git helpers
# ------------------------------------------------------------------ #


def get_git_context(working_dir: str) -> dict[str, object]:
    """Collect git context for *working_dir*.

    Returns a dict with keys:

    * ``is_git`` -- *bool*
    * ``branch`` -- current branch name
    * ``modified_count`` -- number of tracked-modified files
    * ``untracked_count`` -- number of untracked files
    * ``recent_commits`` -- up to 5 one-line commit summaries
    """
    result: dict[str, object] = {
        "is_git": False,
        "git_root": "",
        "branch": "",
        "modified_count": 0,
        "untracked_count": 0,
        "recent_commits": [],
    }

    try:
        cp = _run_git("rev-parse", "--is-inside-work-tree", cwd=working_dir)
        if cp.returncode != 0:
            return result
        result["is_git"] = True

        # Git root directory
        cp = _run_git("rev-parse", "--show-toplevel", cwd=working_dir)
        if cp.returncode == 0:
            result["git_root"] = cp.stdout.strip()

        # Current branch
        cp = _run_git("branch", "--show-current", cwd=working_dir)
        if cp.returncode == 0:
            result["branch"] = cp.stdout.strip()

        # Status counts
        cp = _run_git("status", "--porcelain", cwd=working_dir)
        if cp.returncode == 0:
            modified = 0
            untracked = 0
            for line in cp.stdout.strip().splitlines():
                if line.startswith("??"):
                    untracked += 1
                elif line.strip():
                    modified += 1
            result["modified_count"] = modified
            result["untracked_count"] = untracked

        # Recent commits (up to 5)
        cp = _run_git("log", "--oneline", "-5", cwd=working_dir)
        if cp.returncode == 0 and cp.stdout.strip():
            result["recent_commits"] = cp.stdout.strip().splitlines()

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Git unavailable -- return safe defaults
        pass

    return result


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _run_git(*args: str, cwd: str) -> subprocess.CompletedProcess[str]:
    """Run a git sub-command with standard options."""
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _build_git_snapshot(git_info: dict[str, object]) -> str:
    """Format the ``<git_status>`` section from *git_info*."""
    if not git_info.get("is_git"):
        return ""

    parts: list[str] = ["<git_status>"]

    branch = git_info.get("branch", "")
    if branch:
        parts.append(f"Branch: {branch}")

    parts.append(f"Modified files: {git_info.get('modified_count', 0)}")
    parts.append(f"Untracked files: {git_info.get('untracked_count', 0)}")

    commits = git_info.get("recent_commits", [])
    if commits and isinstance(commits, list):
        parts.append("Recent commits:")
        for commit in commits:
            parts.append(f"  {commit}")

    parts.append("</git_status>")
    return "\n".join(parts)
