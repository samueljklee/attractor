"""apply_patch tool -- OpenAI's v4a unified diff format parser.

Parses and applies unified diff patches in the format that OpenAI
codex-rs models produce. This is the preferred way for OpenAI models
to make code changes (instead of edit_file).

The v4a format is a simplified unified diff:
    --- a/path/to/file.py
    +++ b/path/to/file.py
    @@ -start,count +start,count @@
    -old line
    +new line
     context line

Supports:
- Multiple file patches in one apply_patch call
- File creation (--- /dev/null)
- File deletion (+++ /dev/null)
- Multiple hunks per file

Spec reference: coding-agent-loop-spec Appendix A.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attractor_agent.tools.core import _check_path_allowed


@dataclass
class Hunk:
    """A single hunk in a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    """A patch for a single file."""

    old_path: str  # "a/path" or "/dev/null" for new files
    new_path: str  # "b/path" or "/dev/null" for deletions
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_creation(self) -> bool:
        return self.old_path == "/dev/null"

    @property
    def is_deletion(self) -> bool:
        return self.new_path == "/dev/null"

    @property
    def target_path(self) -> str:
        """The actual file path (strips a/ or b/ prefix)."""
        if self.is_deletion:
            path = self.old_path
        else:
            path = self.new_path

        # Strip a/ or b/ prefix
        if path.startswith(("a/", "b/")):
            return path[2:]
        return path


@dataclass
class PatchSet:
    """A set of file patches parsed from a unified diff."""

    patches: list[FilePatch] = field(default_factory=list)


class PatchParseError(Exception):
    """Error parsing a unified diff."""


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #

_DIFF_HEADER = re.compile(r"^---\s+(.+)$")
_DIFF_HEADER_NEW = re.compile(r"^\+\+\+\s+(.+)$")
_HUNK_HEADER = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def parse_patch(patch_text: str) -> PatchSet:
    """Parse a unified diff string into a PatchSet.

    Args:
        patch_text: Unified diff text (one or more file patches).

    Returns:
        PatchSet with parsed file patches and hunks.

    Raises:
        PatchParseError: On invalid diff syntax.
    """
    patches: list[FilePatch] = []
    lines = patch_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Look for --- header
        m_old = _DIFF_HEADER.match(line)
        if m_old:
            old_path = m_old.group(1).strip()

            # Next line should be +++ header
            i += 1
            if i >= len(lines):
                raise PatchParseError(f"Expected +++ header after --- at line {i}")

            m_new = _DIFF_HEADER_NEW.match(lines[i])
            if not m_new:
                raise PatchParseError(f"Expected +++ header, got: {lines[i]!r} at line {i + 1}")
            new_path = m_new.group(1).strip()

            # Parse hunks
            file_patch = FilePatch(old_path=old_path, new_path=new_path)
            i += 1

            while i < len(lines):
                line = lines[i]

                # Check for next file or end
                if _DIFF_HEADER.match(line):
                    break

                m_hunk = _HUNK_HEADER.match(line)
                if m_hunk:
                    hunk = Hunk(
                        old_start=int(m_hunk.group(1)),
                        old_count=int(m_hunk.group(2) or "1"),
                        new_start=int(m_hunk.group(3)),
                        new_count=int(m_hunk.group(4) or "1"),
                    )

                    i += 1
                    while i < len(lines):
                        hunk_line = lines[i]
                        if _HUNK_HEADER.match(hunk_line) or _DIFF_HEADER.match(hunk_line):
                            break
                        if hunk_line == "" and i == len(lines) - 1:
                            # Trailing empty line
                            i += 1
                            break
                        hunk.lines.append(hunk_line)
                        i += 1

                    file_patch.hunks.append(hunk)
                else:
                    i += 1

            patches.append(file_patch)
        else:
            i += 1

    return PatchSet(patches=patches)


# ------------------------------------------------------------------ #
# Applicator
# ------------------------------------------------------------------ #


def apply_patch_to_file(
    file_path: Path,
    patch: FilePatch,
) -> str:
    """Apply a single file patch to disk.

    Returns:
        Description of what was done.

    Raises:
        FileNotFoundError: If patching an existing file that doesn't exist.
        PatchParseError: If the patch doesn't match the file content.
        PermissionError: If the path is outside allowed roots.
    """
    target = file_path / patch.target_path
    resolved = target.resolve()

    # Security: path confinement
    path_error = _check_path_allowed(resolved)
    if path_error:
        raise PermissionError(path_error)

    # Security: reject symlink traversal -- resolved path must be under file_path
    base_resolved = file_path.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise PermissionError(
            f"Path traversal detected: {patch.target_path} resolves outside base directory"
        ) from None

    if patch.is_creation:
        # Create new file
        resolved.parent.mkdir(parents=True, exist_ok=True)
        content = _build_new_content(patch)
        resolved.write_text(content, encoding="utf-8")
        return f"Created {patch.target_path}"

    if patch.is_deletion:
        # Delete file
        if resolved.exists():
            resolved.unlink()
            return f"Deleted {patch.target_path}"
        return f"Already deleted: {patch.target_path}"

    # Modify existing file
    if not resolved.exists():
        raise FileNotFoundError(f"Cannot patch: {patch.target_path} does not exist")

    original = resolved.read_text(encoding="utf-8")
    patched = _apply_hunks(original, patch.hunks)
    resolved.write_text(patched, encoding="utf-8")
    return f"Patched {patch.target_path} ({len(patch.hunks)} hunk(s))"


def _build_new_content(patch: FilePatch) -> str:
    """Build file content from a creation patch (all + lines)."""
    lines: list[str] = []
    for hunk in patch.hunks:
        for line in hunk.lines:
            if line.startswith("+"):
                lines.append(line[1:])
            elif line.startswith(" "):
                lines.append(line[1:])
    return "\n".join(lines) + "\n" if lines else ""


def _apply_hunks(original: str, hunks: list[Hunk]) -> str:
    """Apply hunks to original content, producing patched content.

    Hunks are applied in reverse order (bottom-up) to preserve line
    numbers for subsequent hunks. Context and removal lines are verified
    against actual file content before applying -- mismatches raise
    PatchParseError to prevent silent corruption from stale patches.
    """
    # Snapshot original for verification (immutable reference).
    # Splicing mutates result_lines, so verifying against it would
    # produce false mismatches on adjacent hunks.
    original_lines = original.split("\n")
    result_lines = list(original_lines)

    # Validate hunks don't overlap (prevents silent corruption from
    # malformed LLM-generated patches). Sort by old_start descending
    # and verify no overlap between adjacent pairs.
    sorted_hunks = sorted(hunks, key=lambda h: h.old_start)
    for i in range(len(sorted_hunks) - 1):
        curr_end = sorted_hunks[i].old_start + sorted_hunks[i].old_count
        next_start = sorted_hunks[i + 1].old_start
        if curr_end > next_start:
            raise PatchParseError(
                f"Overlapping hunks: hunk at line {sorted_hunks[i].old_start} "
                f"(count {sorted_hunks[i].old_count}) overlaps with hunk at "
                f"line {sorted_hunks[i + 1].old_start}"
            )

    # Apply hunks in reverse order to preserve line numbers
    for hunk in reversed(sorted_hunks):
        old_start = hunk.old_start - 1  # 0-indexed
        old_end = old_start + hunk.old_count

        # Verify context and removal lines match ORIGINAL content
        file_idx = old_start
        for line in hunk.lines:
            if line.startswith(" ") or line.startswith("-"):
                expected = line[1:]
                if file_idx >= len(original_lines):
                    raise PatchParseError(
                        f"Hunk at line {hunk.old_start}: file has "
                        f"{len(original_lines)} lines but hunk references "
                        f"line {file_idx + 1}"
                    )
                actual = original_lines[file_idx]
                if actual != expected:
                    raise PatchParseError(
                        f"Hunk at line {hunk.old_start}: context mismatch "
                        f"at file line {file_idx + 1}.\n"
                        f"  Expected: {expected!r}\n"
                        f"  Actual:   {actual!r}"
                    )
                file_idx += 1
            elif line.startswith("+"):
                pass  # added line, doesn't consume file lines
            elif line.startswith("\\"):
                pass  # "\ No newline at end of file"

        # Build replacement lines
        new_lines: list[str] = []
        for line in hunk.lines:
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith(" "):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                pass  # removed line
            elif line.startswith("\\"):
                pass  # "\ No newline at end of file"

        result_lines[old_start:old_end] = new_lines

    return "\n".join(result_lines)


# ------------------------------------------------------------------ #
# Tool interface
# ------------------------------------------------------------------ #

_APPLY_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "string",
            "description": "Unified diff text to apply",
        },
    },
    "required": ["patch"],
}


async def _apply_patch_execute(
    patch: str,
    working_dir: str | None = None,
) -> str:
    """Execute the apply_patch tool."""
    import os

    cwd = Path(working_dir or os.getcwd())

    try:
        patch_set = parse_patch(patch)
    except PatchParseError as e:
        raise ValueError(f"Invalid patch format: {e}") from e

    if not patch_set.patches:
        return "No patches found in input"

    results: list[str] = []
    for file_patch in patch_set.patches:
        result = apply_patch_to_file(cwd, file_patch)
        results.append(result)

    return "\n".join(results)
