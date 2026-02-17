"""Tool output truncation engine.

Implements the two-pass truncation strategy from coding-agent-loop §5.1-5.3:
1. Character-based truncation (head/tail split)
2. Line-based truncation (keep first N + last M lines)

The goal is to keep tool output within token budgets while preserving
the most useful information (beginning and end of output).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TruncationLimits:
    """Per-tool truncation limits. Spec §5.2."""

    max_chars: int = 30_000
    max_lines: int = 500
    head_ratio: float = 0.7  # proportion of budget for head vs tail

    # Common presets per tool type
    @classmethod
    def for_tool(
        cls,
        tool_name: str,
        output_limits: dict[str, int] | None = None,
        line_limits: dict[str, int] | None = None,
    ) -> TruncationLimits:
        """Get truncation limits for a specific tool.

        Config overrides (output_limits, line_limits) take precedence
        over built-in presets.  Spec §5.2, §5.3.
        """
        presets: dict[str, TruncationLimits] = {
            "read_file": cls(max_chars=50_000, max_lines=1000),
            "shell": cls(max_chars=30_000, max_lines=256),
            "grep": cls(max_chars=20_000, max_lines=200),
            "glob": cls(max_chars=20_000, max_lines=500),
            "write_file": cls(max_chars=1_000, max_lines=50),
            "edit_file": cls(max_chars=10_000, max_lines=50),
            "apply_patch": cls(max_chars=10_000, max_lines=200),
            "spawn_agent": cls(max_chars=20_000, max_lines=500),
        }
        preset = presets.get(tool_name, cls())

        # Apply per-tool config overrides
        max_chars = (
            output_limits[tool_name]
            if output_limits and tool_name in output_limits
            else preset.max_chars
        )
        max_lines = (
            line_limits[tool_name] if line_limits and tool_name in line_limits else preset.max_lines
        )

        if max_chars != preset.max_chars or max_lines != preset.max_lines:
            return cls(max_chars=max_chars, max_lines=max_lines, head_ratio=preset.head_ratio)
        return preset


def truncate_output(
    output: str,
    limits: TruncationLimits | None = None,
) -> tuple[str, bool]:
    """Truncate tool output using two-pass strategy.

    Pass 1: Character-based truncation (head + tail split)
    Pass 2: Line-based truncation (keep first N + last M lines)

    Args:
        output: Raw tool output string.
        limits: Truncation configuration. Uses defaults if None.

    Returns:
        Tuple of (truncated_output, was_truncated).
    """
    if not output:
        return output, False

    if limits is None:
        limits = TruncationLimits()

    truncated = False

    # Pass 1: Character-based truncation
    if len(output) > limits.max_chars:
        head_size = int(limits.max_chars * limits.head_ratio)
        tail_size = limits.max_chars - head_size
        head = output[:head_size]
        tail = output[-tail_size:] if tail_size > 0 else ""
        omitted = len(output) - head_size - tail_size
        output = (
            f"{head}\n\n[WARNING: Tool output was truncated. {omitted:,} characters "
            f"were removed from the middle. The full output is available in the "
            f"event stream. If you need to see specific parts, re-run the tool "
            f"with more targeted parameters.]\n\n{tail}"
        )
        truncated = True

    # Pass 2: Line-based truncation
    lines = output.split("\n")
    if len(lines) > limits.max_lines:
        head_lines = int(limits.max_lines * limits.head_ratio)
        tail_lines = limits.max_lines - head_lines
        head = lines[:head_lines]
        tail = lines[-tail_lines:] if tail_lines > 0 else []
        omitted_lines = len(lines) - head_lines - tail_lines
        output = (
            "\n".join(head) + f"\n\n[WARNING: Tool output was truncated. {omitted_lines:,} lines "
            f"were removed from the middle. The full output is available in the "
            f"event stream. If you need to see specific parts, re-run the tool "
            f"with more targeted parameters.]\n\n" + "\n".join(tail)
        )
        truncated = True

    return output, truncated
