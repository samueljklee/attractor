"""Tests for P15: Gemini profile tools – list_dir and read_many_files.

Covers:
- _list_dir: directory listing with depth control, error handling, path confinement
- _read_many_files: batch file reading with headers, graceful error handling, path confinement
- GeminiProfile.get_tools(): injects list_dir + read_many_files
- Other profiles (Anthropic, OpenAI) do NOT get these tools injected
"""

from __future__ import annotations

import os

import pytest

from attractor_agent.tools.core import (
    LIST_DIR,
    READ_MANY_FILES,
    _list_dir,
    _read_many_files,
    set_allowed_roots,
)
from attractor_llm.types import Tool

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _allow_tmp_path(tmp_path):
    """Allow tmp_path in file tool roots so tests can use it freely."""
    set_allowed_roots([str(tmp_path)])
    yield
    set_allowed_roots([os.getcwd()])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_dummy_tool(name: str = "shell") -> Tool:
    """Minimal Tool stub for profile injection tests."""
    return Tool(
        name=name,
        description=f"Dummy {name} tool.",
        parameters={"type": "object", "properties": {}},
        handler=None,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------ #
# _list_dir tests
# ------------------------------------------------------------------ #


async def test_list_dir_returns_directory_contents(tmp_path):
    """Listing a directory returns the names of its children."""
    (tmp_path / "alpha.py").write_text("x")
    (tmp_path / "beta.txt").write_text("y")
    (tmp_path / "subdir").mkdir()

    result = await _list_dir(str(tmp_path), depth=0)

    assert "alpha.py" in result
    assert "beta.txt" in result
    assert "subdir/" in result


async def test_list_dir_respects_depth_zero(tmp_path):
    """depth=0 shows only immediate children; sub-dir contents are NOT shown."""
    subdir = tmp_path / "mydir"
    subdir.mkdir()
    (subdir / "hidden.py").write_text("x")
    (tmp_path / "top.py").write_text("y")

    result = await _list_dir(str(tmp_path), depth=0)

    assert "top.py" in result
    assert "mydir/" in result
    # The file inside the sub-dir must NOT appear
    assert "hidden.py" not in result


async def test_list_dir_respects_depth_one(tmp_path):
    """depth=1 shows one level of subdirectory contents."""
    subdir = tmp_path / "mydir"
    subdir.mkdir()
    (subdir / "inner.py").write_text("x")
    (tmp_path / "top.txt").write_text("y")

    result = await _list_dir(str(tmp_path), depth=1)

    assert "top.txt" in result
    assert "mydir/" in result
    assert "inner.py" in result


async def test_list_dir_depth_two_shows_two_levels(tmp_path):
    """depth=2 expands two levels of sub-directories."""
    level1 = tmp_path / "a"
    level1.mkdir()
    level2 = level1 / "b"
    level2.mkdir()
    (level2 / "deep.py").write_text("x")

    result = await _list_dir(str(tmp_path), depth=2)

    assert "deep.py" in result


async def test_list_dir_depth_two_does_not_show_three_levels(tmp_path):
    """depth=2 stops before the third level."""
    level1 = tmp_path / "a"
    level1.mkdir()
    level2 = level1 / "b"
    level2.mkdir()
    level3 = level2 / "c"
    level3.mkdir()
    (level3 / "too_deep.py").write_text("x")

    result = await _list_dir(str(tmp_path), depth=2)

    assert "too_deep.py" not in result


async def test_list_dir_nonexistent_path_returns_error(tmp_path):
    """Graceful error message for a path that does not exist."""
    # Use a path inside allowed roots so we test the "not found" path,
    # not the confinement path.
    result = await _list_dir(str(tmp_path / "does_not_exist"), depth=0)

    assert "Error" in result
    assert "not found" in result.lower()


async def test_list_dir_file_path_returns_error(tmp_path):
    """Passing a file path (not a directory) returns an error."""
    f = tmp_path / "file.txt"
    f.write_text("hello")

    result = await _list_dir(str(f), depth=0)

    assert "Error" in result
    assert "directory" in result.lower() or "not a directory" in result.lower()


async def test_list_dir_default_path_is_cwd(tmp_path, monkeypatch):
    """Calling _list_dir with no path argument uses the current directory."""
    (tmp_path / "sentinel_file.py").write_text("x")
    monkeypatch.chdir(tmp_path)

    result = await _list_dir()  # no path arg → defaults to "."

    assert "sentinel_file.py" in result


async def test_list_dir_empty_directory(tmp_path):
    """An empty directory returns a meaningful message."""
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    result = await _list_dir(str(empty), depth=0)

    # Should indicate emptiness rather than error
    assert "Error" not in result
    assert "empty" in result.lower() or result.strip().endswith("/")


async def test_list_dir_dirs_listed_with_trailing_slash(tmp_path):
    """Directories are marked with a trailing slash."""
    (tmp_path / "adir").mkdir()
    (tmp_path / "afile.py").write_text("x")

    result = await _list_dir(str(tmp_path), depth=0)

    assert "adir/" in result
    # Regular files must NOT get a trailing slash
    assert "afile.py/" not in result


# ------------------------------------------------------------------ #
# _list_dir path confinement tests
# ------------------------------------------------------------------ #


async def test_list_dir_rejects_path_outside_allowed_roots(tmp_path):
    """list_dir on a path outside allowed roots returns an error, not contents."""
    result = await _list_dir("/etc", depth=0)

    assert "Error" in result
    assert "outside allowed" in result.lower() or "allowed" in result.lower()


async def test_list_dir_allows_path_inside_allowed_roots(tmp_path):
    """list_dir within the allowed root works normally."""
    (tmp_path / "ok.txt").write_text("hi")

    result = await _list_dir(str(tmp_path), depth=0)

    assert "ok.txt" in result
    assert "Error" not in result


async def test_list_dir_caps_depth_at_five(tmp_path):
    """Depth is capped at 5 to prevent enormous output."""
    # Build a 7-level deep hierarchy
    current = tmp_path
    for i in range(7):
        current = current / f"level{i}"
        current.mkdir()
    (current / "bottom.txt").write_text("x")

    result = await _list_dir(str(tmp_path), depth=100)

    # level6/bottom.txt is at depth 7, beyond the cap of 5
    assert "bottom.txt" not in result


# ------------------------------------------------------------------ #
# LIST_DIR Tool constant
# ------------------------------------------------------------------ #


def test_list_dir_tool_constant_name():
    assert LIST_DIR.name == "list_dir"


def test_list_dir_tool_has_execute():
    assert LIST_DIR.execute is not None


# ------------------------------------------------------------------ #
# _read_many_files tests
# ------------------------------------------------------------------ #


async def test_read_many_files_returns_all_contents(tmp_path):
    """All three files are present in the output."""
    files = []
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text(f"content-{i}")
        files.append(str(f))

    result = await _read_many_files(files)

    for i in range(3):
        assert f"content-{i}" in result


async def test_read_many_files_includes_file_headers(tmp_path):
    """Each file section is preceded by '=== file: <path> ===' header."""
    f = tmp_path / "hello.py"
    f.write_text("print('hi')")

    result = await _read_many_files([str(f)])

    assert f"=== file: {f}" in result


async def test_read_many_files_missing_file_graceful(tmp_path):
    """When one file is missing the others are still returned."""
    good1 = tmp_path / "good1.txt"
    good1.write_text("good content 1")
    missing = str(tmp_path / "does_not_exist.txt")
    good2 = tmp_path / "good2.txt"
    good2.write_text("good content 2")

    result = await _read_many_files([str(good1), missing, str(good2)])

    assert "good content 1" in result
    assert "good content 2" in result
    # Missing file should produce an error message, not abort
    assert "Error" in result or "not found" in result.lower()


async def test_read_many_files_empty_list():
    """An empty paths list returns a sensible message (not a crash)."""
    result = await _read_many_files([])

    assert result  # non-empty string
    assert "no files" in result.lower() or result == "(no files requested)"


async def test_read_many_files_uses_line_numbers(tmp_path):
    """Content is line-numbered like read_file."""
    f = tmp_path / "numbered.txt"
    f.write_text("line one\nline two\nline three")

    result = await _read_many_files([str(f)])

    # Line numbers appear in the output (right-justified with tab separator)
    assert "1\t" in result or "     1\t" in result
    assert "line one" in result
    assert "line two" in result


async def test_read_many_files_directory_path_shows_error(tmp_path):
    """Passing a directory path returns a per-file error, not a crash."""
    result = await _read_many_files([str(tmp_path)])

    assert "Error" in result


async def test_read_many_files_multiple_headers_present(tmp_path):
    """When reading N files, N headers appear in the output."""
    paths = []
    for i in range(4):
        f = tmp_path / f"f{i}.txt"
        f.write_text(f"data {i}")
        paths.append(str(f))

    result = await _read_many_files(paths)

    header_count = result.count("=== file:")
    assert header_count == 4


# ------------------------------------------------------------------ #
# _read_many_files path confinement tests
# ------------------------------------------------------------------ #


async def test_read_many_files_rejects_path_outside_allowed_roots(tmp_path):
    """read_many_files on a path outside allowed roots returns a per-file error."""
    result = await _read_many_files(["/etc/passwd"])

    assert "Error" in result
    assert "outside allowed" in result.lower() or "allowed" in result.lower()


async def test_read_many_files_mixed_allowed_and_disallowed(tmp_path):
    """Allowed files succeed, disallowed files get per-file errors."""
    good = tmp_path / "good.txt"
    good.write_text("safe content")

    result = await _read_many_files([str(good), "/etc/passwd"])

    assert "safe content" in result
    assert "Error" in result  # /etc/passwd blocked


async def test_read_many_files_allows_path_inside_allowed_roots(tmp_path):
    """read_many_files within the allowed root works normally."""
    f = tmp_path / "ok.txt"
    f.write_text("allowed content")

    result = await _read_many_files([str(f)])

    assert "allowed content" in result
    assert "outside allowed" not in result.lower()


# ------------------------------------------------------------------ #
# READ_MANY_FILES Tool constant
# ------------------------------------------------------------------ #


def test_read_many_files_tool_constant_name():
    assert READ_MANY_FILES.name == "read_many_files"


def test_read_many_files_tool_has_required_paths():
    assert "paths" in READ_MANY_FILES.parameters.get("required", [])


def test_read_many_files_tool_has_execute():
    assert READ_MANY_FILES.execute is not None


# ------------------------------------------------------------------ #
# GeminiProfile.get_tools() injection tests
# ------------------------------------------------------------------ #


def test_gemini_profile_includes_list_dir():
    """GeminiProfile.get_tools() always appends list_dir."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    base = [_make_dummy_tool("shell")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "list_dir" in names


def test_gemini_profile_includes_read_many_files():
    """GeminiProfile.get_tools() always appends read_many_files."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    base = [_make_dummy_tool("shell")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "read_many_files" in names


def test_gemini_profile_does_not_duplicate_list_dir():
    """If list_dir is already in base_tools, it is not added twice."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    base = [_make_dummy_tool("shell"), LIST_DIR]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert names.count("list_dir") == 1


def test_gemini_profile_does_not_duplicate_read_many_files():
    """If read_many_files is already in base_tools, it is not added twice."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    base = [_make_dummy_tool("shell"), READ_MANY_FILES]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert names.count("read_many_files") == 1


def test_gemini_profile_empty_base_tools_injects_nothing():
    """When base_tools is empty, no Gemini-specific tools are injected."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    tools = profile.get_tools([])

    assert tools == []


def test_gemini_profile_preserves_base_tool_count():
    """Base tools are all present alongside the two injected tools."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    base = [_make_dummy_tool("shell"), _make_dummy_tool("grep"), _make_dummy_tool("glob")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "shell" in names
    assert "grep" in names
    assert "glob" in names
    # Plus the two injected ones
    assert "list_dir" in names
    assert "read_many_files" in names


def test_gemini_profile_injected_tools_have_execute():
    """The injected tools carry working execute handlers."""
    from attractor_agent.profiles.gemini import GeminiProfile

    profile = GeminiProfile()
    tools = profile.get_tools([_make_dummy_tool()])

    by_name = {t.name: t for t in tools}
    assert by_name["list_dir"].execute is not None
    assert by_name["read_many_files"].execute is not None


# ------------------------------------------------------------------ #
# Other profiles do NOT inject Gemini-specific tools
# ------------------------------------------------------------------ #


def test_anthropic_profile_does_not_include_gemini_tools():
    """AnthropicProfile.get_tools() must not inject list_dir or read_many_files."""
    from attractor_agent.profiles.anthropic import AnthropicProfile

    profile = AnthropicProfile()
    base = [_make_dummy_tool("shell"), _make_dummy_tool("grep")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "list_dir" not in names
    assert "read_many_files" not in names


def test_openai_profile_does_not_include_gemini_tools():
    """OpenAIProfile.get_tools() must not inject list_dir or read_many_files."""
    from attractor_agent.profiles.openai import OpenAIProfile

    profile = OpenAIProfile()
    base = [_make_dummy_tool("shell"), _make_dummy_tool("grep")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "list_dir" not in names
    assert "read_many_files" not in names


def test_openai_profile_still_injects_apply_patch():
    """Regression: OpenAI apply_patch injection still works after P15."""
    from attractor_agent.profiles.openai import OpenAIProfile

    profile = OpenAIProfile()
    base = [_make_dummy_tool("shell")]
    tools = profile.get_tools(base)

    names = [t.name for t in tools]
    assert "apply_patch" in names
