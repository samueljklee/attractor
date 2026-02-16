"""Tests for Wave 2 system prompt enrichment: environment context + project docs.

Covers:
- build_environment_context() XML block generation
- get_git_context() in git and non-git directories
- discover_project_docs() with provider filtering, budget, ordering
- Session._call_llm() integration (enriched system prompt reaches the LLM)

Spec reference: coding-agent-loop-spec §6.3-6.5.
"""

from __future__ import annotations

import subprocess

import pytest

from attractor_agent.env_context import (
    build_environment_context,
    get_git_context,
)
from attractor_agent.project_docs import (
    _walk_path,
    discover_project_docs,
)
from attractor_agent.session import Session, SessionConfig
from attractor_llm.client import Client
from tests.helpers import MockAdapter, make_text_response

# ================================================================== #
# build_environment_context
# ================================================================== #


class TestBuildEnvironmentContext:
    """Tests for the <environment> block builder."""

    def test_basic_structure(self):
        result = build_environment_context(
            working_dir="/home/user/project",
            model="claude-sonnet-4-5",
            platform="linux",
            os_version="Linux-6.8.0",
            today="2026-01-15",
            git_info={"is_git": False},
        )
        assert result.startswith("<environment>")
        assert result.strip().endswith("</environment>")
        assert "Working directory: /home/user/project" in result
        assert "Is git repository: false" in result
        assert "Platform: linux" in result
        assert "OS version: Linux-6.8.0" in result
        assert "Today's date: 2026-01-15" in result
        assert "Model: claude-sonnet-4-5" in result

    def test_git_branch_included_when_in_repo(self):
        result = build_environment_context(
            working_dir="/project",
            model="gpt-5.2",
            platform="darwin",
            os_version="macOS-15.0",
            today="2026-02-01",
            git_info={"is_git": True, "branch": "main"},
        )
        assert "Is git repository: true" in result
        assert "Git branch: main" in result

    def test_git_branch_omitted_when_not_in_repo(self):
        result = build_environment_context(
            working_dir="/tmp/scratch",
            model="test-model",
            platform="linux",
            os_version="test",
            today="2026-01-01",
            git_info={"is_git": False},
        )
        assert "Git branch" not in result

    def test_knowledge_cutoff_included_when_provided(self):
        result = build_environment_context(
            working_dir="/project",
            model="claude-sonnet-4-5",
            platform="linux",
            os_version="test",
            today="2026-01-01",
            knowledge_cutoff="2025-04-01",
            git_info={"is_git": False},
        )
        assert "Knowledge cutoff: 2025-04-01" in result

    def test_knowledge_cutoff_omitted_when_none(self):
        result = build_environment_context(
            working_dir="/project",
            model="test-model",
            platform="linux",
            os_version="test",
            today="2026-01-01",
            git_info={"is_git": False},
        )
        assert "Knowledge cutoff" not in result

    def test_git_snapshot_appended_for_repo(self):
        result = build_environment_context(
            working_dir="/project",
            model="test-model",
            platform="linux",
            os_version="test",
            today="2026-01-01",
            git_info={
                "is_git": True,
                "branch": "feat/test",
                "modified_count": 3,
                "untracked_count": 1,
                "recent_commits": ["abc1234 first commit", "def5678 second commit"],
            },
        )
        assert "<git_status>" in result
        assert "Branch: feat/test" in result
        assert "Modified files: 3" in result
        assert "Untracked files: 1" in result
        assert "abc1234 first commit" in result
        assert "def5678 second commit" in result
        assert "</git_status>" in result

    def test_no_git_snapshot_for_non_repo(self):
        result = build_environment_context(
            working_dir="/tmp",
            model="test",
            platform="linux",
            os_version="test",
            today="2026-01-01",
            git_info={"is_git": False},
        )
        assert "<git_status>" not in result


# ================================================================== #
# get_git_context
# ================================================================== #


class TestGetGitContext:
    """Tests for git context collection."""

    def test_in_git_repo(self, tmp_path):
        """Create a real git repo and verify context extraction."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Init repo, configure, and make a commit
        subprocess.run(  # noqa: S603
            ["git", "init"],  # noqa: S607
            cwd=str(repo),
            capture_output=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "config", "user.email", "test@test.com"],  # noqa: S607
            cwd=str(repo),
            capture_output=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "config", "user.name", "Test"],  # noqa: S607
            cwd=str(repo),
            capture_output=True,
        )

        # Create and commit a file
        (repo / "hello.txt").write_text("hello")
        subprocess.run(  # noqa: S603
            ["git", "add", "."],  # noqa: S607
            cwd=str(repo),
            capture_output=True,
        )
        subprocess.run(  # noqa: S603
            ["git", "commit", "-m", "initial commit"],  # noqa: S607
            cwd=str(repo),
            capture_output=True,
        )

        # Create an untracked file and modify existing
        (repo / "new.txt").write_text("new")
        (repo / "hello.txt").write_text("modified")

        info = get_git_context(str(repo))

        assert info["is_git"] is True
        assert isinstance(info["branch"], str)
        assert info["modified_count"] == 1  # hello.txt modified
        assert info["untracked_count"] == 1  # new.txt
        commits = info["recent_commits"]
        assert isinstance(commits, list)
        assert len(commits) == 1
        assert "initial commit" in commits[0]

    def test_not_in_git_repo(self, tmp_path):
        """Non-git directory returns safe defaults."""
        info = get_git_context(str(tmp_path))
        assert info["is_git"] is False
        assert info["branch"] == ""
        assert info["modified_count"] == 0
        assert info["untracked_count"] == 0
        assert info["recent_commits"] == []


# ================================================================== #
# discover_project_docs
# ================================================================== #


class TestDiscoverProjectDocs:
    """Tests for project documentation discovery."""

    def test_agents_md_loaded_for_any_provider(self, tmp_path):
        """AGENTS.md is universal -- loaded regardless of provider."""
        (tmp_path / "AGENTS.md").write_text("# Agent instructions")

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "Agent instructions" in result
        assert "<project_instructions>" in result

        result = discover_project_docs(str(tmp_path), provider_id="openai")
        assert "Agent instructions" in result

        result = discover_project_docs(str(tmp_path), provider_id=None)
        assert "Agent instructions" in result

    def test_anthropic_loads_claude_md(self, tmp_path):
        """Anthropic provider loads CLAUDE.md."""
        (tmp_path / "CLAUDE.md").write_text("# Claude instructions")

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "Claude instructions" in result

    def test_anthropic_does_not_load_gemini_md(self, tmp_path):
        """Anthropic provider must NOT load GEMINI.md."""
        (tmp_path / "GEMINI.md").write_text("# Gemini instructions")
        (tmp_path / "AGENTS.md").write_text("# Universal")

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "Gemini instructions" not in result
        assert "Universal" in result

    def test_openai_loads_codex_instructions(self, tmp_path):
        """OpenAI provider loads .codex/instructions.md."""
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "instructions.md").write_text("# Codex instructions")

        result = discover_project_docs(str(tmp_path), provider_id="openai")
        assert "Codex instructions" in result

    def test_gemini_loads_gemini_md(self, tmp_path):
        """Gemini provider loads GEMINI.md."""
        (tmp_path / "GEMINI.md").write_text("# Gemini instructions")

        result = discover_project_docs(str(tmp_path), provider_id="gemini")
        assert "Gemini instructions" in result

    def test_no_files_returns_empty(self, tmp_path):
        """No recognised files -> empty string."""
        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert result == ""

    def test_root_to_cwd_walk_ordering(self, tmp_path):
        """Files at root are loaded first, subdirectory files appended."""
        # Simulate git root at tmp_path with a subdirectory
        root = tmp_path
        sub = root / "src" / "lib"
        sub.mkdir(parents=True)

        (root / "AGENTS.md").write_text("root agents")
        (sub / "AGENTS.md").write_text("sub agents")

        # Since we're not in a git repo, discover_project_docs will use
        # working_dir as both root and target.  To test walk ordering,
        # use _walk_path directly + call with the sub directory.
        # We'll set up a git repo to get proper root detection.
        subprocess.run(  # noqa: S603
            ["git", "init"],  # noqa: S607
            cwd=str(root),
            capture_output=True,
        )

        result = discover_project_docs(str(sub), provider_id=None)
        # Root file should come before sub file
        root_pos = result.find("root agents")
        sub_pos = result.find("sub agents")
        assert root_pos != -1, "root AGENTS.md should be included"
        assert sub_pos != -1, "sub AGENTS.md should be included"
        assert root_pos < sub_pos, "root files must precede subdirectory files"

    def test_32kb_budget_truncation(self, tmp_path):
        """Content exceeding 32KB budget is truncated with marker."""
        # Create a file that's > 32KB
        big_content = "x" * (33 * 1024)
        (tmp_path / "AGENTS.md").write_text(big_content)

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "[Project instructions truncated at 32KB]" in result

    def test_multiple_files_within_budget(self, tmp_path):
        """Multiple files under budget are all included."""
        (tmp_path / "AGENTS.md").write_text("# Agents")
        (tmp_path / "CLAUDE.md").write_text("# Claude")

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "# Agents" in result
        assert "# Claude" in result
        assert "truncated" not in result

    def test_budget_spans_multiple_files(self, tmp_path):
        """Budget enforcement across multiple files."""
        # First file takes most of the budget
        (tmp_path / "AGENTS.md").write_text("a" * (31 * 1024))
        # Second file pushes over the limit
        (tmp_path / "CLAUDE.md").write_text("b" * (2 * 1024))

        result = discover_project_docs(str(tmp_path), provider_id="anthropic")
        assert "[Project instructions truncated at 32KB]" in result
        # First file should be fully included
        assert "a" * 100 in result

    def test_display_path_in_output(self, tmp_path):
        """Output includes relative file path as header."""
        (tmp_path / "AGENTS.md").write_text("instructions here")

        result = discover_project_docs(str(tmp_path), provider_id=None)
        assert "# From" in result
        assert "AGENTS.md" in result


# ================================================================== #
# _walk_path (internal helper)
# ================================================================== #


class TestWalkPath:
    """Tests for the root-to-cwd directory walker."""

    def test_same_root_and_target(self, tmp_path):
        dirs = _walk_path(tmp_path, tmp_path)
        assert dirs == [tmp_path]

    def test_nested_path(self, tmp_path):
        sub = tmp_path / "a" / "b" / "c"
        sub.mkdir(parents=True)

        dirs = _walk_path(tmp_path, sub)
        assert len(dirs) == 4
        assert dirs[0] == tmp_path
        assert dirs[-1] == sub
        # Intermediate directories are included
        assert tmp_path / "a" in dirs
        assert tmp_path / "a" / "b" in dirs

    def test_target_not_under_root_fallback(self, tmp_path):
        from pathlib import Path

        root = Path("/some/root")
        target = Path("/other/path")
        dirs = _walk_path(root, target)
        assert dirs == [target]


# ================================================================== #
# Session integration
# ================================================================== #


class TestSessionSystemPromptEnrichment:
    """Verify that Session._call_llm() injects environment context."""

    @pytest.mark.asyncio
    async def test_call_llm_includes_environment_block(self, tmp_path):
        """The system prompt sent to the LLM must contain <environment>."""
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(
            model="mock-model",
            provider="mock",
            system_prompt="Base instructions here.",
            working_dir=str(tmp_path),
        )
        session = Session(client=client, config=config)
        await session.submit("hello")

        # Inspect the request that was sent to the adapter
        assert adapter.call_count == 1
        request = adapter.requests[0]
        system = request.system or ""

        assert "<environment>" in system
        assert "Working directory:" in system
        assert "Platform:" in system
        assert "Model: mock-model" in system
        assert "</environment>" in system
        # Base prompt still present
        assert "Base instructions here." in system

    @pytest.mark.asyncio
    async def test_call_llm_includes_project_docs(self, tmp_path):
        """Project docs are appended when instruction files exist."""
        (tmp_path / "AGENTS.md").write_text("project-level agent rules")

        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(
            model="mock-model",
            provider="mock",
            system_prompt="Base prompt.",
            working_dir=str(tmp_path),
        )
        session = Session(client=client, config=config)
        await session.submit("hello")

        request = adapter.requests[0]
        system = request.system or ""
        assert "<project_instructions>" in system
        assert "project-level agent rules" in system

    @pytest.mark.asyncio
    async def test_call_llm_no_project_docs_when_none_exist(self, tmp_path):
        """No <project_instructions> block when no files found."""
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(
            model="mock-model",
            provider="mock",
            working_dir=str(tmp_path),
        )
        session = Session(client=client, config=config)
        await session.submit("hello")

        request = adapter.requests[0]
        system = request.system or ""
        assert "<environment>" in system
        assert "<project_instructions>" not in system

    @pytest.mark.asyncio
    async def test_enriched_prompt_ordering(self, tmp_path):
        """Base prompt first, environment block middle, docs last (spec §6.1)."""
        (tmp_path / "AGENTS.md").write_text("DOCS_MARKER")

        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(
            model="mock-model",
            provider="mock",
            system_prompt="BASE_MARKER",
            working_dir=str(tmp_path),
        )
        session = Session(client=client, config=config)
        await session.submit("hello")

        request = adapter.requests[0]
        system = request.system or ""

        base_pos = system.find("BASE_MARKER")
        env_pos = system.find("<environment>")
        docs_pos = system.find("DOCS_MARKER")

        assert base_pos < env_pos < docs_pos, (
            f"Expected base({base_pos}) < env({env_pos}) < docs({docs_pos})"
        )

    @pytest.mark.asyncio
    async def test_provider_filtering_in_session(self, tmp_path):
        """Session passes provider_id to discover_project_docs correctly."""
        (tmp_path / "CLAUDE.md").write_text("CLAUDE_CONTENT")
        (tmp_path / "GEMINI.md").write_text("GEMINI_CONTENT")

        # Use provider="mock" -- not in _PROVIDER_FILES, so only AGENTS.md
        # would be loaded.  CLAUDE.md and GEMINI.md should NOT appear.
        adapter = MockAdapter(responses=[make_text_response("ok")])
        client = Client()
        client.register_adapter("mock", adapter)

        config = SessionConfig(
            model="mock-model",
            provider="mock",
            working_dir=str(tmp_path),
        )
        session = Session(client=client, config=config)
        await session.submit("hello")

        request = adapter.requests[0]
        system = request.system or ""
        # Neither provider-specific file should be loaded for "mock" provider
        assert "CLAUDE_CONTENT" not in system
        assert "GEMINI_CONTENT" not in system
