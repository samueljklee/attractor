"""Tests for Wave 6 content types and miscellaneous spec compliance items.

Covers:
- Item #18: AUDIO and DOCUMENT content types (Spec §3.3)
- Item #20: Lazy default client initialization (Spec §2.2, §8.1)
- Item #26: supports_parallel_tool_calls profile flag (Spec §2.5)
- Item #27: Shell timeout_ms parameter (Spec §4.2)
- Item #28: ImageData.from_file() local file path (Spec §3.3)
- Item #30: Tool argument schema validation (Spec §5.5)
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    ImageData,
    Tool,
)

# ================================================================== #
# Item #18: AUDIO and DOCUMENT content types
# ================================================================== #


class TestAudioDocumentContentTypes:
    """Spec §3.3: AUDIO and DOCUMENT exist in ContentPartKind."""

    def test_audio_enum_exists(self):
        assert hasattr(ContentPartKind, "AUDIO")
        assert ContentPartKind.AUDIO == "audio"

    def test_document_enum_exists(self):
        assert hasattr(ContentPartKind, "DOCUMENT")
        assert ContentPartKind.DOCUMENT == "document"

    def test_audio_content_part_accepted(self):
        """AUDIO kind can be constructed (future-proof, no required fields yet)."""
        part = ContentPart(kind=ContentPartKind.AUDIO)
        assert part.kind == ContentPartKind.AUDIO

    def test_document_content_part_accepted(self):
        """DOCUMENT kind can be constructed (future-proof, no required fields yet)."""
        part = ContentPart(kind=ContentPartKind.DOCUMENT)
        assert part.kind == ContentPartKind.DOCUMENT

    def test_all_original_kinds_still_present(self):
        """Ensure adding new kinds didn't break existing ones."""
        expected = {
            "text",
            "image",
            "audio",
            "document",
            "tool_call",
            "tool_result",
            "thinking",
            "redacted_thinking",
        }
        actual = {k.value for k in ContentPartKind}
        assert expected == actual


# ================================================================== #
# Item #20: Lazy default client initialization
# ================================================================== #


class TestLazyDefaultClient:
    """Spec §2.2: get_default_client() auto-creates from env vars."""

    def setup_method(self):
        """Reset the module-level default client before each test."""
        import attractor_llm.client as client_mod

        self._original = client_mod._default_client
        client_mod._default_client = None

    def teardown_method(self):
        """Restore the original default client."""
        import attractor_llm.client as client_mod

        client_mod._default_client = self._original

    def test_lazy_init_creates_client_from_env(self):
        """When env vars are set, get_default_client() auto-creates."""
        from attractor_llm.client import get_default_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key"}, clear=False):
            client = get_default_client()
            assert client is not None
            assert "openai" in client._adapters

    def test_lazy_init_raises_when_no_env_vars(self):
        """When no env vars are set, get_default_client() raises ConfigurationError."""
        from attractor_llm.client import get_default_client
        from attractor_llm.errors import ConfigurationError

        # Clear all provider API key env vars
        env_overrides = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            # Remove the keys entirely (patch.dict with "" still sets them)
            for key in env_overrides:
                os.environ.pop(key, None)
            with pytest.raises(ConfigurationError, match="No default client configured"):
                get_default_client()

    def test_lazy_init_caches_client(self):
        """Once auto-created, the same client is returned on subsequent calls."""
        from attractor_llm.client import get_default_client

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False):
            client1 = get_default_client()
            client2 = get_default_client()
            assert client1 is client2

    def test_set_default_client_still_works(self):
        """Explicit set_default_client() takes precedence."""
        from attractor_llm.client import Client, get_default_client, set_default_client

        explicit = Client()
        set_default_client(explicit)
        assert get_default_client() is explicit


# ================================================================== #
# Item #26: supports_parallel_tool_calls profile flag
# ================================================================== #


class TestSupportsParallelToolCalls:
    """Spec §2.5: profiles expose supports_parallel_tool_calls."""

    def test_openai_supports_parallel(self):
        from attractor_agent.profiles.openai import OpenAIProfile

        assert OpenAIProfile().supports_parallel_tool_calls is True

    def test_anthropic_supports_parallel(self):
        from attractor_agent.profiles.anthropic import AnthropicProfile

        assert AnthropicProfile().supports_parallel_tool_calls is True

    def test_gemini_supports_parallel(self):
        from attractor_agent.profiles.gemini import GeminiProfile

        assert GeminiProfile().supports_parallel_tool_calls is True

    def test_base_profile_no_parallel(self):
        from attractor_agent.profiles.base import BaseProfile

        assert BaseProfile().supports_parallel_tool_calls is False

    @pytest.mark.asyncio
    async def test_registry_sequential_when_disabled(self):
        """ToolRegistry executes sequentially when parallel is disabled."""
        from attractor_agent.tools.registry import ToolRegistry

        call_order: list[str] = []

        async def slow_tool(**kwargs):
            call_order.append(f"start-{kwargs.get('id', '?')}")
            await asyncio.sleep(0.05)
            call_order.append(f"end-{kwargs.get('id', '?')}")
            return "ok"

        registry = ToolRegistry(supports_parallel_tool_calls=False)
        tool = Tool(
            name="slow",
            description="slow tool",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            execute=slow_tool,
        )
        registry.register(tool)

        tc1 = ContentPart.tool_call_part("tc-1", "slow", {"id": "A"})
        tc2 = ContentPart.tool_call_part("tc-2", "slow", {"id": "B"})

        results = await registry.execute_tool_calls([tc1, tc2])
        assert len(results) == 2
        # Sequential: A starts and ends before B starts
        assert call_order == ["start-A", "end-A", "start-B", "end-B"]

    @pytest.mark.asyncio
    async def test_registry_parallel_when_enabled(self):
        """ToolRegistry executes in parallel when parallel is enabled."""
        from attractor_agent.tools.registry import ToolRegistry

        call_order: list[str] = []

        async def slow_tool(**kwargs):
            call_order.append(f"start-{kwargs.get('id', '?')}")
            await asyncio.sleep(0.05)
            call_order.append(f"end-{kwargs.get('id', '?')}")
            return "ok"

        registry = ToolRegistry(supports_parallel_tool_calls=True)
        tool = Tool(
            name="slow",
            description="slow tool",
            parameters={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            execute=slow_tool,
        )
        registry.register(tool)

        tc1 = ContentPart.tool_call_part("tc-1", "slow", {"id": "A"})
        tc2 = ContentPart.tool_call_part("tc-2", "slow", {"id": "B"})

        results = await registry.execute_tool_calls([tc1, tc2])
        assert len(results) == 2
        # Parallel: both start before either ends
        assert call_order[0].startswith("start-")
        assert call_order[1].startswith("start-")


# ================================================================== #
# Item #27: Shell timeout_ms parameter
# ================================================================== #


class TestShellTimeoutMs:
    """Spec §4.2: timeout_ms alias for shell tool."""

    def test_timeout_ms_in_schema(self):
        """timeout_ms is declared in the SHELL tool's parameter schema."""
        from attractor_agent.tools.core import SHELL

        props = SHELL.parameters.get("properties", {})
        assert "timeout_ms" in props
        assert props["timeout_ms"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_timeout_ms_takes_precedence(self):
        """When timeout_ms is provided, it overrides timeout."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.tools.core import _shell

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        with (
            patch("attractor_agent.tools.core._environment", mock_env),
            patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
        ):
            await _shell("echo hi", timeout=999, timeout_ms=5000)

        # timeout_ms=5000 -> 5 seconds should be passed
        call_kwargs = mock_env.exec_shell.call_args
        assert call_kwargs[1]["timeout"] == 5

    @pytest.mark.asyncio
    async def test_timeout_ms_rounds_up(self):
        """Sub-second timeout_ms values round up to at least 1s."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.tools.core import _shell

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        with (
            patch("attractor_agent.tools.core._environment", mock_env),
            patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
        ):
            await _shell("echo hi", timeout_ms=500)

        call_kwargs = mock_env.exec_shell.call_args
        assert call_kwargs[1]["timeout"] == 1  # ceiling(500/1000) = 1

    @pytest.mark.asyncio
    async def test_timeout_default_when_no_timeout_ms(self):
        """When timeout_ms is not provided, timeout (seconds) is used as-is."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.tools.core import _shell

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.output = "ok"
        mock_result.stderr = ""

        mock_env = MagicMock()
        mock_env.exec_shell = AsyncMock(return_value=mock_result)

        with (
            patch("attractor_agent.tools.core._environment", mock_env),
            patch("attractor_agent.tools.core._check_path_allowed", return_value=None),
        ):
            await _shell("echo hi", timeout=30)

        call_kwargs = mock_env.exec_shell.call_args
        assert call_kwargs[1]["timeout"] == 30


# ================================================================== #
# Item #28: ImageData.from_file() local file path
# ================================================================== #


class TestImageDataFromFile:
    """Spec §3.3: ImageData.from_file() reads local file."""

    def test_from_file_reads_png(self):
        """from_file() populates data and infers PNG media type."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()
            try:
                img = ImageData.from_file(f.name)
                assert img.data is not None
                assert img.data.startswith(b"\x89PNG")
                assert img.media_type == "image/png"
                assert img.file_path == str(Path(f.name).resolve())
            finally:
                os.unlink(f.name)

    def test_from_file_reads_jpeg(self):
        """from_file() infers JPEG media type."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()
            try:
                img = ImageData.from_file(f.name)
                assert img.media_type == "image/jpeg"
                assert img.data is not None
            finally:
                os.unlink(f.name)

    def test_from_file_not_found(self):
        """from_file() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            ImageData.from_file("/nonexistent/image.png")

    def test_from_file_unknown_extension(self):
        """from_file() falls back to application/octet-stream for unknown ext."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"some data")
            f.flush()
            try:
                img = ImageData.from_file(f.name)
                assert img.media_type == "application/octet-stream"
            finally:
                os.unlink(f.name)

    def test_file_path_field_on_imagedata(self):
        """ImageData accepts file_path as a valid source."""
        img = ImageData(file_path="/some/path.png")
        assert img.file_path == "/some/path.png"

    def test_imagedata_requires_some_source(self):
        """ImageData still requires at least one of data, url, or file_path."""
        with pytest.raises(ValueError, match="requires either"):
            ImageData()


# ================================================================== #
# Item #30: Tool argument schema validation
# ================================================================== #


class TestToolArgumentSchemaValidation:
    """Spec §5.5: validate tool args before execution."""

    def test_validate_catches_missing_required_fields(self):
        """Missing required fields produce an error string."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        result = validate_tool_arguments({"path": "foo.py"}, schema)
        assert result is not None
        assert "content" in result

    def test_validate_catches_wrong_type(self):
        """Type mismatch produces an error string."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
            "required": ["count"],
        }
        result = validate_tool_arguments({"count": "not-a-number"}, schema)
        assert result is not None
        assert "count" in result
        assert "str" in result

    def test_validate_passes_valid_args(self):
        """Valid arguments return None (no error)."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
            },
            "required": ["path"],
        }
        result = validate_tool_arguments({"path": "foo.py", "offset": 10}, schema)
        assert result is None

    def test_validate_allows_extra_keys(self):
        """Extra keys not in the schema are tolerated."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        result = validate_tool_arguments({"path": "x", "extra": True}, schema)
        assert result is None

    def test_validate_bool_not_accepted_as_integer(self):
        """Boolean values should not pass integer type checks."""
        from attractor_agent.tools.registry import validate_tool_arguments

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        result = validate_tool_arguments({"count": True}, schema)
        assert result is not None
        assert "bool" in result

    @pytest.mark.asyncio
    async def test_registry_returns_error_result_on_validation_failure(self):
        """Registry returns is_error=True result instead of executing."""
        from attractor_agent.tools.registry import ToolRegistry

        async def should_not_run(**kwargs):
            raise AssertionError("Tool should not have been called")

        registry = ToolRegistry()
        tool = Tool(
            name="write_file",
            description="write",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            execute=should_not_run,
        )
        registry.register(tool)

        # Missing 'content' required field
        tc = ContentPart.tool_call_part("tc-1", "write_file", {"path": "foo.py"})
        result = await registry.execute_tool_call(tc)
        assert result.is_error is True
        assert "content" in (result.output or "")
        assert "Missing required" in (result.output or "")

    @pytest.mark.asyncio
    async def test_registry_executes_when_args_valid(self):
        """Registry executes the tool when validation passes."""
        from attractor_agent.tools.registry import ToolRegistry

        async def echo_tool(**kwargs):
            return f"got: {kwargs}"

        registry = ToolRegistry()
        tool = Tool(
            name="echo",
            description="echo",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
            },
            execute=echo_tool,
        )
        registry.register(tool)

        tc = ContentPart.tool_call_part("tc-1", "echo", {"message": "hello"})
        result = await registry.execute_tool_call(tc)
        assert result.is_error is False
        assert "hello" in (result.output or "")

    def test_validate_empty_schema_always_passes(self):
        """Tools with empty/no schema pass validation (backward compat)."""
        from attractor_agent.tools.registry import validate_tool_arguments

        assert validate_tool_arguments({"anything": "goes"}, {}) is None
        assert validate_tool_arguments({}, {}) is None
