"""Tests for spec compliance gaps — Wave Final.

Groups:
  TestMaxTurnsDefaults        — Task 1
  TestShellProcessCallback    — Task 2
  TestParallelToolCalls       — Task 3
  TestSessionEndEvent         — Task 4
  TestMiddlewareChain         — Task 5
  TestHttpServer              — Task 6
  TestInterviewerAnswer       — Task 7
  TestR13Validation           — Task 8
  TestAnthropicDescriptions   — Task 9
  TestApplyPatchV4a           — Task 10
"""

from __future__ import annotations

import pytest

from attractor_agent.session import SessionConfig
from attractor_agent.subagent import spawn_subagent


class TestMaxTurnsDefaults:
    """Task 1 — §9 SessionConfig defaults."""

    def test_session_config_max_turns_defaults_to_zero(self):
        """SessionConfig() with no args must have max_turns=0 (unlimited)."""
        config = SessionConfig()
        assert config.max_turns == 0, (
            f"Expected max_turns=0 (unlimited per spec §9), got {config.max_turns}"
        )

    def test_session_config_max_tool_rounds_defaults_to_zero(self):
        """SessionConfig() with no args must have max_tool_rounds_per_turn=0."""
        config = SessionConfig()
        assert config.max_tool_rounds_per_turn == 0, (
            f"Expected max_tool_rounds_per_turn=0, got {config.max_tool_rounds_per_turn}"
        )

    def test_spawn_subagent_max_turns_defaults_to_zero(self):
        """spawn_subagent() max_turns default must be 0 per spec §9."""
        import inspect

        sig = inspect.signature(spawn_subagent)
        assert sig.parameters["max_turns"].default == 0, (
            f"Expected spawn_subagent max_turns default=0, "
            f"got {sig.parameters['max_turns'].default}"
        )

    def test_spawn_subagent_max_tool_rounds_defaults_to_zero(self):
        """spawn_subagent() max_tool_rounds default must be 0."""
        import inspect

        sig = inspect.signature(spawn_subagent)
        assert sig.parameters["max_tool_rounds"].default == 0

    @pytest.mark.asyncio
    async def test_session_zero_max_turns_does_not_limit(self):
        """With max_turns=0, a session must NOT hit the turn limit on turn 1."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import Message, Response, Usage

        config = SessionConfig(max_turns=0, max_tool_rounds_per_turn=0)
        mock_client = MagicMock()
        mock_response = Response(
            id="resp-1",
            model="test-model",
            content=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="test",
        )
        mock_response.message = Message.assistant("Done.")
        mock_client.complete = AsyncMock(return_value=mock_response)

        session = Session(client=mock_client, config=config)
        result = await session.submit("Hello")
        assert "[Turn limit reached]" not in result, (
            "max_turns=0 should mean unlimited, not zero turns allowed"
        )

    def test_subagent_manager_spawn_max_turns_defaults_to_zero(self):
        """SubagentManager.spawn() max_turns and max_tool_rounds defaults must be 0."""
        import inspect

        from attractor_agent.subagent_manager import SubagentManager

        sig = inspect.signature(SubagentManager.spawn)
        assert sig.parameters["max_turns"].default == 0, (
            f"Expected SubagentManager.spawn max_turns default=0, "
            f"got {sig.parameters['max_turns'].default}"
        )
        assert sig.parameters["max_tool_rounds"].default == 0, (
            f"Expected SubagentManager.spawn max_tool_rounds default=0, "
            f"got {sig.parameters['max_tool_rounds'].default}"
        )

    def test_all_max_turns_defaults_in_subagent_module_are_zero(self):
        """No function in subagent.py may default max_turns to a non-zero value."""
        import ast
        import inspect

        import attractor_agent.subagent as subagent_module

        source = inspect.getsource(subagent_module)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg, default in zip(
                    reversed(node.args.args), reversed(node.args.defaults), strict=False
                ):
                    if arg.arg == "max_turns":
                        if isinstance(default, ast.Constant):
                            assert default.value == 0, (
                                f"Function '{node.name}' in subagent.py has "
                                f"max_turns default={default.value}, expected 0 (spec §9)"
                            )

    def test_spawn_agent_tool_max_turns_defaults_to_zero(self):
        """The LLM-callable spawn_agent tool inner function must default max_turns to 0."""
        import ast
        import inspect
        import textwrap

        from attractor_agent import subagent as subagent_module

        # Find the inner execute function — defined inside spawn_subagent
        # or as a module-level helper (e.g. spawn_agent_execute).
        source = inspect.getsource(subagent_module)
        # Verify '= 20' does not appear as a max_turns default anywhere in the module
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg, default in zip(
                    reversed(node.args.args), reversed(node.args.defaults), strict=False
                ):
                    if arg.arg == "max_turns":
                        if isinstance(default, ast.Constant):
                            assert default.value == 0, (
                                f"Function '{node.name}' has max_turns "
                                f"default={default.value}, expected 0 (spec §9)"
                            )

    @pytest.mark.asyncio
    async def test_session_zero_max_tool_rounds_does_not_limit(self):
        """With max_tool_rounds_per_turn=0, tool rounds must not be limited on round 0."""
        from unittest.mock import AsyncMock, MagicMock, patch  # noqa: F401

        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import Message, Response, Usage

        config = SessionConfig(max_turns=0, max_tool_rounds_per_turn=0)
        mock_client = MagicMock()
        # Return a text-only response immediately (no tool calls) so loop exits naturally.
        mock_response = Response(
            id="resp-1",
            model="test-model",
            content=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="test",
        )
        mock_response.message = Message.assistant("Done.")
        mock_client.complete = AsyncMock(return_value=mock_response)

        session = Session(client=mock_client, config=config)
        result = await session.submit("Hello")
        assert "[Tool round limit reached]" not in result, (
            "max_tool_rounds_per_turn=0 should mean unlimited, not zero rounds allowed"
        )


class TestShellProcessCallback:
    """Task 2 — §9.1.6, §9.11.5: shell processes registered for abort cleanup."""

    @pytest.fixture(autouse=True)
    def reset_process_callback(self):
        """Reset the module-level process callback after each test to avoid global state leakage."""
        from attractor_agent.tools.core import set_process_callback

        yield
        set_process_callback(None)

    @pytest.mark.asyncio
    async def test_session_wires_process_callback_on_init(self):
        """After Session.__init__, the module-level process callback must point
        to session.register_process so shell commands auto-register."""
        from unittest.mock import MagicMock

        from attractor_agent.session import Session, SessionConfig
        from attractor_agent.tools.core import get_process_callback

        mock_client = MagicMock()
        session = Session(client=mock_client, config=SessionConfig())
        cb = get_process_callback()
        assert cb is not None, "Session.__init__ must call set_process_callback()"
        assert cb == session.register_process, "Process callback must be session.register_process"

    @pytest.mark.asyncio
    async def test_shell_command_registers_process(self, tmp_path):
        """Running a shell command via the Session must populate _tracked_processes."""
        from unittest.mock import MagicMock

        from attractor_agent.session import Session, SessionConfig
        from attractor_agent.tools.core import set_allowed_roots

        set_allowed_roots([str(tmp_path)])
        mock_client = MagicMock()
        session = Session(client=mock_client, config=SessionConfig())

        from attractor_agent.tools import core as tool_core

        await tool_core._shell("echo hello", working_dir=str(tmp_path))

        assert len(session._tracked_processes) > 0, (
            "shell command must register its subprocess with the session"
        )


class TestParallelToolCalls:
    """Task 3 — §9.3.5: supports_parallel_tool_calls propagated to ToolRegistry."""

    @pytest.fixture(autouse=True)
    def reset_process_callback(self):
        """Reset module-level process callback after each test to avoid global state leakage."""
        from attractor_agent.tools.core import set_process_callback

        yield
        set_process_callback(None)

    def test_profile_parallel_false_propagates_to_registry(self):
        """When profile.supports_parallel_tool_calls=False, registry must also be False."""
        from unittest.mock import MagicMock

        from attractor_agent.session import Session, SessionConfig

        mock_profile = MagicMock()
        mock_profile.supports_parallel_tool_calls = False
        mock_profile.get_tools.return_value = []
        mock_profile.apply_to_config.side_effect = lambda c: c

        session = Session(
            client=MagicMock(),
            config=SessionConfig(),
            profile=mock_profile,
        )
        assert session._tool_registry.supports_parallel_tool_calls is False, (
            "ToolRegistry.supports_parallel_tool_calls must reflect the profile's value"
        )

    def test_profile_parallel_true_propagates_to_registry(self):
        """When profile.supports_parallel_tool_calls=True, registry must be True."""
        from unittest.mock import MagicMock

        from attractor_agent.session import Session, SessionConfig

        mock_profile = MagicMock()
        mock_profile.supports_parallel_tool_calls = True
        mock_profile.get_tools.return_value = []
        mock_profile.apply_to_config.side_effect = lambda c: c

        session = Session(
            client=MagicMock(),
            config=SessionConfig(),
            profile=mock_profile,
        )
        assert session._tool_registry.supports_parallel_tool_calls is True, (
            "ToolRegistry.supports_parallel_tool_calls must reflect the profile's True value"
        )

    def test_no_profile_registry_defaults_true(self):
        """Without a profile, ToolRegistry.supports_parallel_tool_calls defaults True."""
        from unittest.mock import MagicMock

        from attractor_agent.session import Session, SessionConfig

        session = Session(client=MagicMock(), config=SessionConfig())
        assert session._tool_registry.supports_parallel_tool_calls is True, (
            "ToolRegistry.supports_parallel_tool_calls must default to True when no profile given"
        )


class TestSessionEndEvent:
    """Task 4 — §9.10.4: SESSION_END emitted on all CLOSED transitions."""

    @pytest.fixture(autouse=True)
    def reset_process_callback(self):
        from attractor_agent.tools.core import set_process_callback

        yield
        set_process_callback(None)

    @pytest.mark.asyncio
    async def test_session_end_emitted_on_auth_error(self):
        """SESSION_END must be emitted when submit() encounters an auth error."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.events import EventKind, SessionEvent
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.errors import AuthenticationError

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=AuthenticationError("Invalid API key"))

        received: list[EventKind] = []
        session = Session(client=mock_client, config=SessionConfig())

        async def capture(e: SessionEvent) -> None:
            received.append(e.kind)

        session._emitter.on(capture)
        await session.submit("Hello")

        assert EventKind.SESSION_END in received, (
            "SESSION_END must be emitted when submit() results in an auth error "
            f"(CLOSED transition). Got events: {received}"
        )

    @pytest.mark.asyncio
    async def test_session_end_not_emitted_on_normal_turn(self):
        """SESSION_END must NOT be emitted after a normal turn (state stays IDLE)."""
        from unittest.mock import AsyncMock, MagicMock

        from attractor_agent.events import EventKind, SessionEvent
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import Message, Response, Usage

        mock_client = MagicMock()
        resp = Response(
            id="r1",
            model="m",
            content=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=5, output_tokens=5),
            provider="test",
        )
        resp.message = Message.assistant("Hi.")
        mock_client.complete = AsyncMock(return_value=resp)

        received: list[EventKind] = []
        session = Session(client=mock_client, config=SessionConfig())

        async def capture(e: SessionEvent) -> None:
            received.append(e.kind)

        session._emitter.on(capture)
        await session.submit("Hello")

        assert EventKind.SESSION_END not in received, (
            f"SESSION_END must NOT fire after a normal IDLE turn. Got: {received}"
        )

    @pytest.mark.asyncio
    async def test_session_end_emitted_on_abort(self):
        """SESSION_END must be emitted when session is aborted mid-run."""
        from unittest.mock import MagicMock

        from attractor_agent.events import EventKind, SessionEvent
        from attractor_agent.session import Session, SessionConfig
        from attractor_llm.types import Message, Response, Usage

        mock_client = MagicMock()

        # The complete() call will trigger abort on the session, then return normally.
        # abort_signal kwarg must be accepted because _call_llm passes it.
        async def complete_and_abort(request, abort_signal=None):
            session._abort.set()
            resp = Response(
                id="r1",
                model="m",
                content=[],
                stop_reason="end_turn",
                usage=Usage(input_tokens=5, output_tokens=5),
                provider="test",
            )
            resp.message = Message.assistant("Aborted.")
            return resp

        mock_client.complete = complete_and_abort

        received: list[EventKind] = []
        session = Session(client=mock_client, config=SessionConfig())

        async def capture(e: SessionEvent) -> None:
            received.append(e.kind)

        session._emitter.on(capture)
        await session.submit("Hello")

        assert EventKind.SESSION_END in received, (
            "SESSION_END must be emitted when session is aborted (CLOSED transition). "
            f"Got events: {received}"
        )


class TestMiddlewareChain:
    """Task 5 — §8.1.6: apply_middleware() wraps client correctly."""

    @pytest.mark.asyncio
    async def test_apply_middleware_calls_middleware_in_order(self):
        """apply_middleware() must call registered middleware in registration order:
        A(B(core)) means A_before -> B_before -> B_after -> A_after."""
        from typing import Any
        from unittest.mock import AsyncMock, MagicMock

        from attractor_llm.client import Client
        from attractor_llm.middleware import apply_middleware
        from attractor_llm.types import Message, Request, Response, Usage

        call_order: list[str] = []

        async def middleware_a(request: Any, call_next: Any) -> Any:
            call_order.append("A_before")
            response = await call_next(request)
            call_order.append("A_after")
            return response

        async def middleware_b(request: Any, call_next: Any) -> Any:
            call_order.append("B_before")
            response = await call_next(request)
            call_order.append("B_after")
            return response

        mock_resp = Response(
            id="r1",
            model="m",
            content=[],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="test",
        )
        mock_resp.message = Message.assistant("Hi.")

        base_client = MagicMock(spec=Client)
        base_client.complete = AsyncMock(return_value=mock_resp)

        wrapped = apply_middleware(base_client, [middleware_a, middleware_b])

        req = Request(model="test", messages=[Message.user("Hello")])
        await wrapped.complete(req)

        assert call_order == ["A_before", "B_before", "B_after", "A_after"], (
            f"Middleware must wrap in order: A(B(core)). Got: {call_order}"
        )

    def test_client_middleware_param_emits_deprecation_warning(self):
        """Client(middleware=[...]) must emit DeprecationWarning per §8.1.6."""
        import warnings
        from typing import Any

        from attractor_llm.client import Client

        async def noop(req: Any, call_next: Any) -> Any:
            return await call_next(req)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Client(middleware=[noop])
        assert len(w) >= 1, "Client(middleware=[...]) must emit at least one warning"
        categories = [x.category for x in w]
        assert DeprecationWarning in categories, f"Expected DeprecationWarning, got: {categories}"
        messages = [str(x.message).lower() for x in w]
        assert any("deprecated" in m or "apply_middleware" in m for m in messages), (
            f"Warning must mention 'deprecated' or 'apply_middleware'. Got: {messages}"
        )
