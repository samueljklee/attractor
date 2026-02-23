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

import asyncio
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from attractor_agent.session import SessionConfig
from attractor_agent.subagent import spawn_subagent
from attractor_pipeline.server.app import app


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
        from unittest.mock import AsyncMock

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
        from unittest.mock import AsyncMock

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

    def test_apply_middleware_mixed_styles_raises_type_error(self):
        """apply_middleware() must raise TypeError for mixed protocol/functional styles."""
        from unittest.mock import MagicMock

        from attractor_llm.client import Client
        from attractor_llm.middleware import Middleware, apply_middleware

        class ProtocolMiddleware(Middleware):
            def before_request(self, req):
                return req

            def after_response(self, req, resp):
                return resp

        async def functional_mw(req, call_next):
            return await call_next(req)

        base = MagicMock(spec=Client)
        with pytest.raises(TypeError, match="mixing protocol-style"):
            apply_middleware(base, [ProtocolMiddleware(), functional_mw])

    def test_apply_middleware_empty_list_returns_wrapped_client(self):
        """apply_middleware([]) must return a wrapped client (not raise)."""
        from unittest.mock import MagicMock

        from attractor_llm.client import Client
        from attractor_llm.middleware import MiddlewareClient, apply_middleware

        base = MagicMock(spec=Client)
        result = apply_middleware(base, [])
        assert isinstance(result, MiddlewareClient), (
            "apply_middleware with empty list must return a MiddlewareClient"
        )


class TestHttpServer:
    """Task 6 — §11.11.5: POST /run calls run_pipeline, not a stub."""

    @pytest.fixture(autouse=True)
    def clear_runs(self) -> Generator[None, None, None]:
        """Reset module-level _runs between tests (§11.11.5)."""
        from attractor_pipeline.server import app as server_module

        server_module._runs.clear()
        yield
        server_module._runs.clear()

    @pytest.mark.asyncio
    async def test_post_run_calls_run_pipeline(self):
        """POST /run must call run_pipeline(), not just sleep(0)."""
        pipeline_called = []

        async def mock_run_pipeline(*args, **kwargs):
            pipeline_called.append(args)
            mock_result = MagicMock()
            mock_result.status = "completed"
            mock_result.context = {}
            return mock_result

        # Patch wherever run_pipeline is imported in app.py
        with patch("attractor_pipeline.server.app.run_pipeline", mock_run_pipeline):
            client = TestClient(app)
            response = client.post("/run", json={"pipeline": {}, "input": {}})

        assert response.status_code == 202, f"Expected 202, got {response.status_code}"
        run_id = response.json()["id"]
        # Direct task inspection: wait for background task (§11.11.5)
        from attractor_pipeline.server.app import _runs

        task = _runs[run_id]["task"]
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        assert len(pipeline_called) > 0, "run_pipeline must be called when POST /run is received"

    @pytest.mark.asyncio
    async def test_status_transitions_to_completed(self):
        """Status must transition pending -> running -> completed after run_pipeline returns."""

        async def mock_run_pipeline(*args: Any, **kwargs: Any) -> Any:
            result = MagicMock()
            result.status = "completed"
            result.context = {"output": "done"}
            return result

        with patch("attractor_pipeline.server.app.run_pipeline", mock_run_pipeline):
            client = TestClient(app)
            response = client.post("/run", json={"pipeline": {}, "input": "hello"})
            assert response.status_code == 202
            run_id = response.json()["id"]

        # Direct task inspection: wait for background task (§11.11.5)
        from attractor_pipeline.server.app import _runs

        task = _runs[run_id]["task"]
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

        status_resp = client.get(f"/status/{run_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "completed", (
            f"Expected 'completed', got: {status_resp.json()}"
        )

    @pytest.mark.asyncio
    async def test_status_transitions_to_failed_on_exception(self):
        """Status must transition to 'failed' when run_pipeline raises."""

        async def failing_run_pipeline(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Pipeline execution failed")

        with patch("attractor_pipeline.server.app.run_pipeline", failing_run_pipeline):
            client = TestClient(app)
            response = client.post("/run", json={"pipeline": {}, "input": "hello"})
            assert response.status_code == 202
            run_id = response.json()["id"]

        # Direct task inspection: wait for background task (§11.11.5)
        from attractor_pipeline.server.app import _runs

        task = _runs[run_id]["task"]
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

        status_resp = client.get(f"/status/{run_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed", f"Expected 'failed', got: {data}"
        assert "error" in data, "Failed run must include error field"

    @pytest.mark.asyncio
    async def test_null_pipeline_completes_immediately(self):
        """POST /run with null pipeline field completes without calling run_pipeline."""
        pipeline_called = []

        async def mock_run_pipeline(*args: Any, **kwargs: Any) -> Any:
            pipeline_called.append(True)
            return MagicMock(status="completed", context={})

        with patch("attractor_pipeline.server.app.run_pipeline", mock_run_pipeline):
            client = TestClient(app)
            response = client.post("/run", json={"pipeline": None, "input": "hello"})
            assert response.status_code == 202
            run_id = response.json()["id"]

        # Direct task inspection: wait for background task (§11.11.5)
        from attractor_pipeline.server.app import _runs

        task = _runs[run_id]["task"]
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)

        status_resp = client.get(f"/status/{run_id}")
        assert status_resp.json()["status"] == "completed"
        assert len(pipeline_called) == 0, "run_pipeline must NOT be called when pipeline is null"


class TestInterviewerAnswer:
    """Task 7 — §11.8.1: Interviewer.ask() must accept Question and return Answer."""

    @pytest.mark.asyncio
    async def test_queue_interviewer_ask_returns_answer(self):
        """QueueInterviewer.ask() must return Answer per spec §6.1."""
        from attractor_pipeline.handlers.human import Answer, Question, QueueInterviewer

        interviewer = QueueInterviewer(["yes"])
        question = Question(text="Are you sure?")
        result = await interviewer.ask(question)
        assert isinstance(result, Answer), (
            f"ask() must return Answer per spec §6.1, got {type(result)}"
        )
        assert result.value == "yes"

    @pytest.mark.asyncio
    async def test_queue_interviewer_ask_question_returns_answer(self):
        """QueueInterviewer.ask_question() must return Answer (rich API)."""
        from attractor_pipeline.handlers.human import Answer, Question, QueueInterviewer

        interviewer = QueueInterviewer(["yes"])
        question = Question(text="Are you sure?")
        answer = await interviewer.ask_question(question)
        assert isinstance(answer, Answer), f"ask_question() must return Answer, got {type(answer)}"
        assert answer.value == "yes"

    @pytest.mark.asyncio
    async def test_auto_approve_interviewer_ask_returns_answer(self):
        """AutoApproveInterviewer.ask() must return Answer per spec §6.4."""
        from attractor_pipeline.handlers.human import Answer, AutoApproveInterviewer, Question

        interviewer = AutoApproveInterviewer()
        question = Question(text="Approve?")
        result = await interviewer.ask(question)
        assert isinstance(result, Answer), (
            f"AutoApproveInterviewer.ask() must return Answer, got {type(result)}"
        )


class TestAnthropicDescriptions:
    """Task 9 — §9.2.6: Anthropic profile must not overwrite caller tool descriptions."""

    def test_caller_description_preserved_over_anthropic_override(self):
        """When caller supplies a description for edit_file, it must not be replaced."""
        from attractor_agent.profiles.anthropic import AnthropicProfile
        from attractor_llm.types import Tool

        caller_desc = "MY CUSTOM edit_file description that must be preserved"
        tool = Tool(
            name="edit_file",
            description=caller_desc,
            parameters={"type": "object", "properties": {}},
            execute=lambda **kw: "ok",
        )
        profile = AnthropicProfile()
        result_tools = profile.get_tools([tool])
        assert result_tools[0].description == caller_desc, (
            f"Caller description '{caller_desc}' was overwritten with "
            f"'{result_tools[0].description}'"
        )

    def test_anthropic_override_applied_when_description_empty(self):
        """Anthropic override is applied when caller description is None/empty."""
        from attractor_agent.profiles.anthropic import (
            _ANTHROPIC_TOOL_DESCRIPTIONS,
            AnthropicProfile,
        )
        from attractor_llm.types import Tool

        assert "edit_file" in _ANTHROPIC_TOOL_DESCRIPTIONS, (
            "edit_file must be in Anthropic override map"
        )

        tool_no_desc = Tool(
            name="edit_file",
            description="",
            parameters={"type": "object", "properties": {}},
            execute=lambda **kw: "ok",
        )
        profile = AnthropicProfile()
        result_tools = profile.get_tools([tool_no_desc])
        assert result_tools[0].description == _ANTHROPIC_TOOL_DESCRIPTIONS["edit_file"], (
            "Anthropic override must be applied when caller description is empty"
        )

    def test_unknown_tool_description_preserved(self):
        """Tools not in the Anthropic override map keep their original description."""
        from attractor_agent.profiles.anthropic import AnthropicProfile
        from attractor_llm.types import Tool

        tool = Tool(
            name="my_custom_tool",
            description="Does something custom",
            parameters={"type": "object", "properties": {}},
            execute=lambda **kw: "ok",
        )
        profile = AnthropicProfile()
        result_tools = profile.get_tools([tool])
        assert result_tools[0].description == "Does something custom", (
            "Unknown tool description must be preserved unchanged"
        )


class TestApplyPatchV4a:
    """Task 10 — Appendix A: apply_patch handles v4a '*** Begin Patch' format."""

    @pytest.mark.asyncio
    async def test_v4a_add_file(self, tmp_path):
        """*** Add File: creates a new file with the given lines."""
        from attractor_agent.tools.core import _apply_patch, set_allowed_roots

        set_allowed_roots([str(tmp_path)])

        patch = (
            "*** Begin Patch\n"
            f"*** Add File: {tmp_path}/hello.py\n"
            "+def greet():\n"
            '+    return "Hello"\n'
            "*** End Patch\n"
        )
        await _apply_patch(patch)

        result_file = tmp_path / "hello.py"
        assert result_file.exists(), "*** Add File must create the file"
        content = result_file.read_text()
        assert "def greet" in content

    @pytest.mark.asyncio
    async def test_v4a_update_file(self, tmp_path):
        """*** Update File: modifies an existing file using context-based hunks."""
        from attractor_agent.tools.core import _apply_patch, set_allowed_roots

        set_allowed_roots([str(tmp_path)])

        target = tmp_path / "config.py"
        target.write_text("DEBUG = False\nTIMEOUT = 30\n")

        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {tmp_path}/config.py\n"
            "@@ DEBUG\n"
            "-DEBUG = False\n"
            "+DEBUG = True\n"
            "*** End Patch\n"
        )
        await _apply_patch(patch)

        content = target.read_text()
        assert "DEBUG = True" in content
        assert "DEBUG = False" not in content
        assert "TIMEOUT = 30" in content

    @pytest.mark.asyncio
    async def test_v4a_delete_file(self, tmp_path):
        """*** Delete File: removes the target file."""
        from attractor_agent.tools.core import _apply_patch, set_allowed_roots

        set_allowed_roots([str(tmp_path)])

        target = tmp_path / "old.py"
        target.write_text("print('old')\n")
        assert target.exists()

        patch = f"*** Begin Patch\n*** Delete File: {tmp_path}/old.py\n*** End Patch\n"
        await _apply_patch(patch)

        assert not target.exists(), "*** Delete File must remove the file"

    @pytest.mark.asyncio
    async def test_standard_unified_diff_still_works(self, tmp_path):
        """Standard --- a/ +++ b/ patches must still be handled (backward compat)."""
        from attractor_agent.tools.core import _apply_patch, set_allowed_roots

        set_allowed_roots([str(tmp_path)])

        target = tmp_path / "foo.py"
        target.write_text("x = 1\ny = 2\n")

        patch = (
            f"--- a/{tmp_path}/foo.py\n"
            f"+++ b/{tmp_path}/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-x = 1\n"
            "+x = 99\n"
            " y = 2\n"
        )
        await _apply_patch(patch)

        content = target.read_text()
        assert "x = 99" in content
