"""Tests for Wave 8: Interactive subagent tools (Spec §2.8, §9.9).

Validates SubagentManager tracking, send_input/wait/close_agent
operations, and Tool registration via create_interactive_tools().
"""

from __future__ import annotations

import asyncio

import pytest

from attractor_agent.abort import AbortSignal
from attractor_agent.session import Session
from attractor_agent.subagent_manager import (
    SubagentManager,
    TrackedSubagent,
    create_interactive_tools,
)
from attractor_llm.client import Client

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _register_fake_agent(
    manager: SubagentManager,
    agent_id: str,
    *,
    result: str = "test output",
    abort: AbortSignal | None = None,
    delay: float = 0,
) -> TrackedSubagent:
    """Register a fake tracked subagent directly in the manager.

    Uses a simple coroutine instead of a real LLM-backed session loop,
    allowing deterministic testing of the manager's tracking and
    communication logic.
    """
    abort = abort or AbortSignal()
    session = Session(client=Client(), abort_signal=abort)

    async def _fake_work() -> str:
        if delay > 0:
            await asyncio.sleep(delay)
        return result

    task = asyncio.create_task(_fake_work(), name=f"fake-{agent_id}")
    tracked = TrackedSubagent(
        agent_id=agent_id,
        session=session,
        task=task,
        abort_signal=abort,
    )
    manager._agents[agent_id] = tracked
    return tracked


def _register_failing_agent(
    manager: SubagentManager,
    agent_id: str,
    *,
    error: Exception | None = None,
) -> TrackedSubagent:
    """Register a subagent whose task will raise an exception."""
    abort = AbortSignal()
    session = Session(client=Client(), abort_signal=abort)

    async def _fail() -> str:
        raise (error or RuntimeError("boom"))

    task = asyncio.create_task(_fail(), name=f"fail-{agent_id}")
    tracked = TrackedSubagent(
        agent_id=agent_id,
        session=session,
        task=task,
        abort_signal=abort,
    )
    manager._agents[agent_id] = tracked
    return tracked


# ================================================================== #
# SubagentManager -- tracking
# ================================================================== #


class TestSubagentManagerTracking:
    """SubagentManager tracks agents by ID."""

    def test_initial_state_empty(self):
        manager = SubagentManager()
        assert manager.active_agents == {}

    @pytest.mark.asyncio
    async def test_tracks_registered_agent(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001")
        assert "agent-001" in manager.active_agents
        assert len(manager.active_agents) == 1

    @pytest.mark.asyncio
    async def test_tracks_multiple_agents(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001")
        _register_fake_agent(manager, "agent-002")
        assert len(manager.active_agents) == 2


# ================================================================== #
# send_input
# ================================================================== #


class TestSendInput:
    """send_input queues a message via Session.steer()."""

    @pytest.mark.asyncio
    async def test_steers_session(self):
        manager = SubagentManager()
        tracked = _register_fake_agent(manager, "agent-001", delay=5)

        result = manager.send_input("agent-001", "do something else")
        assert "sent" in result.lower()
        # Verify the steering queue received the message
        assert tracked.session._steer_queue == ["do something else"]

    @pytest.mark.asyncio
    async def test_multiple_messages_queue(self):
        manager = SubagentManager()
        tracked = _register_fake_agent(manager, "agent-001", delay=5)

        manager.send_input("agent-001", "first")
        manager.send_input("agent-001", "second")
        assert tracked.session._steer_queue == ["first", "second"]

    @pytest.mark.asyncio
    async def test_nonexistent_agent_returns_error(self):
        manager = SubagentManager()
        result = manager.send_input("no-such-agent", "hello")
        assert "error" in result.lower()
        assert "no-such-agent" in result

    @pytest.mark.asyncio
    async def test_completed_agent_returns_error(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-done", delay=0)
        # Let the task complete
        await asyncio.sleep(0.02)

        result = manager.send_input("agent-done", "too late")
        assert "error" in result.lower()
        assert "completed" in result.lower()


# ================================================================== #
# wait_for_output
# ================================================================== #


class TestWaitForOutput:
    """wait_for_output returns a JSON SubAgentResult (Spec §7.3)."""

    @pytest.mark.asyncio
    async def test_returns_result(self):
        import json

        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="task complete")

        output = await manager.wait_for_output("agent-001")
        parsed = json.loads(output)
        assert parsed["output"] == "task complete"
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_removes_agent_after_wait(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="done")

        await manager.wait_for_output("agent-001")
        assert "agent-001" not in manager.active_agents
        assert manager.active_agents == {}

    @pytest.mark.asyncio
    async def test_nonexistent_agent_returns_error(self):
        import json

        manager = SubagentManager()
        result = await manager.wait_for_output("no-such-agent")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "no-such-agent" in parsed["output"]

    @pytest.mark.asyncio
    async def test_failed_agent_returns_error(self):
        import json

        manager = SubagentManager()
        _register_failing_agent(manager, "agent-fail", error=ValueError("bad input"))

        result = await manager.wait_for_output("agent-fail")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "ValueError" in parsed["output"]
        assert "bad input" in parsed["output"]

    @pytest.mark.asyncio
    async def test_failed_agent_still_removed(self):
        manager = SubagentManager()
        _register_failing_agent(manager, "agent-fail")

        await manager.wait_for_output("agent-fail")
        assert "agent-fail" not in manager.active_agents

    @pytest.mark.asyncio
    async def test_multiple_agents_independent(self):
        import json

        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="result-1")
        _register_fake_agent(manager, "agent-002", result="result-2")

        r1 = await manager.wait_for_output("agent-001")
        assert json.loads(r1)["output"] == "result-1"
        assert len(manager.active_agents) == 1

        r2 = await manager.wait_for_output("agent-002")
        assert json.loads(r2)["output"] == "result-2"
        assert len(manager.active_agents) == 0


# ================================================================== #
# close_agent
# ================================================================== #


class TestCloseAgent:
    """close_agent sets the abort signal and removes tracking."""

    @pytest.mark.asyncio
    async def test_sets_abort_signal(self):
        manager = SubagentManager()
        abort = AbortSignal()
        _register_fake_agent(manager, "agent-001", abort=abort, delay=10)

        assert not abort.is_set
        result = manager.close_agent("agent-001")
        assert abort.is_set
        assert "terminated" in result.lower()

    @pytest.mark.asyncio
    async def test_removes_from_tracking(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", delay=10)

        manager.close_agent("agent-001")
        assert "agent-001" not in manager.active_agents

    @pytest.mark.asyncio
    async def test_nonexistent_agent_returns_error(self):
        manager = SubagentManager()
        result = manager.close_agent("no-such-agent")
        assert "error" in result.lower()
        assert "no-such-agent" in result


# ================================================================== #
# Tool definitions via create_interactive_tools
# ================================================================== #


class TestInteractiveTools:
    """create_interactive_tools produces correct Tool objects."""

    def test_creates_four_tools(self):
        # spawn_agent was added in Audit 2 Wave 4 (§9.12.34-36)
        manager = SubagentManager()
        tools = create_interactive_tools(manager)
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"spawn_agent", "send_input", "wait", "close_agent"}

    def test_tools_have_execute_handlers(self):
        manager = SubagentManager()
        tools = create_interactive_tools(manager)
        for tool in tools:
            assert tool.execute is not None

    def test_tool_parameters(self):
        manager = SubagentManager()
        tools = create_interactive_tools(manager)
        tool_map = {t.name: t for t in tools}

        # send_input requires agent_id and message
        send = tool_map["send_input"]
        assert "agent_id" in send.parameters["properties"]
        assert "message" in send.parameters["properties"]
        assert send.parameters["required"] == ["agent_id", "message"]

        # wait requires agent_id
        wait = tool_map["wait"]
        assert "agent_id" in wait.parameters["properties"]
        assert wait.parameters["required"] == ["agent_id"]

        # close_agent requires agent_id
        close = tool_map["close_agent"]
        assert "agent_id" in close.parameters["properties"]
        assert close.parameters["required"] == ["agent_id"]

    @pytest.mark.asyncio
    async def test_send_input_tool_delegates(self):
        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", delay=5)
        tools = create_interactive_tools(manager)
        send_tool = next(t for t in tools if t.name == "send_input")

        result = await send_tool.execute(agent_id="agent-001", message="hello")
        assert "sent" in result.lower()

    @pytest.mark.asyncio
    async def test_wait_tool_delegates(self):
        import json

        manager = SubagentManager()
        _register_fake_agent(manager, "agent-001", result="final answer")
        tools = create_interactive_tools(manager)
        wait_tool = next(t for t in tools if t.name == "wait")

        result = await wait_tool.execute(agent_id="agent-001")
        parsed = json.loads(result)
        assert parsed["output"] == "final answer"
        assert parsed["success"] is True

    @pytest.mark.asyncio
    async def test_close_agent_tool_delegates(self):
        manager = SubagentManager()
        abort = AbortSignal()
        _register_fake_agent(manager, "agent-001", abort=abort, delay=10)
        tools = create_interactive_tools(manager)
        close_tool = next(t for t in tools if t.name == "close_agent")

        result = await close_tool.execute(agent_id="agent-001")
        assert "terminated" in result.lower()
        assert abort.is_set
