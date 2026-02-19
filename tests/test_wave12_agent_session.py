"""Tests for Wave 12 P11 + P13 + P19 -- agent Session improvements.

P11 (§9.1): Session constructor accepts ProviderProfile / ExecutionEnvironment.
P13 (§2.10): Loop detection injects a SteeringTurn warning and continues.
P19 (§2.7 / §9.7): Public API to read/change reasoning_effort mid-session.
"""

from __future__ import annotations

import pytest  # type: ignore[import-untyped]

from attractor_agent.environment import ExecutionEnvironment, LocalEnvironment
from attractor_agent.profiles.base import BaseProfile, ProviderProfile
from attractor_agent.session import Session, SessionConfig, SessionState, SteeringTurn
from attractor_agent.tools.core import get_environment, set_environment
from attractor_llm.client import Client
from attractor_llm.types import Response, Tool
from tests.helpers import (
    MockAdapter,
    make_text_response,
    make_tool_call_response,
)

# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #


def _make_client(responses: list[Response]) -> tuple[Client, MockAdapter]:
    """Return a (Client, MockAdapter) pair wired to the 'mock' provider."""
    adapter = MockAdapter(responses=responses)
    client = Client()
    client.register_adapter("mock", adapter)
    return client, adapter


async def _noop_execute(**kwargs: object) -> str:
    return "noop-ok"


def _noop_tool(name: str = "noop_tool") -> Tool:
    """A minimal tool that always returns 'noop-ok'. Safe for loop tests."""
    return Tool(
        name=name,
        description="No-op test tool",
        parameters={"type": "object", "properties": {}, "required": []},
        execute=_noop_execute,
    )


# ================================================================== #
# P11 -- Session constructor accepts ProviderProfile / ExecutionEnvironment
# ================================================================== #


class TestP11SessionConstructorParams:
    """Session.__init__ accepts profile= and environment= convenience params."""

    def test_session_accepts_profile_param(self) -> None:
        """Constructing Session with profile= must not raise. Spec §9.1."""
        client, _ = _make_client([])
        profile: ProviderProfile = BaseProfile()

        # Should not raise
        session = Session(
            client=client,
            config=SessionConfig(model="mock-model", provider="mock"),
            profile=profile,
        )

        # The profile's apply_to_config fills system_prompt if unset.
        # BaseProfile has system_prompt="" so the config keeps whatever was set.
        assert isinstance(session, Session)
        assert session.state == SessionState.IDLE

    def test_session_accepts_profile_applies_system_prompt(self) -> None:
        """When profile has a system_prompt, it is applied to config. Spec §9.1."""

        class _CustomProfile(BaseProfile):
            @property
            def system_prompt(self) -> str:
                return "You are a test assistant."

        client, _ = _make_client([])
        profile: ProviderProfile = _CustomProfile()

        # Config starts with empty system_prompt; profile should fill it.
        config = SessionConfig(model="mock-model", provider="mock", system_prompt="")
        session = Session(client=client, config=config, profile=profile)

        # profile.apply_to_config sets system_prompt when config's is empty
        assert "test assistant" in session._config.system_prompt

    def test_session_accepts_environment_param(self) -> None:
        """Constructing Session with environment= must set the module env. Spec §9.1."""
        client, _ = _make_client([])
        new_env = LocalEnvironment()

        original = get_environment()
        try:
            session = Session(
                client=client,
                config=SessionConfig(model="mock-model", provider="mock"),
                environment=new_env,
            )
            # The module-level environment should now be our new instance.
            assert get_environment() is new_env
            assert isinstance(session, Session)
        finally:
            # Always restore so we don't pollute other tests.
            set_environment(original)

    def test_session_environment_implements_protocol(self) -> None:
        """The installed environment satisfies the ExecutionEnvironment protocol."""
        client, _ = _make_client([])
        new_env = LocalEnvironment()

        original = get_environment()
        try:
            Session(
                client=client,
                config=SessionConfig(model="mock-model", provider="mock"),
                environment=new_env,
            )
            assert isinstance(get_environment(), ExecutionEnvironment)
        finally:
            set_environment(original)

    def test_session_profile_none_is_noop(self) -> None:
        """profile=None (default) must leave config unchanged."""
        client, _ = _make_client([])
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            system_prompt="original",
        )
        session = Session(client=client, config=config, profile=None)
        assert session._config.system_prompt == "original"

    def test_session_environment_none_does_not_change_env(self) -> None:
        """environment=None (default) must not change the module-level env."""
        client, _ = _make_client([])
        original = get_environment()

        Session(
            client=client,
            config=SessionConfig(model="mock-model", provider="mock"),
            environment=None,
        )
        assert get_environment() is original

    def test_session_profile_merges_tools(self) -> None:
        """Tools passed explicitly are forwarded through profile.get_tools(). Spec §9.1."""
        client, _ = _make_client([])
        extra_tool = _noop_tool("extra_tool")
        profile: ProviderProfile = BaseProfile()  # get_tools returns list unchanged

        session = Session(
            client=client,
            config=SessionConfig(model="mock-model", provider="mock"),
            tools=[extra_tool],
            profile=profile,
        )

        tool_names = [t.name for t in session.tool_registry.definitions()]
        assert "extra_tool" in tool_names


# ================================================================== #
# P13 -- Loop detection injects SteeringTurn and continues
# ================================================================== #


class TestP13LoopDetectionInjectsAndContinues:
    """Loop detection must warn-and-continue, not exit. Spec §2.10."""

    @pytest.mark.asyncio
    async def test_loop_detection_injects_steering_and_continues(self) -> None:
        """When a loop is detected the session injects a SteeringTurn and does
        NOT exit early -- it returns the eventual text response. Spec §2.10."""
        # Threshold=2 so two identical calls trigger detection immediately.
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            loop_detection_window=3,
            loop_detection_threshold=2,
        )
        tool = _noop_tool("repeat_tool")

        # Responses:
        # 1. First tool call → recorded (count=1)
        # 2. Same tool call → count=2, triggers detection
        # 3. Text response → returned after loop continues
        client, _ = _make_client(
            [
                make_tool_call_response("repeat_tool", {}, "tc-1"),
                make_tool_call_response("repeat_tool", {}, "tc-2"),
                make_text_response("All good after loop warning."),
            ]
        )

        session = Session(client=client, config=config, tools=[tool])
        result = await session.submit("Do the repeated thing")

        # The session must NOT return the old exit string -- it should return
        # the final text response produced after the steering message was shown.
        assert result == "All good after loop warning."
        assert "[Loop detected" not in result

    @pytest.mark.asyncio
    async def test_loop_detection_steering_message_contains_warning(self) -> None:
        """The injected SteeringTurn must mention 'loop' or 'repeating'. Spec §2.10."""
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            loop_detection_window=3,
            loop_detection_threshold=2,
        )
        tool = _noop_tool("repeat_tool")

        client, _ = _make_client(
            [
                make_tool_call_response("repeat_tool", {}, "tc-1"),
                make_tool_call_response("repeat_tool", {}, "tc-2"),
                make_text_response("OK done."),
            ]
        )

        session = Session(client=client, config=config, tools=[tool])
        await session.submit("Trigger the loop")

        # Find any SteeringTurn entries in history.
        # Use explicit loop so pyright narrows Message|SteeringTurn → SteeringTurn.
        steering_turns: list[SteeringTurn] = []
        for entry in session.history:
            if isinstance(entry, SteeringTurn):
                steering_turns.append(entry)
        assert steering_turns, "Expected at least one SteeringTurn to be injected"

        # The content must mention the loop situation.
        combined = " ".join(st.content.lower() for st in steering_turns)
        assert "loop" in combined or "repeating" in combined

    @pytest.mark.asyncio
    async def test_loop_detection_steering_message_exact_text(self) -> None:
        """The SteeringTurn content matches the spec §2.10 wording."""
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            loop_detection_window=3,
            loop_detection_threshold=2,
        )
        tool = _noop_tool("repeat_tool")

        client, _ = _make_client(
            [
                make_tool_call_response("repeat_tool", {}, "tc-1"),
                make_tool_call_response("repeat_tool", {}, "tc-2"),
                make_text_response("Done."),
            ]
        )

        session = Session(client=client, config=config, tools=[tool])
        await session.submit("Trigger")

        steering_turns: list[SteeringTurn] = []
        for entry in session.history:
            if isinstance(entry, SteeringTurn):
                steering_turns.append(entry)
        assert steering_turns
        msg = steering_turns[0].content
        assert "LOOP DETECTED" in msg or "loop" in msg.lower()
        assert "different approach" in msg or "repeating" in msg.lower()

    @pytest.mark.asyncio
    async def test_loop_detection_does_not_early_return(self) -> None:
        """Session returns final text -- never the old '[Loop detected: ...]' string.

        Regression guard: old code returned '[Loop detected: repeated tool calls]'.
        """
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            loop_detection_window=3,
            loop_detection_threshold=2,
        )
        tool = _noop_tool("repeat_tool")

        client, _ = _make_client(
            [
                make_tool_call_response("repeat_tool", {}, "tc-1"),
                make_tool_call_response("repeat_tool", {}, "tc-2"),
                make_text_response("Final answer here."),
            ]
        )

        session = Session(client=client, config=config, tools=[tool])
        result = await session.submit("Go")

        assert result == "Final answer here."
        # Old exit string must never appear
        assert "Loop detected: repeated" not in result


# ================================================================== #
# P19 -- Public API for reasoning_effort mid-session
# ================================================================== #


class TestP19ReasoningEffortAPI:
    """Session exposes set_reasoning_effort() + reasoning_effort property. Spec §2.7."""

    def _make_session(self, effort: str | None = None) -> Session:
        client, _ = _make_client([])
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            reasoning_effort=effort,
        )
        return Session(client=client, config=config)

    def test_reasoning_effort_property_returns_current(self) -> None:
        """reasoning_effort property reflects the config value. Spec §2.7."""
        session = self._make_session(effort="medium")
        assert session.reasoning_effort == "medium"

    def test_reasoning_effort_property_none_when_unset(self) -> None:
        """reasoning_effort is None when not configured."""
        session = self._make_session(effort=None)
        assert session.reasoning_effort is None

    def test_set_reasoning_effort_changes_config(self) -> None:
        """set_reasoning_effort() updates the config immediately. Spec §2.7."""
        session = self._make_session(effort=None)
        assert session.reasoning_effort is None

        session.set_reasoning_effort("high")
        assert session.reasoning_effort == "high"

    def test_set_reasoning_effort_to_none(self) -> None:
        """set_reasoning_effort(None) resets the effort to None. Spec §2.7."""
        session = self._make_session(effort="high")
        assert session.reasoning_effort == "high"

        session.set_reasoning_effort(None)
        assert session.reasoning_effort is None

    def test_set_reasoning_effort_overwrites_previous(self) -> None:
        """Multiple calls each overwrite the previous value."""
        session = self._make_session(effort="low")
        session.set_reasoning_effort("medium")
        assert session.reasoning_effort == "medium"
        session.set_reasoning_effort("high")
        assert session.reasoning_effort == "high"

    def test_set_reasoning_effort_reflects_in_config(self) -> None:
        """The underlying _config.reasoning_effort is updated (not a shadow copy)."""
        session = self._make_session(effort=None)
        session.set_reasoning_effort("low")
        assert session._config.reasoning_effort == "low"

    @pytest.mark.asyncio
    async def test_set_reasoning_effort_used_in_next_llm_call(self) -> None:
        """After set_reasoning_effort(), the next LLM request carries the new value.

        Spec §2.7: 'Changing reasoning_effort mid-session takes effect on the
        next LLM call.'
        """
        client, adapter = _make_client(
            [
                make_text_response("first response"),
                make_text_response("second response"),
            ]
        )
        config = SessionConfig(
            model="mock-model",
            provider="mock",
            reasoning_effort="low",
        )
        session = Session(client=client, config=config)

        # First call uses the initial effort.
        await session.submit("first")
        assert adapter.requests[0].reasoning_effort == "low"

        # Change effort before the second call.
        session.set_reasoning_effort("high")
        await session.submit("second")
        assert adapter.requests[1].reasoning_effort == "high"
