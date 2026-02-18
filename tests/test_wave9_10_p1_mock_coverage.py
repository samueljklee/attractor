# Mock tests for Wave 9, 10, and P1 behaviors not yet covered elsewhere.
#
# Wave 9 behaviors:
#   - Anthropic tool_choice=none omits tools key (S8.7)
#   - user_instructions appended last in enriched system prompt (S9.8)
#   - spawn_subagent working_dir wired through to SessionConfig (S7.x)
#   - Truncation marker text matches spec (S5.1)
#   - Truncation char limits per tool (S5.2)
#   - Start node shape is Mdiamond (S2.4)
#   - Validation R03/R04 severity is ERROR (S7.2)
#
# Wave 10 behaviors (complementary to test_wave10_streaming_generation.py):
#   - StreamResult.response() accumulates events into a Response
#   - StreamResult.text_stream yields only text strings
#   - StreamResult.__aiter__ yields strings for backward compat
#   - StreamResult.iter_events() yields raw StreamEvent objects
#   - abort_signal on generate() raises AbortError
#   - abort_signal checked in tool execution loop
#   - TimeoutConfig default field values
#   - AdapterTimeout default field values
#   - New StreamEventKind enum values exist
#
# P1: Already covered in test_wave11_default_provider.py -- skipped here.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Wave 9 imports
from attractor_agent.session import Session, SessionConfig
from attractor_agent.subagent import spawn_subagent
from attractor_agent.truncation import TruncationLimits, truncate_output
from attractor_llm.adapters.anthropic import AnthropicAdapter
from attractor_llm.adapters.base import ProviderConfig
from attractor_llm.client import Client
from attractor_llm.errors import AbortError
from attractor_llm.generate import generate
from attractor_llm.streaming import StreamResult
from attractor_llm.types import (
    AdapterTimeout,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    StreamEvent,
    StreamEventKind,
    TimeoutConfig,
    Tool,
    Usage,
)
from attractor_pipeline.graph import Edge, Graph, Node, NodeShape
from attractor_pipeline.validation import Severity, validate

# ================================================================== #
# Helpers
# ================================================================== #


async def _event_stream(*events: StreamEvent):
    """Async generator that yields a fixed sequence of StreamEvents."""
    for event in events:
        yield event


def _text_response(text: str) -> Response:
    """Create a minimal text Response for mock use."""
    return Response(message=Message.assistant(text), usage=Usage())


# ================================================================== #
# Wave 9 S8.7 -- Anthropic tool_choice=none omits "tools" key
# ================================================================== #


class TestAnthropicToolChoiceNone:
    """When tool_choice='none', the Anthropic request body must NOT include 'tools'."""

    def _make_adapter(self) -> AnthropicAdapter:
        return AnthropicAdapter(ProviderConfig(api_key="test-key"))

    def _make_tool(self, name: str = "my_tool") -> Tool:
        return Tool(
            name=name,
            description="A test tool.",
            parameters={"type": "object", "properties": {}},
            execute=None,
        )

    def test_tool_choice_none_omits_tools_key(self):
        """Request body must have no 'tools' key when tool_choice='none'."""
        adapter = self._make_adapter()
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            tools=[self._make_tool()],
            tool_choice="none",
        )
        body = adapter._translate_request(request)

        assert "tools" not in body, "tools key must be absent when tool_choice='none'"

    def test_tool_choice_none_also_omits_tool_choice_key(self):
        """The tool_choice key itself must not appear either (Anthropic rejects it)."""
        adapter = self._make_adapter()
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            tools=[self._make_tool()],
            tool_choice="none",
        )
        body = adapter._translate_request(request)

        assert "tool_choice" not in body

    def test_tool_choice_none_without_tools_is_harmless(self):
        """tool_choice='none' with no tools provided produces a clean body."""
        adapter = self._make_adapter()
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            tool_choice="none",
        )
        body = adapter._translate_request(request)

        assert "tools" not in body
        assert "tool_choice" not in body

    def test_tool_choice_auto_keeps_tools(self):
        """Contrast: tool_choice='auto' must keep the tools array."""
        adapter = self._make_adapter()
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            tools=[self._make_tool()],
            tool_choice="auto",
        )
        body = adapter._translate_request(request)

        assert "tools" in body
        assert body["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required_maps_to_any(self):
        """tool_choice='required' maps to Anthropic type='any'."""
        adapter = self._make_adapter()
        request = Request(
            model="claude-sonnet-4-5",
            messages=[Message.user("Hello")],
            tools=[self._make_tool()],
            tool_choice="required",
        )
        body = adapter._translate_request(request)

        assert "tools" in body
        assert body["tool_choice"] == {"type": "any"}


# ================================================================== #
# Wave 9 S9.8 -- user_instructions appended LAST
# ================================================================== #


class TestUserInstructionsAppendedLast:
    """user_instructions must be the final segment of the enriched system prompt."""

    def test_user_instructions_after_system_prompt(self):
        """user_instructions appears after system_prompt in the enriched output."""
        from tests.helpers import MockAdapter

        config = SessionConfig(
            system_prompt="Base system instructions.",
            user_instructions="User override: be concise.",
        )
        client = Client()
        client.register_adapter("mock", MockAdapter(responses=[]))

        session = Session(client=client, config=config)
        enriched = session._build_enriched_system_prompt()

        # Both parts must be present
        assert "Base system instructions." in enriched
        assert "User override: be concise." in enriched

        # user_instructions must come AFTER system_prompt
        base_idx = enriched.index("Base system instructions.")
        user_idx = enriched.index("User override: be concise.")
        assert user_idx > base_idx, (
            "user_instructions should appear after system_prompt, "
            f"but system_prompt at {base_idx}, user_instructions at {user_idx}"
        )

    def test_user_instructions_is_final_part(self):
        """user_instructions is literally the last text in the enriched prompt."""
        from tests.helpers import MockAdapter

        config = SessionConfig(
            system_prompt="Profile prompt.",
            user_instructions="FINAL OVERRIDE TEXT.",
        )
        client = Client()
        client.register_adapter("mock", MockAdapter(responses=[]))

        session = Session(client=client, config=config)
        enriched = session._build_enriched_system_prompt()

        assert enriched.endswith("FINAL OVERRIDE TEXT."), (
            f"Enriched prompt should end with user_instructions, got: ...{enriched[-100:]!r}"
        )

    def test_empty_user_instructions_not_appended(self):
        """Empty user_instructions does not pollute the end of the prompt."""
        from tests.helpers import MockAdapter

        config = SessionConfig(
            system_prompt="Base prompt.",
            user_instructions="",
        )
        client = Client()
        client.register_adapter("mock", MockAdapter(responses=[]))

        session = Session(client=client, config=config)
        enriched = session._build_enriched_system_prompt()

        # Should contain the base prompt
        assert "Base prompt." in enriched
        # Should not end with empty whitespace introduced by a missing instruction
        assert enriched.strip()


# ================================================================== #
# Wave 9 S7.x -- spawn_subagent working_dir wired through
# ================================================================== #


class TestSpawnSubagentWorkingDir:
    """spawn_subagent() passes working_dir into SessionConfig."""

    def test_session_config_has_working_dir_field(self):
        """SessionConfig exposes a working_dir field (defaults to None)."""
        cfg = SessionConfig()
        assert cfg.working_dir is None

        cfg2 = SessionConfig(working_dir="/some/custom/path")
        assert cfg2.working_dir == "/some/custom/path"

    @pytest.mark.asyncio
    async def test_spawn_subagent_wires_working_dir_to_session_config(self):
        """working_dir passed to spawn_subagent ends up in the Session's config."""
        from tests.helpers import MockAdapter

        captured_configs: list[SessionConfig] = []

        # Capture the config that Session receives, then short-circuit submit()
        original_init = Session.__init__

        def capturing_init(self, *, client, config=None, tools=None, abort_signal=None):
            original_init(
                self, client=client, config=config, tools=tools, abort_signal=abort_signal
            )
            captured_configs.append(self._config)

        with (
            patch.object(Session, "__init__", capturing_init),
            patch.object(Session, "submit", new_callable=AsyncMock, return_value="done"),
        ):
            client = Client()
            client.register_adapter("mock", MockAdapter(responses=[]))

            await spawn_subagent(
                client,
                "do something",
                working_dir="/custom/work/dir",
            )

        assert len(captured_configs) >= 1
        assert captured_configs[0].working_dir == "/custom/work/dir"

    @pytest.mark.asyncio
    async def test_spawn_subagent_none_working_dir_is_none_in_config(self):
        """If working_dir is not supplied, SessionConfig.working_dir is None."""
        from tests.helpers import MockAdapter

        captured_configs: list[SessionConfig] = []

        original_init = Session.__init__

        def capturing_init(self, *, client, config=None, tools=None, abort_signal=None):
            original_init(
                self, client=client, config=config, tools=tools, abort_signal=abort_signal
            )
            captured_configs.append(self._config)

        with (
            patch.object(Session, "__init__", capturing_init),
            patch.object(Session, "submit", new_callable=AsyncMock, return_value="done"),
        ):
            client = Client()
            client.register_adapter("mock", MockAdapter(responses=[]))

            await spawn_subagent(client, "do something")

        assert len(captured_configs) >= 1
        assert captured_configs[0].working_dir is None


# ================================================================== #
# Wave 9 S5.1 -- Truncation marker text matches spec
# ================================================================== #


class TestTruncationMarkerText:
    """The WARNING banner inserted by truncate_output matches the spec text."""

    def test_char_truncation_marker_contains_removed_from_middle(self):
        """Character truncation banner says 'characters were removed from the middle'."""
        limits = TruncationLimits(max_chars=100, max_lines=100_000)
        content = "A" * 300  # 3x over the limit

        result, was_truncated = truncate_output(content, limits)

        assert was_truncated
        assert "characters were removed from the middle" in result

    def test_char_truncation_marker_contains_re_run_tip(self):
        """Character truncation banner says 're-run the tool with more targeted parameters'."""
        limits = TruncationLimits(max_chars=100, max_lines=100_000)
        content = "B" * 300

        result, was_truncated = truncate_output(content, limits)

        assert was_truncated
        assert "re-run the tool with more targeted parameters" in result

    def test_char_truncation_marker_contains_warning(self):
        """Character truncation banner includes the word WARNING."""
        limits = TruncationLimits(max_chars=50, max_lines=100_000)
        content = "C" * 200

        result, was_truncated = truncate_output(content, limits)

        assert was_truncated
        assert "WARNING" in result

    def test_char_truncation_marker_includes_omitted_count(self):
        """Omitted character count appears numerically in the banner."""
        limits = TruncationLimits(max_chars=100, max_lines=100_000, head_ratio=0.5)
        # 200 chars: head=50, tail=50, omitted=100
        content = "D" * 200

        result, was_truncated = truncate_output(content, limits)

        assert was_truncated
        # The omitted count (100) must appear in the marker
        assert "100" in result

    def test_line_truncation_marker_contains_re_run_tip(self):
        """Line-based truncation banner also contains the re-run tip."""
        # Huge char limit so only lines trigger truncation
        limits = TruncationLimits(max_chars=500_000, max_lines=5)
        content = "\n".join(["line"] * 20)  # 20 lines, over 5-line limit

        result, was_truncated = truncate_output(content, limits)

        assert was_truncated
        assert "re-run the tool with more targeted parameters" in result


# ================================================================== #
# Wave 9 S5.2 -- Truncation char limits per tool
# ================================================================== #


class TestTruncationCharLimits:
    """Default max_chars for each built-in tool preset matches the spec."""

    def test_glob_char_limit_is_20000(self):
        limits = TruncationLimits.for_tool("glob")
        assert limits.max_chars == 20_000

    def test_edit_file_char_limit_is_10000(self):
        limits = TruncationLimits.for_tool("edit_file")
        assert limits.max_chars == 10_000

    def test_write_file_char_limit_is_1000(self):
        limits = TruncationLimits.for_tool("write_file")
        assert limits.max_chars == 1_000

    def test_apply_patch_char_limit_is_10000(self):
        limits = TruncationLimits.for_tool("apply_patch")
        assert limits.max_chars == 10_000

    def test_spawn_agent_char_limit_is_20000(self):
        limits = TruncationLimits.for_tool("spawn_agent")
        assert limits.max_chars == 20_000

    def test_unknown_tool_falls_back_to_class_default(self):
        """Unrecognised tool name returns the class-level defaults."""
        limits = TruncationLimits.for_tool("unknown_tool_xyz")
        default = TruncationLimits()
        assert limits.max_chars == default.max_chars

    def test_config_override_beats_preset(self):
        """output_limits dict overrides the built-in preset for a named tool."""
        overrides = {"glob": 5_000}
        limits = TruncationLimits.for_tool("glob", output_limits=overrides)
        assert limits.max_chars == 5_000


# ================================================================== #
# Wave 9 S2.4 -- Start node shape is Mdiamond
# ================================================================== #


class TestStartNodeShape:
    """Graph.get_start_node() identifies the node with shape='Mdiamond'."""

    def test_get_start_node_finds_mdiamond(self):
        """The Mdiamond node is returned by get_start_node()."""
        graph = Graph(name="test")
        graph.nodes["start"] = Node(id="start", shape="Mdiamond")
        graph.nodes["task"] = Node(id="task", shape="box")
        graph.nodes["end"] = Node(id="end", shape="Msquare")

        start = graph.get_start_node()

        assert start is not None
        assert start.id == "start"
        assert start.shape == "Mdiamond"

    def test_get_start_node_returns_none_if_absent(self):
        """get_start_node() returns None when no Mdiamond node exists."""
        graph = Graph(name="test")
        graph.nodes["task"] = Node(id="task", shape="box")

        assert graph.get_start_node() is None

    def test_node_shape_enum_mdiamond_value(self):
        """NodeShape.MDIAMOND has the string value 'Mdiamond'."""
        assert NodeShape.MDIAMOND == "Mdiamond"

    def test_mdiamond_maps_to_start_handler(self):
        """shape='Mdiamond' maps to the 'start' handler type."""
        assert NodeShape.handler_for_shape("Mdiamond") == "start"

    def test_effective_handler_of_mdiamond_node(self):
        """A Node with shape='Mdiamond' reports effective_handler='start'."""
        node = Node(id="entry", shape="Mdiamond")
        assert node.effective_handler == "start"


# ================================================================== #
# Wave 9 S7.2 -- Validation R03/R04 severity is ERROR
# ================================================================== #


class TestValidationR03R04Severity:
    """R03 and R04 violations produce ERROR-level diagnostics."""

    def _minimal_valid_graph(self) -> Graph:
        """A small graph: start -> task -> end."""
        g = Graph(name="test")
        g.nodes["start"] = Node(id="start", shape="Mdiamond")
        g.nodes["task"] = Node(id="task", shape="box")
        g.nodes["end"] = Node(id="end", shape="Msquare")
        g.edges.append(Edge(source="start", target="task"))
        g.edges.append(Edge(source="task", target="end"))
        return g

    def test_r03_start_with_incoming_is_error(self):
        """R03: an incoming edge to the start node produces ERROR severity."""
        graph = self._minimal_valid_graph()
        # Add an illegal incoming edge to the start node
        graph.edges.append(Edge(source="task", target="start"))

        diagnostics = validate(graph)
        r03 = [d for d in diagnostics if d.rule == "R03"]

        assert len(r03) >= 1
        assert all(d.severity == Severity.ERROR for d in r03)

    def test_r03_references_start_node_id(self):
        """R03 diagnostic carries the start node's id in node_id field."""
        graph = self._minimal_valid_graph()
        graph.edges.append(Edge(source="task", target="start"))

        diagnostics = validate(graph)
        r03 = [d for d in diagnostics if d.rule == "R03"]

        assert any(d.node_id == "start" for d in r03)

    def test_r04_exit_with_outgoing_is_error(self):
        """R04: an outgoing edge from an exit node produces ERROR severity."""
        graph = self._minimal_valid_graph()
        # Add an illegal outgoing edge from the exit node
        graph.edges.append(Edge(source="end", target="task"))

        diagnostics = validate(graph)
        r04 = [d for d in diagnostics if d.rule == "R04"]

        assert len(r04) >= 1
        assert all(d.severity == Severity.ERROR for d in r04)

    def test_r04_references_exit_node_id(self):
        """R04 diagnostic carries the exit node's id in node_id field."""
        graph = self._minimal_valid_graph()
        graph.edges.append(Edge(source="end", target="task"))

        diagnostics = validate(graph)
        r04 = [d for d in diagnostics if d.rule == "R04"]

        assert any(d.node_id == "end" for d in r04)

    def test_valid_graph_produces_no_r03_or_r04(self):
        """A well-formed graph produces no R03 or R04 diagnostics."""
        graph = self._minimal_valid_graph()

        diagnostics = validate(graph)
        rule_ids = {d.rule for d in diagnostics}

        assert "R03" not in rule_ids
        assert "R04" not in rule_ids


# ================================================================== #
# Wave 10 -- StreamResult.response() accumulates (complementary)
# ================================================================== #


class TestStreamResultResponseAccumulates:
    """StreamResult.response() collects all events and returns a complete Response."""

    @pytest.mark.asyncio
    async def test_response_accumulates_text_usage_model_provider(self):
        """response() returns a Response with full text, usage, model, and provider."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m1", provider="p1"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="Hello "),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="world"),
            StreamEvent(kind=StreamEventKind.USAGE, usage=Usage(input_tokens=4, output_tokens=2)),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))
        resp = await result.response()

        assert resp.text == "Hello world"
        assert resp.model == "m1"
        assert resp.provider == "p1"
        assert resp.usage.input_tokens == 4
        assert resp.usage.output_tokens == 2
        assert resp.finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_response_is_idempotent(self):
        """Calling response() twice returns the same accumulated data."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="ok"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        resp1 = await result.response()
        resp2 = await result.response()

        assert resp1.text == resp2.text == "ok"

    @pytest.mark.asyncio
    async def test_response_after_text_stream_matches(self):
        """response() after iterating text_stream still returns the full accumulated text."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="part1"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="part2"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        chunks = [c async for c in result.text_stream]
        assert chunks == ["part1", "part2"]

        resp = await result.response()
        assert resp.text == "part1part2"


# ================================================================== #
# Wave 10 -- StreamResult.text_stream yields text only
# ================================================================== #


class TestStreamResultTextStreamYieldsTextOnly:
    """text_stream only emits the text string from TEXT_DELTA events."""

    @pytest.mark.asyncio
    async def test_text_stream_skips_non_delta_events(self):
        """START, USAGE, and FINISH events are not emitted by text_stream."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="chunk1"),
            StreamEvent(kind=StreamEventKind.USAGE, usage=Usage(input_tokens=2, output_tokens=1)),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="chunk2"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        chunks = [c async for c in result.text_stream]

        assert chunks == ["chunk1", "chunk2"]
        assert all(isinstance(c, str) for c in chunks)

    @pytest.mark.asyncio
    async def test_text_stream_items_are_str_not_events(self):
        """Items from text_stream are plain str, not StreamEvent instances."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="hello"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        async for item in result.text_stream:
            assert isinstance(item, str), f"Expected str, got {type(item).__name__}"
            assert not isinstance(item, StreamEvent)


# ================================================================== #
# Wave 10 -- StreamResult.__aiter__ backward compat
# ================================================================== #


class TestStreamResultAiterBackwardCompat:
    """async for over StreamResult yields strings, not StreamEvents."""

    @pytest.mark.asyncio
    async def test_aiter_yields_strings_not_events(self):
        """__aiter__ delegates to text_stream, yielding plain str objects."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="foo"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="bar"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        collected = []
        async for item in result:
            assert isinstance(item, str), f"Expected str, got {type(item).__name__}"
            collected.append(item)

        assert collected == ["foo", "bar"]


# ================================================================== #
# Wave 10 -- StreamResult.iter_events() yields StreamEvents
# ================================================================== #


class TestStreamResultIterEvents:
    """iter_events() provides raw access to the underlying event stream."""

    @pytest.mark.asyncio
    async def test_iter_events_yields_stream_event_objects(self):
        """iter_events() yields StreamEvent instances, not strings."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="hi"),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        collected = []
        async for event in result.iter_events():
            assert isinstance(event, StreamEvent), (
                f"Expected StreamEvent, got {type(event).__name__}"
            )
            collected.append(event)

        assert len(collected) == 3
        assert collected[0].kind == StreamEventKind.START
        assert collected[1].kind == StreamEventKind.TEXT_DELTA
        assert collected[2].kind == StreamEventKind.FINISH

    @pytest.mark.asyncio
    async def test_iter_events_preserves_all_event_kinds(self):
        """All event types in the stream are passed through by iter_events()."""
        events = [
            StreamEvent(kind=StreamEventKind.START, model="m", provider="p"),
            StreamEvent(kind=StreamEventKind.TEXT_DELTA, text="a"),
            StreamEvent(kind=StreamEventKind.USAGE, usage=Usage(input_tokens=1, output_tokens=1)),
            StreamEvent(kind=StreamEventKind.FINISH, finish_reason=FinishReason.STOP),
        ]
        result = StreamResult(_event_stream(*events))

        kinds = [e.kind async for e in result.iter_events()]

        assert StreamEventKind.START in kinds
        assert StreamEventKind.TEXT_DELTA in kinds
        assert StreamEventKind.USAGE in kinds
        assert StreamEventKind.FINISH in kinds


# ================================================================== #
# Wave 10 -- abort_signal on generate() raises AbortError
# ================================================================== #


class TestAbortSignalOnGenerate:
    """abort_signal=<already-set> causes generate() to raise AbortError."""

    @pytest.mark.asyncio
    async def test_preset_abort_signal_raises_abort_error(self):
        """generate() raises AbortError when abort_signal.is_set is True post-complete."""
        abort = MagicMock()
        abort.is_set = True

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hi"))

        with pytest.raises(AbortError, match="aborted"):
            await generate(client, "test-model", "Hello", abort_signal=abort)

    @pytest.mark.asyncio
    async def test_no_abort_signal_returns_normally(self):
        """Without abort_signal, generate() returns a GenerateResult normally."""
        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hello"))

        result = await generate(client, "test-model", "Say hello")
        assert result.text == "Hello"

    @pytest.mark.asyncio
    async def test_abort_signal_false_returns_normally(self):
        """abort_signal present but is_set=False allows normal completion."""
        abort = MagicMock()
        abort.is_set = False

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(return_value=_text_response("Hello"))

        result = await generate(client, "test-model", "Say hello", abort_signal=abort)
        assert result.text == "Hello"


# ================================================================== #
# Wave 10 -- abort_signal checked in tool loop
# ================================================================== #


class TestAbortSignalInToolLoop:
    """abort_signal is checked after each complete() call in the tool-call loop."""

    @pytest.mark.asyncio
    async def test_abort_after_first_tool_call_raises_abort_error(self):
        """AbortError is raised when the signal becomes True after the first tool call."""
        call_count = 0
        abort = MagicMock()

        # is_set becomes True only after the first complete() call
        def _is_set():
            return call_count >= 1

        type(abort).is_set = property(lambda self: _is_set())

        tool_response = Response(
            message=Message(
                role="assistant",
                content=[ContentPart.tool_call_part("tc1", "echo_tool", '{"msg": "hi"}')],
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
        )

        async def _mock_complete(request, **kwargs):
            nonlocal call_count
            call_count += 1
            return tool_response

        client = AsyncMock(spec=Client)
        client.complete = AsyncMock(side_effect=_mock_complete)

        async def _exec(**kwargs):
            return "tool result"

        tool = Tool(
            name="echo_tool",
            description="Echo tool",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            execute=_exec,
        )

        with pytest.raises(AbortError):
            await generate(
                client,
                "test-model",
                "Use the tool",
                tools=[tool],
                abort_signal=abort,
            )


# ================================================================== #
# Wave 10 -- TimeoutConfig field defaults
# ================================================================== #


class TestTimeoutConfigDefaults:
    """TimeoutConfig dataclass has the correct default field values."""

    def test_total_defaults_to_none(self):
        assert TimeoutConfig().total is None

    def test_per_step_defaults_to_none(self):
        assert TimeoutConfig().per_step is None

    def test_custom_values_are_stored(self):
        tc = TimeoutConfig(total=120.0, per_step=30.0)
        assert tc.total == 120.0
        assert tc.per_step == 30.0


# ================================================================== #
# Wave 10 -- AdapterTimeout field defaults
# ================================================================== #


class TestAdapterTimeoutDefaults:
    """AdapterTimeout dataclass has the correct default field values."""

    def test_connect_defaults_to_10(self):
        assert AdapterTimeout().connect == 10.0

    def test_request_defaults_to_120(self):
        assert AdapterTimeout().request == 120.0

    def test_stream_read_defaults_to_30(self):
        assert AdapterTimeout().stream_read == 30.0

    def test_custom_values_are_stored(self):
        at = AdapterTimeout(connect=5.0, request=60.0, stream_read=15.0)
        assert at.connect == 5.0
        assert at.request == 60.0
        assert at.stream_read == 15.0


# ================================================================== #
# Wave 10 -- New StreamEventKind values
# ================================================================== #


class TestStreamEventKindNewValues:
    """New StreamEventKind members added in Wave 10 exist with correct string values."""

    def test_text_start_value(self):
        assert StreamEventKind.TEXT_START == "text_start"

    def test_text_end_value(self):
        assert StreamEventKind.TEXT_END == "text_end"

    def test_reasoning_start_value(self):
        assert StreamEventKind.REASONING_START == "reasoning_start"

    def test_reasoning_end_value(self):
        assert StreamEventKind.REASONING_END == "reasoning_end"

    def test_provider_event_value(self):
        assert StreamEventKind.PROVIDER_EVENT == "provider_event"

    def test_provider_event_stream_event_carries_raw_event(self):
        """A PROVIDER_EVENT StreamEvent can carry arbitrary raw provider data."""
        event = StreamEvent(
            kind=StreamEventKind.PROVIDER_EVENT,
            raw_event={"type": "content_block_start", "index": 0},
        )
        assert event.raw_event == {"type": "content_block_start", "index": 0}

    def test_all_new_kinds_are_distinct(self):
        """All five new kinds have distinct string values from each other."""
        new_kinds = {
            StreamEventKind.TEXT_START,
            StreamEventKind.TEXT_END,
            StreamEventKind.REASONING_START,
            StreamEventKind.REASONING_END,
            StreamEventKind.PROVIDER_EVENT,
        }
        assert len(new_kinds) == 5  # All distinct
