"""Audit 2 Wave 1 - Foundation Fixes regression tests.

Covers all 7 surgical fixes:
  Item 1  Sec 8.8.3    - Jitter range [0.5x, 1.5x]
  Item 2  Sec 3.14     - StreamEventKind.STREAM_START / STREAM_END
  Item 3  Sec 8.4.10   - Anthropic adapter respects AdapterTimeout
  Item 4+5 Sec 8.2.5/6 - OpenAI system messages go to top-level `instructions`
  Item 6  Sec 11.12.6  - Orphan-node severity is WARNING
  Item 7  Sec 11.2.2   - Multiple exit nodes are rejected
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Item 1 - Jitter range Sec 8.8.3
# ---------------------------------------------------------------------------


class TestJitterRange:
    """compute_delay() jitter must produce values in [0.5x, 1.5x]."""

    def test_jitter_range_covers_full_spectrum(self) -> None:
        """Statistical: 2000 samples must span near-0.5 and near-1.5 of base."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=1.0,  # attempt=0 always yields base=1.0
            jitter=True,
        )
        samples = [policy.compute_delay(0) for _ in range(2000)]

        lo, hi = min(samples), max(samples)

        # Must never exceed [0.5, 1.5]
        assert lo >= 0.5 - 1e-9, f"min sample {lo} is below 0.5"
        assert hi <= 1.5 + 1e-9, f"max sample {hi} exceeds 1.5"

        # With 2000 draws uniform on [0.5, 1.5] the probability that
        # we never see a value < 0.65 or > 1.35 is astronomically small.
        assert lo < 0.65, f"min sample {lo:.4f} suspiciously high (old 0.5*0.5 bug?)"
        assert hi > 1.35, f"max sample {hi:.4f} suspiciously low (old 1.0 cap bug?)"

    def test_no_jitter_returns_exact_delay(self) -> None:
        """When jitter=False compute_delay returns the exact exponential backoff."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(
            max_retries=3,
            initial_delay=0.4,
            backoff_factor=2.0,
            jitter=False,
        )
        assert policy.compute_delay(0) == pytest.approx(0.4)
        assert policy.compute_delay(1) == pytest.approx(0.8)
        assert policy.compute_delay(2) == pytest.approx(1.6)

    def test_jitter_upper_bound_is_1_5x(self) -> None:
        """Formula must be delay*(0.5 + rng*1.0) so the cap is 1.5, not 1.0."""
        import random

        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(initial_delay=10.0, backoff_factor=1.0, jitter=True)
        original = random.random
        random.random = lambda: 1.0  # type: ignore[assignment]
        try:
            d = policy.compute_delay(0)
        finally:
            random.random = original  # type: ignore[assignment]

        assert d == pytest.approx(15.0), (
            f"With rng=1.0 expected 10.0*(0.5+1.0*1.0)=15.0, got {d}"
        )

    def test_jitter_lower_bound_is_0_5x(self) -> None:
        """Formula must be delay*(0.5 + rng*1.0) so the floor is 0.5."""
        import random

        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(initial_delay=10.0, backoff_factor=1.0, jitter=True)
        original = random.random
        random.random = lambda: 0.0  # type: ignore[assignment]
        try:
            d = policy.compute_delay(0)
        finally:
            random.random = original  # type: ignore[assignment]

        assert d == pytest.approx(5.0), (
            f"With rng=0.0 expected 10.0*(0.5+0.0*1.0)=5.0, got {d}"
        )


# ---------------------------------------------------------------------------
# Item 2 - StreamEventKind aliases Sec 3.14
# ---------------------------------------------------------------------------


class TestStreamEventKindAliases:
    """STREAM_START and STREAM_END must exist as distinct enum members."""

    def test_stream_event_kind_has_stream_start_and_end(self) -> None:
        """Both STREAM_START and STREAM_END are accessible on StreamEventKind."""
        from attractor_llm.types import StreamEventKind

        assert hasattr(StreamEventKind, "STREAM_START"), "STREAM_START missing"
        assert hasattr(StreamEventKind, "STREAM_END"), "STREAM_END missing"

    def test_stream_start_value(self) -> None:
        from attractor_llm.types import StreamEventKind

        assert StreamEventKind.STREAM_START == "stream_start"

    def test_stream_end_value(self) -> None:
        from attractor_llm.types import StreamEventKind

        assert StreamEventKind.STREAM_END == "stream_end"

    def test_legacy_start_finish_still_present(self) -> None:
        """Backward-compat: START and FINISH must not be removed."""
        from attractor_llm.types import StreamEventKind

        assert StreamEventKind.START == "start"
        assert StreamEventKind.FINISH == "finish"

    def test_stream_start_is_separate_from_start(self) -> None:
        """STREAM_START and START are distinct members with distinct values."""
        from attractor_llm.types import StreamEventKind

        assert StreamEventKind.STREAM_START != StreamEventKind.START
        assert StreamEventKind.STREAM_END != StreamEventKind.FINISH


# ---------------------------------------------------------------------------
# Item 3 - Anthropic adapter respects AdapterTimeout Sec 8.4.10
# ---------------------------------------------------------------------------


class TestAnthropicAdapterTimeout:
    """AnthropicAdapter must honour AdapterTimeout when supplied."""

    def _make_config(self, **kwargs):  # type: ignore[no-untyped-def]
        from attractor_llm.adapters.base import ProviderConfig

        return ProviderConfig(api_key="test-key", **kwargs)

    def test_anthropic_adapter_uses_adapter_timeout(self) -> None:
        """When adapter_timeout is set the httpx client gets those values."""
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.types import AdapterTimeout

        at = AdapterTimeout(connect=5.0, request=99.0, stream_read=45.0)
        config = self._make_config(adapter_timeout=at)
        adapter = AnthropicAdapter(config)

        t = adapter._client.timeout
        assert t.connect == pytest.approx(5.0), f"connect={t.connect}"
        assert t.read == pytest.approx(45.0), f"read={t.read}"

    def test_anthropic_fallback_to_legacy_timeout(self) -> None:
        """Without AdapterTimeout the httpx client uses config.timeout."""
        from attractor_llm.adapters.anthropic import AnthropicAdapter

        config = self._make_config(timeout=77.0)
        adapter = AnthropicAdapter(config)

        t = adapter._client.timeout
        # Legacy path: httpx.Timeout(config.timeout, connect=10.0)
        assert t.connect == pytest.approx(10.0), f"connect should be 10.0, got {t.connect}"
        # read/write inherit from the scalar timeout
        assert t.read == pytest.approx(77.0) or t.write == pytest.approx(77.0), (
            f"legacy timeout not applied; timeout={t!r}"
        )

    def test_anthropic_adapter_timeout_overrides_legacy(self) -> None:
        """adapter_timeout takes priority over the scalar config.timeout."""
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.types import AdapterTimeout

        at = AdapterTimeout(connect=3.0, request=55.0, stream_read=20.0)
        config = self._make_config(timeout=999.0, adapter_timeout=at)
        adapter = AnthropicAdapter(config)

        t = adapter._client.timeout
        # Must NOT use 999.0 from the scalar field when AdapterTimeout is present
        assert t.connect != pytest.approx(999.0), "connect should not be 999.0"
        assert t.connect == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Item 4+5 - OpenAI system messages go to `instructions` Sec 8.2.5 / 8.2.6
# ---------------------------------------------------------------------------


class TestOpenAISystemInstructions:
    """SYSTEM role messages must appear in body['instructions'], not body['input']."""

    def _make_adapter(self):  # type: ignore[no-untyped-def]
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        config = ProviderConfig(api_key="test-key")
        return OpenAIAdapter(config)

    def test_openai_system_in_instructions_field(self) -> None:
        """System messages are extracted into body['instructions']."""
        from attractor_llm.types import Message, Request

        adapter = self._make_adapter()
        req = Request(
            model="gpt-4o",
            messages=[
                Message.system("You are a helpful assistant."),
                Message.user("Hello!"),
            ],
        )
        body = adapter._translate_request(req)

        assert "instructions" in body, "body must contain 'instructions'"
        assert "You are a helpful assistant." in body["instructions"]

    def test_system_message_not_in_input_array(self) -> None:
        """No {role:system} items appear in the input array."""
        from attractor_llm.types import Message, Request

        adapter = self._make_adapter()
        req = Request(
            model="gpt-4o",
            messages=[
                Message.system("System prompt here."),
                Message.user("User turn."),
            ],
        )
        body = adapter._translate_request(req)

        for item in body.get("input", []):
            assert item.get("role") != "system", (
                f"Found system role in input array: {item}"
            )

    def test_multiple_system_messages_concatenated(self) -> None:
        """Multiple system messages are joined with double-newline."""
        from attractor_llm.types import Message, Request

        adapter = self._make_adapter()
        req = Request(
            model="gpt-4o",
            messages=[
                Message.system("First directive."),
                Message.system("Second directive."),
                Message.user("Go."),
            ],
        )
        body = adapter._translate_request(req)

        instr = body.get("instructions", "")
        assert "First directive." in instr
        assert "Second directive." in instr

    def test_no_system_no_instructions_key(self) -> None:
        """When there is no system message, 'instructions' key must be absent."""
        from attractor_llm.types import Message, Request

        adapter = self._make_adapter()
        req = Request(
            model="gpt-4o",
            messages=[Message.user("Just a user message.")],
        )
        body = adapter._translate_request(req)

        assert "instructions" not in body, (
            f"Unexpected 'instructions' key: {body.get('instructions')!r}"
        )

    def test_user_message_remains_in_input(self) -> None:
        """Non-system messages still appear in the input array."""
        from attractor_llm.types import Message, Request

        adapter = self._make_adapter()
        req = Request(
            model="gpt-4o",
            messages=[
                Message.system("Sys."),
                Message.user("Hello."),
            ],
        )
        body = adapter._translate_request(req)

        user_items = [i for i in body.get("input", []) if i.get("role") == "user"]
        assert user_items, "User message missing from input array"


# ---------------------------------------------------------------------------
# Item 6 - Orphan-node severity is WARNING Sec 11.12.6
# ---------------------------------------------------------------------------


class TestOrphanNodeSeverity:
    """_rule_no_orphan_nodes must emit WARNING, not ERROR."""

    def _graph_with_orphan(self):  # type: ignore[no-untyped-def]
        from attractor_pipeline.graph import Edge, Graph, Node

        start = Node(id="start", shape="Mdiamond")
        a = Node(id="a", shape="box")
        exit_ = Node(id="exit", shape="Msquare")
        orphan = Node(id="orphan", shape="box")
        return Graph(
            nodes={"start": start, "a": a, "exit": exit_, "orphan": orphan},
            edges=[Edge(source="start", target="a"), Edge(source="a", target="exit")],
        )

    def test_orphan_node_severity_is_warning(self) -> None:
        """Unreachable node must produce WARNING diagnostic, not ERROR."""
        from attractor_pipeline.validation import Severity, _rule_no_orphan_nodes

        graph = self._graph_with_orphan()
        diags = _rule_no_orphan_nodes(graph)

        assert diags, "Expected at least one diagnostic for orphan node"
        for d in diags:
            assert d.severity == Severity.WARNING, (
                f"Expected WARNING, got {d.severity!r} for node {d.node_id!r}"
            )

    def test_orphan_node_not_error(self) -> None:
        """Severity must not be ERROR (regression guard)."""
        from attractor_pipeline.validation import Severity, _rule_no_orphan_nodes

        graph = self._graph_with_orphan()
        diags = _rule_no_orphan_nodes(graph)

        for d in diags:
            assert d.severity != Severity.ERROR, (
                f"Got ERROR for node {d.node_id!r}; expected WARNING per Sec 11.12.6"
            )

    def test_fully_connected_graph_has_no_orphan_diagnostics(self) -> None:
        """A fully-connected graph produces zero R05 diagnostics."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.validation import _rule_no_orphan_nodes

        start = Node(id="start", shape="Mdiamond")
        exit_ = Node(id="exit", shape="Msquare")
        graph = Graph(
            nodes={"start": start, "exit": exit_},
            edges=[Edge(source="start", target="exit")],
        )
        assert _rule_no_orphan_nodes(graph) == []


# ---------------------------------------------------------------------------
# Item 7 - Multiple exit nodes rejected Sec 11.2.2
# ---------------------------------------------------------------------------


class TestMultipleExitNodes:
    """_rule_has_exit_node must reject graphs with more than one exit node."""

    def _make_graph(self, exit_count: int):  # type: ignore[no-untyped-def]
        """Build a graph with `exit_count` Msquare nodes reachable from start."""
        from attractor_pipeline.graph import Edge, Graph, Node

        start = Node(id="start", shape="Mdiamond")
        nodes: dict = {"start": start}
        edges = []
        for i in range(exit_count):
            eid = f"exit{i}"
            nodes[eid] = Node(id=eid, shape="Msquare")
            edges.append(Edge(source="start", target=eid))
        return Graph(nodes=nodes, edges=edges)

    def test_multiple_exit_nodes_rejected(self) -> None:
        """Graph with 2 exit nodes must produce an ERROR diagnostic."""
        from attractor_pipeline.validation import Severity, _rule_has_exit_node

        graph = self._make_graph(2)
        diags = _rule_has_exit_node(graph)

        assert diags, "Expected an ERROR diagnostic for 2 exit nodes"
        assert any(d.severity == Severity.ERROR for d in diags)

    def test_multiple_exit_nodes_error_message_contains_count(self) -> None:
        """Error message must mention the node count."""
        from attractor_pipeline.validation import _rule_has_exit_node

        graph = self._make_graph(3)
        diags = _rule_has_exit_node(graph)

        assert diags
        combined = " ".join(d.message for d in diags)
        assert "3" in combined, f"Count '3' not found in message: {combined!r}"

    def test_single_exit_node_passes(self) -> None:
        """Graph with exactly 1 exit node must produce no diagnostics."""
        from attractor_pipeline.validation import _rule_has_exit_node

        graph = self._make_graph(1)
        diags = _rule_has_exit_node(graph)
        assert diags == [], f"Unexpected diagnostics for 1 exit node: {diags}"

    def test_zero_exit_nodes_still_reported(self) -> None:
        """No exit nodes must still be an ERROR (pre-existing behaviour)."""
        from attractor_pipeline.graph import Edge, Graph, Node
        from attractor_pipeline.validation import Severity, _rule_has_exit_node

        start = Node(id="start", shape="Mdiamond")
        a = Node(id="a", shape="box")
        graph = Graph(
            nodes={"start": start, "a": a},
            edges=[Edge(source="start", target="a")],
        )
        diags = _rule_has_exit_node(graph)
        assert diags
        assert all(d.severity == Severity.ERROR for d in diags)

    def test_exactly_one_exit_node_no_false_positive(self) -> None:
        """Single-exit graphs must not trigger the new upper-bound check."""
        from attractor_pipeline.validation import _rule_has_exit_node

        graph = self._make_graph(1)
        diags = _rule_has_exit_node(graph)
        multi_exit_diags = [
            d for d in diags if "exit nodes but exactly one" in d.message
        ]
        assert multi_exit_diags == [], (
            f"False-positive multi-exit diagnostic: {multi_exit_diags}"
        )
