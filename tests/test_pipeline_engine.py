"""Tests for the Attractor Pipeline Engine: parser, conditions, engine, handlers, stylesheet."""

from __future__ import annotations

import pytest

from attractor_agent.abort import AbortSignal
from attractor_pipeline import (
    Checkpoint,
    CodergenHandler,
    ExitHandler,
    HandlerRegistry,
    HandlerResult,
    Outcome,
    PipelineStatus,
    StartHandler,
    evaluate_condition,
    parse_dot,
    register_default_handlers,
    run_pipeline,
    select_edge,
)
from attractor_pipeline.parser.parser import ParseError
from attractor_pipeline.stylesheet import (
    StylesheetParseError,
    apply_stylesheet,
    parse_stylesheet,
)
from attractor_pipeline.validation import Severity, validate, validate_or_raise

# ================================================================== #
# DOT Parser
# ================================================================== #


class TestDotParser:
    def test_simple_pipeline(self):
        g = parse_dot("""
        digraph Simple {
            graph [goal="Test"]
            start [shape=Mdiamond]
            task [shape=box]
            done [shape=Msquare]
            start -> task -> done
        }
        """)
        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        assert g.goal == "Test"

    def test_chained_edges_expand(self):
        g = parse_dot("""
        digraph Chain {
            a [shape=Mdiamond]
            b [shape=box]
            c [shape=box]
            d [shape=Msquare]
            a -> b -> c -> d
        }
        """)
        assert len(g.edges) == 3
        assert g.edges[0].source == "a" and g.edges[0].target == "b"
        assert g.edges[1].source == "b" and g.edges[1].target == "c"
        assert g.edges[2].source == "c" and g.edges[2].target == "d"

    def test_edge_attributes(self):
        g = parse_dot("""
        digraph Attrs {
            a [shape=Mdiamond]
            b [shape=box]
            a -> b [label="yes", condition="outcome = success", weight="2.0"]
        }
        """)
        e = g.edges[0]
        assert e.label == "yes"
        assert e.condition == "outcome = success"
        assert e.weight == 2.0

    def test_default_node_attributes(self):
        g = parse_dot("""
        digraph Defaults {
            node [shape=box, llm_model="claude-sonnet-4-5"]
            a [label="A"]
            b [label="B"]
        }
        """)
        assert g.nodes["a"].shape == "box"
        assert g.nodes["a"].llm_model == "claude-sonnet-4-5"

    def test_subgraph_class_inheritance(self):
        g = parse_dot("""
        digraph Sub {
            subgraph code_tasks {
                a [shape=box]
                b [shape=box]
            }
            c [shape=diamond]
        }
        """)
        assert g.nodes["a"].node_class == "code_tasks"
        assert g.nodes["b"].node_class == "code_tasks"
        assert g.nodes["c"].node_class == ""

    def test_comments_stripped(self):
        g = parse_dot("""
        // line comment
        digraph Comments {
            /* block comment */
            start [shape=Mdiamond]
            done [shape=Msquare]
            start -> done
        }
        """)
        assert len(g.nodes) == 2

    def test_graph_level_attributes(self):
        g = parse_dot("""
        digraph Config {
            graph [goal="X", default_max_retry="10", max_goal_gate_redirects="3"]
            start [shape=Mdiamond]
        }
        """)
        assert g.goal == "X"
        assert g.default_max_retry == 10
        assert g.max_goal_gate_redirects == 3

    def test_invalid_syntax_raises(self):
        with pytest.raises(ParseError):
            parse_dot("not a digraph")

    def test_start_node_lookup(self):
        g = parse_dot("""
        digraph S { start [shape=Mdiamond]; done [shape=Msquare]; start -> done }
        """)
        assert g.get_start_node() is not None
        assert g.get_start_node().id == "start"

    def test_exit_nodes_lookup(self):
        g = parse_dot("""
        digraph E {
            s [shape=Mdiamond]
            d1 [shape=Msquare]
            d2 [shape=Msquare]
            s -> d1; s -> d2
        }
        """)
        exits = g.get_exit_nodes()
        assert len(exits) == 2


# ================================================================== #
# Condition Evaluator
# ================================================================== #


class TestConditionEvaluator:
    def test_equality(self):
        assert evaluate_condition("outcome = SUCCESS", {"outcome": "SUCCESS"})
        assert not evaluate_condition("outcome = FAIL", {"outcome": "SUCCESS"})

    def test_inequality(self):
        assert evaluate_condition("key != bad", {"key": "good"})
        assert not evaluate_condition("key != good", {"key": "good"})

    def test_conjunction(self):
        assert evaluate_condition(
            "outcome = success && ready = true",
            {"outcome": "success", "ready": "true"},
        )
        assert not evaluate_condition(
            "outcome = success && ready = false",
            {"outcome": "success", "ready": "true"},
        )

    def test_case_insensitive(self):
        assert evaluate_condition("outcome = SUCCESS", {"outcome": "success"})

    def test_empty_is_true(self):
        assert evaluate_condition("", {})
        assert evaluate_condition("  ", {})

    def test_dotted_path(self):
        assert evaluate_condition(
            "context.ready = true",
            {"context": {"ready": "true"}},
        )


# ================================================================== #
# Edge Selection
# ================================================================== #


class TestEdgeSelection:
    def _make_graph(self):
        return parse_dot("""
        digraph E {
            a [shape=box]
            b [shape=box]
            c [shape=box]
            d [shape=box]
            a -> b [condition="outcome = success", label="win"]
            a -> c [label="lose", weight="2.0"]
            a -> d [weight="1.0"]
        }
        """)

    def test_step1_condition_match(self):
        g = self._make_graph()
        hr = HandlerResult(status=Outcome.SUCCESS)
        edge = select_edge(g.nodes["a"], hr, g, {"outcome": "success"})
        assert edge is not None and edge.target == "b"

    def test_step2_preferred_label(self):
        g = self._make_graph()
        hr = HandlerResult(status=Outcome.FAIL, preferred_label="lose")
        edge = select_edge(g.nodes["a"], hr, g, {"outcome": "fail"})
        assert edge is not None and edge.target == "c"

    def test_step4_weight(self):
        g = self._make_graph()
        hr = HandlerResult(status=Outcome.FAIL)
        edge = select_edge(g.nodes["a"], hr, g, {"outcome": "fail"})
        assert edge is not None and edge.target == "c"  # weight 2.0 > 1.0

    def test_no_edges_returns_none(self):
        g = parse_dot("""
        digraph N { a [shape=box] }
        """)
        hr = HandlerResult(status=Outcome.SUCCESS)
        assert select_edge(g.nodes["a"], hr, g, {}) is None


# ================================================================== #
# Pipeline Execution
# ================================================================== #


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_linear_pipeline(self):
        g = parse_dot("""
        digraph L {
            graph [goal="Test"]
            start [shape=Mdiamond]
            task [shape=box, prompt="Hello"]
            done [shape=Msquare]
            start -> task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "start" in result.completed_nodes
        assert "done" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_conditional_branching(self):
        g = parse_dot("""
        digraph B {
            graph [goal="Branch"]
            start [shape=Mdiamond]
            check [shape=diamond]
            yes [shape=box, prompt="Y"]
            no [shape=box, prompt="N"]
            done [shape=Msquare]
            start -> check
            check -> yes [condition="outcome = success"]
            check -> no [condition="outcome = fail"]
            yes -> done
            no -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "yes" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_goal_gate_circuit_breaker(self):
        g = parse_dot("""
        digraph G {
            graph [goal="Gate", max_goal_gate_redirects="2"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Code"]
            done [shape=Msquare, goal_gate="outcome = fail", retry_target="code"]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.FAILED
        assert "redirects" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_abort_cancels_pipeline(self):
        g = parse_dot("""
        digraph A {
            start [shape=Mdiamond]
            done [shape=Msquare]
            start -> done
        }
        """)
        abort = AbortSignal()
        abort.set()

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, abort_signal=abort)
        assert result.status == PipelineStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_load(self, tmp_path):
        g = parse_dot("""
        digraph C {
            graph [goal="Checkpoint"]
            start [shape=Mdiamond]
            t1 [shape=box, prompt="T1"]
            done [shape=Msquare]
            start -> t1 -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry, logs_root=tmp_path)
        assert result.status == PipelineStatus.COMPLETED
        assert (tmp_path / "checkpoint.json").exists()

        ckpt = Checkpoint.load(tmp_path / "checkpoint.json")
        assert ckpt.graph_name == "C"
        assert len(ckpt.completed_nodes) > 0

    @pytest.mark.asyncio
    async def test_tool_handler_executes_command(self):
        g = parse_dot("""
        digraph T {
            start [shape=Mdiamond]
            run [shape=parallelogram, prompt="echo hello_pipeline"]
            done [shape=Msquare]
            start -> run -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "hello_pipeline" in result.context.get("tool.run.output", "")

    @pytest.mark.asyncio
    async def test_human_handler_auto_approve(self):
        g = parse_dot("""
        digraph H {
            start [shape=Mdiamond]
            gate [shape=house, prompt="Approve?"]
            task [shape=box, prompt="Do it"]
            done [shape=Msquare]
            start -> gate
            gate -> task [label="approved"]
            task -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        assert "gate" in result.completed_nodes

    @pytest.mark.asyncio
    async def test_missing_handler_fails(self):
        g = parse_dot("""
        digraph M {
            start [shape=Mdiamond]
            custom [shape=box, handler="nonexistent"]
            done [shape=Msquare]
            start -> custom -> done
        }
        """)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.FAILED
        assert "nonexistent" in (result.error or "")

    @pytest.mark.asyncio
    async def test_codergen_backend_with_mock(self):
        class MockBackend:
            async def run(self, node, prompt, context, abort_signal=None):
                return f"Generated: {prompt[:30]}"

        g = parse_dot("""
        digraph CB {
            graph [goal="Build feature"]
            start [shape=Mdiamond]
            code [shape=box, prompt="Implement $goal"]
            done [shape=Msquare]
            start -> code -> done
        }
        """)
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())
        registry.register("codergen", CodergenHandler(backend=MockBackend()))

        result = await run_pipeline(g, registry)
        assert result.status == PipelineStatus.COMPLETED
        output = result.context.get("codergen.code.output", "")
        assert "Build feature" in output


# ================================================================== #
# Validation Rules
# ================================================================== #


class TestValidation:
    def test_valid_graph_passes(self):
        g = parse_dot("""
        digraph V { graph [goal="X"]; start [shape=Mdiamond]; done [shape=Msquare]; start -> done }
        """)
        diags = validate(g)
        errors = [d for d in diags if d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_start_node(self):
        g = parse_dot("""
        digraph NoStart { done [shape=Msquare] }
        """)
        diags = validate(g)
        r01 = [d for d in diags if d.rule == "R01"]
        assert len(r01) == 1 and r01[0].severity == Severity.ERROR

    def test_missing_exit_node(self):
        g = parse_dot("""
        digraph NoExit { start [shape=Mdiamond]; task [shape=box]; start -> task }
        """)
        diags = validate(g)
        r02 = [d for d in diags if d.rule == "R02"]
        assert len(r02) == 1 and r02[0].severity == Severity.ERROR

    def test_exit_unreachable(self):
        g = parse_dot("""
        digraph Unreach { start [shape=Mdiamond]; a [shape=box]; done [shape=Msquare]; start -> a }
        """)
        diags = validate(g)
        r07 = [d for d in diags if d.rule == "R07"]
        assert len(r07) == 1 and r07[0].severity == Severity.ERROR

    def test_validate_or_raise(self):
        g = parse_dot("digraph Bad { task [shape=box] }")
        with pytest.raises(ValueError, match="R01"):
            validate_or_raise(g)

    def test_retry_target_nonexistent(self):
        g = parse_dot("""
        digraph BadRetry {
            graph [goal="X"]
            start [shape=Mdiamond]
            done [shape=Msquare, goal_gate="outcome = success", retry_target="ghost"]
            start -> done
        }
        """)
        diags = validate(g)
        r10 = [d for d in diags if d.rule == "R10"]
        assert len(r10) == 1 and r10[0].severity == Severity.ERROR

    def test_self_loop_warning(self):
        g = parse_dot("""
        digraph Loop {
            graph [goal="X"]
            start [shape=Mdiamond]
            task [shape=box]
            done [shape=Msquare]
            start -> task -> done
            task -> task
        }
        """)
        diags = validate(g)
        r11 = [d for d in diags if d.rule == "R11"]
        assert len(r11) == 1

    def test_no_goal_info(self):
        g = parse_dot("""
        digraph NoGoal { start [shape=Mdiamond]; done [shape=Msquare]; start -> done }
        """)
        diags = validate(g)
        r12 = [d for d in diags if d.rule == "R12"]
        assert len(r12) == 1 and r12[0].severity == Severity.INFO


# ================================================================== #
# Stylesheet
# ================================================================== #


class TestStylesheet:
    def test_universal_selector(self):
        ss = parse_stylesheet("* { llm_model: test; }")
        assert len(ss.rules) == 1
        assert ss.rules[0].selector.specificity == 0

    def test_shape_selector(self):
        ss = parse_stylesheet("box { reasoning_effort: medium; }")
        assert ss.rules[0].selector.kind == "shape"
        assert ss.rules[0].selector.specificity == 1

    def test_class_selector(self):
        ss = parse_stylesheet(".critical { llm_model: opus; }")
        assert ss.rules[0].selector.kind == "class"
        assert ss.rules[0].selector.specificity == 2

    def test_id_selector(self):
        ss = parse_stylesheet("#review { llm_model: gpt; }")
        assert ss.rules[0].selector.kind == "id"
        assert ss.rules[0].selector.specificity == 3

    def test_unknown_shape_raises(self):
        with pytest.raises(StylesheetParseError, match="Unknown selector"):
            parse_stylesheet("unknown_thing { llm_model: x; }")

    def test_specificity_cascade(self):
        g = parse_dot("""
        digraph S {
            graph [model_stylesheet="* { llm_model: base; }\nbox { llm_model: shape; }\n.special { llm_model: class; }\n#node1 { llm_model: id; }"]
            start [shape=Mdiamond]
            node1 [shape=box, class="special"]
            node2 [shape=box, class="special"]
            node3 [shape=box]
            done [shape=Msquare]
            start -> node1 -> node2 -> node3 -> done
        }
        """)
        apply_stylesheet(g)
        assert g.nodes["node1"].llm_model == "id"  # #id wins
        assert g.nodes["node2"].llm_model == "class"  # .class wins
        assert g.nodes["node3"].llm_model == "shape"  # shape wins
        assert g.nodes["start"].llm_model == "base"  # * wins

    def test_explicit_attr_overrides_stylesheet(self):
        g = parse_dot("""
        digraph O {
            graph [model_stylesheet="* { llm_model: stylesheet; }"]
            start [shape=Mdiamond]
            explicit [shape=box, llm_model="from-dot"]
            implicit [shape=box]
            done [shape=Msquare]
            start -> explicit -> implicit -> done
        }
        """)
        apply_stylesheet(g)
        assert g.nodes["explicit"].llm_model == "from-dot"
        assert g.nodes["implicit"].llm_model == "stylesheet"

    def test_empty_stylesheet_noop(self):
        ss = parse_stylesheet("")
        assert len(ss.rules) == 0

    def test_comments_stripped(self):
        ss = parse_stylesheet("/* comment */ * { llm_model: test; }")
        assert len(ss.rules) == 1


# ================================================================== #
# §11.12 Integration-Test Coverage Gaps
# ================================================================== #


class TestDoD11_12_3_MultiLineNodeAttributes:
    """§11.12.3: Parse multi-line node attribute blocks."""

    def test_multiline_node_attributes_parsed_correctly(self):
        """Node attributes spanning multiple lines within [...] are parsed correctly."""
        g = parse_dot("""
        digraph ML {
            graph [goal="Multi-line attrs"]
            start [shape=Mdiamond]
            task [
                shape=box,
                prompt="A multi-line
                        prompt attribute",
                max_retries=2,
                label="Task Node"
            ]
            done [shape=Msquare]
            start -> task -> done
        }
        """)
        assert "task" in g.nodes
        node = g.nodes["task"]
        assert node.shape == "box"
        assert node.max_retries == 2
        assert node.label == "Task Node"
        # prompt spans lines but should be captured as a single string
        assert "multi-line" in node.prompt

    def test_multiline_edge_attributes_parsed_correctly(self):
        """Edge attributes spanning multiple lines within [...] are parsed."""
        g = parse_dot("""
        digraph MLEdge {
            graph [goal="Multi-line edge attrs"]
            start [shape=Mdiamond]
            check [shape=diamond]
            yes   [shape=box, prompt="Y"]
            no    [shape=box, prompt="N"]
            done  [shape=Msquare]
            start -> check
            check -> yes [
                condition="outcome = success",
                weight=2.0,
                label="success"
            ]
            check -> no [
                condition="outcome = fail",
                label="fail"
            ]
            yes -> done
            no  -> done
        }
        """)
        edges_to_yes = [e for e in g.edges if e.target == "yes"]
        assert len(edges_to_yes) == 1
        edge = edges_to_yes[0]
        assert edge.condition == "outcome = success"
        assert edge.weight == 2.0
        assert edge.label == "success"


class TestDoD11_12_9_PerNodeMaxRetries:
    """§11.12.9: Execute with retry on failure (max_retries=2 per node)."""

    @pytest.mark.asyncio
    async def test_per_node_max_retries_from_dot_attribute(self):
        """Node with max_retries=2 in DOT retries exactly 2 times after initial fail.

        Total attempts = 3: 1 initial + 2 retries. After exhaustion the node
        uses its final outcome (FAIL) for edge selection.
        """
        from unittest.mock import patch

        call_counts: dict[str, int] = {}

        class _FailTwiceHandler:
            """Fails the first 2 calls, succeeds on the 3rd."""

            async def execute(self, node, context, graph, logs_root, abort_signal=None):
                key = node.id
                call_counts[key] = call_counts.get(key, 0) + 1
                if call_counts[key] <= 2:
                    return HandlerResult(status=Outcome.FAIL, failure_reason="deliberate fail")
                return HandlerResult(status=Outcome.SUCCESS, output="recovered")

        g = parse_dot("""
        digraph R2 {
            graph [goal="Per-node retries"]
            start [shape=Mdiamond]
            task  [shape=box, prompt="Retry me", max_retries=2]
            done  [shape=Msquare]
            start -> task -> done
        }
        """)
        assert g.nodes["task"].max_retries == 2, (
            "DOT max_retries=2 should be parsed onto the node"
        )

        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("codergen", _FailTwiceHandler())
        registry.register("exit", ExitHandler())

        with patch("attractor_pipeline.engine.runner.anyio.sleep"):
            result = await run_pipeline(g, registry)

        assert result.status == PipelineStatus.COMPLETED, (
            "Pipeline should complete — task recovered on attempt 3"
        )
        got = call_counts.get("task", 0)
        assert got == 3, (
            f"Expected 3 calls (1 initial + 2 retries), got {got}"
        )


class TestDoD11_12_15_LexicalTiebreak:
    """§11.12.15: Lexical tiebreak as final fallback in edge selection."""

    def test_lexical_tiebreak_selects_alphabetically_first_target(self):
        """When multiple unconditional edges have equal weight, the lexicographically
        smaller target node ID wins.

        Edge a->'b' and a->'a_alt': 'a_alt' < 'b' alphabetically → 'a_alt' selected.
        """
        from attractor_pipeline.engine.runner import HandlerResult, Outcome, select_edge
        from attractor_pipeline.graph import Edge, Graph, Node

        g = Graph(name="lex")
        node_a = Node(id="a", shape="box")
        g.nodes["a"] = node_a
        g.nodes["b"] = Node(id="b", shape="box")
        g.nodes["a_alt"] = Node(id="a_alt", shape="box")

        # Both edges: no condition, equal weight (default 1.0)
        g.edges = [
            Edge(source="a", target="b", weight=1.0),
            Edge(source="a", target="a_alt", weight=1.0),
        ]

        result = HandlerResult(status=Outcome.SUCCESS)
        chosen = select_edge(node_a, result, g, {})

        assert chosen is not None
        assert chosen.target == "a_alt", (
            f"Lexical tiebreak must select 'a_alt' (comes before 'b'), got '{chosen.target}'"
        )

    def test_lexical_tiebreak_does_not_fire_when_weights_differ(self):
        """Weight takes priority over lexical order — higher-weight edge wins."""
        from attractor_pipeline.engine.runner import HandlerResult, Outcome, select_edge
        from attractor_pipeline.graph import Edge, Graph, Node

        g = Graph(name="weight_priority")
        node_a = Node(id="a", shape="box")
        g.nodes["a"] = node_a
        g.nodes["b"] = Node(id="b", shape="box")
        g.nodes["a_alt"] = Node(id="a_alt", shape="box")

        # 'b' has higher weight despite lexically later name
        g.edges = [
            Edge(source="a", target="b", weight=2.0),
            Edge(source="a", target="a_alt", weight=1.0),
        ]

        result = HandlerResult(status=Outcome.SUCCESS)
        chosen = select_edge(node_a, result, g, {})

        assert chosen is not None
        assert chosen.target == "b", (
            f"Weight 2.0 must beat weight 1.0 regardless of lexical order, got '{chosen.target}'"
        )


class TestDoD11_12_17_CheckpointResume:
    """§11.12.17: Checkpoint save and resume produces same result."""

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint_produces_same_result(self, tmp_path):
        """A pipeline that runs to completion and then resumes from its saved
        checkpoint should reach the same final status and completed_nodes.
        """
        dot_src = """
        digraph ResumableTest {
            graph [goal="Test checkpoint resume"]
            start [shape=Mdiamond]
            t1    [shape=box, prompt="Step 1"]
            t2    [shape=box, prompt="Step 2"]
            done  [shape=Msquare]
            start -> t1 -> t2 -> done
        }
        """

        g = parse_dot(dot_src)
        registry = HandlerRegistry()
        register_default_handlers(registry)

        # Full run — produces checkpoint
        logs_root_1 = tmp_path / "run1"
        logs_root_1.mkdir()
        result_1 = await run_pipeline(g, registry, logs_root=logs_root_1)
        assert result_1.status == PipelineStatus.COMPLETED

        ckpt_path = logs_root_1 / "checkpoint.json"
        assert ckpt_path.exists(), "Checkpoint file must be written after run"

        # Load checkpoint and resume
        ckpt = Checkpoint.load(ckpt_path)
        logs_root_2 = tmp_path / "run2"
        logs_root_2.mkdir()

        # Parse fresh graph (simulates agent restart)
        g2 = parse_dot(dot_src)
        result_2 = await run_pipeline(g2, registry, checkpoint=ckpt, logs_root=logs_root_2)

        # Both runs must have the same outcome
        assert result_2.status == result_1.status, (
            f"Resumed run status {result_2.status!r} != fresh run status {result_1.status!r}"
        )
        # Same nodes completed (order may differ, use sets)
        assert set(result_2.completed_nodes) == set(result_1.completed_nodes), (
            f"Resumed completed_nodes {result_2.completed_nodes!r} != "
            f"fresh completed_nodes {result_1.completed_nodes!r}"
        )

    @pytest.mark.asyncio
    async def test_resume_from_mid_pipeline_checkpoint(self, tmp_path):
        """Resuming from a mid-pipeline checkpoint skips already-completed nodes.

        This is the canonical crash-recovery scenario: the pipeline is
        interrupted after t1 completes, then resumed. The resumed run must
        start at t2, complete t2 → done, and NOT re-execute t1.
        """
        dot_src = """
        digraph ResumableTest {
            graph [goal="Test mid-pipeline resume"]
            start [shape=Mdiamond]
            t1    [shape=box, prompt="Step 1"]
            t2    [shape=box, prompt="Step 2"]
            done  [shape=Msquare]
            start -> t1 -> t2 -> done
        }
        """

        call_counts: dict[str, int] = {}

        class _TrackingCodergen:
            """Records which nodes are called during the resumed run."""

            async def execute(self, node, context, graph, logs_root, abort_signal=None):
                call_counts[node.id] = call_counts.get(node.id, 0) + 1
                return HandlerResult(status=Outcome.SUCCESS, output=f"{node.id}-done")

        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("codergen", _TrackingCodergen())
        registry.register("exit", ExitHandler())

        # Construct a partial checkpoint: start and t1 have completed,
        # pipeline is interrupted before t2 runs.
        partial_ckpt = Checkpoint(
            graph_name="ResumableTest",
            current_node_id="t2",  # resume point
            completed_nodes=[
                {"node_id": "start"},
                {"node_id": "t1"},
            ],
            context_values={"goal": "Test mid-pipeline resume"},
            node_retry_counts={},
            goal_gate_redirect_count=0,
            status="running",
        )

        g = parse_dot(dot_src)
        logs_root = tmp_path / "resume"
        logs_root.mkdir()
        result = await run_pipeline(g, registry, checkpoint=partial_ckpt, logs_root=logs_root)

        assert result.status == PipelineStatus.COMPLETED

        # t1 was already in the checkpoint — must NOT be re-executed
        assert call_counts.get("t1", 0) == 0, (
            "t1 was completed before the checkpoint — must not be re-executed on resume"
        )
        # t2 must run exactly once (resume point)
        assert call_counts.get("t2", 1) == 1, "t2 must run exactly once on resume"
        # Final completed_nodes must contain all four nodes
        assert set(result.completed_nodes) == {"start", "t1", "t2", "done"}


class TestDoD11_12_22_LargePipeline:
    """§11.12.22: Pipeline with 10+ nodes completes without errors."""

    @pytest.mark.asyncio
    async def test_twelve_node_pipeline_completes(self):
        """A 12-node linear pipeline (start + 10 box nodes + done) must run
        to completion without errors or infinite loops.
        """
        # Build: start -> n1 -> n2 -> ... -> n10 -> done
        nodes = ["n" + str(i) for i in range(1, 11)]
        node_decls = "\n            ".join(
            f'{n} [shape=box, prompt="Node {n}"]' for n in nodes
        )
        edge_chain = " -> ".join(["start"] + nodes + ["done"])

        dot_src = f"""
        digraph BigPipeline {{
            graph [goal="Large pipeline test"]
            start [shape=Mdiamond]
            {node_decls}
            done  [shape=Msquare]
            {edge_chain}
        }}
        """

        g = parse_dot(dot_src)
        assert len(g.nodes) == 12, f"Expected 12 nodes, got {len(g.nodes)}"

        registry = HandlerRegistry()
        register_default_handlers(registry)

        result = await run_pipeline(g, registry)

        assert result.status == PipelineStatus.COMPLETED, (
            f"12-node pipeline must complete, got status: {result.status}"
        )
        # All 10 middle nodes should appear in completed_nodes
        for n in nodes:
            assert n in result.completed_nodes, f"Node '{n}' missing from completed_nodes"
