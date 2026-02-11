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
            start [shape=ellipse]
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
            a [shape=ellipse]
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
            a [shape=ellipse]
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
            start [shape=ellipse]
            done [shape=Msquare]
            start -> done
        }
        """)
        assert len(g.nodes) == 2

    def test_graph_level_attributes(self):
        g = parse_dot("""
        digraph Config {
            graph [goal="X", default_max_retry="10", max_goal_gate_redirects="3"]
            start [shape=ellipse]
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
        digraph S { start [shape=ellipse]; done [shape=Msquare]; start -> done }
        """)
        assert g.get_start_node() is not None
        assert g.get_start_node().id == "start"

    def test_exit_nodes_lookup(self):
        g = parse_dot("""
        digraph E {
            s [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
        digraph V { graph [goal="X"]; start [shape=ellipse]; done [shape=Msquare]; start -> done }
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
        digraph NoExit { start [shape=ellipse]; task [shape=box]; start -> task }
        """)
        diags = validate(g)
        r02 = [d for d in diags if d.rule == "R02"]
        assert len(r02) == 1 and r02[0].severity == Severity.ERROR

    def test_exit_unreachable(self):
        g = parse_dot("""
        digraph Unreach { start [shape=ellipse]; a [shape=box]; done [shape=Msquare]; start -> a }
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
        digraph NoGoal { start [shape=ellipse]; done [shape=Msquare]; start -> done }
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
            start [shape=ellipse]
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
            start [shape=ellipse]
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
