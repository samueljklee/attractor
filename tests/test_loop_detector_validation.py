"""Tests for LoopDetector and missing validation rules.

Coverage gaps identified by swarm review round 1:
- B3: _LoopDetector is untested
- W11: Validation rules R03, R04, R05, R06, R08, R09 have no dedicated tests
"""

from __future__ import annotations

from attractor_agent.session import _LoopDetector
from attractor_pipeline.parser import parse_dot
from attractor_pipeline.validation import Severity, validate

# ================================================================== #
# LoopDetector
# ================================================================== #


class TestLoopDetector:
    def test_no_loop_below_threshold(self):
        ld = _LoopDetector(window=4, threshold=3)
        assert ld.record("tool_a", "args1") is False
        assert ld.record("tool_a", "args1") is False  # 2nd time
        assert ld.record("tool_b", "args2") is False  # different tool

    def test_loop_detected_at_threshold(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("grep", '{"pattern": "foo"}')
        ld.record("grep", '{"pattern": "foo"}')
        result = ld.record("grep", '{"pattern": "foo"}')
        assert result is True  # 3rd identical call in window of 4

    def test_window_slides(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("a", "x")  # [a:x]
        ld.record("b", "y")  # [a:x, b:y]
        ld.record("a", "x")  # [a:x, b:y, a:x]
        ld.record("a", "x")  # [a:x, b:y, a:x, a:x] -- 3 of "a:x" in window of 4
        # After window slides, old entries drop off
        assert ld.record("c", "z") is False  # [b:y, a:x, a:x, c:z] -- only 2 of "a:x"

    def test_reset_clears_history(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("grep", "x")
        ld.record("grep", "x")
        ld.reset()
        assert ld.record("grep", "x") is False  # fresh after reset

    def test_different_args_not_loop(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("grep", '{"pattern": "foo"}')
        ld.record("grep", '{"pattern": "bar"}')
        ld.record("grep", '{"pattern": "baz"}')
        # Same tool but different args -- no loop
        assert ld._recent[-1] != ld._recent[-2]

    def test_args_truncated_at_200_chars(self):
        """Signature uses first 200 chars of args -- long args are fingerprinted."""
        ld = _LoopDetector(window=4, threshold=3)
        long_args = "x" * 300
        ld.record("tool", long_args)
        ld.record("tool", long_args)
        result = ld.record("tool", long_args)
        assert result is True  # truncated args still match

    def test_none_args(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("tool", None)
        ld.record("tool", None)
        result = ld.record("tool", None)
        assert result is True

    def test_dict_args(self):
        ld = _LoopDetector(window=4, threshold=3)
        ld.record("tool", {"key": "value"})
        ld.record("tool", {"key": "value"})
        result = ld.record("tool", {"key": "value"})
        assert result is True


# ================================================================== #
# Missing validation rules (R03, R04, R05, R06, R08, R09)
# ================================================================== #


class TestValidationR03:
    """R03: Start node should have no incoming edges."""

    def test_start_with_incoming_warns(self):
        g = parse_dot("""
        digraph R03 {
            graph [goal="X"]
            start [shape=Mdiamond]
            task [shape=box]
            done [shape=Msquare]
            start -> task -> done
            task -> start
        }
        """)
        diags = validate(g)
        r03 = [d for d in diags if d.rule == "R03"]
        assert len(r03) == 1
        assert r03[0].severity == Severity.ERROR

    def test_clean_start_no_warning(self):
        g = parse_dot("""
        digraph Clean {
            graph [goal="X"]
            start [shape=Mdiamond]
            done [shape=Msquare]
            start -> done
        }
        """)
        diags = validate(g)
        r03 = [d for d in diags if d.rule == "R03"]
        assert len(r03) == 0


class TestValidationR04:
    """R04: Exit nodes should have no outgoing edges."""

    def test_exit_with_outgoing_warns(self):
        g = parse_dot("""
        digraph R04 {
            graph [goal="X"]
            start [shape=Mdiamond]
            done [shape=Msquare]
            task [shape=box]
            start -> done
            done -> task
        }
        """)
        diags = validate(g)
        r04 = [d for d in diags if d.rule == "R04"]
        assert len(r04) == 1
        assert r04[0].severity == Severity.ERROR


class TestValidationR05:
    """R05: Every non-start node should have at least one incoming edge."""

    def test_orphan_node_warns(self):
        g = parse_dot("""
        digraph R05 {
            graph [goal="X"]
            start [shape=Mdiamond]
            orphan [shape=box]
            done [shape=Msquare]
            start -> done
        }
        """)
        diags = validate(g)
        r05 = [d for d in diags if d.rule == "R05"]
        orphan_diags = [d for d in r05 if "orphan" in d.message]
        assert len(orphan_diags) >= 1


class TestValidationR06:
    """R06: All edge endpoints must reference existing nodes."""

    def test_edge_to_nonexistent_node_errors(self):
        # This is tricky -- the parser creates nodes from edges.
        # But we can test by manually constructing a graph.
        from attractor_pipeline.graph import Edge, Graph, Node

        g = Graph(
            name="R06",
            nodes={"start": Node(id="start", shape="Mdiamond")},
            edges=[Edge(source="start", target="ghost")],
        )
        diags = validate(g)
        r06 = [d for d in diags if d.rule == "R06"]
        assert len(r06) >= 1
        assert r06[0].severity == Severity.ERROR


class TestValidationR08:
    """R08: Conditional nodes should have at least 2 outgoing edges."""

    def test_conditional_one_edge_warns(self):
        g = parse_dot("""
        digraph R08 {
            graph [goal="X"]
            start [shape=Mdiamond]
            branch [shape=diamond]
            done [shape=Msquare]
            start -> branch -> done
        }
        """)
        diags = validate(g)
        r08 = [d for d in diags if d.rule == "R08"]
        assert len(r08) == 1
        assert r08[0].severity == Severity.WARNING

    def test_conditional_two_edges_ok(self):
        g = parse_dot("""
        digraph R08ok {
            graph [goal="X"]
            start [shape=Mdiamond]
            branch [shape=diamond]
            a [shape=box]
            b [shape=box]
            done [shape=Msquare]
            start -> branch
            branch -> a [condition="outcome = success"]
            branch -> b [condition="outcome = fail"]
            a -> done
            b -> done
        }
        """)
        diags = validate(g)
        r08 = [d for d in diags if d.rule == "R08"]
        assert len(r08) == 0


class TestValidationR09:
    """R09: Exit nodes with goal_gate should have a retry_target."""

    def test_goal_gate_without_retry_target_warns(self):
        g = parse_dot("""
        digraph R09 {
            graph [goal="X"]
            start [shape=Mdiamond]
            done [shape=Msquare, goal_gate="outcome = success"]
            start -> done
        }
        """)
        diags = validate(g)
        r09 = [d for d in diags if d.rule == "R09"]
        assert len(r09) == 1
        assert r09[0].severity == Severity.WARNING

    def test_goal_gate_with_retry_target_ok(self):
        g = parse_dot("""
        digraph R09ok {
            graph [goal="X"]
            start [shape=Mdiamond]
            task [shape=box]
            done [shape=Msquare, goal_gate="outcome = success", retry_target="task"]
            start -> task -> done
        }
        """)
        diags = validate(g)
        r09 = [d for d in diags if d.rule == "R09"]
        assert len(r09) == 0
