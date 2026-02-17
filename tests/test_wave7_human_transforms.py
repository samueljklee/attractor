"""Tests for Wave 7 spec compliance: Items #15, #16, #17.

Item #15: CallbackInterviewer and QueueInterviewer (Spec §6, §11.8)
Item #16: QuestionType enum and ask() protocol update (Spec §6, §11.8)
Item #17: GraphTransform pipeline (Spec §9, §11.11)
"""

from __future__ import annotations

import pytest

from attractor_pipeline.graph import Edge, Graph, Node
from attractor_pipeline.handlers.human import (
    AutoApproveInterviewer,
    CallbackInterviewer,
    QuestionType,
    QueueInterviewer,
)
from attractor_pipeline.transforms import (
    GraphTransform,
    VariableExpansionTransform,
    apply_transforms,
)

# ------------------------------------------------------------------ #
# Item #15: CallbackInterviewer
# ------------------------------------------------------------------ #


class TestCallbackInterviewer:
    """CallbackInterviewer delegates to the provided async callback."""

    @pytest.mark.anyio
    async def test_callback_receives_arguments(self) -> None:
        """The callback receives question, options, and node_id."""
        received: dict[str, object] = {}

        async def cb(
            question: str,
            options: list[str] | None,
            node_id: str | None,
        ) -> str:
            received["question"] = question
            received["options"] = options
            received["node_id"] = node_id
            return "callback-answer"

        interviewer = CallbackInterviewer(cb)
        result = await interviewer.ask("What?", ["a", "b"], node_id="n1")

        assert result == "callback-answer"
        assert received["question"] == "What?"
        assert received["options"] == ["a", "b"]
        assert received["node_id"] == "n1"

    @pytest.mark.anyio
    async def test_callback_with_defaults(self) -> None:
        """Callback works with default option/node_id values."""

        async def cb(
            question: str,
            options: list[str] | None,
            node_id: str | None,
        ) -> str:
            return f"got: {question}"

        interviewer = CallbackInterviewer(cb)
        result = await interviewer.ask("Hello?")
        assert result == "got: Hello?"

    @pytest.mark.anyio
    async def test_callback_none_node_id_when_empty(self) -> None:
        """Empty node_id string is passed as None to the callback."""
        received_node_id: list[str | None] = []

        async def cb(
            question: str,
            options: list[str] | None,
            node_id: str | None,
        ) -> str:
            received_node_id.append(node_id)
            return "ok"

        interviewer = CallbackInterviewer(cb)
        await interviewer.ask("q")
        assert received_node_id[0] is None


# ------------------------------------------------------------------ #
# Item #15: QueueInterviewer
# ------------------------------------------------------------------ #


class TestQueueInterviewer:
    """QueueInterviewer returns answers in order from a pre-filled queue."""

    @pytest.mark.anyio
    async def test_returns_answers_in_order(self) -> None:
        interviewer = QueueInterviewer(["first", "second", "third"])

        assert await interviewer.ask("q1") == "first"
        assert await interviewer.ask("q2") == "second"
        assert await interviewer.ask("q3") == "third"

    @pytest.mark.anyio
    async def test_returns_skipped_when_exhausted(self) -> None:
        """Spec §6.4: returns 'SKIPPED' when queue is empty."""
        interviewer = QueueInterviewer(["only-one"])
        await interviewer.ask("q1")

        result = await interviewer.ask("q2")
        assert result == "SKIPPED"

    @pytest.mark.anyio
    async def test_empty_queue_returns_skipped_immediately(self) -> None:
        """Spec §6.4: empty queue returns 'SKIPPED' on first ask."""
        interviewer = QueueInterviewer([])

        result = await interviewer.ask("q1")
        assert result == "SKIPPED"

    @pytest.mark.anyio
    async def test_ignores_options_and_node_id(self) -> None:
        """Queue answers are returned regardless of question content."""
        interviewer = QueueInterviewer(["yes"])
        result = await interviewer.ask("Approve?", options=["yes", "no"], node_id="gate1")
        assert result == "yes"

    @pytest.mark.anyio
    async def test_original_list_not_mutated(self) -> None:
        """QueueInterviewer copies the input list."""
        answers = ["a", "b"]
        interviewer = QueueInterviewer(answers)
        await interviewer.ask("q")
        assert answers == ["a", "b"]  # original unchanged


# ------------------------------------------------------------------ #
# Item #16: QuestionType enum
# ------------------------------------------------------------------ #


class TestQuestionType:
    """QuestionType enum has all 4 required values."""

    def test_has_all_four_values(self) -> None:
        assert QuestionType.SINGLE_SELECT == "single_select"
        assert QuestionType.MULTI_SELECT == "multi_select"
        assert QuestionType.FREE_TEXT == "free_text"
        assert QuestionType.CONFIRM == "confirm"

    def test_exactly_four_members(self) -> None:
        assert len(QuestionType) == 4

    def test_is_str_enum(self) -> None:
        """QuestionType values are strings (StrEnum)."""
        for member in QuestionType:
            assert isinstance(member, str)

    @pytest.mark.anyio
    async def test_auto_approve_accepts_question_type(self) -> None:
        """Existing AutoApproveInterviewer accepts the new parameter."""
        interviewer = AutoApproveInterviewer()
        result = await interviewer.ask(
            "OK?",
            options=["yes", "no"],
            question_type=QuestionType.CONFIRM,
        )
        assert result == "yes"

    @pytest.mark.anyio
    async def test_queue_accepts_question_type(self) -> None:
        """QueueInterviewer accepts the new question_type parameter."""
        interviewer = QueueInterviewer(["answer"])
        result = await interviewer.ask(
            "Pick one",
            options=["a", "b"],
            question_type=QuestionType.SINGLE_SELECT,
        )
        assert result == "answer"

    @pytest.mark.anyio
    async def test_callback_accepts_question_type(self) -> None:
        """CallbackInterviewer accepts the new question_type parameter."""

        async def cb(q: str, opts: list[str] | None, nid: str | None) -> str:
            return "cb"

        interviewer = CallbackInterviewer(cb)
        result = await interviewer.ask("Pick", question_type=QuestionType.MULTI_SELECT)
        assert result == "cb"


# ------------------------------------------------------------------ #
# Item #17: GraphTransform protocol
# ------------------------------------------------------------------ #


def _make_graph(**kwargs: object) -> Graph:
    """Create a minimal valid graph for testing."""
    g = Graph(name="test")
    g.nodes["start"] = Node(id="start", shape="ellipse", label="Start")
    g.nodes["end"] = Node(id="end", shape="Msquare", label="End")
    g.edges.append(Edge(source="start", target="end"))
    for k, v in kwargs.items():
        setattr(g, k, v)
    return g


class TestGraphTransformProtocol:
    """GraphTransform protocol and apply_transforms()."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """GraphTransform can be used with isinstance()."""

        class MyTransform:
            def apply(self, graph: Graph) -> Graph:
                return graph

        assert isinstance(MyTransform(), GraphTransform)

    def test_class_without_apply_is_not_transform(self) -> None:
        class NotATransform:
            pass

        assert not isinstance(NotATransform(), GraphTransform)

    def test_empty_transforms_is_noop(self) -> None:
        """An empty transform list returns the graph unchanged."""
        graph = _make_graph()
        result = apply_transforms(graph, [])
        assert result is graph

    def test_single_transform_applied(self) -> None:
        """A single transform modifies the graph."""

        class AddGoal:
            def apply(self, graph: Graph) -> Graph:
                graph.goal = "transformed"
                return graph

        graph = _make_graph(goal="original")
        result = apply_transforms(graph, [AddGoal()])
        assert result.goal == "transformed"

    def test_transforms_applied_in_order(self) -> None:
        """Transforms are applied in list order."""
        log: list[str] = []

        class LogTransform:
            def __init__(self, name: str) -> None:
                self._name = name

            def apply(self, graph: Graph) -> Graph:
                log.append(self._name)
                return graph

        graph = _make_graph()
        apply_transforms(
            graph, [LogTransform("first"), LogTransform("second"), LogTransform("third")]
        )
        assert log == ["first", "second", "third"]

    def test_chained_transforms_accumulate(self) -> None:
        """Each transform sees the result of previous transforms."""

        class AppendToGoal:
            def __init__(self, suffix: str) -> None:
                self._suffix = suffix

            def apply(self, graph: Graph) -> Graph:
                graph.goal = (graph.goal or "") + self._suffix
                return graph

        graph = _make_graph(goal="")
        result = apply_transforms(graph, [AppendToGoal("A"), AppendToGoal("B"), AppendToGoal("C")])
        assert result.goal == "ABC"


# ------------------------------------------------------------------ #
# Item #17: VariableExpansionTransform
# ------------------------------------------------------------------ #


class TestVariableExpansionTransform:
    """Built-in transform that expands $variables in node prompts."""

    def test_expands_variables_in_prompts(self) -> None:
        graph = _make_graph()
        graph.nodes["start"].prompt = "Goal is $goal"
        graph.nodes["end"].prompt = "Done with $project"

        transform = VariableExpansionTransform({"goal": "test-goal", "project": "attractor"})
        result = transform.apply(graph)

        assert result.nodes["start"].prompt == "Goal is test-goal"
        assert result.nodes["end"].prompt == "Done with attractor"

    def test_undefined_variables_kept(self) -> None:
        graph = _make_graph()
        graph.nodes["start"].prompt = "Hello $undefined"

        transform = VariableExpansionTransform({})
        result = transform.apply(graph)

        assert result.nodes["start"].prompt == "Hello $undefined"

    def test_empty_prompts_unchanged(self) -> None:
        graph = _make_graph()
        graph.nodes["start"].prompt = ""

        transform = VariableExpansionTransform({"goal": "x"})
        result = transform.apply(graph)

        assert result.nodes["start"].prompt == ""

    def test_works_in_pipeline(self) -> None:
        """VariableExpansionTransform integrates with apply_transforms."""
        graph = _make_graph()
        graph.nodes["start"].prompt = "Do $task"

        result = apply_transforms(
            graph,
            [VariableExpansionTransform({"task": "testing"})],
        )
        assert result.nodes["start"].prompt == "Do testing"


# ------------------------------------------------------------------ #
# Item #17: Integration with run_pipeline
# ------------------------------------------------------------------ #


class TestRunPipelineTransforms:
    """Transforms parameter is accepted by run_pipeline()."""

    @pytest.mark.anyio
    async def test_transforms_param_accepted(self) -> None:
        """run_pipeline accepts the transforms kwarg without error."""
        from attractor_pipeline.engine.runner import (
            HandlerRegistry,
            run_pipeline,
        )
        from attractor_pipeline.handlers.basic import ExitHandler, StartHandler

        graph = _make_graph()
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())

        # Empty transforms -- should work like before
        result = await run_pipeline(graph, registry, transforms=[])
        assert result.status.value == "completed"

    @pytest.mark.anyio
    async def test_transforms_applied_before_execution(self) -> None:
        """Transforms modify the graph before the engine runs."""
        from attractor_pipeline.engine.runner import (
            HandlerRegistry,
            run_pipeline,
        )
        from attractor_pipeline.handlers.basic import ExitHandler, StartHandler

        graph = _make_graph(goal="original")
        registry = HandlerRegistry()
        registry.register("start", StartHandler())
        registry.register("exit", ExitHandler())

        class SetGoal:
            def apply(self, g: Graph) -> Graph:
                g.goal = "transformed-goal"
                return g

        result = await run_pipeline(graph, registry, transforms=[SetGoal()])
        # The context should have the transformed goal
        assert result.context.get("goal") == "transformed-goal"
