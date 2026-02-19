"""Tests for Wave 15: Pipeline Execution & Types (P31-P34).

Covers:
- P31: Named RetryPolicy presets (spec 11.5)
- P32: Jitter field on RetryPolicy (spec 11.5)
- P33: PipelineContext class replacing bare dict (spec 11.7)
- P34: Question / Answer dataclasses for Interviewer (spec 11.8)
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------------------------------------------------------------------------
# P31: Named retry presets (spec 11.5)
# ---------------------------------------------------------------------------


class TestRetryPresets:
    """P31: RETRY_PRESETS dict exposes the five named presets."""

    def test_retry_presets_all_five_exist(self) -> None:
        """All five preset names are present in RETRY_PRESETS."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        expected = {"none", "standard", "aggressive", "linear", "patient"}
        assert expected == set(RETRY_PRESETS.keys())

    def test_retry_preset_none_has_zero_retries(self) -> None:
        """The 'none' preset never retries."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        assert RETRY_PRESETS["none"].max_retries == 0

    def test_retry_preset_standard_values(self) -> None:
        """The 'standard' preset has the spec-mandated values."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        std = RETRY_PRESETS["standard"]
        assert std.max_retries == 3
        assert std.initial_delay == pytest.approx(1.0)
        assert std.backoff_factor == pytest.approx(2.0)
        assert std.max_delay == pytest.approx(30.0)
        assert std.jitter is True

    def test_retry_preset_aggressive_values(self) -> None:
        """The 'aggressive' preset has 5 retries with tighter backoff."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        agg = RETRY_PRESETS["aggressive"]
        assert agg.max_retries == 5
        assert agg.initial_delay == pytest.approx(0.5)
        assert agg.backoff_factor == pytest.approx(1.5)
        assert agg.max_delay == pytest.approx(10.0)
        assert agg.jitter is True

    def test_retry_preset_linear_no_jitter(self) -> None:
        """The 'linear' preset uses backoff_factor=1.0 and jitter=False."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        lin = RETRY_PRESETS["linear"]
        assert lin.backoff_factor == pytest.approx(1.0)
        assert lin.jitter is False

    def test_retry_preset_patient_values(self) -> None:
        """The 'patient' preset waits up to 120 s and retries 10 times."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS

        pat = RETRY_PRESETS["patient"]
        assert pat.max_retries == 10
        assert pat.initial_delay == pytest.approx(5.0)
        assert pat.max_delay == pytest.approx(120.0)
        assert pat.jitter is True

    def test_get_retry_preset_known_name(self) -> None:
        """get_retry_preset returns the correct policy for a known name."""
        from attractor_pipeline.engine.runner import RETRY_PRESETS, get_retry_preset

        for name in RETRY_PRESETS:
            assert get_retry_preset(name) is RETRY_PRESETS[name]

    def test_get_retry_preset_unknown_returns_none(self) -> None:
        """get_retry_preset returns None for an unrecognised name."""
        from attractor_pipeline.engine.runner import get_retry_preset

        assert get_retry_preset("bogus") is None
        assert get_retry_preset("") is None
        assert get_retry_preset("STANDARD") is None  # case-sensitive


# ---------------------------------------------------------------------------
# P32: Jitter field on RetryPolicy (spec 11.5)
# ---------------------------------------------------------------------------


class TestRetryPolicyJitter:
    """P32: RetryPolicy must have a boolean jitter field that controls delay variance."""

    def test_retry_policy_has_jitter_field(self) -> None:
        """RetryPolicy exposes a 'jitter' attribute."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy()
        assert hasattr(policy, "jitter"), "RetryPolicy must have a 'jitter' attribute"

    def test_jitter_default_is_true(self) -> None:
        """jitter defaults to True on a freshly constructed RetryPolicy."""
        from attractor_llm.retry import RetryPolicy

        assert RetryPolicy().jitter is True

    def test_jitter_can_be_disabled(self) -> None:
        """jitter=False is accepted and stored correctly."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(jitter=False)
        assert policy.jitter is False

    def test_jitter_false_gives_deterministic_delay(self) -> None:
        """With jitter=False, compute_delay is deterministic across calls."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(initial_delay=1.0, backoff_factor=2.0, jitter=False)
        delays = [policy.compute_delay(0) for _ in range(20)]
        # Without jitter every call at the same attempt must return the same value.
        assert len(set(delays)) == 1, "jitter=False must produce identical delays"

    def test_jitter_true_introduces_variance(self) -> None:
        """With jitter=True, compute_delay varies across repeated calls."""
        from attractor_llm.retry import RetryPolicy

        policy = RetryPolicy(initial_delay=10.0, backoff_factor=1.0, jitter=True)
        delays = [policy.compute_delay(0) for _ in range(50)]
        # With jitter it is astronomically unlikely that 50 draws are identical.
        assert len(set(delays)) > 1, "jitter=True must introduce variance in delays"

    def test_jitter_field_is_dataclass_field(self) -> None:
        """jitter is a proper dataclass field (not a property or class var)."""
        from attractor_llm.retry import RetryPolicy

        fields = {f.name for f in dataclasses.fields(RetryPolicy)}
        assert "jitter" in fields


# ---------------------------------------------------------------------------
# P33: PipelineContext class (spec 11.7)
# ---------------------------------------------------------------------------


class TestPipelineContext:
    """P33: PipelineContext wraps a dict with a structured interface."""

    def test_pipeline_context_get_set(self) -> None:
        """get() and set() work like dict.__getitem__ / __setitem__."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        ctx.set("answer", 42)
        assert ctx.get("answer") == 42

    def test_pipeline_context_get_default(self) -> None:
        """get() returns the supplied default for missing keys."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "fallback") == "fallback"

    def test_pipeline_context_snapshot_is_copy(self) -> None:
        """snapshot() returns a shallow copy; mutating it does not affect the context."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        ctx.set("x", 1)
        snap = ctx.snapshot()

        # Mutate the snapshot -- the context must be unchanged.
        snap["x"] = 999
        snap["new_key"] = "intrusion"

        assert ctx.get("x") == 1
        assert ctx.get("new_key") is None

    def test_pipeline_context_clone_is_independent(self) -> None:
        """clone() produces a PipelineContext whose mutations do not leak back."""
        from attractor_pipeline.engine.runner import PipelineContext

        original = PipelineContext()
        original.set("shared", "original")

        clone = original.clone()
        assert isinstance(clone, PipelineContext)
        assert clone.get("shared") == "original"

        # Mutate the clone.
        clone.set("shared", "cloned")
        clone.set("extra", "only-in-clone")

        # Original must be unaffected.
        assert original.get("shared") == "original"
        assert original.get("extra") is None

    def test_pipeline_context_apply_updates(self) -> None:
        """apply_updates() merges a dict into the context (last-write wins)."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        ctx.set("a", 1)
        ctx.apply_updates({"a": 99, "b": 2})

        assert ctx.get("a") == 99
        assert ctx.get("b") == 2

    def test_pipeline_context_append_log(self) -> None:
        """append_log() accumulates entries under the '_log' key."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        ctx.append_log("step 1")
        ctx.append_log("step 2")

        log = ctx.get("_log")
        assert isinstance(log, list)
        assert log == ["step 1", "step 2"]

    def test_pipeline_context_append_log_order_preserved(self) -> None:
        """append_log() preserves insertion order across many entries."""
        from attractor_pipeline.engine.runner import PipelineContext

        ctx = PipelineContext()
        entries = [f"entry-{i}" for i in range(10)]
        for e in entries:
            ctx.append_log(e)

        assert ctx.get("_log") == entries

    @pytest.mark.asyncio
    async def test_run_pipeline_accepts_dict_context(self) -> None:
        """run_pipeline() accepts a bare dict for context (backward compat)."""
        from attractor_pipeline.engine.runner import (
            HandlerRegistry,
            HandlerResult,
            Outcome,
            PipelineStatus,
            run_pipeline,
        )
        from attractor_pipeline.graph import Edge, Graph, Node

        # Minimal two-node graph: start (Mdiamond) -> exit (Msquare).
        start_node = Node(id="start", shape="Mdiamond")
        exit_node = Node(id="exit", shape="Msquare")
        edge = Edge(source="start", target="exit")
        graph = Graph(
            nodes={"start": start_node, "exit": exit_node},
            edges=[edge],
        )

        class _NopHandler:
            async def execute(self, node, context, graph, logs_root, abort_signal=None):
                return HandlerResult(status=Outcome.SUCCESS)

        registry = HandlerRegistry()
        registry.register(start_node.effective_handler, _NopHandler())
        registry.register(exit_node.effective_handler, _NopHandler())

        # Pass a plain dict -- must not raise TypeError.
        result = await run_pipeline(graph, registry, context={"key": "value"})

        assert result.status == PipelineStatus.COMPLETED
        # The value set in the input dict must survive into the result context.
        assert result.context.get("key") == "value"

    @pytest.mark.asyncio
    async def test_run_pipeline_accepts_pipeline_context(self) -> None:
        """run_pipeline() also accepts a PipelineContext object directly."""
        from attractor_pipeline.engine.runner import (
            HandlerRegistry,
            HandlerResult,
            Outcome,
            PipelineContext,
            PipelineStatus,
            run_pipeline,
        )
        from attractor_pipeline.graph import Edge, Graph, Node

        start_node = Node(id="start", shape="Mdiamond")
        exit_node = Node(id="exit", shape="Msquare")
        edge = Edge(source="start", target="exit")
        graph = Graph(
            nodes={"start": start_node, "exit": exit_node},
            edges=[edge],
        )

        class _NopHandler:
            async def execute(self, node, context, graph, logs_root, abort_signal=None):
                return HandlerResult(status=Outcome.SUCCESS)

        registry = HandlerRegistry()
        registry.register(start_node.effective_handler, _NopHandler())
        registry.register(exit_node.effective_handler, _NopHandler())

        ctx = PipelineContext()
        ctx.set("from_ctx", True)

        result = await run_pipeline(graph, registry, context=ctx)

        assert result.status == PipelineStatus.COMPLETED
        assert result.context.get("from_ctx") is True


# ---------------------------------------------------------------------------
# P34: Question / Answer dataclasses for Interviewer (spec 11.8)
# ---------------------------------------------------------------------------


class TestQuestionAnswerDataclasses:
    """P34: Question and Answer are proper dataclasses with the spec fields."""

    def test_question_dataclass_fields(self) -> None:
        """Question exposes the spec-mandated fields with correct defaults."""
        from attractor_pipeline.handlers.human import Question, QuestionType

        q = Question(text="Do you approve?")

        assert q.text == "Do you approve?"
        assert q.question_type == QuestionType.FREE_TEXT  # default
        assert q.options is None
        assert q.default is None
        assert q.timeout_seconds is None
        assert q.stage == ""
        assert q.metadata == {}

    def test_question_dataclass_all_fields(self) -> None:
        """Question can be constructed with every field supplied explicitly."""
        from attractor_pipeline.handlers.human import Question, QuestionType

        q = Question(
            text="Pick one",
            question_type=QuestionType.SINGLE_SELECT,
            options=["yes", "no"],
            default="yes",
            timeout_seconds=30.0,
            stage="review",
            metadata={"node": "approval"},
        )

        assert q.text == "Pick one"
        assert q.question_type == QuestionType.SINGLE_SELECT
        assert q.options == ["yes", "no"]
        assert q.default == "yes"
        assert q.timeout_seconds == pytest.approx(30.0)
        assert q.stage == "review"
        assert q.metadata == {"node": "approval"}

    def test_question_is_dataclass(self) -> None:
        """Question is a real dataclass (not just a plain class)."""
        from attractor_pipeline.handlers.human import Question

        assert dataclasses.is_dataclass(Question)

    def test_answer_dataclass_fields(self) -> None:
        """Answer exposes the spec-mandated fields with correct defaults."""
        from attractor_pipeline.handlers.human import Answer

        a = Answer(value="yes")

        assert a.value == "yes"
        assert a.selected_option is None  # default
        assert a.text == ""  # default

    def test_answer_dataclass_all_fields(self) -> None:
        """Answer can be constructed with every field supplied explicitly."""
        from attractor_pipeline.handlers.human import Answer

        a = Answer(value="yes", selected_option="yes", text="User selected yes")

        assert a.value == "yes"
        assert a.selected_option == "yes"
        assert a.text == "User selected yes"

    def test_answer_is_dataclass(self) -> None:
        """Answer is a real dataclass (not just a plain class)."""
        from attractor_pipeline.handlers.human import Answer

        assert dataclasses.is_dataclass(Answer)


class TestInterviewerAcceptsQuestionReturnsAnswer:
    """P34: Interviewer implementations expose ask_question() -> Answer."""

    @pytest.mark.asyncio
    async def test_interviewer_accepts_question_returns_answer(self) -> None:
        """AutoApproveInterviewer satisfies the Question -> Answer API."""
        from attractor_pipeline.handlers.human import (
            Answer,
            AutoApproveInterviewer,
            Question,
        )

        interviewer = AutoApproveInterviewer()
        q = Question(text="What is your name?")
        answer = await interviewer.ask_question(q)

        assert isinstance(answer, Answer)
        assert isinstance(answer.value, str)
        assert len(answer.value) > 0

    @pytest.mark.asyncio
    async def test_auto_approve_ask_question_with_options(self) -> None:
        """AutoApproveInterviewer.ask_question() returns first option when options given."""
        from attractor_pipeline.handlers.human import (
            Answer,
            AutoApproveInterviewer,
            Question,
            QuestionType,
        )

        interviewer = AutoApproveInterviewer()
        q = Question(
            text="Approve?",
            question_type=QuestionType.SINGLE_SELECT,
            options=["yes", "no"],
        )
        result = await interviewer.ask_question(q)

        assert isinstance(result, Answer)
        assert result.value == "yes"  # first option auto-approved
        assert result.selected_option == "yes"

    @pytest.mark.asyncio
    async def test_queue_interviewer_ask_question(self) -> None:
        """QueueInterviewer.ask_question() draws answers from the pre-filled queue."""
        from attractor_pipeline.handlers.human import (
            Answer,
            Question,
            QueueInterviewer,
        )

        interviewer = QueueInterviewer(answers=["approved", "rejected"])
        q = Question(text="First question?")
        first = await interviewer.ask_question(q)
        second = await interviewer.ask_question(q)
        third = await interviewer.ask_question(q)  # queue exhausted

        assert isinstance(first, Answer)
        assert first.value == "approved"
        assert second.value == "rejected"
        assert third.value == "SKIPPED"

    @pytest.mark.asyncio
    async def test_ask_question_selected_option_set_when_in_options(self) -> None:
        """selected_option is populated when the returned value is one of the options."""
        from attractor_pipeline.handlers.human import (
            AutoApproveInterviewer,
            Question,
            QuestionType,
        )

        interviewer = AutoApproveInterviewer()
        q = Question(
            text="Choose",
            question_type=QuestionType.SINGLE_SELECT,
            options=["alpha", "beta", "gamma"],
        )
        answer = await interviewer.ask_question(q)

        assert q.options is not None
        assert answer.value in q.options
        assert answer.selected_option == answer.value

    @pytest.mark.asyncio
    async def test_ask_question_selected_option_none_for_free_text(self) -> None:
        """selected_option is None when there are no options (free-text question)."""
        from attractor_pipeline.handlers.human import AutoApproveInterviewer, Question

        interviewer = AutoApproveInterviewer()
        q = Question(text="Any thoughts?")
        answer = await interviewer.ask_question(q)

        # No options list -> selected_option must be None.
        assert answer.selected_option is None

    @pytest.mark.asyncio
    async def test_callback_interviewer_ask_question(self) -> None:
        """CallbackInterviewer.ask_question() delegates to the underlying callback."""
        from attractor_pipeline.handlers.human import (
            Answer,
            CallbackInterviewer,
            Question,
        )

        async def _cb(question: str, options, node_id) -> str:
            return f"echo:{question}"

        interviewer = CallbackInterviewer(_cb)
        q = Question(text="ping")
        answer = await interviewer.ask_question(q)

        assert isinstance(answer, Answer)
        assert answer.value == "echo:ping"

    def test_question_backward_compat_flat_ask_unchanged(self) -> None:
        """The flat ask() method still exists on all interviewer implementations."""
        import inspect

        from attractor_pipeline.handlers.human import (
            AutoApproveInterviewer,
            ConsoleInterviewer,
            QueueInterviewer,
        )

        for cls in (AutoApproveInterviewer, ConsoleInterviewer, QueueInterviewer):
            assert hasattr(cls, "ask"), f"{cls.__name__} must retain ask()"
            sig = inspect.signature(cls.ask)
            params = list(sig.parameters.keys())
            assert "question" in params, f"{cls.__name__}.ask() must have 'question' param"
