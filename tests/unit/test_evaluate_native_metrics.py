"""evaluate() runs modern metrics natively (#214).

A ``SimpleBaseMetric`` (``ragas.metrics.collections``, ``NumericMetric``, the
``@numeric_metric`` decorator) is no longer wrapped as a legacy ``Metric``:
evaluate() maps each sample's fields onto the metric's ``ascore(**inputs)``
and, where ``ascore`` takes a model as an argument, passes the run's.
"""

from __future__ import annotations

import math
import typing as t
import warnings

import instructor
import pytest

from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.cost import TokenUsage
from ragas.llms import llm_factory
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import NumericMetric, numeric_metric
from ragas.metrics.collections import AspectCritic
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.run_config import RunConfig

VERDICT = '{"reason": "It answers the question.", "verdict": 1}'


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def make_llm(response: str = VERDICT):
    return llm_factory("gpt-6-luna", mode=instructor.Mode.JSON, mock_response=response)


def run(dataset, metrics, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return evaluate(dataset, metrics=metrics, show_progress=False, **kwargs)


def single_turn(n: int = 2) -> EvaluationDataset:
    return EvaluationDataset(
        samples=[
            SingleTurnSample(user_input=f"q{i}", response=f"r{i}", reference="ref")
            for i in range(n)
        ]
    )


class Recording(BaseMetric):
    """Scores 0.5 and records what ascore() was called with."""

    def __init__(self, name: str = "recording", fail_on: t.Optional[str] = None):
        self.calls: t.List[t.Dict[str, t.Any]] = []
        self.fail_on = fail_on
        super().__init__(name=name)

    async def ascore(  # type: ignore[override]
        self, user_input: str, response: str, reference: t.Optional[str] = None
    ) -> MetricResult:
        self.calls.append(
            dict(user_input=user_input, response=response, reference=reference)
        )
        if user_input == self.fail_on:
            raise RuntimeError(f"boom on {user_input}")
        return MetricResult(value=0.5, reason=f"saw {response}")


class Conversation(BaseMetric):
    """Takes ``user_input`` as text, like the criteria and rubric metrics."""

    def __init__(self):
        self.seen: t.List[t.Any] = []
        super().__init__(name="conversation")

    async def ascore(self, user_input: str) -> MetricResult:  # type: ignore[override]
        self.seen.append(user_input)
        return MetricResult(value=1.0)


class Messages(BaseMetric):
    """Walks ``user_input`` as messages, like the agent metrics."""

    def __init__(self):
        self.seen: t.List[t.Any] = []
        super().__init__(name="messages")

    async def ascore(  # type: ignore[override]
        self, user_input: t.List[t.Union[HumanMessage, AIMessage]]
    ) -> MetricResult:
        self.seen.append(user_input)
        return MetricResult(value=float(len(user_input)))


def multi_turn() -> EvaluationDataset:
    return EvaluationDataset(
        samples=[
            MultiTurnSample(
                user_input=[HumanMessage(content="hi"), AIMessage(content="hello")]
            )
        ]
    )


# --- sample mapping -------------------------------------------------------------


def test_sample_fields_are_passed_to_ascore_by_name():
    metric = Recording()
    result = run(single_turn(), [metric])
    assert result["recording"] == [0.5, 0.5]
    assert sorted(metric.calls, key=lambda c: c["user_input"]) == [
        dict(user_input="q0", response="r0", reference="ref"),
        dict(user_input="q1", response="r1", reference="ref"),
    ]


def test_an_optional_input_the_sample_lacks_is_left_to_its_default():
    metric = Recording()
    dataset = EvaluationDataset(
        samples=[SingleTurnSample(user_input="q", response="r")]
    )
    run(dataset, [metric])
    assert metric.calls == [dict(user_input="q", response="r", reference=None)]


def test_the_metric_object_is_used_as_is():
    """No copy and no wrapper: the caller's metric is the one called."""
    metric = Recording()
    before = dict(metric.__dict__)
    run(single_turn(1), [metric])
    assert len(metric.calls) == 1
    assert {k: v for k, v in metric.__dict__.items() if k != "calls"} == {
        k: v for k, v in before.items() if k != "calls"
    }


def test_reasons_are_carried_through():
    result = run(single_turn(1), [Recording()])
    assert result.reasons["recording"] == ["saw r0"]


def test_missing_required_column_is_rejected_before_scoring():
    metric = Recording()
    dataset = EvaluationDataset(samples=[SingleTurnSample(user_input="q")])
    with pytest.raises(ValueError, match=r"recording.*\['response'\]"):
        run(dataset, [metric])
    assert metric.calls == []


# --- multi-turn -----------------------------------------------------------------


def test_multi_turn_text_metric_gets_the_rendered_conversation():
    metric = Conversation()
    result = run(multi_turn(), [metric])
    assert result["conversation"] == [1.0]
    assert metric.seen == [multi_turn().samples[0].pretty_repr()]


def test_multi_turn_message_metric_gets_the_messages():
    metric = Messages()
    result = run(multi_turn(), [metric])
    assert result["messages"] == [2.0]
    assert [type(m) for m in metric.seen[0]] == [HumanMessage, AIMessage]


def test_single_turn_only_metric_is_rejected_for_multi_turn_data():
    with pytest.raises(ValueError, match="does not support the sample type"):
        run(multi_turn(), [Recording()])


# --- failures -------------------------------------------------------------------


def test_a_failing_row_is_nan_and_the_others_score():
    result = run(single_turn(), [Recording(fail_on="q1")])
    scores = result["recording"]
    assert scores[0] == 0.5
    assert math.isnan(scores[1])


def test_raise_exceptions_propagates_the_metrics_error():
    with pytest.raises(RuntimeError, match="boom on q1"):
        run(single_turn(), [Recording(fail_on="q1")], raise_exceptions=True)


def test_a_non_numeric_value_is_nan():
    class Label(BaseMetric):
        async def ascore(self, response: str) -> MetricResult:  # type: ignore[override]
            return MetricResult(value="pass")

    result = run(single_turn(1), [Label(name="label")])
    assert math.isnan(result["label"][0])


def test_the_run_timeout_applies():
    import asyncio

    class Slow(BaseMetric):
        async def ascore(self, response: str) -> MetricResult:  # type: ignore[override]
            await asyncio.sleep(5)
            return MetricResult(value=1.0)

    result = run(
        single_turn(1),
        [Slow(name="slow")],
        run_config=RunConfig(timeout=0.05),  # type: ignore[arg-type]
    )
    assert math.isnan(result["slow"][0])


# --- models ---------------------------------------------------------------------


NUMERIC = '{"reason": "fine", "value": 0.7}'


def numeric(name: str = "quality"):
    return NumericMetric(
        name=name,
        prompt="Rate the response {response} to {user_input} from 0 to 1.",
        allowed_values=(0.0, 1.0),
    )


def test_a_prompt_metric_gets_the_runs_llm():
    """NumericMetric takes ``llm`` per call; it used to score NaN in evaluate()."""
    result = run(single_turn(1), [numeric()], llm=make_llm(NUMERIC))
    assert result["quality"] == [0.7]
    assert result.reasons["quality"] == ["fine"]


def test_a_prompt_metric_gets_the_default_llm_when_none_was_passed(monkeypatch):
    built = []

    def fake_default():
        built.append(1)
        return make_llm(NUMERIC)

    monkeypatch.setattr("ragas.evaluation.default_llm", fake_default)
    result = run(single_turn(2), [numeric("a"), numeric("b")])
    assert result["a"] == [0.7, 0.7] and result["b"] == [0.7, 0.7]
    assert built == [1]  # one default LLM per run, shared


def test_a_metric_with_its_own_llm_does_not_build_the_default(monkeypatch):
    def fail():
        raise AssertionError("default_llm() should not be called")

    monkeypatch.setattr("ragas.evaluation.default_llm", fail)
    critic = AspectCritic(name="answers", definition="Answers?", llm=make_llm())
    assert run(single_turn(1), [critic])["answers"] == [1.0]


def test_a_non_ragas_llm_for_a_prompt_metric_is_rejected_up_front():
    with pytest.raises(TypeError, match="llm_factory"):
        run(single_turn(1), [numeric()], llm=object())


def test_a_decorated_metric_gets_sample_fields_and_the_llm():
    llm = make_llm()
    seen = []

    @numeric_metric(name="length", allowed_values=(0.0, 100.0))
    def length(response: str, llm) -> float:
        seen.append(llm)
        return float(len(response))

    result = run(single_turn(1), [length], llm=llm)
    assert result["length"] == [2.0]
    assert seen == [llm]


# --- usage and tracing ------------------------------------------------------------


def test_llm_usage_is_collected():
    result = run(
        single_turn(3),
        [AspectCritic(name="answers", definition="Answers?", llm=make_llm())],
        token_usage_parser=lambda raw: TokenUsage(input_tokens=1, output_tokens=0),
    )
    assert result.total_tokens().input_tokens == 3


def test_each_row_holds_one_metric_run_with_the_llm_call_under_it():
    critic = AspectCritic(name="answers", definition="Answers?", llm=make_llm())
    result = run(single_turn(2), [critic])
    assert len(result.traces) == 2
    assert list(dict(result.traces[0]["answers"])) == ["AspectCriticOutput"]
    assert result.traces[1].scores == {"answers": 1.0}


# --- default metrics ----------------------------------------------------------------


def test_default_metrics_are_collections_metrics_built_with_the_runs_models():
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.evaluation import _default_metrics_for, _Models
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm = make_llm()
    metrics = _default_metrics_for(_Models(llm, None))
    assert [type(m) for m in metrics] == [
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
        ContextRecall,
    ]
    assert [m.name for m in metrics] == [
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "context_recall",
    ]
    assert all(m.llm is llm for m in metrics)
    assert isinstance(metrics[0].embeddings, BaseRagasEmbedding)
