"""``evaluate()`` runs ragas.metrics.collections metrics through their signature.

A collections metric is a ``SimpleBaseMetric`` with only ``ascore(**kwargs)``.
``ragas.metrics.sample_inputs`` reads that signature and maps sample fields
onto the call, so these tests pin the behaviour that makes it generic, not any
one metric.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.dataset_schema import MultiTurnSample as MTS, SingleTurnSample as STS
from ragas.llms.base import InstructorLLM
from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics import collections as collections_module
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.sample_inputs import _as_score, input_spec


def collections_classes():
    """Every concrete metric class in ragas.metrics.collections."""
    out = []
    for name in sorted(dir(collections_module)):
        if name.startswith("_"):
            continue
        obj = getattr(collections_module, name)
        if isinstance(obj, type) and name not in ("BaseMetric", "DistanceMeasure"):
            if issubclass(obj, SimpleBaseMetric):
                out.append((name, obj))
    return out


def fixed_score(metric, value=0.75):
    """Replace the LLM round-trip, keeping the real ``ascore`` signature intact."""
    real = type(metric).ascore
    metric._real_ascore_signature = inspect.signature(real)
    metric.ascore = lambda **kw: asyncio.sleep(0, MetricResult(value=value))
    return metric


@pytest.fixture
def llm():
    return MagicMock(spec=InstructorLLM)


@pytest.fixture
def single_turn_dataset():
    return EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=f"q{i}",
                retrieved_contexts=["ctx"],
                reference="ref",
                response="resp",
            )
            for i in range(3)
        ]
    )


class TestAdapterCoverage:
    """The bridge must handle the whole collections namespace, not a sample of it."""

    @pytest.mark.parametrize("name,cls", collections_classes())
    def test_every_metric_maps_onto_a_sample_type(self, name, cls):
        """An ascore parameter that is on neither sample type would be unreachable."""
        params = {
            p
            for p, spec in inspect.signature(cls.ascore).parameters.items()
            if p != "self" and spec.kind not in (spec.VAR_KEYWORD, spec.VAR_POSITIONAL)
        }
        unknown = params - set(STS.model_fields) - set(MTS.model_fields)
        assert not unknown, (
            f"{name}.ascore takes {sorted(unknown)}, which no sample can supply"
        )


class TestInputSpec:
    def test_required_columns_come_from_the_signature(self, llm):
        from ragas.metrics.collections import ContextRecall

        spec = input_spec(ContextRecall(llm=llm))
        assert spec.required_columns() == {
            "user_input",
            "retrieved_contexts",
            "reference",
        }
        assert spec.single_turn and not spec.multi_turn

    def test_optional_parameters_are_not_required(self, llm):
        """DomainSpecificRubrics takes everything optionally; nothing is demanded."""
        from ragas.metrics.collections import DomainSpecificRubrics

        assert input_spec(DomainSpecificRubrics(llm=llm)).required_columns() == set()

    def test_multi_turn_metric_is_multi_turn_only(self):
        from ragas.metrics.collections import ToolCallAccuracy

        spec = input_spec(ToolCallAccuracy())
        assert spec.multi_turn and not spec.single_turn

    def test_metric_valid_for_both_sample_types_supports_both(self, llm):
        from ragas.metrics.collections import AgentGoalAccuracyWithoutReference

        spec = input_spec(AgentGoalAccuracyWithoutReference(llm=llm))
        assert spec.single_turn and spec.multi_turn

    def test_unmappable_signature_is_rejected_with_guidance(self):
        class Weird(SimpleBaseMetric):
            def score(self, **kwargs):  # pragma: no cover - not called
                ...

            async def ascore(self, not_a_sample_field: str):  # type: ignore[override]
                ...  # pragma: no cover - not called

        with pytest.raises(ValueError, match="not_a_sample_field"):
            input_spec(Weird(name="weird"))


class TestEvaluateAcceptsCollectionsMetrics:
    def test_scores_are_produced(self, llm, single_turn_dataset):
        from ragas.metrics.collections import ContextRecall

        metric = fixed_score(ContextRecall(llm=llm), 0.8)
        result = evaluate(single_turn_dataset, metrics=[metric])
        assert result["context_recall"] == [0.8, 0.8, 0.8]

    def test_llm_and_non_llm_metrics_together(self, llm, single_turn_dataset):
        from ragas.metrics.collections import BleuScore, ContextRecall

        metric = fixed_score(ContextRecall(llm=llm), 0.9)
        result = evaluate(single_turn_dataset, metrics=[metric, BleuScore()])
        df = result.to_pandas()
        assert "context_recall" in df.columns
        assert "bleu_score" in df.columns

    def test_traces_are_populated(self, llm, single_turn_dataset):
        """EvaluationResult.traces must not go empty for collections metrics."""
        metric = fixed_score(
            __import__(
                "ragas.metrics.collections", fromlist=["ContextRecall"]
            ).ContextRecall(llm=llm)
        )
        result = evaluate(single_turn_dataset, metrics=[metric])
        assert len(result.traces) == 3
        assert "context_recall" in result.traces[0]

    def test_missing_column_is_reported_before_any_llm_call(self, llm):
        from ragas.metrics.collections import ContextRecall

        dataset = EvaluationDataset(
            samples=[SingleTurnSample(user_input="q", response="r")]
        )
        with pytest.raises(ValueError, match="retrieved_contexts|reference"):
            evaluate(dataset, metrics=[ContextRecall(llm=llm)])

    def test_sample_type_mismatch_is_rejected(self, single_turn_dataset):
        from ragas.metrics.collections import ToolCallAccuracy

        with pytest.raises(ValueError, match="does not support the sample type"):
            evaluate(single_turn_dataset, metrics=[ToolCallAccuracy()])

    def test_multi_turn_metric_runs_unstubbed(self):
        """ToolCallAccuracy needs no LLM, so this exercises the real scoring path."""
        from ragas.metrics.collections import ToolCallAccuracy

        dataset = EvaluationDataset(
            samples=[
                MultiTurnSample(
                    user_input=[
                        HumanMessage(content="weather in Paris?"),
                        AIMessage(
                            content="checking",
                            tool_calls=[
                                ToolCall(name="weather", args={"city": "Paris"})
                            ],
                        ),
                    ],
                    reference_tool_calls=[
                        ToolCall(name="weather", args={"city": "Paris"})
                    ],
                )
            ]
        )
        result = evaluate(dataset, metrics=[ToolCallAccuracy()])
        assert result["tool_call_accuracy"] == [1.0]

    def test_a_plain_non_metric_is_still_rejected(self, single_turn_dataset):
        """The wrap must not swallow the existing type guard."""
        with pytest.raises(TypeError, match="initialised metric objects"):
            evaluate(single_turn_dataset, metrics=["not a metric"])  # type: ignore[list-item]


class TestScoreConversion:
    @pytest.mark.parametrize(
        "value,expected", [(0.5, 0.5), (1, 1.0), (True, 1.0), ("0.25", 0.25)]
    )
    def test_numeric_values_pass_through(self, value, expected):
        assert _as_score(value, "m") == expected

    def test_non_numeric_value_names_the_metric_and_the_alternative(self):
        with pytest.raises(ValueError, match="ascore"):
            _as_score("pass", "my_metric")
