"""Collections ports of the four non-LLM context metrics.

These four had no unit tests at all in their legacy form, so the port is also
the first coverage they have ever had.

The load-bearing tests are the equivalence ones: a port that quietly changes
scores is worse than no port, because users comparing runs across a version
bump would see drift with no signal. Each new metric is checked against its
legacy counterpart over randomised inputs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
)
from ragas.metrics.collections._average_precision import average_precision
from ragas.metrics.collections._string import DistanceMeasure

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CORPUS = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Berlin is the capital of Germany.",
    "Paris is the capitol of France",  # near-miss spelling
    "Completely unrelated sentence about turtles.",
]


def legacy(name):
    """Import the legacy counterpart lazily; importing warns by design."""
    import ragas.metrics as m

    return getattr(m, name)


def same(a, b):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) < 1e-9


class TestIDBasedMetrics:
    @pytest.mark.asyncio
    async def test_precision_counts_relevant_retrieved(self):
        result = await IDBasedContextPrecision().ascore(
            retrieved_context_ids=["d1", "d2", "d3"],
            reference_context_ids=["d1", "d2"],
        )
        assert result.value == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_recall_counts_references_retrieved(self):
        result = await IDBasedContextRecall().ascore(
            retrieved_context_ids=["d1", "d2"],
            reference_context_ids=["d1", "d2", "d3"],
        )
        assert result.value == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_int_and_str_ids_compare_equal(self):
        """Documented behaviour: IDs are normalised to strings."""
        result = await IDBasedContextPrecision().ascore(
            retrieved_context_ids=[1, 2], reference_context_ids=["1", "2"]
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_duplicate_ids_are_deduplicated(self):
        """Sets, not lists -- a context retrieved twice counts once."""
        result = await IDBasedContextPrecision().ascore(
            retrieved_context_ids=["d1", "d1", "d2"],
            reference_context_ids=["d1"],
        )
        assert result.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_empty_retrieved_is_nan_not_zero(self):
        """Nothing retrieved is undefined precision, not perfect or zero."""
        result = await IDBasedContextPrecision().ascore(
            retrieved_context_ids=[], reference_context_ids=["d1"]
        )
        assert np.isnan(result.value)

    @pytest.mark.asyncio
    async def test_empty_reference_is_nan_for_recall(self):
        result = await IDBasedContextRecall().ascore(
            retrieved_context_ids=["d1"], reference_context_ids=[]
        )
        assert np.isnan(result.value)


class TestNonLLMContextMetrics:
    @pytest.mark.asyncio
    async def test_exact_match_scores_one(self):
        ctx = ["Paris is the capital of France."]
        assert (
            await NonLLMContextRecall().ascore(
                retrieved_contexts=ctx, reference_contexts=ctx
            )
        ).value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_unrelated_context_scores_zero(self):
        result = await NonLLMContextPrecisionWithReference().ascore(
            retrieved_contexts=["Completely unrelated sentence about turtles."],
            reference_contexts=["Paris is the capital of France."],
        )
        assert result.value == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_ranking_matters_for_precision(self):
        """Average precision rewards relevant contexts appearing earlier."""
        relevant = "Paris is the capital of France."
        noise = "Completely unrelated sentence about turtles."
        metric = NonLLMContextPrecisionWithReference()

        first = await metric.ascore(
            retrieved_contexts=[relevant, noise], reference_contexts=[relevant]
        )
        last = await metric.ascore(
            retrieved_contexts=[noise, relevant], reference_contexts=[relevant]
        )
        assert first.value > last.value

    @pytest.mark.asyncio
    async def test_distance_measure_is_configurable(self):
        metric = NonLLMContextPrecisionWithReference(
            distance_measure=DistanceMeasure.JARO_WINKLER
        )
        assert metric.distance_measure is DistanceMeasure.JARO_WINKLER
        result = await metric.ascore(
            retrieved_contexts=["Paris is the capital of France."],
            reference_contexts=["Paris is the capital of France."],
        )
        assert result.value == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.asyncio
    async def test_empty_reference_contexts_is_nan_for_recall(self):
        result = await NonLLMContextRecall().ascore(
            retrieved_contexts=["anything"], reference_contexts=[]
        )
        assert np.isnan(result.value)


class TestAveragePrecisionHelper:
    """Extracted from two identical copies inside context_precision/metric.py."""

    @pytest.mark.parametrize(
        "verdicts,expected",
        [
            ([], 0.0),
            ([0, 0, 0], 0.0),
            ([1], 1.0),
            ([1, 1], 1.0),
            ([1, 0], 1.0),
            ([0, 1], 0.5),
        ],
    )
    def test_known_values(self, verdicts, expected):
        assert average_precision(verdicts) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize("verdicts", [[1], [1, 1], [1, 1, 1, 0], [1, 0, 0]])
    def test_perfect_ranking_is_exactly_one(self, verdicts):
        # no epsilon: relevant contexts all ranked first must give exactly 1.0 (#57)
        assert average_precision(verdicts) == 1.0

    def test_earlier_hits_score_higher(self):
        assert average_precision([1, 0, 0]) > average_precision([0, 0, 1])


class TestEquivalenceWithLegacy:
    """A port that moves scores is a silent regression for anyone comparing runs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "new_cls,legacy_name",
        [
            (IDBasedContextPrecision, "IDBasedContextPrecision"),
            (IDBasedContextRecall, "IDBasedContextRecall"),
        ],
    )
    async def test_id_based_matches_legacy(self, new_cls, legacy_name):
        rng = random.Random(0)
        ids = ["a", "b", "c", "d", 1, 2]
        old_metric, new_metric = legacy(legacy_name)(), new_cls()

        for _ in range(100):
            retrieved = rng.sample(ids, rng.randint(0, 4))
            reference = rng.sample(ids, rng.randint(0, 4))
            old = await old_metric._single_turn_ascore(
                SingleTurnSample(
                    retrieved_context_ids=retrieved, reference_context_ids=reference
                ),
                None,
            )
            new = await new_metric.ascore(
                retrieved_context_ids=retrieved, reference_context_ids=reference
            )
            assert same(old, new.value), (
                f"{legacy_name}: retrieved={retrieved} reference={reference} "
                f"legacy={old} port={new.value}"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "new_cls,legacy_name",
        [
            (
                NonLLMContextPrecisionWithReference,
                "NonLLMContextPrecisionWithReference",
            ),
            (NonLLMContextRecall, "NonLLMContextRecall"),
        ],
    )
    async def test_non_llm_matches_legacy(self, new_cls, legacy_name):
        rng = random.Random(1)
        old_metric, new_metric = legacy(legacy_name)(), new_cls()

        for _ in range(40):
            retrieved = rng.sample(CORPUS, rng.randint(1, 4))
            reference = rng.sample(CORPUS, rng.randint(1, 3))
            old = await old_metric._single_turn_ascore(
                SingleTurnSample(
                    retrieved_contexts=retrieved, reference_contexts=reference
                ),
                None,
            )
            new = await new_metric.ascore(
                retrieved_contexts=retrieved, reference_contexts=reference
            )
            assert same(old, new.value), (
                f"{legacy_name}: retrieved={retrieved} reference={reference} "
                f"legacy={old} port={new.value}"
            )

    @pytest.mark.asyncio
    async def test_threshold_is_inclusive_for_both(self):
        """Both use `>=` (#64); legacy recall used `>`, so an exact match at
        threshold=1.0 counted for precision but scored 0 for recall."""
        exact = ["Paris is the capital of France."]
        precision = NonLLMContextPrecisionWithReference(threshold=1.0)
        recall = NonLLMContextRecall(threshold=1.0)

        assert (
            await precision.ascore(retrieved_contexts=exact, reference_contexts=exact)
        ).value == pytest.approx(1.0)
        assert (
            await recall.ascore(retrieved_contexts=exact, reference_contexts=exact)
        ).value == pytest.approx(1.0)


class TestRunsThroughEvaluate:
    """The bridge from #25 should carry these with no per-metric work."""

    def test_id_based_metric_runs_in_evaluate(self):
        from ragas import EvaluationDataset, evaluate

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input="q",
                    retrieved_context_ids=["d1", "d2"],
                    reference_context_ids=["d1"],
                )
            ]
        )
        result = evaluate(dataset, metrics=[IDBasedContextPrecision()])
        assert result["id_based_context_precision"] == [0.5]

    def test_non_llm_metric_runs_in_evaluate(self):
        from ragas import EvaluationDataset, evaluate

        ctx = ["Paris is the capital of France."]
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input="q", retrieved_contexts=ctx, reference_contexts=ctx
                )
            ]
        )
        result = evaluate(dataset, metrics=[NonLLMContextRecall()])
        assert result["non_llm_context_recall"] == [1.0]
