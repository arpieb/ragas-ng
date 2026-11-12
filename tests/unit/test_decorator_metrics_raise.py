"""Decorator metrics raise, the same way ``SimpleLLMMetric`` does (#239).

``@discrete_metric`` / ``@numeric_metric`` / ``@ranking_metric`` used to catch
every exception and return ``MetricResult(value=None, reason=...)``, so
``raise_exceptions`` could never surface their errors. An invalid value now
raises ``ValueError`` with ``SimpleLLMMetric``'s wording, and errors from the
function propagate.
"""

from __future__ import annotations

import math
import typing as t

import pytest

from ragas import SingleTurnSample, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.experiment import experiment
from ragas.metrics import (
    DiscreteMetric,
    NumericMetric,
    RankingMetric,
    discrete_metric,
    numeric_metric,
    ranking_metric,
)


class Boom(Exception):
    pass


# (SimpleLLMMetric class, decorator, allowed_values, an invalid value)
CASES = [
    pytest.param(
        DiscreteMetric, discrete_metric, ["pass", "fail"], "maybe", id="discrete"
    ),
    pytest.param(NumericMetric, numeric_metric, (0.0, 1.0), 1.5, id="numeric"),
    pytest.param(NumericMetric, numeric_metric, range(0, 5), 7.0, id="numeric-range"),
    pytest.param(RankingMetric, ranking_metric, 3, ["a"], id="ranking"),
]


@pytest.mark.parametrize("cls, decorator, allowed, invalid", CASES)
def test_invalid_value_gives_the_same_error_in_both_styles(
    cls, decorator, allowed, invalid
):
    llm_metric = cls(name="m", prompt="Rate {response}.", allowed_values=allowed)

    @decorator(name="m", allowed_values=allowed)
    def decorated(response: str) -> t.Any:
        return invalid

    # SimpleLLMMetric.ascore raises ValueError(validate_result_value(...)).
    # (A DiscreteMetric's response model is a Literal, so its LLM can't
    # return an invalid value; compare the validator's message directly.)
    expected = llm_metric.validate_result_value(invalid)
    assert expected
    with pytest.raises(ValueError) as e:
        decorated.score(response="r")
    assert str(e.value) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_errors_from_the_function_propagate(is_async):
    if is_async:

        @numeric_metric(name="m", allowed_values=(0.0, 1.0))
        async def metric(response: str) -> float:
            raise Boom("broken")

    else:

        @numeric_metric(name="m", allowed_values=(0.0, 1.0))
        def metric(response: str) -> float:
            raise Boom("broken")

    with pytest.raises(Boom, match="broken"):
        await metric.ascore(response="r")
    with pytest.raises(Boom, match="broken"):
        metric.score(response="r")


@numeric_metric(name="fails", allowed_values=(0.0, 1.0))
def fails(response: str) -> float:
    if response == "bad":
        raise Boom("bad row")
    return 2.0 if response == "high" else 0.5


def _score(*responses: str, **kwargs: t.Any):
    dataset = Dataset(
        name="rows",
        backend=InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[SingleTurnSample(response=r) for r in responses],
    )
    return metrics_experiment([fails], **kwargs).run(dataset)


def test_metrics_experiment_scores_failed_rows_nan():
    scores = [row["fails"] for row in _score("ok", "bad", "high")]
    assert scores[0] == 0.5
    assert math.isnan(scores[1]) and math.isnan(scores[2])


def test_metrics_experiment_raise_exceptions_surfaces_the_error():
    with pytest.raises(Boom, match="bad row"):
        _score("ok", "bad", raise_exceptions=True)
    with pytest.raises(ValueError, match="expected value in range"):
        _score("ok", "high", raise_exceptions=True)


@pytest.mark.asyncio
async def test_experiment_raise_exceptions_surfaces_an_invalid_value():
    @experiment()
    async def run(row):
        result = await fails.ascore(response=row["response"])
        return {"score": result.value}

    data = Dataset(name="rows", backend=InMemoryBackend(), data=[{"response": "high"}])
    with pytest.raises(ValueError, match="expected value in range"):
        await run.arun(data, name="exp", raise_exceptions=True)
