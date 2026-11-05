"""EvaluationResult reports how many rows each metric's mean covers (#53).

The mean skips rows that errored or scored NaN. Without a count, a run where
half the rows failed reports the same headline as a clean one, and can even
look better as the system degrades.
"""

from __future__ import annotations

import logging
import math

import pytest

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

NAN = float("nan")


def make_result(scores):
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input=f"q{i}", response="r")
            for i in range(len(scores))
        ]
    )
    return EvaluationResult(scores=scores, dataset=dataset)


def test_repr_is_unchanged_when_every_row_scored():
    result = make_result([{"m": 1.0}, {"m": 0.5}])
    assert repr(result) == "{'m': 0.7500}"


def test_repr_shows_coverage_when_rows_are_missing():
    result = make_result(
        [{"m": 1.0, "k": 1.0}, {"m": NAN, "k": 0.0}, {"m": 0.5, "k": 1.0}]
    )
    assert repr(result) == "{'m': 0.7500 (n=2/3), 'k': 0.6667}"


def test_summary_reports_mean_count_and_missing():
    result = make_result([{"m": 1.0}, {"m": NAN}, {"m": None}, {"m": 0.0}])
    assert result.summary() == {"m": {"mean": 0.5, "count": 2, "missing": 2}}


def test_all_rows_missing():
    result = make_result([{"m": NAN}, {"m": NAN}])
    s = result.summary()["m"]
    assert math.isnan(s["mean"]) and s["count"] == 0 and s["missing"] == 2
    assert repr(result) == "{'m': nan (n=0/2)}"


def test_degraded_run_is_distinguishable_from_a_clean_one():
    # Same headline mean; only the coverage tells them apart.
    clean = make_result([{"m": 0.8}, {"m": 0.8}, {"m": 0.8}, {"m": 0.8}])
    degraded = make_result([{"m": 0.8}, {"m": NAN}, {"m": NAN}, {"m": NAN}])
    assert clean.summary()["m"]["mean"] == degraded.summary()["m"]["mean"]
    assert repr(clean) != repr(degraded)


class _FailsOnSecondRow(BaseMetric):
    def __init__(self):
        super().__init__(name="flaky")

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        if user_input == "q1":
            raise RuntimeError("judge timed out")
        return MetricResult(value=1.0)


def test_evaluate_warns_when_rows_have_no_score(caplog):
    ds = EvaluationDataset(
        samples=[SingleTurnSample(user_input=f"q{i}", response="r") for i in range(3)]
    )
    with caplog.at_level(logging.WARNING, logger="ragas.evaluation"):
        result = evaluate(
            ds,
            metrics=[_FailsOnSecondRow()],
            raise_exceptions=False,
            show_progress=False,
        )
    assert result.summary()["flaky"] == {"mean": 1.0, "count": 2, "missing": 1}
    assert "flaky: 1 of 3 rows have no score" in caplog.text


@pytest.mark.parametrize(
    "value, expected", [(0.0, True), (NAN, False), (None, False), ("yes", True)]
)
def test_non_numeric_outputs_count_as_present(value, expected):
    from ragas.dataset_schema import _has_score

    assert _has_score(value) is expected
