"""summarize() reports how many rows each metric's mean covers (#53).

The mean skips rows that errored or scored NaN. Without a count, a run where
half the rows failed reports the same headline as a clean one, and can even
look better as the system degrades.
"""

from __future__ import annotations

import math

import pytest

from ragas import summarize
from ragas._summary import has_score
from ragas.backends import InMemoryBackend
from ragas.experiment import Experiment

NAN = float("nan")


def make_exp(scores):
    rows = [
        {"user_input": f"q{i}", **row, **{f"{k}_reason": None for k in row}}
        for i, row in enumerate(scores)
    ]
    return Experiment("exp", InMemoryBackend(), data=rows)


def test_summary_reports_mean_count_and_missing():
    exp = make_exp([{"m": 1.0}, {"m": NAN}, {"m": None}, {"m": 0.0}])
    assert summarize(exp) == {"m": {"mean": 0.5, "count": 2, "missing": 2}}


def test_all_rows_missing():
    s = summarize(make_exp([{"m": NAN}, {"m": NAN}]))["m"]
    assert math.isnan(s["mean"]) and s["count"] == 0 and s["missing"] == 2


def test_degraded_run_is_distinguishable_from_a_clean_one():
    # Same headline mean; only the coverage tells them apart.
    clean = summarize(make_exp([{"m": 0.8}] * 4))["m"]
    degraded = summarize(make_exp([{"m": 0.8}, {"m": NAN}, {"m": NAN}, {"m": NAN}]))[
        "m"
    ]
    assert clean["mean"] == degraded["mean"]
    assert (clean["count"], clean["missing"]) == (4, 0)
    assert (degraded["count"], degraded["missing"]) == (1, 3)


@pytest.mark.parametrize(
    "value, expected", [(0.0, True), (NAN, False), (None, False), ("yes", True)]
)
def test_non_numeric_outputs_count_as_present(value, expected):
    assert has_score(value) is expected
