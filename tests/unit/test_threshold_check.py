"""ragas.check(): the one-call CI threshold check (#169).

The CI guide asserted on the summary by hand. ``check()`` does the
common case -- lower/upper bounds on each metric's mean, and no unscored rows
-- and raises a ``ThresholdError`` (an ``AssertionError``) with a per-metric
table.
"""

import math

import pytest

from ragas import check
from ragas.backends import InMemoryBackend
from ragas.experiment import Experiment

NAN = math.nan


class _Result:
    """An experiment with the given metric columns, and ``check`` on it."""

    def __init__(self, exp: Experiment):
        self.exp = exp

    def check(self, **kwargs):
        return check(self.exp, **kwargs)


def _result(**columns):
    n = len(next(iter(columns.values())))
    rows = []
    for i in range(n):
        row = {"user_input": f"q{i}"}
        for name, values in columns.items():
            row[name] = values[i]
            row[f"{name}_reason"] = None
        rows.append(row)
    return _Result(Experiment("exp", InMemoryBackend(), data=rows))


def _threshold_error():
    from ragas.exceptions import ThresholdError

    return ThresholdError


def test_passing_returns_none():
    result = _result(faithfulness=[0.8, 0.9], answer_relevancy=[0.95, 0.9])
    assert result.check(min_mean={"faithfulness": 0.8, "answer_relevancy": 0.9}) is None


def test_a_mean_below_its_bound_fails_with_a_table():
    result = _result(faithfulness=[0.2, 0.4], answer_relevancy=[0.95, 0.9])
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean={"faithfulness": 0.5, "answer_relevancy": 0.9})
    message = str(e.value)
    assert message.startswith("1 of 2 metrics failed their thresholds:")
    lines = message.splitlines()
    assert lines[1].split() == [
        "metric",
        "mean",
        "required",
        "count",
        "missing",
        "result",
    ]
    faith = next(line for line in lines if line.startswith("faithfulness"))
    assert "0.3000" in faith and ">= 0.5" in faith and "mean below 0.5" in faith
    relevancy = next(line for line in lines if line.startswith("answer_relevancy"))
    assert relevancy.endswith("ok")
    assert e.value.failures == [("faithfulness", "mean below 0.5")]


def test_it_is_an_assertion_error():
    result = _result(faithfulness=[0.2])
    with pytest.raises(AssertionError):
        result.check(min_mean={"faithfulness": 0.5})


def test_max_mean_and_a_range():
    result = _result(faithfulness=[0.6, 0.6])
    result.check(min_mean={"faithfulness": 0.3}, max_mean={"faithfulness": 0.7})
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean={"faithfulness": 0.3}, max_mean={"faithfulness": 0.5})
    assert e.value.failures == [("faithfulness", "mean above 0.5")]
    assert "0.3 .. 0.5" in str(e.value)


def test_one_number_applies_to_every_metric():
    result = _result(a=[0.9], b=[0.4])
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean=0.5)
    assert e.value.failures == [("b", "mean below 0.5")]


def test_missing_rows_fail_by_default():
    # the mean skips unscored rows, so it alone would look fine
    result = _result(faithfulness=[0.9, NAN, 0.9])
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean={"faithfulness": 0.5})
    assert e.value.failures == [("faithfulness", "1 row missing (allowed 0)")]
    with pytest.raises(_threshold_error()):
        result.check()  # no bounds: still checks coverage


def test_allow_missing():
    result = _result(faithfulness=[0.9, NAN, NAN])
    result.check(min_mean={"faithfulness": 0.5}, allow_missing=2)
    result.check(min_mean={"faithfulness": 0.5}, allow_missing=None)
    with pytest.raises(_threshold_error()) as e:
        result.check(allow_missing=1)
    assert e.value.failures == [("faithfulness", "2 rows missing (allowed 1)")]


def test_a_metric_with_no_scores_fails_any_bound():
    result = _result(faithfulness=[NAN, NAN])
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean={"faithfulness": 0.0}, max_mean=1.0, allow_missing=None)
    assert e.value.failures == [("faithfulness", "no scores")]
    assert "nan" in str(e.value)


def test_an_unknown_metric_name_is_an_error_not_a_pass():
    result = _result(faithfulness=[0.9])
    with pytest.raises(ValueError, match="faithfulnes.*Metrics: faithfulness"):
        result.check(min_mean={"faithfulnes": 0.5})
    with pytest.raises(ValueError, match="max_mean"):
        result.check(max_mean={"nope": 0.5})


@pytest.mark.parametrize("bound", [True, "0.5", [0.5]])
def test_a_bound_that_is_not_a_number_or_mapping_is_rejected(bound):
    with pytest.raises(TypeError):
        _result(a=[1.0]).check(min_mean=bound)


def test_negative_allow_missing_is_rejected():
    with pytest.raises(ValueError):
        _result(a=[1.0]).check(allow_missing=-1)


def test_only_checked_metrics_are_in_the_table():
    result = _result(a=[0.1], b=[0.9])
    with pytest.raises(_threshold_error()) as e:
        result.check(min_mean={"a": 0.5}, allow_missing=None)
    assert "1 of 1 metric failed" in str(e.value)
    assert not any(line.startswith("b ") for line in str(e.value).splitlines())


def test_reasons_do_not_affect_the_check():
    result = _result(faithfulness=[0.9, 0.7])
    for row, reason in zip(result.exp, ["ok", "fine"]):
        row["faithfulness_reason"] = reason
    result.check(min_mean={"faithfulness": 0.8})
