"""metrics_experiment / summarize / check (#253).

The expected scores, NaN rows, reasons, summaries and ThresholdError tables
are the ones evaluate() gave on the same rows and metrics before it was
removed (#257).
"""

from __future__ import annotations

import asyncio
import logging
import math
import typing as t

import instructor
import pytest

from ragas import (
    Dataset,
    check,
    metrics_experiment,
    summarize,
)
from ragas.backends import InMemoryBackend
from ragas.cost import TokenUsage, TokenUsageCollector
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.exceptions import ThresholdError
from ragas.llms import llm_factory
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import AspectCritic, ExactMatch, NonLLMStringSimilarity
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.run_config import RunConfig

VERDICT = '{"reason": "It answers the question.", "verdict": 1}'


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def make_llm(response: str = VERDICT):
    return llm_factory("gpt-6-luna", mode=instructor.Mode.JSON, mock_response=response)


class Recording(BaseMetric):
    """Scores 0.5 with a reason, and raises on ``fail_on``."""

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


class Messages(BaseMetric):
    """Walks ``user_input`` as messages, like the agent metrics."""

    def __init__(self):
        super().__init__(name="messages")

    async def ascore(  # type: ignore[override]
        self, user_input: t.List[t.Union[HumanMessage, AIMessage]]
    ) -> MetricResult:
        return MetricResult(value=float(len(user_input)))


class Slow(BaseMetric):
    def __init__(self):
        self.cancelled = False
        super().__init__(name="slow")

    async def ascore(self, user_input: str) -> MetricResult:  # type: ignore[override]
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return MetricResult(value=1.0)


def samples(n: int = 4) -> t.List[SingleTurnSample]:
    return [
        SingleTurnSample(
            user_input=f"q{i}",
            response="ref" if i % 2 else f"r{i}",
            reference="ref",
        )
        for i in range(n)
    ]


def dataset(rows: t.Optional[t.List[t.Any]] = None, **kwargs: t.Any) -> Dataset:
    return Dataset(
        "rows",
        InMemoryBackend(),
        data=list(samples() if rows is None else rows),
        **kwargs,
    )


def metrics() -> t.List[t.Any]:
    return [
        ExactMatch(),
        NonLLMStringSimilarity(),
        Recording(fail_on="q2"),
        AspectCritic(name="answers", definition="Answers?", llm=make_llm()),
    ]


def column(exp, name):
    return [row[name] for row in exp]


def same(a: t.List[float], b: t.List[float]) -> bool:
    return all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b))


# --- scores, reasons, summary and check ----------------------------------------------

NAN = float("nan")
REASON = "It answers the question."


def test_scores_reasons_and_summary():
    exp = metrics_experiment(metrics()).run(dataset(), name="scores")

    assert column(exp, "exact_match") == [0.0, 1.0, 0.0, 1.0]
    assert same(
        column(exp, "non_llm_string_similarity"),
        [0.33333333333333337, 1.0, 0.33333333333333337, 1.0],
    )
    assert same(column(exp, "recording"), [0.5, 0.5, NAN, 0.5])  # q2 raised
    assert column(exp, "answers") == [1.0] * 4
    assert column(exp, "exact_match_reason") == [None] * 4
    assert column(exp, "recording_reason") == ["saw r0", "saw ref", None, "saw ref"]
    assert column(exp, "answers_reason") == [REASON] * 4
    assert summarize(exp) == {
        "exact_match": {"mean": 0.5, "count": 4, "missing": 0},
        "non_llm_string_similarity": {
            "mean": 0.6666666666666667,
            "count": 4,
            "missing": 0,
        },
        "recording": {"mean": 0.5, "count": 3, "missing": 1},
        "answers": {"mean": 1.0, "count": 4, "missing": 0},
    }


CHECK_TABLES = [
    (
        dict(min_mean=0.9),
        "3 of 4 metrics failed their thresholds:\n"
        "metric                     mean    required  count  missing  result\n"
        "exact_match                0.5000  >= 0.9    4      0        mean below 0.9\n"
        "non_llm_string_similarity  0.6667  >= 0.9    4      0        mean below 0.9\n"
        "recording                  0.5000  >= 0.9    3      1        "
        "mean below 0.9; 1 row missing (allowed 0)\n"
        "answers                    1.0000  >= 0.9    4      0        ok",
        [
            ("exact_match", "mean below 0.9"),
            ("non_llm_string_similarity", "mean below 0.9"),
            ("recording", "mean below 0.9"),
            ("recording", "1 row missing (allowed 0)"),
        ],
    ),
    (
        # passes the mean, fails on missing
        dict(min_mean={"recording": 0.4}),
        "1 of 4 metrics failed their thresholds:\n"
        "metric                     mean    required  count  missing  result\n"
        "exact_match                0.5000            4      0        ok\n"
        "non_llm_string_similarity  0.6667            4      0        ok\n"
        "recording                  0.5000  >= 0.4    3      1        "
        "1 row missing (allowed 0)\n"
        "answers                    1.0000            4      0        ok",
        [("recording", "1 row missing (allowed 0)")],
    ),
    (
        dict(max_mean={"exact_match": 0.2}, allow_missing=None),
        "1 of 1 metric failed their thresholds:\n"
        "metric       mean    required  count  missing  result\n"
        "exact_match  0.5000  <= 0.2    4      0        mean above 0.2",
        [("exact_match", "mean above 0.2")],
    ),
]


@pytest.mark.parametrize("bounds, table, failures", CHECK_TABLES)
def test_check_raises_the_threshold_table(bounds, table, failures):
    exp = metrics_experiment(metrics()).run(dataset())
    with pytest.raises(ThresholdError) as got:
        check(exp, **bounds)
    assert str(got.value) == table
    assert got.value.failures == failures


def test_check_passes_and_rejects_unknown_metrics():
    exp = metrics_experiment([ExactMatch()]).run(dataset())
    check(exp, min_mean=0.5)
    with pytest.raises(ValueError, match="not in this result: typo"):
        check(exp, min_mean={"typo": 0.5})


# --- rows ---------------------------------------------------------------------------


def test_dict_rows_and_extra_columns_are_kept():
    rows = [dict(s.to_dict(), id=i) for i, s in enumerate(samples())]
    exp = metrics_experiment([ExactMatch()]).run(dataset(rows))
    assert column(exp, "id") == [0, 1, 2, 3]
    assert column(exp, "exact_match") == [0.0, 1.0, 0.0, 1.0]
    assert column(exp, "response")[1] == "ref"


def test_typed_rows_score_like_dicts():
    typed = metrics_experiment([ExactMatch()]).run(dataset(data_model=SingleTurnSample))
    assert column(typed, "exact_match") == [0.0, 1.0, 0.0, 1.0]


def test_column_map_names_the_dataset_columns():
    rows = [{"question": "q", "answer": "a", "truth": "a"}]
    exp = metrics_experiment(
        [ExactMatch()], column_map={"response": "answer", "reference": "truth"}
    ).run(dataset(rows))
    assert column(exp, "exact_match") == [1.0]


def test_multi_turn_rows():
    conversation = [HumanMessage(content="hi"), AIMessage(content="hello")]
    exp = metrics_experiment([Messages()]).run(
        dataset([MultiTurnSample(user_input=conversation)])
    )
    assert column(exp, "messages") == [2.0]


def test_a_per_call_llm_is_passed_to_the_metric():
    metric = DiscreteMetric(
        name="tone", prompt="Rate {response}.", allowed_values=["good", "bad"]
    )
    llm = make_llm('{"reason": "polite", "value": "good"}')
    with pytest.raises(ValueError, match=r"takes `llm` .* llm=\.\.\."):
        metrics_experiment([metric])
    exp = metrics_experiment([metric], llm=llm).run(dataset(samples(1)))
    # a discrete label isn't a number: it scores NaN
    assert math.isnan(column(exp, "tone")[0])


def test_sync_run_and_async_arun_agree():
    async def go():
        return await metrics_experiment([ExactMatch()]).arun(dataset())

    asynchronous = asyncio.run(go())
    synchronous = metrics_experiment([ExactMatch()]).run(dataset())
    assert list(asynchronous) == list(synchronous)


# --- up-front checks -----------------------------------------------------------------


def test_a_required_column_no_row_has_fails_before_any_row_runs():
    metric = Recording()
    rows = [{"user_input": "q"}]
    with pytest.raises(ValueError, match=r"requires .*\['response'\]"):
        metrics_experiment([metric]).run(dataset(rows))
    assert metric.calls == []


def test_a_metric_that_cannot_score_the_sample_type_fails_up_front():
    with pytest.raises(ValueError, match="does not support the sample type"):
        metrics_experiment([ExactMatch()]).run(
            dataset([MultiTurnSample(user_input=[HumanMessage(content="hi")])])
        )


@pytest.mark.parametrize(
    "rows, match",
    [
        (
            [
                SingleTurnSample(user_input="q"),
                MultiTurnSample(user_input=[HumanMessage(content="hi")]),
            ],
            "mixes SingleTurnSample and MultiTurnSample",
        ),
        ([{"user_input": 3}], "Row 0 is not a valid sample"),
        (
            [{"response": "r", "reference": "r", "exact_match": 1}],
            "already has a column",
        ),
    ],
)
def test_bad_datasets_fail_up_front(rows, match):
    with pytest.raises(ValueError, match=match):
        metrics_experiment([ExactMatch()]).run(dataset(rows))


@pytest.mark.parametrize(
    "build, error, match",
    [
        (lambda: metrics_experiment("exact_match"), TypeError, "list of metrics"),
        (lambda: metrics_experiment([]), ValueError, "at least one"),
        (lambda: metrics_experiment([object()]), TypeError, "metric objects"),
        (
            lambda: metrics_experiment([ExactMatch(), ExactMatch()]),
            ValueError,
            "unique",
        ),
        (
            lambda: metrics_experiment([ExactMatch()], llm=object()),
            TypeError,
            "llm_factory",
        ),
    ],
)
def test_bad_arguments_fail_when_built(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_only_a_ragas_dataset_runs():
    with pytest.raises(TypeError, match="ragas Dataset"):
        metrics_experiment([ExactMatch()]).run(samples())  # type: ignore[arg-type]


def test_rows_take_no_extra_arguments():
    with pytest.raises(TypeError, match="no extra arguments"):
        metrics_experiment([ExactMatch()]).run(dataset(), "name", None, 1)


# --- errors ---------------------------------------------------------------------------


def test_a_failed_metric_is_logged_and_its_row_kept(caplog):
    caplog.set_level(logging.WARNING, logger="ragas.scoring")
    exp = metrics_experiment([Recording(fail_on="q2"), ExactMatch()]).run(dataset())
    assert len(exp) == 4
    assert column(exp, "exact_match") == [0.0, 1.0, 0.0, 1.0]
    messages = [r.getMessage() for r in caplog.records]
    assert any("boom on q2" in m for m in messages)
    assert any("recording: 1 of 4 rows have no score" in m for m in messages)


@pytest.mark.parametrize("where", ["decorator", "arun"])
def test_raise_exceptions_raises_the_metrics_own_error(where):
    slow = Slow()
    rows = [{"user_input": "q2", "response": "r"}]
    if where == "decorator":
        run = metrics_experiment([Recording(fail_on="q2"), slow], raise_exceptions=True)
        call = lambda: run.run(dataset(rows))  # noqa: E731
    else:
        run = metrics_experiment([Recording(fail_on="q2"), slow])
        call = lambda: run.run(dataset(rows), raise_exceptions=True)  # noqa: E731
    with pytest.raises(RuntimeError, match="boom on q2"):
        call()
    assert slow.cancelled  # the row's other metric was cancelled


def test_timeout_bounds_each_metric_call():
    exp = metrics_experiment([Slow()], run_config=RunConfig(timeout=0.05)).run(
        dataset(samples(1))
    )
    assert math.isnan(column(exp, "slow")[0])


# --- token usage ------------------------------------------------------------------------


def test_usage_collects_the_runs_tokens():
    usage = TokenUsageCollector(
        token_usage_parser=lambda raw: TokenUsage(input_tokens=1, output_tokens=0)
    )
    critic = AspectCritic(name="answers", definition="Answers?", llm=make_llm())
    metrics_experiment([critic], usage=usage).run(dataset())
    assert usage.total_tokens().input_tokens == 4


# --- summarize / check on saved experiments ---------------------------------------------


@pytest.mark.parametrize("backend", ["local/csv", "local/jsonl"])
def test_summarize_a_reloaded_experiment(tmp_path, backend):
    run = metrics_experiment(
        [ExactMatch(), Recording(fail_on="q2")],
        backend=backend,
        backend_kwargs={"root_dir": str(tmp_path)},
    )
    exp = run.run(dataset(), name="saved")
    exp.reload()
    assert summarize(exp) == {
        "exact_match": {"mean": 0.5, "count": 4, "missing": 0},
        "recording": {"mean": 0.5, "count": 3, "missing": 1},
    }


def test_summarize_names_metrics_explicitly():
    rows = [{"score": 1.0}, {"score": None}]
    with pytest.raises(ValueError, match="No metric columns"):
        summarize(rows)
    assert summarize(rows, metrics=["score"]) == {
        "score": {"mean": 1.0, "count": 1, "missing": 1}
    }
    assert (
        summarize([dict(exact_match=1.0)], metrics=[ExactMatch()])["exact_match"][
            "count"
        ]
        == 1
    )


def test_exported_from_ragas_experiment():
    import ragas.experiment as module
    from ragas.experiment import check as c, metrics_experiment as m, summarize as s

    assert (m, s, c) == (metrics_experiment, summarize, check)
    assert not hasattr(module, "no_such_name")


def test_a_blank_string_counts_as_a_value_through_csv(tmp_path):
    """Documented: only None is missing, and CSV loads a missing str as "".

    A sample with no reference raises up front from inmemory/jsonl, but after
    a CSV round trip its reference is "" and it is scored (storage_backends.md,
    "What each backend keeps").
    """
    from ragas.dataset import Dataset

    row = SingleTurnSample(user_input="q", response="a")
    run = metrics_experiment([ExactMatch()])
    results = {}
    for backend in ("local/jsonl", "local/csv"):
        saved = Dataset(
            "d",
            backend,
            data_model=SingleTurnSample,
            data=[row],
            root_dir=str(tmp_path / backend.replace("/", "_")),
        )
        saved.save()
        loaded = Dataset.load(
            "d",
            backend,
            data_model=SingleTurnSample,
            root_dir=str(tmp_path / backend.replace("/", "_")),
        )
        results[backend] = loaded[0].reference
        if backend == "local/jsonl":
            with pytest.raises(ValueError, match=r"\['reference'\]"):
                run.run(loaded)
        else:
            exp = run.run(loaded)
            assert summarize(exp)["exact_match"] == {
                "mean": 0.0,
                "count": 1,
                "missing": 0,
            }
    assert results == {"local/jsonl": None, "local/csv": ""}
