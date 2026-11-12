"""Scoring-run input checks, repeated prompts in traces, empty usage (#77, #78, #79)."""

from __future__ import annotations

import logging
import math

import pytest

from ragas import Dataset, SingleTurnSample, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.callbacks import ChainRun, parse_run_traces
from ragas.cost import TokenUsage, TokenUsageCollector
from ragas.metrics.collections import BleuScore

SAMPLE = dict(
    response="the cat sat on the mat today", reference="the cat sat on the mat today"
)


def dataset():
    return Dataset(
        "rows",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[SingleTurnSample(**SAMPLE)],
    )


# --- #77: llm / embeddings / metrics ----------------------------------------------


class RawClient:
    """Stands in for e.g. openai.OpenAI passed where a ragas LLM belongs."""

    def __init__(self):
        self.called = False

    def __getattr__(self, name):
        self.called = True
        raise AttributeError(name)


def test_a_raw_client_as_llm_is_rejected_before_the_run():
    # used to be accepted, then fail in every job with NaN scores
    client = RawClient()
    with pytest.raises(TypeError, match="llm_factory"):
        metrics_experiment([BleuScore()], llm=client)
    assert not client.called


def test_a_raw_client_as_embeddings_is_rejected():
    with pytest.raises(TypeError, match="embedding_factory"):
        metrics_experiment([BleuScore()], embeddings=RawClient())


def test_metrics_may_be_a_tuple():
    exp = metrics_experiment((BleuScore(),)).run(dataset())
    assert [row["bleu_score"] for row in exp] == pytest.approx([1.0])


def test_metrics_as_a_string_is_rejected():
    with pytest.raises(TypeError, match="list"):
        metrics_experiment("bleu_score")  # type: ignore[arg-type]


# --- #78: repeated prompts in traces -----------------------------------------------


def chain(run_id, parent, name, children=(), inputs=None, outputs=None):
    return ChainRun(
        run_id=run_id,
        parent_run_id=parent,
        name=name,
        inputs=inputs or {},
        metadata={},
        outputs=outputs or {},
        children=list(children),
    )


def test_repeated_prompt_names_are_all_kept():
    traces = {
        "root": chain("root", None, "evaluation", ["row"]),
        "row": chain("row", "root", "row 0", ["metric"]),
        "metric": chain(
            "metric",
            "row",
            "factual_correctness",
            ["p1", "p2", "p3", "p4"],
            outputs={"output": 0.5},
        ),
        "p1": chain("p1", "metric", "claim_decomposition", inputs={"data": "resp"}),
        "p2": chain("p2", "metric", "claim_decomposition", inputs={"data": "ref"}),
        "p3": chain("p3", "metric", "nli", inputs={"data": "a"}),
        "p4": chain("p4", "metric", "nli", inputs={"data": "b"}),
    }
    [row] = parse_run_traces(traces)
    prompts = row["factual_correctness"]
    assert list(prompts) == [
        "claim_decomposition",
        "claim_decomposition (2)",
        "nli",
        "nli (2)",
    ]
    assert prompts["claim_decomposition"]["input"] == "resp"
    assert prompts["claim_decomposition (2)"]["input"] == "ref"


# --- #79: no usage recorded -----------------------------------------------------


def test_total_tokens_with_no_usage_is_zero(caplog):
    with caplog.at_level(logging.WARNING, logger="ragas.cost"):
        total = TokenUsageCollector().total_tokens()
    assert total == TokenUsage(input_tokens=0, output_tokens=0)
    assert "No token usage was recorded" in caplog.text


def test_total_cost_with_no_usage_is_zero():
    assert TokenUsageCollector().total_cost(cost_per_input_token=1e-6) == 0.0


def test_total_cost_still_requires_prices():
    with pytest.raises(ValueError, match="cost"):
        TokenUsageCollector().total_cost()


def test_a_run_whose_metrics_record_no_usage():
    usage = TokenUsageCollector(lambda raw: TokenUsage(input_tokens=1, output_tokens=1))
    exp = metrics_experiment([BleuScore()], usage=usage).run(dataset())
    assert usage.total_tokens() == TokenUsage(input_tokens=0, output_tokens=0)
    assert usage.total_cost(cost_per_input_token=1.0) == 0.0
    assert not math.isnan(exp[0]["bleu_score"])
