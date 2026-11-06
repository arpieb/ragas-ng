"""evaluate() input checks and result plumbing (#77, #78, #79)."""

from __future__ import annotations

import logging
import math
import warnings

import pytest

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.callbacks import ChainRun, parse_run_traces
from ragas.cost import TokenUsage, TokenUsageCollector
from ragas.metrics.collections import BleuScore


def run(dataset, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return evaluate(
            dataset,
            metrics=kwargs.pop("metrics", [BleuScore()]),
            show_progress=False,
            **kwargs,
        )


SAMPLE = dict(
    response="the cat sat on the mat today", reference="the cat sat on the mat today"
)


# --- #77: dataset -------------------------------------------------------------


def test_a_list_of_samples_is_converted():
    result = run([SingleTurnSample(**SAMPLE), SingleTurnSample(**SAMPLE)])
    assert result["bleu_score"] == pytest.approx([1.0, 1.0])


def test_a_list_of_dicts_is_converted():
    # used to fail with "'list' object has no attribute 'get_sample_type'"
    result = run([dict(SAMPLE)])
    assert result["bleu_score"] == pytest.approx([1.0])


def test_an_empty_list_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        run([])


def test_a_mixed_list_is_rejected():
    with pytest.raises(TypeError, match="only samples"):
        run([SingleTurnSample(**SAMPLE), dict(SAMPLE)])


@pytest.mark.parametrize("bad", ["user_input,response", 42, {"response": "x"}])
def test_other_dataset_types_are_rejected_with_the_fix(bad):
    with pytest.raises(TypeError, match="EvaluationDataset"):
        run(bad)


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
        run([SingleTurnSample(**SAMPLE)], llm=client)
    assert not client.called


def test_a_raw_client_as_embeddings_is_rejected():
    with pytest.raises(TypeError, match="embedding_factory"):
        run([SingleTurnSample(**SAMPLE)], embeddings=RawClient())


def test_metrics_may_be_a_tuple():
    result = run(
        EvaluationDataset(samples=[SingleTurnSample(**SAMPLE)]),
        metrics=(BleuScore(),),
    )
    assert result["bleu_score"] == pytest.approx([1.0])


def test_metrics_as_a_string_is_rejected():
    with pytest.raises(TypeError, match="list"):
        run([SingleTurnSample(**SAMPLE)], metrics="bleu_score")


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


def test_evaluate_result_with_an_llm_that_records_nothing():
    result = run(
        [SingleTurnSample(**SAMPLE)],
        token_usage_parser=lambda raw: TokenUsage(input_tokens=1, output_tokens=1),
    )
    assert result.total_tokens() == TokenUsage(input_tokens=0, output_tokens=0)
    assert result.total_cost(cost_per_input_token=1.0) == 0.0
    assert not math.isnan(result["bleu_score"][0])
