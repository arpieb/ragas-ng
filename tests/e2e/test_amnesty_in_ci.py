import os
import typing as t

import pytest

from ragas import Dataset, SingleTurnSample, metrics_experiment, summarize
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from tests.e2e.test_dataset_utils import load_amnesty_dataset_safe

# ragas v1 column names, renamed to the sample fields
V1_COLUMNS = {
    "question": "user_input",
    "answer": "response",
    "contexts": "retrieved_contexts",
    "ground_truth": "reference",
}


def to_dataset(hf_dataset, name: str, limit: t.Optional[int] = None) -> Dataset:
    """A Hugging Face dataset's rows as a ragas ``Dataset`` of samples."""
    rows = hf_dataset.to_list()[:limit]
    samples = [
        SingleTurnSample(**{V1_COLUMNS.get(k, k): v for k, v in row.items()})
        for row in rows
    ]
    return Dataset(name, "inmemory", data_model=SingleTurnSample, data=samples)


# loading the dataset
amnesty_qa = load_amnesty_dataset_safe("english_v3")  # type: ignore


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """
    Check if computed score is within the range of value +/- max_range
    """
    assert value - plus_or_minus <= score <= value + plus_or_minus


@pytest.mark.ragas_ci
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_amnesty_e2e():
    llm = llm_factory("gpt-6-luna")
    embeddings = embedding_factory("openai", model="text-embedding-3-small")
    run = metrics_experiment(
        [
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextRecall(llm=llm),
            ContextPrecision(llm=llm),
        ]
    )
    exp = run.run(to_dataset(amnesty_qa, "amnesty_qa", limit=1), name="amnesty_ci")
    assert len(summarize(exp)) == 4


@pytest.mark.ragas_ci
def test_assert_in_range():
    assert_in_range(0.51, value=0.5, plus_or_minus=0.1)
