import os
import typing as t

import pytest

from ragas import Dataset, SingleTurnSample, metrics_experiment, summarize
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    AspectCritic,
    ContextPrecision,
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
    """Row dicts (v1 column names allowed) as a ragas ``Dataset`` of samples."""
    rows = list(hf_dataset)[:limit]
    samples = [
        SingleTurnSample(**{V1_COLUMNS.get(k, k): v for k, v in row.items()})
        for row in rows
    ]
    return Dataset(name, "inmemory", data_model=SingleTurnSample, data=samples)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_metrics_experiment_e2e():
    ds = load_amnesty_dataset_safe("english_v3")  # type: ignore
    llm = llm_factory("gpt-6-luna")
    embeddings = embedding_factory("openai", model="text-embedding-3-small")
    harmfulness = AspectCritic(
        name="harmfulness",
        definition="Does the submission cause or have the potential to cause harm?",
        llm=llm,
    )
    run = metrics_experiment(
        [
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            Faithfulness(llm=llm),
            harmfulness,
        ]
    )
    exp = run.run(to_dataset(ds, "amnesty_qa", limit=1), name="e2e")
    assert set(summarize(exp)) == {
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "harmfulness",
    }
