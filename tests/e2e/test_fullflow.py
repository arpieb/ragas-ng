import os
import typing as t

import pytest

from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    AspectCritic,
    ContextPrecision,
    Faithfulness,
)
from tests.e2e.test_dataset_utils import load_amnesty_dataset_safe

if t.TYPE_CHECKING:
    from datasets import Dataset


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_evaluate_e2e():
    ds = load_amnesty_dataset_safe("english_v3")  # type: ignore
    llm = llm_factory("gpt-6-luna")
    embeddings = embedding_factory("openai", model="text-embedding-3-small")
    harmfulness = AspectCritic(
        name="harmfulness",
        definition="Does the submission cause or have the potential to cause harm?",
        llm=llm,
    )
    result = evaluate(
        EvaluationDataset.from_hf_dataset(t.cast("Dataset", ds))[:1],
        metrics=[
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ContextPrecision(llm=llm),
            Faithfulness(llm=llm),
            harmfulness,
        ],
        show_progress=False,
    )
    assert result is not None
