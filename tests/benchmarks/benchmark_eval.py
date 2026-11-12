import time
import typing as t

from ragas import Dataset, SingleTurnSample, metrics_experiment
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    Faithfulness,
    SemanticSimilarity,
)

from ..e2e.test_dataset_utils import load_amnesty_dataset_safe

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


# from ragas.metrics.critique import harmfulness  # Import unavailable

# data - using safe dataset loading
eval_dataset = to_dataset(load_amnesty_dataset_safe("english_v2"), "amnesty_qa")

# metrics
llm = llm_factory("gpt-6-luna")
embeddings = embedding_factory("openai", model="text-embedding-3-small")
metrics = [
    Faithfulness(llm=llm),
    ContextRecall(llm=llm),
    AnswerRelevancy(llm=llm, embeddings=embeddings),
    AnswerCorrectness(llm=llm, embeddings=embeddings),
    ContextPrecision(llm=llm),
    ContextUtilization(llm=llm),
    SemanticSimilarity(embeddings=embeddings),
]

# os.environ["PYTHONASYNCIODEBUG"] = "1"
IGNORE_ASYNCIO = False

if __name__ == "__main__":
    # asyncio
    print("Starting [Asyncio]")
    start = time.time()
    _ = metrics_experiment(metrics).run(eval_dataset, name="benchmark")
    print(f"Time taken [Asyncio]: {time.time() - start:.2f}s")
