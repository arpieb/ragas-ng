import time

from ragas import evaluate
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

# from ragas.metrics.critique import harmfulness  # Import unavailable

# data - using safe dataset loading
eval_dataset = load_amnesty_dataset_safe("english_v2")

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
    _ = evaluate(
        eval_dataset,
        metrics=metrics,
    )
    print(f"Time taken [Asyncio]: {time.time() - start:.2f}s")
