---
search:
  exclude: true
---

# Adding to your CI pipeline with Pytest

You can add Ragas evaluations to your continuous integration pipeline to track the quality of your RAG pipeline over time. Treat them as part of your end-to-end test suite, and run them before major changes and releases.

A Ragas CI test is an ordinary Pytest test that runs `evaluate()` and asserts on `result.summary()`. That gives each metric's mean, and how many rows it covers.

!!! note
    This example uses a dataset that already holds outputs from a reference RAG pipeline. When testing your own system, generate the `response` and `retrieved_contexts` columns with the pipeline you want to test. See [Evaluation Dataset](../../concepts/components/eval_dataset.md) for how to build one from your own data.

```python
import pytest
from datasets import load_dataset
from openai import AsyncOpenAI

from ragas import EvaluationDataset, evaluate
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


def assert_in_range(score: float, value: float, plus_or_minus: float):
    """Check that score is within value +/- plus_or_minus."""
    assert value - plus_or_minus <= score <= value + plus_or_minus


def test_amnesty_e2e():
    # columns: user_input, response, retrieved_contexts, reference
    dataset = EvaluationDataset.from_hf_dataset(
        load_dataset("vibrantlabsai/amnesty_qa", "english_v3")["eval"]
    )

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)  # pin the judge model
    embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=client
    )

    result = evaluate(
        dataset,
        metrics=[
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            Faithfulness(llm=llm),
            ContextRecall(llm=llm),
            ContextPrecision(llm=llm),
        ],
    )
    summary = result.summary()

    # every row must be scored: a metric that errors or can't judge a row
    # is left out of the mean, so check coverage before trusting it
    for name, stats in summary.items():
        assert stats["missing"] == 0, f"{name}: {stats['missing']} rows unscored"

    assert summary["answer_relevancy"]["mean"] >= 0.9
    assert summary["context_recall"]["mean"] >= 0.95
    assert summary["context_precision"]["mean"] >= 0.95
    assert_in_range(summary["faithfulness"]["mean"], value=0.4, plus_or_minus=0.1)
```

`result["faithfulness"]` is the list of per-row scores, so compare it per row, or use `summary()` for the mean as above.

## Keeping CI scores stable

LLM-judged scores vary a little from run to run. To keep CI from flapping:

- **Pin the judge model and its version**, as above. A model upgrade is a deliberate change, and you re-baseline your thresholds when you make it.
- **Keep the defaults deterministic.** `llm_factory()` defaults to a low temperature. Metrics that sample on purpose (`strictness` > 1) average or vote over several samples, which smooths the result at the cost of more calls.
- **Assert with tolerances**, not exact values: a lower bound, or `assert_in_range` around a baseline you measured.
- **Fail on missing rows**, as above. A provider outage or parse failure otherwise shows up only as a smaller `count`.

## Using Pytest Markers for Ragas E2E tests

These end-to-end tests are slow and call an LLM. [Pytest markers](https://docs.pytest.org/en/latest/example/markers.html) let you tag them so that they run only when needed.

To add a `ragas_ci` marker, add the following to your `conftest.py`:

```python
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "ragas_ci: Set of tests that will be run as part of Ragas CI"
    )
```

Now mark the Ragas tests with it:

```python
import pytest


@pytest.mark.ragas_ci
def test_amnesty_e2e(): ...  # as above
```

Then run only them with `pytest -m ragas_ci`, or skip them in fast runs with `pytest -m "not ragas_ci"`.
