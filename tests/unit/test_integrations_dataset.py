"""The integrations return ragas ``Dataset``s and score with metrics_experiment (#255).

griptape and r2r's ``transform_to_ragas_dataset()`` return a ``Dataset`` of
``SingleTurnSample`` rows instead of an ``EvaluationDataset``, and
llama_index's ``evaluate()`` is now ``evaluate_query_engine()``, which returns
the ``Experiment``. No network and no optional packages: the query engine,
R2R responses and griptape module are fakes.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest

from ragas import metrics_experiment, summarize
from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.experiment import Experiment
from ragas.metrics.collections import ExactMatch

# --- llama_index -----------------------------------------------------------------


@dataclass
class Node:
    text: str

    def get_text(self) -> str:
        return self.text


@dataclass
class Response:
    response: str
    source_nodes: list = field(default_factory=list)


class QueryEngine:
    """Answers "q<i>" with "a<i>"; raises for "boom"."""

    def __init__(self):
        self.queries = []

    async def aquery(self, query: str) -> Response:
        self.queries.append(query)
        if query == "boom":
            raise RuntimeError("engine down")
        return Response(response="a" + query[1:], source_nodes=[Node(f"ctx {query}")])


def questions(*rows) -> Dataset:
    return Dataset(
        "questions",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[SingleTurnSample(user_input=q, reference=r) for q, r in rows],
    )


def test_evaluate_query_engine_answers_and_scores_each_row():
    from ragas.integrations.llama_index import evaluate_query_engine

    dataset = questions(("q1", "a1"), ("q2", "wrong"))
    engine = QueryEngine()

    exp = evaluate_query_engine(engine, dataset, [ExactMatch()], name="li")

    assert isinstance(exp, Experiment) and exp.name == "li"
    assert sorted(engine.queries) == ["q1", "q2"]
    rows = sorted(exp, key=lambda r: r["user_input"])
    assert [r["response"] for r in rows] == ["a1", "a2"]
    assert [r["retrieved_contexts"] for r in rows] == [["ctx q1"], ["ctx q2"]]
    assert [r["exact_match"] for r in rows] == [1.0, 0.0]
    assert summarize(exp)["exact_match"] == {"mean": 0.5, "count": 2, "missing": 0}
    # the input dataset is left alone
    assert all(s.response is None for s in dataset)


@pytest.mark.asyncio
async def test_aevaluate_query_engine_with_dict_rows_and_a_column_map():
    from ragas.integrations.llama_index import aevaluate_query_engine

    dataset = Dataset(
        "questions",
        InMemoryBackend(),
        data=[{"question": "q1", "reference": "a1"}],
    )
    exp = await aevaluate_query_engine(
        QueryEngine(),
        dataset,
        [ExactMatch()],
        column_map={"user_input": "question"},
    )
    (row,) = list(exp)
    assert row["user_input"] == "q1" and row["exact_match"] == 1.0


def test_a_failed_query_leaves_the_row_unanswered_unless_raise_exceptions():
    from ragas.integrations.llama_index import evaluate_query_engine

    dataset = questions(("q1", "a1"), ("boom", "x"))
    exp = evaluate_query_engine(QueryEngine(), dataset, [ExactMatch()])
    failed = next(r for r in exp if r["user_input"] == "boom")
    assert failed.get("response") is None

    with pytest.raises(RuntimeError, match="engine down"):
        evaluate_query_engine(
            QueryEngine(), dataset, [ExactMatch()], raise_exceptions=True
        )


def test_evaluate_query_engine_rejects_other_inputs():
    from ragas.integrations.llama_index import evaluate_query_engine

    with pytest.raises(TypeError, match="ragas Dataset"):
        evaluate_query_engine(QueryEngine(), [{"user_input": "q"}], [ExactMatch()])
    multi = Dataset("m", InMemoryBackend(), data=[{"user_input": [{"content": "hi"}]}])
    with pytest.raises(NotImplementedError, match="Multi-turn"):
        evaluate_query_engine(QueryEngine(), multi, [ExactMatch()])


# --- r2r ---------------------------------------------------------------------------


def r2r_response(answer: str, chunks):
    search = types.SimpleNamespace(
        as_dict=lambda: {"chunk_search_results": [{"text": c} for c in chunks]}
    )
    return types.SimpleNamespace(
        results=types.SimpleNamespace(
            generated_answer=answer,
            search_results=search,
        )
    )


def test_r2r_transform_returns_a_dataset(tmp_path):
    from ragas.integrations.r2r import transform_to_ragas_dataset

    dataset = transform_to_ragas_dataset(
        user_inputs=["q1", "q2"],
        r2r_responses=[r2r_response("a1", ["c1"]), r2r_response("a2", ["c2", "c3"])],
        references=["a1", "x"],
    )
    assert isinstance(dataset, Dataset) and dataset.name == "r2r"
    assert dataset.data_model is SingleTurnSample
    assert [s.response for s in dataset] == ["a1", "a2"]
    assert [s.retrieved_contexts for s in dataset] == [["c1"], ["c2", "c3"]]

    saved = transform_to_ragas_dataset(
        user_inputs=["q1"],
        references=["a1"],
        name="mine",
        backend="local/jsonl",
        root_dir=str(tmp_path),
    )
    saved.save()
    loaded = Dataset.load(
        "mine", "local/jsonl", data_model=SingleTurnSample, root_dir=str(tmp_path)
    )
    assert [s.reference for s in loaded] == ["a1"]


# --- griptape ----------------------------------------------------------------------


@pytest.fixture
def griptape(monkeypatch):
    """A fake ``griptape.engines.rag``, so the integration imports."""
    rag = types.ModuleType("griptape.engines.rag")
    rag.RagContext = object  # type: ignore[attr-defined]
    for name, module in {
        "griptape": types.ModuleType("griptape"),
        "griptape.engines": types.ModuleType("griptape.engines"),
        "griptape.engines.rag": rag,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "ragas.integrations.griptape", raising=False)
    return importlib.import_module("ragas.integrations.griptape")


@dataclass
class Text:
    value: str

    def to_text(self) -> str:
        return self.value


def test_griptape_transform_returns_a_dataset(griptape):
    contexts = [
        types.SimpleNamespace(
            query="q1", text_chunks=[Text("c1")], outputs=[Text("a1")]
        ),
    ]
    dataset = griptape.transform_to_ragas_dataset(contexts, references=["a1"])
    assert isinstance(dataset, Dataset) and dataset.name == "griptape"
    (sample,) = list(dataset)
    assert sample == SingleTurnSample(
        user_input="q1", retrieved_contexts=["c1"], response="a1", reference="a1"
    )
    exp = metrics_experiment([ExactMatch()]).run(dataset)
    assert summarize(exp)["exact_match"]["mean"] == 1.0
