"""SimpleLLMMetric scoring, validation, persistence and alignment (#217)."""

import math
import shutil
import typing as t
from pathlib import Path

import instructor
import pytest

from ragas import Dataset, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric, NumericMetric, RankingMetric
from ragas.metrics.alignment import Aligner, validate_alignment
from ragas.metrics.serialization import metric_to_dict
from ragas.prompt import DynamicFewShotPrompt, Prompt

FIXTURES = Path(__file__).parent / "fixtures" / "saved_metrics"


class AsyncOnlyLLM:
    """Answers every call with ``value``; has no sync ``generate``."""

    def __init__(self, value: t.Any):
        self.value = value
        self.prompts: t.List[str] = []

    async def agenerate(self, prompt, response_model):
        self.prompts.append(prompt)
        return response_model(value=self.value, reason="because")


class FixedEmbedding:
    def embed_query(self, text: str) -> t.List[float]:
        return [1.0, 0.0]

    async def aembed_query(self, text: str) -> t.List[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return [self.embed_query(x) for x in texts]


def numeric(allowed_values: t.Any = (0.0, 1.0)) -> NumericMetric:
    return NumericMetric(
        name="quality", prompt="Rate {response}.", allowed_values=allowed_values
    )


# --- score / ascore ---------------------------------------------------------------


def test_score_runs_ascore():
    llm = AsyncOnlyLLM(0.5)
    result = numeric().score(llm=llm, response="hi")
    assert result.value == 0.5
    assert result.reason == "because"
    assert llm.prompts and "Rate hi." in llm.prompts[0]
    assert result.traces == {
        "input": llm.prompts[0],
        "output": {"reason": "because", "value": 0.5},
    }


@pytest.mark.asyncio
async def test_ascore_and_score_agree():
    metric = numeric()
    sync = metric.score(llm=AsyncOnlyLLM(0.25), response="hi")
    asynchronous = await metric.ascore(llm=AsyncOnlyLLM(0.25), response="hi")
    assert sync.value == asynchronous.value
    assert sync.traces == asynchronous.traces


def test_batch_score_shares_the_llm():
    results = numeric().batch_score(
        [{"response": "a"}, {"response": "b"}], llm=AsyncOnlyLLM(1.0)
    )
    assert [r.value for r in results] == [1.0, 1.0]


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_numeric_out_of_range_raises(value):
    with pytest.raises(ValueError, match="expected value in range"):
        numeric().score(llm=AsyncOnlyLLM(value), response="hi")


def test_numeric_range_object_is_enforced():
    metric = numeric(range(0, 5))
    assert metric.score(llm=AsyncOnlyLLM(3.0), response="x").value == 3.0
    with pytest.raises(ValueError):
        metric.score(llm=AsyncOnlyLLM(7.0), response="x")


def test_ranking_wrong_length_raises():
    metric = RankingMetric(name="rank", prompt="Rank {items}.", allowed_values=3)
    assert metric.score(llm=AsyncOnlyLLM(["a", "b", "c"]), items="x").value == [
        "a",
        "b",
        "c",
    ]
    with pytest.raises(ValueError, match="expected 3 items"):
        metric.score(llm=AsyncOnlyLLM(["a"]), items="x")


def test_missing_prompt_raises_value_error():
    metric = numeric()
    metric.prompt = None
    with pytest.raises(ValueError, match="has no prompt"):
        metric.score(llm=AsyncOnlyLLM(0.5), response="hi")


def test_out_of_range_is_nan_in_a_scoring_run(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory(
        "gpt-6-luna",
        mode=instructor.Mode.JSON,
        mock_response='{"reason": "too high", "value": 7.0}',
    )
    dataset = Dataset(
        "rows",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[SingleTurnSample(response="r")],
    )
    exp = metrics_experiment([numeric()], llm=llm).run(dataset)
    assert math.isnan(exp[0]["quality"])
    with pytest.raises(ValueError, match="expected value in range"):
        metrics_experiment([numeric()], llm=llm, raise_exceptions=True).run(dataset)


# --- persistence --------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_round_trip(tmp_path, suffix):
    prompt = Prompt("Rate {response}.", examples=[({"response": "a"}, {"value": 1})])
    metric = NumericMetric(name="m", prompt=prompt, allowed_values=(1.0, 5.0))
    path = tmp_path / f"m{suffix}"
    metric.save(str(path))
    loaded = NumericMetric.load(str(path))
    assert metric_to_dict(loaded) == metric_to_dict(metric)
    assert loaded.allowed_values == (1.0, 5.0)


def test_range_allowed_values_round_trip(tmp_path):
    path = tmp_path / "m.json"
    numeric(range(1, 11)).save(str(path))
    assert NumericMetric.load(str(path)).allowed_values == range(1, 11)


def test_load_checks_the_type_in_the_base(tmp_path):
    path = tmp_path / "d.json"
    DiscreteMetric(name="d", prompt="x {a}").save(str(path))
    with pytest.raises(ValueError, match="Cannot load DiscreteMetric as NumericMetric"):
        NumericMetric.load(str(path))
    with pytest.raises(
        ValueError, match="Cannot load DiscreteMetric as SimpleLLMMetric"
    ):
        from ragas.metrics.base import SimpleLLMMetric

        SimpleLLMMetric.load(str(path))


def test_load_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot load metric from"):
        DiscreteMetric.load(str(tmp_path / "nope.json"))


def test_loads_files_saved_by_the_previous_version(tmp_path):
    """The fixtures were written by main before #217's refactor."""
    discrete = DiscreteMetric.load(str(FIXTURES / "discrete_main.json"))
    assert discrete.name == "quality"
    assert discrete.allowed_values == ["pass", "fail"]
    assert isinstance(discrete.prompt, Prompt)
    assert discrete.prompt.examples == [({"response": "ok"}, {"value": "pass"})]

    numeric_metric = NumericMetric.load(str(FIXTURES / "numeric_main.json.gz"))
    assert numeric_metric.allowed_values == (0.0, 10.0)
    assert numeric_metric.score(llm=AsyncOnlyLLM(9.0), response="x").value == 9.0

    ranking = RankingMetric.load(str(FIXTURES / "ranking_main.json"))
    assert ranking.allowed_values == 3

    dynamic = DiscreteMetric.load(
        str(FIXTURES / "dynamic_main.json"), embedding_model=FixedEmbedding()
    )
    assert isinstance(dynamic.prompt, DynamicFewShotPrompt)
    assert dynamic.prompt.max_similar_examples == 2
    assert dynamic.prompt.similarity_threshold == 0.5
    assert dynamic.prompt.example_store._examples == [
        ({"response": "great"}, {"judge": "pass"})
    ]

    # and re-saving writes the same document
    copy = tmp_path / "discrete_main.json"
    shutil.copy(FIXTURES / "discrete_main.json", copy)
    discrete.save(str(copy))
    import json

    assert json.loads(copy.read_text()) == json.loads(
        (FIXTURES / "discrete_main.json").read_text()
    )


# --- alignment ----------------------------------------------------------------------


def labelled(rows: t.List[t.Dict[str, t.Any]]) -> Dataset:
    return Dataset(name="labels", backend="inmemory", data=rows)


def test_aligner_builds_a_few_shot_prompt(monkeypatch):
    metric = DiscreteMetric(name="verdict", prompt="Judge {response}.")
    data = labelled(
        [
            {"response": "good", "verdict": "pass", "verdict_reason": "fine"},
            {"response": "bad", "verdict": "fail"},
            {"response": "unlabelled"},
        ]
    )
    monkeypatch.setattr(data, "reload", lambda: None)
    metric.align(data, FixedEmbedding(), max_similar_examples=2)  # type: ignore[arg-type]
    assert isinstance(metric.prompt, DynamicFewShotPrompt)
    assert metric.prompt.max_similar_examples == 2
    assert metric.prompt.example_store._examples == [
        ({"response": "good"}, {"verdict": "pass", "verdict_reason": "fine"}),
        ({"response": "bad"}, {"verdict": "fail"}),
    ]


def test_aligner_is_usable_directly(monkeypatch):
    metric = DiscreteMetric(name="verdict", prompt="Judge {response}.")
    data = labelled([{"response": "good", "verdict": "pass"}])
    monkeypatch.setattr(data, "reload", lambda: None)
    Aligner(FixedEmbedding(), similarity_threshold=0.1).align(metric, data)  # type: ignore[arg-type]
    assert isinstance(metric.prompt, DynamicFewShotPrompt)
    assert metric.prompt.similarity_threshold == 0.1


def test_validate_alignment_pairs_labels_with_their_predictions(monkeypatch):
    class Echo:
        """Predicts the response text itself."""

        async def agenerate(self, prompt, response_model):
            text = prompt.rsplit("Judge ", 1)[1].split(".")[0]
            return response_model(value=text, reason="r")

    metric = DiscreteMetric(
        name="verdict", prompt="Judge {response}.", allowed_values=["pass", "fail"]
    )
    data = labelled(
        [
            {"response": "pass", "verdict": None},  # unlabelled: left out
            {"response": "pass", "verdict": "pass"},
            {"response": "fail", "verdict": "pass"},
        ]
    )
    monkeypatch.setattr(data, "reload", lambda: None)
    monkeypatch.setattr(type(metric), "get_correlation", lambda self, g, p: 0.0)
    out = validate_alignment(metric, Echo(), data)  # type: ignore[arg-type]
    # Before #217 the None label was dropped from the labels only, so the
    # labels [pass, pass] were zipped with predictions [pass, pass, fail]: 1.0.
    assert out["agreement_rate"] == 0.5
    assert list(out["df"]["verdict_pred"]) == ["pass", "pass", "fail"]
