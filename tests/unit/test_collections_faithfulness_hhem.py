"""Collections port of FaithfulnesswithHHEM.

The metric shares statement generation with Faithfulness and then scores each
statement against the retrieved contexts with a local NLI cross-encoder rather
than a second LLM call.

``transformers`` is not in the dev environment and the model is a large download
that runs third-party code, so every test here injects a stub classifier. The
one thing that is asserted about the real loader is that it is never touched
unless a score is actually computed -- which is the behaviour change from
legacy, where the model was fetched at construction.
"""

from __future__ import annotations

import sys
import types
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import FaithfulnesswithHHEM
from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class FakeTensor:
    """Minimal stand-in for the torch tensor the classifier returns."""

    def __init__(self, values):
        self._values = list(values)

    def cpu(self):
        return self

    def detach(self):
        return self

    def round(self):
        return FakeTensor(round(v) for v in self._values)

    def tolist(self):
        return list(self._values)


class StubClassifier:
    """Returns a fixed verdict per (premise, statement) pair, recording batches."""

    def __init__(self, verdicts: t.Sequence[float]):
        self._verdicts = list(verdicts)
        self.batches: t.List[t.List[t.Tuple[str, str]]] = []
        self._cursor = 0

    def predict(self, pairs):
        self.batches.append(list(pairs))
        out = self._verdicts[self._cursor : self._cursor + len(pairs)]
        self._cursor += len(pairs)
        return FakeTensor(out)


def llm_yielding(statements: t.List[str]):
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(prompt, model):
        return StatementGeneratorOutput(statements=statements)

    llm.agenerate = agenerate
    return llm


class TestScoring:
    @pytest.mark.asyncio
    async def test_all_statements_entailed_scores_one(self):
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(["a", "b"]), classifier=StubClassifier([1.0, 1.0])
        )
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["ctx"]
        )
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_partial_entailment_is_the_proportion(self):
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(["a", "b", "c", "d"]),
            classifier=StubClassifier([1.0, 1.0, 0.0, 0.0]),
        )
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["ctx"]
        )
        assert result.value == 0.5

    @pytest.mark.asyncio
    async def test_predictions_are_rounded_like_legacy(self):
        """Legacy calls .round() on the tensor; 0.6 counts as entailed."""
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(["a", "b"]), classifier=StubClassifier([0.6, 0.4])
        )
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["ctx"]
        )
        assert result.value == 0.5

    @pytest.mark.asyncio
    async def test_no_statements_is_nan(self):
        import math

        metric = FaithfulnesswithHHEM(
            llm=llm_yielding([]), classifier=StubClassifier([])
        )
        result = await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["ctx"]
        )
        assert math.isnan(result.value)

    @pytest.mark.asyncio
    async def test_contexts_are_joined_into_one_premise(self):
        """Legacy joins all retrieved contexts with newlines into a single premise."""
        stub = StubClassifier([1.0])
        metric = FaithfulnesswithHHEM(llm=llm_yielding(["a"]), classifier=stub)
        await metric.ascore(
            user_input="q", response="r", retrieved_contexts=["one", "two"]
        )
        premise, statement = stub.batches[0][0]
        assert premise == "one\ntwo"
        assert statement == "a"

    @pytest.mark.asyncio
    async def test_statements_are_batched(self):
        """Bounded batches keep a long response from exhausting memory."""
        stub = StubClassifier([1.0] * 5)
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(list("abcde")), classifier=stub, batch_size=2
        )
        await metric.ascore(user_input="q", response="r", retrieved_contexts=["ctx"])
        assert [len(b) for b in stub.batches] == [2, 2, 1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "missing", ["user_input", "response", "retrieved_contexts"]
    )
    async def test_missing_input_is_rejected(self, missing):
        kwargs = {
            "user_input": "q",
            "response": "r",
            "retrieved_contexts": ["ctx"],
        }
        kwargs[missing] = [] if missing == "retrieved_contexts" else ""
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(["a"]), classifier=StubClassifier([1.0])
        )
        with pytest.raises(ValueError, match=missing):
            await metric.ascore(**kwargs)


class TestLazyModelLoading:
    """The behaviour change from legacy, which downloaded at construction."""

    def test_construction_does_not_touch_transformers(self):
        """transformers is not installed here; constructing must still work."""
        metric = FaithfulnesswithHHEM(llm=llm_yielding(["a"]))
        assert metric._classifier is None

    def test_a_supplied_classifier_is_used_as_is(self):
        stub = StubClassifier([1.0])
        metric = FaithfulnesswithHHEM(llm=llm_yielding(["a"]), classifier=stub)
        assert metric.classifier is stub

    def test_missing_transformers_names_the_install(self):
        metric = FaithfulnesswithHHEM(llm=llm_yielding(["a"]))
        with pytest.raises(ImportError, match="pip install transformers"):
            _ = metric.classifier


class FakeAutoModel:
    """Records from_pretrained(...) calls in place of transformers' loader."""

    calls: t.List[t.Dict[str, t.Any]] = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.calls.append({"name": name, **kwargs})
        return MagicMock()


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.AutoModelForSequenceClassification = FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    FakeAutoModel.calls = []
    return FakeAutoModel.calls


class TestPinnedRemoteCode:
    """trust_remote_code runs the Hub's modeling file: load a reviewed commit (#195)."""

    def test_the_default_load_is_pinned_to_a_commit(self, fake_transformers):
        from ragas.metrics.collections.faithfulness_hhem.metric import HHEM_REVISION

        _ = FaithfulnesswithHHEM(llm=llm_yielding(["a"])).classifier
        assert fake_transformers == [
            {
                "name": "vectara/hallucination_evaluation_model",
                "trust_remote_code": True,
                "revision": HHEM_REVISION,
            }
        ]
        assert len(HHEM_REVISION) == 40  # a full commit SHA, not a branch

    def test_another_revision_can_be_chosen(self, fake_transformers):
        metric = FaithfulnesswithHHEM(llm=llm_yielding(["a"]), revision="abc123")
        _ = metric.classifier
        assert fake_transformers[0]["revision"] == "abc123"

    def test_the_legacy_metric_loads_the_same_commit(self, fake_transformers):
        from ragas.metrics import FaithfulnesswithHHEM as LegacyHHEM
        from ragas.metrics.collections.faithfulness_hhem.metric import HHEM_REVISION

        LegacyHHEM()
        assert fake_transformers[0]["revision"] == HHEM_REVISION
        assert fake_transformers[0]["trust_remote_code"] is True


class TestRunsThroughEvaluate:
    def test_scores_via_evaluate(self):
        metric = FaithfulnesswithHHEM(
            llm=llm_yielding(["a", "b"]), classifier=StubClassifier([1.0, 0.0])
        )
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input="q", response="r", retrieved_contexts=["ctx"]
                )
            ]
        )
        result = evaluate(dataset, metrics=[metric])
        assert result["faithfulness_with_hhem"] == [0.5]
