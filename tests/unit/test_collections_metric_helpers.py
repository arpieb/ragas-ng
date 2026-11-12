"""Shared helpers behind the collections metrics (#217)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics import collections as C
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


def _metric_classes():
    for name in C.__all__:
        cls = getattr(C, name)
        if (
            inspect.isclass(cls)
            and issubclass(cls, BaseMetric)
            and cls is not BaseMetric
        ):
            yield cls


# --------------------------------------------------------------------------- #
# Components declared on the class
# --------------------------------------------------------------------------- #


class NeedsLLM(BaseMetric):
    required_components = ("llm",)

    def __init__(self, llm, name="needs_llm", **kwargs):
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(self) -> MetricResult:
        return MetricResult(value=1.0)


class MaybeEmbeddings(BaseMetric):
    optional_components = ("embeddings",)

    def __init__(self, embeddings=None, **kwargs):
        super().__init__(embeddings=embeddings, name="maybe", **kwargs)

    async def ascore(self) -> MetricResult:
        return MetricResult(value=1.0)


class NoComponents(BaseMetric):
    async def ascore(self) -> MetricResult:
        return MetricResult(value=1.0)


def test_required_component_is_stored_and_validated():
    llm = MagicMock(spec=InstructorLLM)
    assert NeedsLLM(llm).llm is llm
    with pytest.raises(ValueError, match="InstructorLLM"):
        NeedsLLM(None)
    with pytest.raises(ValueError, match="InstructorLLM"):
        NeedsLLM(object())


def test_optional_component_accepts_none_but_validates_a_value():
    assert MaybeEmbeddings().embeddings is None
    emb = MagicMock(spec=BaseRagasEmbedding)
    assert MaybeEmbeddings(emb).embeddings is emb
    with pytest.raises(ValueError, match="modern embeddings"):
        MaybeEmbeddings(object())


def test_undeclared_component_is_an_unexpected_keyword():
    with pytest.raises(TypeError, match="'llm'"):
        NoComponents(name="x", llm=MagicMock(spec=InstructorLLM))


@pytest.mark.parametrize("cls", list(_metric_classes()), ids=lambda c: c.__name__)
def test_every_metric_declares_the_components_it_takes(cls):
    params = inspect.signature(cls.__init__).parameters
    declared = (*cls.required_components, *cls.optional_components)
    for component in ("llm", "embeddings"):
        assert (component in params) == (component in declared), component


# --------------------------------------------------------------------------- #
# One F-score
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tp,fp,fn,expected",
    [
        (0, 0, 0, (1.0, 1.0, 1.0)),  # nothing expected, nothing produced
        (0, 0, 3, (0.0, 0.0, 0.0)),  # nothing produced
        (0, 3, 0, (0.0, 0.0, 0.0)),  # nothing expected
        (0, 2, 2, (0.0, 0.0, 0.0)),
        (3, 0, 0, (1.0, 1.0, 1.0)),  # exact, no epsilon
        (1, 1, 0, (0.5, 1.0, 2 / 3)),
    ],
)
def test_f_score_edge_case_policy(tp, fp, fn, expected):
    from ragas.metrics.utils import fbeta_score, precision_recall

    precision, recall, f1 = expected
    assert precision_recall(tp, fp, fn) == pytest.approx((precision, recall))
    assert fbeta_score(tp, fp, fn) == pytest.approx(f1)


def test_fbeta_weights_recall_above_one():
    from ragas.metrics.utils import fbeta_score

    # precision 0.5, recall 1.0
    assert fbeta_score(1, 1, 0, beta=2.0) == pytest.approx(5 * 0.5 / (4 * 0.5 + 1))


def test_optimizer_f1_loss_uses_the_same_policy():
    from ragas.losses import BinaryMetricLoss

    loss = BinaryMetricLoss(metric="f1_score")
    assert loss([0, 0], [0, 0]) == 1.0
    assert loss([1, 0], [1, 1]) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "answered,in_scope,mode,expected",
    [
        ([True, True], [True, True], "f1", 1.0),
        ([True, True], [True, True], "precision", 1.0),
        # Every topic out of scope and refused: perfect adherence (was ~0).
        ([False, False], [False, False], "f1", 1.0),
        ([False], [True], "recall", 0.0),
    ],
)
def test_topic_adherence_scores_exactly(answered, in_scope, mode, expected):
    import numpy as np

    metric = C.TopicAdherence(llm=MagicMock(spec=InstructorLLM), mode=mode)
    score = metric._compute_score(np.array(answered), np.array(in_scope))
    assert score == expected
