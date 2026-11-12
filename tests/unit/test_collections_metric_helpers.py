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


# --------------------------------------------------------------------------- #
# Shared input checks
# --------------------------------------------------------------------------- #


def test_require_raises_for_the_first_missing_field_in_order():
    from ragas.metrics.collections._inputs import require

    require(a="x", b=["y"], c=0)  # 0 is a value, not missing
    with pytest.raises(ValueError) as e:
        require(a="x", b=[], c=None)
    assert str(e.value) == "b is missing. Please add b to the test sample."
    with pytest.raises(ValueError, match="^c is missing"):
        require(c=None)
    require(a="  ")  # whitespace is left to the metric


MISSING_CASES = [
    (
        C.Faithfulness,
        dict(user_input="q", response="", retrieved_contexts=["c"]),
        "response",
    ),
    (
        C.ContextRecall,
        dict(user_input="q", reference="r", retrieved_contexts=[]),
        "retrieved_contexts",
    ),
    (
        C.ContextPrecision,
        dict(user_input="", reference="r", retrieved_contexts=["c"]),
        "user_input",
    ),
    (C.AnswerRelevancy, dict(user_input="q", response=""), "response"),
    (
        C.MultiModalRelevance,
        dict(user_input="q", response="", retrieved_contexts=["c"]),
        "response",
    ),
    (C.AgentGoalAccuracyWithReference, dict(user_input=[], reference=""), "reference"),
    (
        C.InstanceSpecificRubrics,
        dict(user_input="q", response="r", rubrics={}),
        "rubrics",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory,kwargs,missing", MISSING_CASES, ids=[c[0].__name__ for c in MISSING_CASES]
)
async def test_missing_inputs_share_one_wording(factory, kwargs, missing):
    llm = MagicMock(spec=InstructorLLM)
    params = inspect.signature(factory.__init__).parameters
    extra = (
        {"embeddings": MagicMock(spec=BaseRagasEmbedding)}
        if "embeddings" in params
        else {}
    )
    with pytest.raises(ValueError) as e:
        await factory(llm=llm, **extra).ascore(**kwargs)
    assert str(e.value) == (
        f"{missing} is missing. Please add {missing} to the test sample."
    )
    llm.agenerate.assert_not_called()


def test_judge_inputs_share_the_sample_fields():
    from ragas.metrics.collections._inputs import SampleFieldsInput
    from ragas.metrics.collections.aspect_critic.util import AspectCriticInput
    from ragas.metrics.collections.domain_specific_rubrics.util import (
        RubricScoreInput,
    )
    from ragas.metrics.collections.instance_specific_rubrics.util import (
        InstanceRubricScoreInput,
    )
    from ragas.metrics.collections.simple_criteria.util import SimpleCriteriaInput

    fields = list(SampleFieldsInput.model_fields)
    for model in (AspectCriticInput, SimpleCriteriaInput, RubricScoreInput):
        assert issubclass(model, SampleFieldsInput)
        assert list(model.model_fields) == fields
    assert list(InstanceRubricScoreInput.model_fields) == [*fields, "rubrics"]
