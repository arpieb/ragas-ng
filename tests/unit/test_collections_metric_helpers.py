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
        if inspect.isclass(cls) and issubclass(cls, BaseMetric) and cls is not BaseMetric:
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
