"""Constructor and ascore contract of collections ``BaseMetric`` (#216)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import (
    AspectCritic,
    ContextPrecision,
    ContextUtilization,
    ExactMatch,
)
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


def fake_llm():
    return MagicMock(spec=InstructorLLM)


@pytest.mark.parametrize(
    "cls, default",
    [
        (ContextPrecision, "context_precision"),
        (ContextUtilization, "context_utilization"),
    ],
)
def test_context_precision_wrappers_accept_a_name(cls, default):
    # was TypeError: got multiple values for keyword argument 'name'
    assert cls(llm=fake_llm(), name="custom").name == "custom"
    assert cls(llm=fake_llm()).name == default


def test_unknown_kwarg_is_rejected_not_dropped():
    # the typo used to build a metric with the default strictness of 1
    with pytest.raises(TypeError, match=r"AspectCritic\(\).*'strictnes'"):
        AspectCritic(llm=fake_llm(), definition="d", strictnes=3)


def test_unknown_kwarg_is_rejected_on_a_non_llm_metric():
    with pytest.raises(TypeError, match="'bogus'"):
        ExactMatch(bogus=True)


def test_known_kwargs_still_work():
    assert AspectCritic(llm=fake_llm(), definition="d", strictness=3).strictness == 3
    assert ExactMatch(allowed_values=(0.0, 1.0)).name == "exact_match"


def test_ascore_is_abstract():
    # a subclass that forgot ascore used to score 0.0 via a placeholder
    class NoAscore(BaseMetric):
        pass

    with pytest.raises(TypeError, match="abstract"):
        NoAscore(name="x")

    class WithAscore(BaseMetric):
        async def ascore(self, **kwargs) -> MetricResult:
            return MetricResult(value=1.0)

    assert WithAscore(name="y").score().value == 1.0
