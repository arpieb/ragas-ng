"""NVIDIA-style dual-judge metrics: LLM errors raise, invalid ratings give NaN (#185)."""

from __future__ import annotations

import math

import pytest

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)


class FakeLLM(InstructorBaseRagasLLM):
    def __init__(self, rating=None, error=None):
        self.rating = rating
        self.error = error
        self.calls = 0

    def generate(self, prompt, response_model, temperature=None, sample_index=None):
        raise NotImplementedError

    async def agenerate(
        self, prompt, response_model, temperature=None, sample_index=None
    ):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return response_model(rating=self.rating)


CASES = [
    (AnswerAccuracy, dict(user_input="q", response="a", reference="b")),
    (
        ResponseGroundedness,
        dict(response="Paris is big.", retrieved_contexts=["Paris is a city."]),
    ),
    (
        ContextRelevance,
        dict(user_input="Where is Paris?", retrieved_contexts=["Paris is in France."]),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls,kwargs", CASES)
async def test_llm_error_propagates(metric_cls, kwargs):
    llm = FakeLLM(error=RuntimeError("400 bad request"))
    with pytest.raises(RuntimeError, match="400 bad request"):
        await metric_cls(llm=llm).ascore(**kwargs)
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls,kwargs", CASES)
async def test_invalid_rating_is_nan(metric_cls, kwargs):
    llm = FakeLLM(rating=99)
    result = await metric_cls(llm=llm, max_retries=2).ascore(**kwargs)
    assert math.isnan(result.value)
    assert llm.calls == 4  # two judges x two attempts
