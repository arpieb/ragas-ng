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
        self.requests = []

    def generate(self, prompt, response_model, temperature=None, sample_index=None):
        raise NotImplementedError

    async def agenerate(
        self, prompt, response_model, temperature=None, sample_index=None
    ):
        self.calls += 1
        self.requests.append((prompt, temperature, sample_index))
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
    assert llm.calls == 2  # the two judges run concurrently, no retry


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls,kwargs", CASES)
async def test_invalid_rating_is_nan(metric_cls, kwargs):
    llm = FakeLLM(rating=99)
    result = await metric_cls(llm=llm, max_retries=2).ascore(**kwargs)
    assert math.isnan(result.value)
    assert llm.calls == 4  # two judges x two attempts


class CachedLLM(FakeLLM):
    """Answers from a cache keyed like InstructorLLM's: one answer per request."""

    def __init__(self, answers):
        super().__init__()
        self.answers = answers  # sample_index -> rating
        self.cache = {}

    async def agenerate(
        self, prompt, response_model, temperature=None, sample_index=None
    ):
        key = (prompt, temperature, sample_index)
        if key not in self.cache:
            self.calls += 1
            self.requests.append(key)
            self.cache[key] = response_model(rating=self.answers[sample_index])
        return self.cache[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls,kwargs", CASES)
async def test_retry_is_a_fresh_sample_so_a_cache_cannot_repeat_it(metric_cls, kwargs):
    """A retry after an off-scale rating used to resend the identical request,
    so under a cache it could only return the same rating (#217, cf. #55)."""
    top = max(metric_cls.ratings)
    llm = CachedLLM({None: 99, 1: 99, 2: top})
    result = await metric_cls(llm=llm, max_retries=3).ascore(**kwargs)
    assert result.value == 1.0
    first, *retries = [r for r in llm.requests if r[0] == llm.requests[0][0]]
    assert first[1:] == (None, None)  # first attempt: configured settings
    assert [r[2] for r in retries] == [1, 2]
    assert all(r[1] is not None for r in retries)


@pytest.mark.asyncio
async def test_judges_run_concurrently():
    import asyncio

    started = []
    gate = asyncio.Event()

    class Blocking(FakeLLM):
        async def agenerate(
            self, prompt, response_model, temperature=None, sample_index=None
        ):
            started.append(prompt)
            if len(started) == 2:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=1)
            return response_model(rating=4)

    result = await AnswerAccuracy(llm=Blocking()).ascore(**CASES[0][1])
    assert result.value == 1.0
    assert len(started) == 2


def test_average_scores_is_a_nan_mean():
    from ragas.metrics.collections._dual_judge import average_scores

    assert average_scores(0.5, 1.0) == 0.75
    assert average_scores(float("nan"), 0.5) == 0.5
    assert math.isnan(average_scores(float("nan"), float("nan")))
