"""strictness draws N samples that can actually differ (#55).

AnswerRelevancy, AspectCritic and SimpleCriteriaScore used to make N identical
calls at temperature 0.01 / top_p 0.1. With a cache, all N returned the same
cached response, so strictness was effectively 1.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ragas.cache import DiskCacheBackend
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import AspectCritic
from ragas.metrics.collections._sampling import SAMPLING_TEMPERATURE, sample_n
from ragas.metrics.collections.aspect_critic.util import AspectCriticOutput


class Out(BaseModel):
    text: str


def recording_llm():
    llm = MagicMock(spec=InstructorLLM)
    llm.calls = []

    async def agenerate(prompt, model, **kwargs):
        llm.calls.append(kwargs)
        return model(text=str(len(llm.calls)))

    llm.agenerate = agenerate
    return llm


# --- sample_n -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_sample_is_the_plain_call():
    llm = recording_llm()
    await sample_n(llm, "p", Out, 1)
    assert llm.calls == [{}]  # configured settings, nothing overridden


@pytest.mark.asyncio
async def test_multiple_samples_raise_temperature_and_are_distinct_for_the_cache():
    llm = recording_llm()
    out = await sample_n(llm, "p", Out, 3)
    assert len(out) == 3
    assert [c["temperature"] for c in llm.calls] == [SAMPLING_TEMPERATURE] * 3
    assert sorted(c["sample_index"] for c in llm.calls) == [0, 1, 2]


# --- per-call overrides reach the provider ---------------------------------------


def instructor_llm(model="gpt-4o-mini", provider="openai", cache=None, **kw):
    client = MagicMock()
    client.chat.completions.create.return_value = Out(text="x")
    client.create.return_value = Out(text="x")
    return InstructorLLM(
        client=client, model=model, provider=provider, cache=cache, **kw
    ), client


def test_default_call_sends_configured_sampling_args():
    llm, client = instructor_llm()
    llm.generate("p", Out)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.01 and kwargs["top_p"] == 0.1


def test_temperature_override_drops_top_p_and_never_reaches_as_sample_index():
    llm, client = instructor_llm()
    llm.generate("p", Out, temperature=0.3, sample_index=2)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert "top_p" not in kwargs
    assert "sample_index" not in kwargs
    # the instance's configuration is untouched
    assert llm.model_args["temperature"] == 0.01 and llm.model_args["top_p"] == 0.1


def test_reasoning_models_still_force_temperature_one():
    llm, client = instructor_llm(model="o3-mini")
    llm.generate("p", Out, temperature=0.3)
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 1.0


def test_google_override_lands_in_generation_config():
    llm, client = instructor_llm(model="gemini-x", provider="google")
    llm.generate("p", Out, temperature=0.3)
    config = client.chat.completions.create.call_args.kwargs["generation_config"]
    assert config["temperature"] == 0.3 and "top_p" not in config


# --- the cache no longer collapses N samples into one ----------------------------


@pytest.mark.asyncio
async def test_cached_strictness_makes_n_distinct_calls_then_replays_them(tmp_path):
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "c"))
    client = MagicMock()
    verdicts = iter([1, 0, 0, 1, 1, 1])

    async def create(**kwargs):
        return AspectCriticOutput(reason="r", verdict=next(verdicts))

    client.chat.completions.create = create
    llm = InstructorLLM(
        client=client,
        model="gpt-4o-mini",
        provider="openai",
        cache=cache,
        model_args=InstructorModelArgs(),
    )
    llm.is_async = True
    metric = AspectCritic(llm=llm, name="harm", definition="harmful?", strictness=3)

    first = await metric.ascore(user_input="q", response="r")
    # 3 real calls (1, 0, 0) -> majority 0. Pre-fix: 1 call, cached twice -> 1.
    assert first.value == 0.0

    second = await metric.ascore(user_input="q", response="r")
    assert second.value == 0.0  # replayed from cache; no further calls consumed
    assert next(verdicts) == 1  # the 4th scripted verdict was never used
