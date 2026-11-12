"""copy.deepcopy / copy.copy of LLMs, embeddings and the metrics holding them (#211).

``copy.deepcopy(llm_factory(...))`` raised ``RecursionError`` (instructor's
client ``__getattr__`` recursing on an object ``copy`` made without
``__init__``). Metrics holding ``LiteLLMEmbeddings`` or a rapidfuzz distance
failed with "cannot pickle 'module' object". And with a cache, a copy kept the
original's ``generate`` wrapper, so its calls ran on the original.

A copy now has its own settings, and shares the clients, cache and rate limit.
"""

import asyncio
import copy

import instructor
import pytest
from pydantic import BaseModel

from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM

copies = pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])


class Answer(BaseModel):
    value: int


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _llm(**kwargs):
    return llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=instructor.Mode.JSON, **kwargs
    )


@pytest.fixture
def calls_by(monkeypatch):
    """The InstructorLLM each provider call ran on."""
    seen = []
    real = InstructorLLM._call_async

    async def spy(self, *args, **kwargs):
        seen.append(self)
        return await real(self, *args, **kwargs)

    monkeypatch.setattr(InstructorLLM, "_call_async", spy)
    return seen


# --- LLMs ---------------------------------------------------------------------


@copies
def test_an_llm_can_be_copied(copier):
    llm = _llm()
    twin = copier(llm)
    assert asyncio.run(twin.agenerate("q", Answer)).value == 1


def test_a_deep_copy_shares_clients_cache_and_rate_limit(tmp_path):
    from ragas.rate_limit import RateLimit

    llm = _llm(cache=DiskCacheBackend(str(tmp_path)), rate_limit=RateLimit(rpm=100))
    twin = copy.deepcopy(llm)
    assert twin.client is llm.client
    assert twin.sync_client is llm.sync_client
    assert twin.cache is llm.cache
    assert twin.rate_limit is llm.rate_limit


def test_a_deep_copy_has_its_own_settings():
    llm = _llm(temperature=0.1)
    twin = copy.deepcopy(llm)
    twin.model_args["temperature"] = 0.9
    assert llm.model_args["temperature"] == 0.1


@copies
def test_a_cached_copy_calls_run_on_the_copy(copier, tmp_path, calls_by):
    llm = _llm(cache=DiskCacheBackend(str(tmp_path)))
    twin = copier(llm)
    asyncio.run(twin.agenerate("only the copy", Answer))
    assert calls_by == [twin]  # was: the original


@copies
def test_an_uncached_copy_calls_run_on_the_copy(copier, calls_by):
    twin = copier(_llm())
    asyncio.run(twin.agenerate("q", Answer))
    assert calls_by == [twin]


def test_objects_referring_to_one_llm_still_share_one_copy():
    llm = _llm()
    holder = {"a": llm, "b": [llm]}
    twin = copy.deepcopy(holder)
    assert twin["a"] is twin["b"][0]
    assert twin["a"] is not llm


# --- embeddings ---------------------------------------------------------------


@copies
def test_litellm_embeddings_can_be_copied(copier, tmp_path):
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.rate_limit import RateLimit

    emb = LiteLLMEmbeddings(
        model="text-embedding-3-small",
        cache=DiskCacheBackend(str(tmp_path)),
        rate_limit=RateLimit(rpm=100),
    )
    twin = copier(emb)
    assert twin.litellm is emb.litellm  # a module: shared, not copied
    assert twin.cache is emb.cache and twin.rate_limit is emb.rate_limit
    # the cache wrapper is the copy's own
    assert twin.embed_text.__wrapped__.__self__ is twin


def test_openai_embeddings_share_their_client():
    from openai import OpenAI

    from ragas.embeddings import OpenAIEmbeddings

    client = OpenAI(api_key="sk-test")
    emb = OpenAIEmbeddings(client=client, model="text-embedding-3-small")
    assert copy.deepcopy(emb).client is client


# --- metrics holding them -----------------------------------------------------


def test_collections_metrics_with_an_llm_and_embeddings_can_be_deep_copied():
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        Faithfulness,
        SemanticSimilarity,
    )

    llm = _llm()
    emb = LiteLLMEmbeddings(model="text-embedding-3-small")
    for metric in (
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        AnswerCorrectness(llm=llm, embeddings=emb),
    ):
        twin = copy.deepcopy(metric)
        assert twin.llm is not llm and twin.llm.client is llm.client
    twin = copy.deepcopy(SemanticSimilarity(embeddings=emb))
    assert twin.embeddings is not emb and twin.embeddings.litellm is emb.litellm


def test_string_metrics_can_be_deep_copied_and_still_score():
    from ragas.metrics.collections import (
        DistanceMeasure,
        NonLLMContextRecall,
        NonLLMStringSimilarity,
    )

    metric = NonLLMStringSimilarity(distance_measure=DistanceMeasure.JARO)
    twin = copy.deepcopy(metric)
    assert (
        asyncio.run(twin.ascore(reference="kitten", response="sitting")).value
        == asyncio.run(metric.ascore(reference="kitten", response="sitting")).value
    )
    copy.deepcopy(NonLLMContextRecall())


def test_string_metric_and_relationship_builder_can_be_deep_copied():
    from ragas.metrics.collections import NonLLMStringSimilarity
    from ragas.testset.transforms.relationship_builders.traditional import (
        OverlapScoreBuilder,
    )

    twin = copy.deepcopy(NonLLMStringSimilarity())
    assert twin.distance_measure_map[twin.distance_measure] is not None
    builder = copy.deepcopy(OverlapScoreBuilder())
    assert builder.distance_measure_map[builder.distance_measure] is not None
