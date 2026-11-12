"""Collections metrics work whatever client the LLM was built from (#68).

Every collections metric is async. The default ``llm_factory("gpt-4o-mini")``
used to wrap the *sync* ``litellm.completion``, and google-genai clients were
wrapped sync too, so ``ascore()`` raised "Cannot use agenerate() with a
synchronous client". The no-client and google-genai paths now build an async
client plus a sync twin for ``generate()``; any other sync client is run on a
worker thread.
"""

from __future__ import annotations

import threading
import typing as t

import pytest
from pydantic import BaseModel

from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import ContextEntityRecall


class Answer(BaseModel):
    value: int


def entities_tool_call(*entities: str) -> t.List[t.Dict[str, t.Any]]:
    import json

    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "EntitiesList",
                "arguments": json.dumps({"entities": list(entities)}),
            },
        }
    ]


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


# --- the default (no-client, LiteLLM) path, end to end with no network -------


@pytest.mark.asyncio
async def test_default_llm_scores_a_collections_metric(api_key):
    # Real litellm + instructor; litellm's mock_* kwargs stand in for the API.
    llm = llm_factory(
        "gpt-4o-mini",
        mock_response="",
        mock_tool_calls=entities_tool_call("Paris", "France"),
    )
    assert llm.is_async  # type: ignore[attr-defined]

    result = await ContextEntityRecall(llm=llm).ascore(
        reference="Paris, France", retrieved_contexts=["Paris is in France."]
    )
    assert result.value == pytest.approx(1.0)


def test_default_llm_sync_generate_uses_the_sync_twin(api_key):
    llm = llm_factory(
        "gpt-4o-mini",
        mock_response="",
        mock_tool_calls=entities_tool_call("Paris"),
    )
    assert isinstance(llm, InstructorLLM) and llm._litellm  # one path (#182)
    assert type(llm.client).__name__ == "AsyncInstructor"
    assert type(llm.sync_client).__name__ == "Instructor"
    assert "sync_client" not in llm.model_args

    result = ContextEntityRecall(llm=llm).score(
        reference="Paris", retrieved_contexts=["Paris"]
    )
    assert result.value == pytest.approx(1.0)


# --- sync clients run on a worker thread under agenerate() --------------------


class SyncCompletions:
    def __init__(self):
        self.threads: t.List[int] = []

    def create(self, response_model, **kwargs):
        self.threads.append(threading.get_ident())
        return response_model(value=1)


class SyncClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = SyncCompletions()


@pytest.mark.asyncio
async def test_instructor_llm_with_sync_client_runs_off_the_event_loop():
    client = SyncClient()
    llm = InstructorLLM(client=client, model="m", provider="openai")
    assert not llm.is_async

    result = await llm.agenerate("p", Answer)
    assert result.value == 1
    assert client.chat.completions.threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_sync_client_metric_scores_under_ascore():
    class EntitiesCompletions:
        def create(self, response_model, **kwargs):
            return response_model(entities=["Paris"])

    client = SyncClient()
    client.chat.completions = EntitiesCompletions()  # type: ignore[assignment]
    llm = InstructorLLM(client=client, model="m", provider="openai")

    result = await ContextEntityRecall(llm=llm).ascore(
        reference="Paris", retrieved_contexts=["Paris"]
    )
    assert result.value == pytest.approx(1.0)


# --- embeddings: a sync OpenAI client works under aembed_* --------------------


class SyncEmbeddingsClient:
    def __init__(self):
        self.threads: t.List[int] = []
        outer = self

        class Embeddings:
            def create(self, input, model, **kwargs):
                outer.threads.append(threading.get_ident())
                items = [input] if isinstance(input, str) else input
                data = [
                    type("Item", (), {"embedding": [float(len(x)), 1.0]})()
                    for x in items
                ]
                return type("Response", (), {"data": data})()

        self.embeddings = Embeddings()


@pytest.mark.asyncio
async def test_openai_embeddings_with_sync_client_work_under_aembed():
    from ragas.embeddings.openai_provider import OpenAIEmbeddings

    client = SyncEmbeddingsClient()
    emb = OpenAIEmbeddings(client=client)
    assert not emb.is_async

    assert await emb.aembed_text("abc") == [3.0, 1.0]
    assert await emb.aembed_texts(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert all(tid != threading.get_ident() for tid in client.threads)
