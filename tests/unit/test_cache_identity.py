"""Cache keys include the wrapper's identity (#72).

Cached methods are wrapped as bound methods, so ``self`` never reached the key:
two LLM (or embedding) wrappers for different models sharing one cache backend
returned each other's responses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from ragas.cache import DiskCacheBackend, cache_identity, cacher
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import InstructorLLM, InstructorModelArgs


class Answer(BaseModel):
    text: str


def fake_instructor_llm(model: str, cache, reply: str, **kw) -> InstructorLLM:
    """InstructorLLM whose underlying call returns `reply` and counts calls."""
    client = MagicMock()
    llm = InstructorLLM(
        client=client, model=model, provider="openai", cache=cache, **kw
    )
    llm.calls = 0  # type: ignore[attr-defined]

    def _create(*a, **k):
        llm.calls += 1  # type: ignore[attr-defined]
        return Answer(text=reply)

    client.chat.completions.create.side_effect = _create
    return llm


@pytest.fixture
def cache(tmp_path):
    return DiskCacheBackend(cache_dir=str(tmp_path / "cache"))


def test_different_models_do_not_share_entries(cache):
    a = fake_instructor_llm("model-a", cache, "from A")
    b = fake_instructor_llm("model-b", cache, "from B")

    assert a.generate("same prompt", Answer).text == "from A"
    # Before #72 this returned A's cached "from A".
    assert b.generate("same prompt", Answer).text == "from B"
    assert (a.calls, b.calls) == (1, 1)  # type: ignore[attr-defined]


def test_same_model_and_settings_still_hits_the_cache(cache):
    a1 = fake_instructor_llm("model-a", cache, "first")
    a2 = fake_instructor_llm("model-a", cache, "second")
    assert a1.generate("p", Answer).text == "first"
    assert a2.generate("p", Answer).text == "first"
    assert a2.calls == 0  # type: ignore[attr-defined]


def test_sampling_args_are_part_of_the_identity(cache):
    cold = fake_instructor_llm("m", cache, "cold")
    hot = fake_instructor_llm(
        "m", cache, "hot", model_args=InstructorModelArgs(temperature=0.9)
    )
    cold.generate("p", Answer)
    assert hot.generate("p", Answer).text == "hot"


def test_identity_is_read_at_call_time(cache):
    llm = fake_instructor_llm("m", cache, "x")
    llm.generate("p", Answer)
    llm.model = "other"  # e.g. reconfigured after construction
    llm.generate("p", Answer)
    assert llm.calls == 2  # type: ignore[attr-defined]


def test_embeddings_for_different_models_do_not_share_entries(tmp_path):
    cache = DiskCacheBackend(cache_dir=str(tmp_path / "emb"))

    def embedder(model, vec):
        client = MagicMock()
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=vec)]
        )
        return embedding_factory("openai", model=model, client=client, cache=cache)

    small = embedder("text-embedding-3-small", [0.1, 0.2])
    large = embedder("text-embedding-3-large", [0.9, 0.8, 0.7])
    assert small.embed_text("t") == [0.1, 0.2]
    assert large.embed_text("t") == [0.9, 0.8, 0.7]


def test_cache_identity_contents():
    obj = SimpleNamespace(
        provider="openai",
        model="gpt-x",
        model_args={"temperature": 0.0},
        client=SimpleNamespace(base_url="http://localhost:8000/v1"),
    )
    ident = cache_identity(obj)
    assert ident["model"] == "gpt-x"
    assert ident["provider"] == "openai"
    assert ident["model_args"] == {"temperature": 0.0}
    assert ident["base_url"] == "http://localhost:8000/v1"


def test_cache_identity_skips_objects_and_unstable_urls():
    obj = SimpleNamespace(model=object(), model_name="all-MiniLM", client=MagicMock())
    ident = cache_identity(obj)
    assert "model" not in ident and ident["model_name"] == "all-MiniLM"
    assert "base_url" not in ident


def test_different_endpoints_same_model_name_do_not_share(cache):
    calls = []

    class Wrapper:
        def __init__(self, url):
            self.model = "llama3"
            self.client = SimpleNamespace(base_url=url)
            self.ask = cacher(cache, identity=lambda: cache_identity(self))(self.ask)

        def ask(self, prompt):
            calls.append(self.client.base_url)
            return self.client.base_url

    assert Wrapper("http://a/v1").ask("p") == "http://a/v1"
    assert Wrapper("http://b/v1").ask("p") == "http://b/v1"
    assert len(calls) == 2
