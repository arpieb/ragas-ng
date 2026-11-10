"""Smaller fixes from #216.

1. ``evaluate()`` warned twice: its own DeprecationWarning, then the one from
   the public ``aevaluate()`` it called, with a stacklevel inside ragas.
2. ``GoogleEmbeddings(use_vertex=True)`` checked for a client and then never
   used it, loading ``TextEmbeddingModel.from_pretrained`` on every call.
3. ``LiteLLMEmbeddings`` asked ``get_optimal_batch_size`` about provider
   "litellm", so every model got the fallback batch size of 10.
"""

import asyncio
import copy
import sys
import types
import warnings
from unittest.mock import MagicMock

import pytest

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

# ---------------------------------------------------------------------------
# 1. evaluate() / aevaluate() deprecation warnings
# ---------------------------------------------------------------------------


def _dataset():
    return EvaluationDataset(
        samples=[SingleTurnSample(user_input="q", response="a", reference="a")]
    )


def _metrics():
    from ragas.metrics import ExactMatch

    return [ExactMatch()]


def _evaluate_deprecations(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "evaluate() is deprecated" in str(w.message)
    ]


def test_evaluate_warns_once_at_the_caller():
    from ragas import evaluate

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluate(_dataset(), metrics=_metrics(), show_progress=False)

    found = _evaluate_deprecations(caught)
    assert [str(w.message).split()[0] for w in found] == ["evaluate()"]
    assert found[0].filename == __file__


def test_aevaluate_still_warns_once_at_the_caller():
    from ragas import aevaluate

    async def main():
        return await aevaluate(_dataset(), metrics=_metrics(), show_progress=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(main())

    found = _evaluate_deprecations(caught)
    assert [str(w.message).split()[0] for w in found] == ["aevaluate()"]
    assert found[0].filename == __file__


# ---------------------------------------------------------------------------
# 2. GoogleEmbeddings(use_vertex=True) uses its client
# ---------------------------------------------------------------------------


def _vector(values):
    return types.SimpleNamespace(values=values)


@pytest.fixture
def fake_vertexai(monkeypatch):
    """A stand-in for vertexai.language_models.TextEmbeddingModel."""
    model = MagicMock(name="TextEmbeddingModel()")
    model.get_embeddings.side_effect = lambda texts, **kw: [
        _vector([float(len(t))]) for t in texts
    ]
    cls = MagicMock(name="TextEmbeddingModel")
    cls.from_pretrained.return_value = model
    language_models = types.ModuleType("vertexai.language_models")
    language_models.TextEmbeddingModel = cls  # type: ignore[attr-defined]
    vertexai = types.ModuleType("vertexai")
    vertexai.language_models = language_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vertexai", vertexai)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", language_models)
    return cls


def test_vertex_text_embedding_model_client_is_used(fake_vertexai):
    from ragas.embeddings import GoogleEmbeddings

    client = MagicMock(spec=["get_embeddings"])
    client.get_embeddings.side_effect = lambda texts, **kw: [
        _vector([1.0, 2.0]) for _ in texts
    ]
    emb = GoogleEmbeddings(client=client, model="text-embedding-005", use_vertex=True)

    assert emb.embed_text("hello") == [1.0, 2.0]
    assert emb.embed_texts(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert client.get_embeddings.call_count == 2
    fake_vertexai.from_pretrained.assert_not_called()


def test_vertex_model_is_loaded_once_and_reused(fake_vertexai):
    from ragas.embeddings import GoogleEmbeddings

    emb = GoogleEmbeddings(client=object(), model="text-embedding-005", use_vertex=True)
    assert emb.embed_text("abc") == [3.0]
    assert emb.embed_texts(["a", "bb"]) == [[1.0], [2.0]]
    assert asyncio.run(emb.aembed_text("abcd")) == [4.0]

    fake_vertexai.from_pretrained.assert_called_once_with("text-embedding-005")
    # a copy (evaluate() copies its embeddings) shares the loaded model
    twin = copy.deepcopy(emb)
    assert twin.embed_text("x") == [1.0]
    fake_vertexai.from_pretrained.assert_called_once()


def test_vertex_genai_client_goes_through_models_embed_content(fake_vertexai):
    from ragas.embeddings import GoogleEmbeddings

    class Client:  # google-genai's genai.Client(vertexai=True, ...)
        def __init__(self):
            self.models = MagicMock()
            self.models.embed_content.return_value = types.SimpleNamespace(
                embeddings=[_vector([0.5])]
            )

    client = Client()
    emb = GoogleEmbeddings(client=client, model="gemini-embedding-001", use_vertex=True)

    assert emb.embed_text("hi") == [0.5]
    client.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001", contents="hi"
    )
    fake_vertexai.from_pretrained.assert_not_called()


def test_vertex_still_requires_a_client():
    from ragas.embeddings import GoogleEmbeddings

    with pytest.raises(ValueError, match="require a client"):
        GoogleEmbeddings(use_vertex=True)


# ---------------------------------------------------------------------------
# 3. LiteLLMEmbeddings batch size follows the model's provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("text-embedding-3-small", 100),  # bare OpenAI name
        ("openai/text-embedding-3-small", 100),
        ("cohere/embed-english-v3.0", 96),
        ("vertex_ai/text-embedding-005", 5),
        ("huggingface/BAAI/bge-small-en-v1.5", 32),
        ("ollama/nomic-embed-text", 10),  # not in the table
    ],
)
def test_litellm_batch_size_follows_the_provider(model, expected):
    from ragas.embeddings import LiteLLMEmbeddings

    assert LiteLLMEmbeddings(model=model).batch_size == expected


def test_litellm_explicit_batch_size_wins_and_shows_in_repr():
    from ragas.embeddings import LiteLLMEmbeddings

    assert "batch_size" not in repr(LiteLLMEmbeddings(model="text-embedding-3-small"))
    emb = LiteLLMEmbeddings(model="text-embedding-3-small", batch_size=7)
    assert emb.batch_size == 7
    assert "batch_size=7" in repr(emb)


def test_litellm_unknown_model_falls_back_to_the_default():
    from ragas.embeddings import LiteLLMEmbeddings

    assert LiteLLMEmbeddings(model="no-such-model-xyz").batch_size == 10


def test_litellm_unknown_model_prints_nothing_and_restores_setting(capsys):
    import litellm

    from ragas.embeddings import LiteLLMEmbeddings

    before = litellm.suppress_debug_info
    LiteLLMEmbeddings(model="no-such-model-xyz")
    out = capsys.readouterr()
    assert "Provider List" not in out.out + out.err
    assert litellm.suppress_debug_info == before
