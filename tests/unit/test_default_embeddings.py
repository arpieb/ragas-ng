"""evaluate()'s default embeddings follow a LiteLLM LLM's provider (#190).

An ``llm_factory()`` LLM built without a client reports ``provider="litellm"``.
evaluate() used to give it OpenAI's embedding model whatever the LLM was, so a
Gemini-only setup needed ``OPENAI_API_KEY``. The embedding model now comes from
the LLM's LiteLLM prefix, where the provider has one.
"""

import warnings

import pytest

from ragas import EvaluationDataset
from ragas.llms import llm_factory


@pytest.fixture(autouse=True)
def no_openai_key(monkeypatch):
    # the point of #190: none of this may need an OpenAI key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")


def _defaults(llm):
    from ragas.embeddings.base import default_embeddings_for_litellm_llm

    return default_embeddings_for_litellm_llm(llm)


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gemini/gemini-3.6-flash", "gemini/gemini-embedding-2"),
        ("vertex_ai/gemini-3.6-flash", "vertex_ai/gemini-embedding-2"),
        ("gpt-6-luna", "text-embedding-ada-002"),
        ("openai/gpt-6-luna", "text-embedding-ada-002"),
    ],
)
def test_embedding_model_follows_the_llm_prefix(model, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback warning for these
        embeddings = _defaults(llm_factory(model))
    assert type(embeddings).__name__ == "LiteLLMEmbeddings"
    assert embeddings.model == expected


def test_vertex_project_and_location_carry_over():
    llm = llm_factory(
        "vertex_ai/gemini-3.6-flash",
        vertex_project="my-project",
        vertex_location="europe-west4",
        temperature=0.3,
    )
    embeddings = _defaults(llm)
    assert embeddings.litellm_params == {
        "vertex_project": "my-project",
        "vertex_location": "europe-west4",
    }  # the provider's address, not the completion's sampling


def test_a_gemini_key_passed_to_the_llm_is_used_for_its_embeddings():
    embeddings = _defaults(llm_factory("gemini/gemini-3.6-flash", api_key="g-arg"))
    assert embeddings.api_key == "g-arg"


def test_openai_defaults_take_no_arguments_from_the_llm():
    # unchanged from before #190: OpenAI credentials come from the environment
    embeddings = _defaults(llm_factory("openai/gpt-6-luna", api_base="http://x/v1"))
    assert embeddings.api_base is None and embeddings.litellm_params == {}


@pytest.mark.parametrize("model", ["ollama/qwen3.8:27b", "anthropic/claude-haiku-5-5"])
def test_a_provider_without_a_default_warns_and_falls_back(model):
    with pytest.warns(UserWarning, match="Pass your own") as record:
        embeddings = _defaults(llm_factory(model))
    assert embeddings.model == "text-embedding-ada-002"
    assert model in str(record[0].message)
    assert "OPENAI_API_KEY" in str(record[0].message)


def test_evaluate_gives_a_gemini_llm_gemini_embeddings(monkeypatch):
    import ragas.evaluation as evaluation
    from ragas.metrics import answer_relevancy

    seen = {}
    real = evaluation.default_embeddings_for_litellm_llm

    def spy(llm):
        seen["embeddings"] = real(llm)
        raise RuntimeError("stop here")

    monkeypatch.setattr(evaluation, "default_embeddings_for_litellm_llm", spy)
    llm = llm_factory("gemini/gemini-3.6-flash")
    ds = EvaluationDataset.from_list([{"user_input": "q", "response": "a"}])
    with pytest.raises(RuntimeError, match="stop here"):
        evaluation.evaluate(ds, metrics=[answer_relevancy], llm=llm)
    assert seen["embeddings"].model == "gemini/gemini-embedding-2"
