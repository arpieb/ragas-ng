"""The quickstart's example RAG app sends the model it is given (#105).

It hard-coded model="gpt-4o", so pointing it at Ollama or another
OpenAI-compatible endpoint still asked for gpt-4o. Loaded from its file: the
examples package isn't installed in the test environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RAG_PY = Path(__file__).resolve().parents[2] / "examples/ragas_examples/rag_eval/rag.py"


@pytest.fixture(scope="module")
def rag():
    spec = importlib.util.spec_from_file_location("_example_rag", RAG_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_example_rag"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("_example_rag", None)


class FakeChatClient:
    """Just enough of an OpenAI-compatible client."""

    def __init__(self):
        self.models = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kwargs):
        self.models.append(model)
        message = SimpleNamespace(content=" an answer ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_default_client_sends_the_given_model(rag, tmp_path):
    client = FakeChatClient()
    app = rag.default_rag_client(client, logdir=str(tmp_path), model="llama3.1")

    result = app.query("What are the main features of Ragas?")

    assert client.models == ["llama3.1"]
    assert result["answer"] == "an answer"


def test_model_defaults_to_gpt_6_luna(rag, tmp_path):
    client = FakeChatClient()
    rag.default_rag_client(client, logdir=str(tmp_path)).query(
        "What are the main features of Ragas?"
    )
    assert client.models == ["gpt-6-luna"]


def test_traces_record_the_model_used(rag, tmp_path):
    client = FakeChatClient()
    app = rag.ExampleRAG(llm_client=client, logdir=str(tmp_path), model="claude-x")
    app.add_documents(["Ragas evaluates LLM applications."])

    app.query("What does Ragas evaluate?")

    models = [e.data.get("model") for e in app.traces if "model" in e.data]
    assert models and set(models) == {"claude-x"}
