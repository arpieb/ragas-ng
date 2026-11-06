"""llm_factory patches every client correctly and honours ``mode=`` (#70, #71).

- ``mode=`` was dropped for anthropic/google/gemini/perplexity and on the
  LiteLLM path, where it was also sent to the model as a request kwarg.
- ``mistralai.Mistral`` clients were rejected by the generic patcher.
- ``adapter="litellm"`` passed the caller's client through unpatched, so a
  raw OpenAI client (e.g. a LiteLLM proxy) failed on ``response_model``.
"""

from __future__ import annotations

import json
import typing as t
import warnings

import httpx
import instructor
import pytest
from pydantic import BaseModel

from ragas.llms import llm_factory
from ragas.llms.litellm_llm import LiteLLMStructuredLLM


class Answer(BaseModel):
    value: int


class Recorder:
    """Stands in for an instructor ``from_*`` factory and records its calls."""

    def __init__(self):
        self.calls: t.List[t.Dict[str, t.Any]] = []

    def __call__(self, client, **kwargs):
        self.calls.append(kwargs)
        name = "AsyncInstructor" if kwargs.get("use_async") else "Instructor"
        return type(name, (), {"chat": None})()


@pytest.fixture
def record(monkeypatch):
    def install(name: str) -> Recorder:
        rec = Recorder()
        monkeypatch.setattr(instructor, name, rec, raising=False)
        return rec

    return install


class FakeGenaiClient:
    models = object()


FakeGenaiClient.__module__ = "google.genai.client"
FakeGenaiClient.__name__ = "Client"


class FakeOldGeminiModel:
    pass


FakeOldGeminiModel.__module__ = "google.generativeai.generative_models"


class FakeMistral:
    """Shaped like mistralai.Mistral: chat.complete, no chat.completions."""

    def __init__(self):
        self.chat = type("Chat", (), {"complete": lambda *a, **k: None})()


FakeMistral.__module__ = "mistralai.sdk"


class FakeAnthropic:
    pass


# --- mode reaches every provider factory (#71) ------------------------------


@pytest.mark.parametrize(
    "provider, factory, client, adapter",
    [
        ("anthropic", "from_anthropic", FakeAnthropic(), "auto"),
        ("google", "from_genai", FakeGenaiClient(), "auto"),
        ("gemini", "from_gemini", FakeOldGeminiModel(), "instructor"),
        ("perplexity", "from_perplexity", FakeAnthropic(), "auto"),
        ("mistral", "from_mistral", FakeMistral(), "auto"),
    ],
)
def test_mode_reaches_the_provider_factory(record, provider, factory, client, adapter):
    rec = record(factory)
    llm_factory(
        "m",
        provider=provider,
        client=client,
        adapter=adapter,
        mode=instructor.Mode.MD_JSON,
    )
    assert rec.calls
    assert all(c["mode"] is instructor.Mode.MD_JSON for c in rec.calls)


@pytest.mark.parametrize(
    "provider, factory, client",
    [
        ("anthropic", "from_anthropic", FakeAnthropic()),
        ("google", "from_genai", FakeGenaiClient()),
        ("mistral", "from_mistral", FakeMistral()),
    ],
)
def test_without_mode_each_provider_keeps_its_own_default(
    record, provider, factory, client
):
    # Forcing Mode.JSON would be rejected by e.g. instructor's Anthropic handler.
    rec = record(factory)
    llm_factory("m", provider=provider, client=client)
    assert all("mode" not in c for c in rec.calls)


def test_openai_still_defaults_to_json(record):
    rec = record("from_openai")
    llm_factory("gpt-4o", client=FakeAnthropic())
    assert rec.calls == [{"mode": instructor.Mode.JSON}]


# --- the LiteLLM path ---------------------------------------------------------


def test_no_client_path_applies_mode_and_never_sends_it(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory("gpt-4o-mini", mode=instructor.Mode.MD_JSON)
    assert isinstance(llm, LiteLLMStructuredLLM)
    assert "mode" not in llm.model_args
    assert llm.client.mode is instructor.Mode.MD_JSON
    assert llm.sync_client.mode is instructor.Mode.MD_JSON


def test_no_client_path_without_mode_sends_no_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory("gpt-4o-mini")
    assert "mode" not in llm.model_args


def test_no_client_mode_takes_effect(monkeypatch):
    # MD_JSON reads JSON from the message text, which TOOLS (the default) can't.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory(
        "gpt-4o-mini",
        mode=instructor.Mode.MD_JSON,
        mock_response='```json\n{"value": 5}\n```',
    )
    assert llm.generate("p", Answer).value == 5


def openai_proxy_client() -> t.Any:
    """A real openai.OpenAI client whose transport answers like a proxy."""
    from openai import OpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "mode" not in body and "response_model" not in body
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"value": 4}'},
                    }
                ],
            },
        )

    return OpenAI(
        api_key="anything",
        base_url="http://proxy.local/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_litellm_adapter_patches_an_openai_proxy_client():
    # The documented "LiteLLM proxy" setup used to fail with
    # "Completions.create() got an unexpected keyword argument 'response_model'".
    llm = llm_factory(
        "gemini-2.0-flash", client=openai_proxy_client(), adapter="litellm"
    )
    assert isinstance(llm, LiteLLMStructuredLLM)
    assert llm.generate("p", Answer).value == 4


def test_litellm_adapter_mode_goes_to_the_patch_not_the_model(record):
    rec = record("from_openai")
    llm = llm_factory(
        "m",
        client=FakeAnthropicWithChat(),
        adapter="litellm",
        mode=instructor.Mode.TOOLS,
    )
    assert rec.calls == [{"mode": instructor.Mode.TOOLS}]
    assert "mode" not in llm.model_args


class FakeAnthropicWithChat:
    def __init__(self):
        completions = type("C", (), {"create": lambda *a, **k: None})()
        self.chat = type("Chat", (), {"completions": completions})()


def test_litellm_adapter_wraps_a_router_async_with_a_sync_twin(record):
    rec = record("from_litellm")

    class Router:
        def completion(self, **kwargs): ...

        async def acompletion(self, **kwargs): ...

    router = Router()
    llm = llm_factory("m", client=router, adapter="litellm")
    assert llm.sync_client is not None
    assert len(rec.calls) == 2


def test_litellm_adapter_rejects_an_unusable_client():
    with pytest.raises(ValueError, match="gemini/gemini-2.0-flash"):
        llm_factory("gemini-2.0-flash", provider="google", client=FakeOldGeminiModel())


def test_litellm_adapter_warns_that_mode_cannot_reach_a_patched_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import litellm

    patched = instructor.from_litellm(litellm.acompletion)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        llm = llm_factory(
            "m", client=patched, adapter="litellm", mode=instructor.Mode.JSON
        )
    assert llm.client is patched
    assert any("mode= is ignored" in str(w.message) for w in caught)


# --- mistralai (#70) ----------------------------------------------------------


def test_mistralai_client_uses_from_mistral_async_with_a_sync_twin(record):
    rec = record("from_mistral")
    llm = llm_factory("mistral-large-latest", provider="mistral", client=FakeMistral())
    assert [c.get("use_async", False) for c in rec.calls] == [True, False]
    assert llm.is_async  # type: ignore[attr-defined]
    assert llm.sync_client is not None  # type: ignore[attr-defined]


def test_openai_compatible_client_with_mistral_provider_keeps_generic_path(record):
    # provider="mistral" with an OpenAI SDK client (Mistral's OpenAI-compatible
    # endpoint) still goes through from_openai, as before.
    openai_rec = record("from_openai")
    mistral_rec = record("from_mistral")
    llm_factory(
        "mistral-large-latest", provider="mistral", client=FakeAnthropicWithChat()
    )
    assert openai_rec.calls and not mistral_rec.calls
