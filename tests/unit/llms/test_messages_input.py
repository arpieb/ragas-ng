"""generate()/agenerate() take a list of chat messages as well as a string (#214).

A messages list (OpenAI format, multimodal content parts included) goes down
the same path as a string prompt: retries, cache, system_prompt, provider
parameter mapping, usage recording and sync clients. String prompts are sent,
and cached, exactly as before.

LiteLLM's ``mock_response`` stands in for the provider; no network calls.
"""

from __future__ import annotations

import typing as t
import warnings

import instructor
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from ragas.cache import CacheInterface
from ragas.cost import TokenUsageCollector
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM, InstructorModelArgs, _build_messages
from ragas.run_config import RunConfig


class Answer(BaseModel):
    value: int


class GoldenAnswer(BaseModel):
    # A fixed module, so str(GoldenAnswer) -- part of the cache key -- doesn't
    # depend on how pytest imports this file.
    __module__ = "golden"
    value: int


IMAGE = "data:image/png;base64,iVBORw0KGgo="
MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in the image?"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ],
    }
]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class MemoryCache(CacheInterface):
    def __init__(self):
        self.data: t.Dict[str, t.Any] = {}
        self.keys: t.List[str] = []

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.keys.append(key)
        self.data[key] = value

    def has_key(self, key):
        return key in self.data


class Recorder:
    """An instructor-shaped client that records what it is sent."""

    def __init__(self, *errors: BaseException, is_async: bool = False):
        self.errors = list(errors)
        self.calls: t.List[t.Dict[str, t.Any]] = []
        self.is_async = is_async

    def _next(self, response_model, kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return response_model(value=len(self.calls))

    def client(self) -> t.Any:
        rec = self

        class AsyncCompletions:
            async def create(self, response_model, **kwargs):
                return rec._next(response_model, kwargs)

        class SyncCompletions:
            def create(self, response_model, **kwargs):
                return rec._next(response_model, kwargs)

        completions = AsyncCompletions() if self.is_async else SyncCompletions()
        client: t.Any = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": completions})()
        return client


class RateLimited(Exception):
    status_code = 429


# --- message building ----------------------------------------------------------


def test_string_prompt_messages_are_unchanged():
    assert _build_messages("p", None) == [{"role": "user", "content": "p"}]
    assert _build_messages("p", "sys") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "p"},
    ]


def test_messages_are_sent_as_given_with_the_system_prompt_prepended():
    assert _build_messages(MESSAGES, None) == MESSAGES
    assert _build_messages(MESSAGES, "sys") == [
        {"role": "system", "content": "sys"},
        *MESSAGES,
    ]


def test_a_system_message_in_the_list_takes_precedence():
    messages = [{"role": "system", "content": "mine"}, *MESSAGES]
    assert _build_messages(messages, "sys") == messages


def test_the_callers_list_is_not_mutated():
    messages = [dict(m) for m in MESSAGES]
    _build_messages(messages, "sys")
    assert messages == MESSAGES


@pytest.mark.parametrize("bad", [b"bytes", 3, [{"content": "no role"}], ["text"]])
def test_other_inputs_are_rejected(bad):
    with pytest.raises(TypeError):
        _build_messages(bad, None)


def test_an_empty_list_is_rejected():
    with pytest.raises(ValueError):
        _build_messages([], None)


# --- string prompts: byte-for-byte as before ----------------------------------


@pytest.mark.asyncio
async def test_string_prompt_cache_keys_are_unchanged():
    """Keys computed by the code before #214; existing disk caches still hit."""
    cache = MemoryCache()
    llm = InstructorLLM(
        client=Recorder().client(),
        model="m",
        provider="openai",
        model_args=InstructorModelArgs(system_prompt="sys"),
        cache=cache,
    )
    llm.generate("p", GoldenAnswer)
    await llm.agenerate("p", GoldenAnswer)
    llm.generate("p", GoldenAnswer, temperature=0.5, sample_index=1)
    assert cache.keys == [
        "7e7db9ba297fe81c9d8ff69de645aeac31c98111cd5b3527aa253f8a1d89792d",
        "8547f38b4b496063b7c2539aa23cca72c36f049418bf9814bb7a86351edd3309",
        "0d1c92e2df9a8ca176d851089b4ac0cb9508478f0127f6966069cb98690b60b3",
    ]


# --- messages through the LLM layer -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [None, OpenAI(api_key="sk-test"), AsyncOpenAI(api_key="sk-test")]
)
async def test_messages_reach_litellm_with_any_client(client):
    llm = llm_factory(
        "gpt-6-luna",
        client=client,
        mode=instructor.Mode.JSON,
        mock_response='{"value": 7}',
    )
    assert (await llm.agenerate(MESSAGES, Answer)).value == 7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert llm.generate(MESSAGES, Answer).value == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_messages_get_provider_params_and_system_prompt(is_async):
    rec = Recorder(is_async=is_async)
    llm = InstructorLLM(
        client=rec.client(),
        model="claude-sonnet-5-5",
        provider="anthropic",
        model_args=InstructorModelArgs(system_prompt="sys"),
    )
    await llm.agenerate(MESSAGES, Answer)
    sent = rec.calls[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["messages"] == [{"role": "system", "content": "sys"}, *MESSAGES]
    # The same mapping as a string prompt gets: Claude takes no sampling params.
    assert "temperature" not in sent and "top_p" not in sent
    llm.generate("p", Answer)
    assert {k: v for k, v in rec.calls[1].items() if k != "messages"} == {
        k: v for k, v in sent.items() if k != "messages"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_messages_are_retried(is_async):
    rec = Recorder(RateLimited(), is_async=is_async)
    llm = InstructorLLM(client=rec.client(), model="m", provider="openai")
    llm.set_run_config(RunConfig(max_wait=0))
    assert (await llm.agenerate(MESSAGES, Answer)).value == 2
    assert len(rec.calls) == 2


@pytest.mark.asyncio
async def test_messages_are_cached_by_content():
    rec = Recorder()
    llm = InstructorLLM(
        client=rec.client(), model="m", provider="openai", cache=MemoryCache()
    )
    first = await llm.agenerate(MESSAGES, Answer)
    assert (await llm.agenerate(MESSAGES, Answer)) == first
    assert len(rec.calls) == 1

    other = [{"role": "user", "content": [{"type": "text", "text": "other"}]}]
    await llm.agenerate(other, Answer)
    assert len(rec.calls) == 2


@pytest.mark.asyncio
async def test_messages_record_usage():
    llm = llm_factory(
        "gpt-6-luna", mode=instructor.Mode.JSON, mock_response='{"value": 7}'
    )
    assert isinstance(llm, InstructorLLM)
    collector = llm.usage_collector = TokenUsageCollector()
    await llm.agenerate(MESSAGES, Answer)
    (usage,) = collector.usage_data
    assert (usage.input_tokens, usage.output_tokens) == (10, 20)
