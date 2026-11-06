"""RunConfig retry/backoff reaches the llm_factory LLMs (#73).

Only the legacy ``BaseRagasLLM`` used to retry. ``InstructorLLM`` and
``LiteLLMStructuredLLM`` -- the default path -- failed a sample on the
first rate limit. They now retry transient errors with exponential backoff
per ``RunConfig``, and never retry a 400.
"""

from __future__ import annotations

import typing as t

import pytest
from pydantic import BaseModel

from ragas.llms._retry import is_transient
from ragas.llms.base import InstructorLLM
from ragas.llms.litellm_llm import LiteLLMStructuredLLM
from ragas.run_config import RunConfig

FAST = dict(max_wait=0)


class Answer(BaseModel):
    value: int


class HTTPError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    """Name-only match, as for SDK errors without a status code."""


class Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


class ErrorWithResponse(Exception):
    def __init__(self, status_code: int):
        super().__init__("wrapped")
        self.response = Response(status_code)


def wrapped(inner: BaseException) -> Exception:
    """Like instructor's InstructorRetryException around a provider error."""
    try:
        raise inner
    except BaseException as e:
        try:
            raise RuntimeError("InstructorRetryException") from e
        except RuntimeError as outer:
            return outer


@pytest.mark.parametrize(
    "exc",
    [
        HTTPError(429),
        HTTPError(408),
        HTTPError(409),
        HTTPError(500),
        HTTPError(503),
        HTTPError(529),
        ErrorWithResponse(429),
        RateLimitError(),
        TimeoutError(),
        ConnectionError(),
        wrapped(HTTPError(429)),
        wrapped(RateLimitError()),
    ],
)
def test_transient_errors(exc):
    assert is_transient(exc)


@pytest.mark.parametrize(
    "exc",
    [
        HTTPError(400),
        HTTPError(401),
        HTTPError(404),
        HTTPError(422),
        ErrorWithResponse(400),
        ValueError("bad output"),
        wrapped(HTTPError(400)),
        wrapped(ValueError("validation")),
    ],
)
def test_permanent_errors(exc):
    assert not is_transient(exc)


class Flaky:
    """A provider endpoint that fails with ``errors`` in turn, then answers."""

    def __init__(self, *errors: BaseException, is_async: bool = False):
        self.errors = list(errors)
        self.attempts = 0
        self.is_async = is_async

    def _next(self, response_model):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return response_model(value=self.attempts)

    def client(self) -> t.Any:
        flaky = self

        if self.is_async:

            class Completions:
                async def create(self, response_model, **kwargs):
                    return flaky._next(response_model)
        else:

            class Completions:  # type: ignore[no-redef]
                def create(self, response_model, **kwargs):
                    return flaky._next(response_model)

        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": Completions()})()
        return client


def instructor_llm(flaky: Flaky, **rc) -> InstructorLLM:
    llm = InstructorLLM(client=flaky.client(), model="m", provider="openai")
    llm.set_run_config(RunConfig(**{**FAST, **rc}))
    return llm


def test_sync_rate_limit_is_retried():
    flaky = Flaky(HTTPError(429), HTTPError(503))
    assert instructor_llm(flaky).generate("p", Answer).value == 3
    assert flaky.attempts == 3


@pytest.mark.asyncio
async def test_async_rate_limit_is_retried():
    flaky = Flaky(HTTPError(429), is_async=True)
    llm = instructor_llm(flaky)
    assert llm.is_async
    assert (await llm.agenerate("p", Answer)).value == 2


@pytest.mark.asyncio
async def test_sync_client_under_agenerate_is_retried():
    flaky = Flaky(RateLimitError())
    llm = instructor_llm(flaky)
    assert not llm.is_async
    assert (await llm.agenerate("p", Answer)).value == 2


def test_sync_twin_is_retried():
    async_flaky = Flaky(is_async=True)
    sync_flaky = Flaky(HTTPError(500))
    llm = InstructorLLM(
        client=async_flaky.client(),
        sync_client=sync_flaky.client(),
        model="m",
        provider="openai",
    )
    llm.set_run_config(RunConfig(**FAST))
    assert llm.generate("p", Answer).value == 2
    assert async_flaky.attempts == 0


def test_bad_request_is_not_retried():
    flaky = Flaky(HTTPError(400))
    with pytest.raises(HTTPError):
        instructor_llm(flaky).generate("p", Answer)
    assert flaky.attempts == 1


def test_max_retries_is_the_attempt_budget():
    flaky = Flaky(*[HTTPError(429)] * 10)
    with pytest.raises(HTTPError):
        instructor_llm(flaky, max_retries=3).generate("p", Answer)
    assert flaky.attempts == 3


def test_custom_exception_types_are_honoured():
    flaky = Flaky(ValueError("x"))
    assert (
        instructor_llm(flaky, exception_types=(ValueError,)).generate("p", Answer).value
        == 2
    )

    flaky = Flaky(HTTPError(429))
    with pytest.raises(HTTPError):
        instructor_llm(flaky, exception_types=(ValueError,)).generate("p", Answer)
    assert flaky.attempts == 1


@pytest.mark.asyncio
async def test_litellm_llm_is_retried():
    flaky = Flaky(HTTPError(429), is_async=True)
    llm = LiteLLMStructuredLLM(client=flaky.client(), model="m", provider="litellm")
    llm.set_run_config(RunConfig(**FAST))
    assert llm.is_async
    assert (await llm.agenerate("p", Answer)).value == 2


def test_default_run_config_is_per_instance():
    a = InstructorLLM(client=Flaky().client(), model="m", provider="openai")
    b = InstructorLLM(client=Flaky().client(), model="m", provider="openai")
    assert a.run_config == RunConfig()
    assert a.run_config is not b.run_config


def test_a_runs_config_applies_inside_the_run_only():
    # evaluate() passes its RunConfig through a run scope (#76). It used to
    # set it on the (possibly shared) LLM, where it stuck after the run.
    from ragas.metrics._collections_bridge import adapt_collections_metric
    from ragas.metrics.collections import ContextEntityRecall
    from ragas.run_config import run_scope

    llm = InstructorLLM(client=Flaky().client(), model="m", provider="openai")
    own = RunConfig(max_retries=7)
    llm.set_run_config(own)
    rc = RunConfig(max_retries=2)
    with run_scope(run_config=rc):
        adapt_collections_metric(ContextEntityRecall(llm=llm)).init(rc)
        assert llm.run_config is rc
    assert llm.run_config is own


def test_litellm_rate_limit_end_to_end(monkeypatch):
    # Real litellm + instructor: a mocked 429 comes back wrapped in
    # InstructorRetryException and is still recognised and retried.
    from ragas.llms import llm_factory

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = llm_factory("gpt-4o-mini", mock_response="litellm.RateLimitError")
    llm.set_run_config(RunConfig(max_retries=2, max_wait=0))

    calls = []
    original = llm.sync_client.chat.completions.create  # type: ignore[attr-defined]

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        llm.sync_client.chat.completions,  # type: ignore[attr-defined]
        "create",
        counting,
    )
    with pytest.raises(Exception) as info:
        llm.generate("p", Answer)
    assert is_transient(info.value)
    assert len(calls) == 2
