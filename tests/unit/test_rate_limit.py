"""RateLimit: requests/tokens per minute for LLM and embedding calls (#171).

``max_workers`` limits jobs at once, not requests per minute. A ``RateLimit``
passed to ``llm_factory(rate_limit=...)`` or ``LiteLLMEmbeddings`` paces each
request those objects send; one object can be shared between them.

The minute is shrunk to 600 ms here (``_MINUTE_MS``), so a ``RateLimit(rpm=3)``
allows 3 requests per 600 ms, at most 1 per 10 ms.
"""

import asyncio
import copy
import threading
import time
import typing as t

import instructor
import pytest
from pydantic import BaseModel

MINUTE = 0.6
JSON = instructor.Mode.JSON


@pytest.fixture(autouse=True)
def short_minute(monkeypatch):
    import ragas.rate_limit

    monkeypatch.setattr(ragas.rate_limit, "_MINUTE_MS", int(MINUTE * 1000))
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "sk-test")


def _max_in_a_minute(stamps: t.List[float]) -> int:
    stamps = sorted(stamps)
    # a little under a window: timer granularity on CI
    return max(sum(1 for y in stamps if x <= y < x + MINUTE * 0.95) for x in stamps)


class Answer(BaseModel):
    value: int


# --- the limit itself ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{}, {"rpm": 0}, {"rpm": -1}, {"tpm": 1.5}, {"rpm": True}]
)
def test_invalid_limits_are_rejected(kwargs):
    from ragas.rate_limit import RateLimit

    with pytest.raises(ValueError):
        RateLimit(**kwargs)


def test_rpm_paces_async_requests():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=3)
    stamps = []

    async def request():
        await limit.aacquire()
        stamps.append(time.monotonic())

    async def main():
        await asyncio.gather(*(request() for _ in range(7)))

    start = time.monotonic()
    asyncio.run(main())
    assert len(stamps) == 7
    assert _max_in_a_minute(stamps) <= 3
    assert time.monotonic() - start >= 2 * MINUTE * 0.95  # 3 + 3 + 1


def test_rpm_paces_sync_requests():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=2)
    stamps = []
    for _ in range(4):
        limit.acquire()
        stamps.append(time.monotonic())
    assert _max_in_a_minute(stamps) <= 2


def test_requests_are_spread_rather_than_sent_in_one_burst():
    from ragas.rate_limit import RateLimit

    # rpm=120 -> at most 2 per 1/60 of the minute (10 ms here)
    limit = RateLimit(rpm=120)
    stamps = []
    for _ in range(6):
        limit.acquire()
        stamps.append(time.monotonic())
    assert stamps[-1] - stamps[0] >= 2 * (MINUTE / 60) * 0.9


def test_waiting_does_not_block_the_event_loop():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=1)
    ticks = []

    async def tick():
        for _ in range(5):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    async def main():
        await limit.aacquire()
        await asyncio.gather(limit.aacquire(), tick())

    asyncio.run(main())
    assert len(ticks) == 5 and ticks[-1] - ticks[0] < MINUTE


def test_one_limit_is_shared_across_threads_and_event_loops():
    # evaluate() runs on run_sync's loop, sync generate() on the caller's
    # thread, a notebook on its own loop: one RateLimit covers them all
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=4)
    stamps = []
    lock = threading.Lock()

    def note():
        with lock:
            stamps.append(time.monotonic())

    async def on_a_loop(n):
        async def one():
            await limit.aacquire()
            note()

        await asyncio.gather(*(one() for _ in range(n)))

    def sync(n):
        for _ in range(n):
            limit.acquire()
            note()

    workers = [
        threading.Thread(target=lambda: asyncio.run(on_a_loop(3))),
        threading.Thread(target=lambda: asyncio.run(on_a_loop(3))),
        threading.Thread(target=sync, args=(3,)),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=10)
    assert len(stamps) == 9
    assert _max_in_a_minute(stamps) <= 4


def test_tpm_counts_tokens():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(tpm=100)
    start = time.monotonic()
    limit.acquire(tokens=60)
    limit.acquire(tokens=30)
    assert time.monotonic() - start < MINUTE / 2  # 90 of 100
    limit.acquire(tokens=30)  # 120: waits for the window
    assert time.monotonic() - start >= MINUTE * 0.9


def test_a_request_larger_than_tpm_is_counted_as_tpm():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(tpm=100)
    start = time.monotonic()
    limit.acquire(tokens=10_000)  # would never fit; must not hang or fail
    assert time.monotonic() - start < MINUTE / 2


def test_a_cancelled_wait_takes_no_slot():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=1)

    async def main():
        await limit.aacquire()
        waiting = asyncio.create_task(limit.aacquire())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        start = time.monotonic()
        await limit.aacquire()
        return time.monotonic() - start

    # the next window (plus pyrate-limiter's 50 ms buffer), not the one after
    assert asyncio.run(main()) < MINUTE * 1.5


def test_copies_share_the_limit():
    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=10)
    assert copy.copy(limit) is limit
    holder = {"limit": limit}
    assert copy.deepcopy(holder)["limit"] is limit


def test_token_estimates():
    from ragas.rate_limit import estimate_chat_tokens, estimate_embedding_tokens

    messages = [{"role": "user", "content": "word " * 100}]
    prompt = estimate_chat_tokens({"model": "gpt-4o", "messages": messages})
    assert 90 <= prompt <= 120
    assert (
        estimate_chat_tokens(
            {"model": "gpt-4o", "messages": messages, "max_tokens": 500}
        )
        == prompt + 500
    )
    assert estimate_embedding_tokens("text-embedding-3-small", ["a b c", "d e"]) > 0


# --- llm_factory(rate_limit=...) ----------------------------------------------


def _spy(rpm=None, tpm=None):
    from ragas.rate_limit import RateLimit

    class Spy(RateLimit):
        def __init__(self):
            super().__init__(rpm=rpm or 10_000, tpm=tpm)
            self.sync_calls: t.List[int] = []
            self.async_calls: t.List[int] = []

        def acquire(self, tokens=0):
            self.sync_calls.append(tokens)
            super().acquire(tokens)

        async def aacquire(self, tokens=0):
            self.async_calls.append(tokens)
            await super().aacquire(tokens)

    return Spy()


def test_llm_factory_paces_async_and_sync_calls():
    from ragas.llms import llm_factory

    limit = _spy()
    llm = llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=limit
    )
    assert llm.rate_limit is limit
    assert asyncio.run(llm.agenerate("q", Answer)).value == 1
    assert llm.generate("q", Answer).value == 1
    assert len(limit.async_calls) == 1
    assert len(limit.sync_calls) == 1


def test_every_attempt_is_counted_including_instructors_retries():
    from ragas.llms import llm_factory

    limit = _spy()
    # not an Answer: instructor retries max_retries times, each a request
    llm = llm_factory(
        "gpt-4o",
        mock_response='{"wrong": 1}',
        mode=JSON,
        rate_limit=limit,
        max_retries=2,
    )
    with pytest.raises(Exception):
        asyncio.run(llm.agenerate("q", Answer))
    assert len(limit.async_calls) >= 2


def test_llm_tokens_are_estimated_from_the_request():
    from ragas.llms import llm_factory

    limit = _spy(tpm=1_000_000)
    llm = llm_factory(
        "gpt-4o",
        mock_response='{"value": 1}',
        mode=JSON,
        rate_limit=limit,
        max_tokens=300,
    )
    asyncio.run(llm.agenerate("word " * 200, Answer))
    (tokens,) = limit.async_calls
    assert tokens >= 200 + 300  # prompt (and instructor's schema) + max_tokens


def test_without_tpm_no_tokens_are_counted():
    from ragas.llms import llm_factory

    limit = _spy()
    llm = llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=limit
    )
    asyncio.run(llm.agenerate("word " * 200, Answer))
    assert limit.async_calls == [0]


def test_llm_factory_actually_waits():
    from ragas.llms import llm_factory
    from ragas.rate_limit import RateLimit

    llm = llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=RateLimit(rpm=2)
    )

    async def main():
        start = time.monotonic()
        await asyncio.gather(*(llm.agenerate("q", Answer) for _ in range(3)))
        return time.monotonic() - start

    assert asyncio.run(main()) >= MINUTE * 0.9


def test_two_llms_share_one_limit():
    from ragas.llms import llm_factory

    limit = _spy()
    a = llm_factory("gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=limit)
    b = llm_factory(
        "anthropic/claude-sonnet-5-5",
        mock_response='{"value": 2}',
        mode=JSON,
        rate_limit=limit,
    )
    asyncio.run(a.agenerate("q", Answer))
    asyncio.run(b.agenerate("q", Answer))
    assert len(limit.async_calls) == 2


def test_rate_limit_is_not_sent_to_the_provider():
    from ragas.llms import llm_factory

    llm = llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=_spy()
    )
    assert "rate_limit" not in llm.model_args


def test_a_router_is_paced(monkeypatch):
    import litellm

    from ragas.llms import llm_factory

    router = litellm.Router(
        model_list=[
            {
                "model_name": "judge",
                "litellm_params": {
                    "model": "gpt-4o",
                    "mock_response": '{"value": 3}',
                },
            }
        ]
    )
    limit = _spy()
    llm = llm_factory("judge", client=router, rate_limit=limit, mode=JSON)
    assert asyncio.run(llm.agenerate("q", Answer)).value == 3
    assert len(limit.async_calls) == 1


# --- embeddings -----------------------------------------------------------------


class _FakeLiteLLM:
    def __init__(self):
        self.calls: t.List[t.List[str]] = []

    def _response(self, input):
        self.calls.append(list(input))
        return type("R", (), {"data": [{"embedding": [0.0]} for _ in input]})()

    def embedding(self, input, **kwargs):
        return self._response(input)

    async def aembedding(self, input, **kwargs):
        return self._response(input)


def test_litellm_embeddings_pace_each_request():
    from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

    limit = _spy(tpm=1_000_000)
    emb = LiteLLMEmbeddings(
        model="text-embedding-3-small", batch_size=2, rate_limit=limit
    )
    emb.litellm = _FakeLiteLLM()
    emb.embed_text("one")
    emb.embed_texts(["a", "b", "c"])  # two batches
    asyncio.run(emb.aembed_text("two"))
    asyncio.run(emb.aembed_texts(["a", "b", "c"]))
    assert len(limit.sync_calls) == 3
    assert len(limit.async_calls) == 3
    assert all(tokens > 0 for tokens in limit.sync_calls + limit.async_calls)
    assert "rate_limit" not in emb.litellm_params


def test_embedding_factory_passes_rate_limit_through():
    from ragas.embeddings.base import embedding_factory

    limit = _spy()
    emb = embedding_factory("litellm", model="text-embedding-3-small", rate_limit=limit)
    assert emb.rate_limit is limit


def test_evaluates_metric_copies_keep_the_same_limit():
    # evaluate() runs shallow copies of your metrics (#76)
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    limit = _spy()
    llm = llm_factory(
        "gpt-4o", mock_response='{"value": 1}', mode=JSON, rate_limit=limit
    )
    metric = copy.copy(Faithfulness(llm=llm))
    assert metric.llm.rate_limit is limit
    asyncio.run(metric.llm.agenerate("q", Answer))
    assert len(limit.async_calls) == 1
