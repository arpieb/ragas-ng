"""Sync calls on async clients all run on one shared loop (#74).

The sync-over-async helpers (three copies: InstructorLLM,
LiteLLMStructuredLLM, embeddings) made a fresh event loop per call inside a
running loop and closed it afterwards. An async client keeps pooled
connections bound to the loop that opened them, so the next sync call failed
with "bound to a different event loop" / "Event loop is closed". The fakes
below fail the same way when used from a second loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import typing as t

import pytest
from pydantic import BaseModel

from ragas.async_utils import run_sync


class LoopBound:
    """Binds to the first loop that uses it, like an httpx connection pool."""

    def __init__(self):
        self.loop: t.Optional[asyncio.AbstractEventLoop] = None
        self.calls = 0

    async def use(self) -> int:
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        elif self.loop is not loop:
            raise RuntimeError("bound to a different event loop")
        if self.loop.is_closed():
            raise RuntimeError("Event loop is closed")
        self.calls += 1
        return self.calls


class Answer(BaseModel):
    value: int


def async_llm(pool: LoopBound):
    from ragas.llms.base import InstructorLLM

    class Completions:
        async def create(self, response_model, **kwargs):
            return response_model(value=await pool.use())

    client = type("C", (), {})()
    client.chat = type("Chat", (), {"completions": Completions()})()
    llm = InstructorLLM(client=client, model="m", provider="openai")
    assert llm.is_async
    return llm


def async_embeddings(pool: LoopBound):
    from ragas.embeddings.openai_provider import OpenAIEmbeddings

    class Embeddings:
        async def create(self, input, model, **kwargs):
            n = await pool.use()
            items = [input] if isinstance(input, str) else input
            data = [type("Item", (), {"embedding": [float(n)]})() for _ in items]
            return type("Response", (), {"data": data})()

    client = type("C", (), {})()
    client.embeddings = Embeddings()
    emb = OpenAIEmbeddings(client=client)
    assert emb.is_async
    return emb


# --- the #74 scenario: repeated sync calls from inside a running loop ----------


@pytest.mark.asyncio
async def test_sync_llm_calls_inside_a_running_loop_share_one_loop():
    pool = LoopBound()
    llm = async_llm(pool)
    assert [llm.generate("p", Answer).value for _ in range(3)] == [1, 2, 3]


@pytest.mark.asyncio
async def test_sync_embed_text_inside_a_running_loop_with_an_async_client():
    pool = LoopBound()
    emb = async_embeddings(pool)
    assert emb.embed_text("a") == [1.0]
    assert emb.embed_text("b") == [2.0]
    assert emb.embed_texts(["c", "d"]) == [[3.0], [3.0]]


def test_sync_calls_with_and_without_a_running_loop_share_one_loop():
    pool = LoopBound()
    llm = async_llm(pool)
    assert llm.generate("p", Answer).value == 1

    async def inside_a_loop():
        return llm.generate("p", Answer).value

    assert asyncio.run(inside_a_loop()) == 2
    assert llm.generate("p", Answer).value == 3


def test_sync_calls_from_several_threads():
    pool = LoopBound()
    llm = async_llm(pool)
    results: t.List[int] = []
    lock = threading.Lock()

    def worker():
        value = llm.generate("p", Answer).value
        with lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(results) == list(range(1, 9))


def test_legacy_embeddings_bridge_uses_the_shared_loop():
    from ragas.embeddings.utils import run_async_in_current_loop

    pool = LoopBound()
    assert run_async_in_current_loop(pool.use()) == 1
    assert run_sync(pool.use()) == 2


# --- run_sync itself ---------------------------------------------------------


def test_exceptions_propagate_with_their_type():
    async def boom():
        raise KeyError("missing")

    with pytest.raises(KeyError, match="missing"):
        run_sync(boom())


def test_caller_contextvars_are_visible():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="unset")
    var.set("caller")

    async def read():
        return var.get()

    assert run_sync(read()) == "caller"


def test_sync_call_from_code_already_on_the_bridge_loop_does_not_deadlock():
    async def inner():
        return 7

    async def outer():
        # async code calling a sync API that bridges back to async
        return run_sync(inner())

    result: t.Dict[str, int] = {}

    def target():
        result["value"] = run_sync(outer())

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout=10)
    assert not th.is_alive(), "run_sync deadlocked on its own loop"
    assert result["value"] == 7


def test_the_bridge_loop_is_reused():
    async def loop_id():
        return id(asyncio.get_running_loop())

    assert run_sync(loop_id()) == run_sync(loop_id())
