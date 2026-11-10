"""Requests- and tokens-per-minute limits for LLM and embedding calls (#171).

``max_workers`` limits how many jobs run at once, not how many requests reach
the provider per minute: one metric or experiment row can make several LLM
calls, and instructor's validation retries add more. A ``RateLimit`` given to
``llm_factory(rate_limit=...)`` (or ``LiteLLMEmbeddings(rate_limit=...)``)
paces every request those objects send, waiting -- never raising -- until the
provider's limit has room::

    from ragas.rate_limit import RateLimit

    limit = RateLimit(rpm=500, tpm=200_000)
    judge = llm_factory("gpt-6-luna", rate_limit=limit)
    # the same key's limit, shared: pass the same object
    other = llm_factory("gpt-6-luna", temperature=0.7, rate_limit=limit)

Because it sits in the LLM layer, ``evaluate()``, metrics, testset generation
and experiments all go through it. The counting is done by
[pyrate-limiter](https://pypi.org/project/pyrate-limiter/) (sliding windows;
thread- and event-loop-safe).
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import typing as t

from pyrate_limiter import InMemoryBucket, Limiter, Rate

logger = logging.getLogger(__name__)

# Window lengths in milliseconds. Module-level so tests can shrink them.
_MINUTE_MS = 60_000

_KEY = "ragas"


def _check(name: str, value: t.Optional[int]) -> t.Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"RateLimit({name}=...) must be a positive int or None")
    return value


class RateLimit:
    """Requests per minute and, optionally, tokens per minute.

    Args:
        rpm: Requests per minute. Requests are also spread over the minute
            (at most ``ceil(rpm / 60)`` per second) rather than sent in one
            burst, since providers often enforce limits over shorter windows.
        tpm: Tokens per minute. Each request is counted before it is sent as
            its prompt's tokens (``litellm.token_counter``) plus the
            ``max_tokens`` it asks for, if you set one -- how OpenAI counts
            against its limit. Without ``max_tokens`` the output is not
            counted, so leave room for it.
            A request larger than ``tpm`` is counted as ``tpm``.

    One object is one limit. Share it between the LLMs and embeddings that
    use the same API key; give a different key its own. Copies of the
    metrics holding the LLM, such as the ones ``evaluate()`` runs, share it.

    Every request is counted, including instructor's validation retries and
    ragas' own retries. Retries made inside a client (the OpenAI SDK's
    ``max_retries``, LiteLLM's ``num_retries``) are not seen.
    """

    def __init__(self, rpm: t.Optional[int] = None, tpm: t.Optional[int] = None):
        self.rpm = _check("rpm", rpm)
        self.tpm = _check("tpm", tpm)
        if self.rpm is None and self.tpm is None:
            raise ValueError("RateLimit needs rpm, tpm or both")
        self._requests = self._limiter(self.rpm, smooth=True) if self.rpm else None
        self._tokens = self._limiter(self.tpm, smooth=False) if self.tpm else None

    @staticmethod
    def _limiter(per_minute: int, smooth: bool) -> Limiter:
        rates = []
        per_second = math.ceil(per_minute / 60)
        if smooth and per_second < per_minute:
            rates.append(Rate(per_second, _MINUTE_MS // 60))
        rates.append(Rate(per_minute, _MINUTE_MS))
        return Limiter(InMemoryBucket(rates))

    def _token_weight(self, tokens: int) -> int:
        assert self.tpm is not None
        return max(1, min(int(tokens), self.tpm))

    def acquire(self, tokens: int = 0) -> None:
        """Block until a request of ``tokens`` tokens may be sent."""
        if self._tokens is not None and tokens > 0:
            self._tokens.try_acquire(_KEY, self._token_weight(tokens))
        if self._requests is not None:
            self._requests.try_acquire(_KEY)

    async def aacquire(self, tokens: int = 0) -> None:
        """Wait, without blocking the event loop, until the request may go."""
        if self._tokens is not None and tokens > 0:
            await self._tokens.try_acquire_async(_KEY, self._token_weight(tokens))
        if self._requests is not None:
            await self._requests.try_acquire_async(_KEY)

    # One limit, however often the object holding it is copied.
    def __copy__(self) -> "RateLimit":
        return self

    def __deepcopy__(self, memo: t.Dict[int, t.Any]) -> "RateLimit":
        return self

    def __repr__(self) -> str:
        return f"RateLimit(rpm={self.rpm}, tpm={self.tpm})"


# --- token estimates ----------------------------------------------------------


def _fallback_count(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_chat_tokens(kwargs: t.Mapping[str, t.Any]) -> int:
    """Prompt tokens plus the requested ``max_tokens`` of a completion call."""
    import litellm

    messages = kwargs.get("messages") or []
    try:
        prompt = litellm.token_counter(model=kwargs.get("model", ""), messages=messages)
    except Exception:  # unknown model or content shape: a rough count
        prompt = _fallback_count(
            " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        )
    completion = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 0
    return int(prompt) + int(completion)


def estimate_embedding_tokens(model: str, texts: t.Sequence[str]) -> int:
    """Tokens of the texts sent in one embedding request."""
    import litellm

    try:
        return sum(int(litellm.token_counter(model=model, text=text)) for text in texts)
    except Exception:
        return sum(_fallback_count(text) for text in texts)


# --- wrapping a transport function ---------------------------------------------


def limit_completion_calls(fn: t.Callable[..., t.Any], limit: RateLimit) -> t.Any:
    """``fn`` (``litellm.acompletion``, ``completion`` or a Router's), paced.

    The wrapper keeps ``fn``'s sync/async nature: instructor decides how to
    call it from that.
    """

    def tokens(kwargs: t.Mapping[str, t.Any]) -> int:
        return estimate_chat_tokens(kwargs) if limit.tpm else 0

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def paced_async(*args: t.Any, **kwargs: t.Any) -> t.Any:
            await limit.aacquire(tokens(kwargs))
            return await fn(*args, **kwargs)

        return paced_async

    @functools.wraps(fn)
    def paced(*args: t.Any, **kwargs: t.Any) -> t.Any:
        limit.acquire(tokens(kwargs))
        return fn(*args, **kwargs)

    return paced


__all__ = ["RateLimit", "estimate_chat_tokens", "estimate_embedding_tokens"]
