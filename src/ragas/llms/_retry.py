"""Retry with backoff for the structured-output LLMs (#73).

``RunConfig``'s ``max_retries`` / ``max_wait`` used to reach only the legacy
``BaseRagasLLM``. ``InstructorLLM`` and ``LiteLLMStructuredLLM`` -- what
``llm_factory`` builds -- had no backoff of their own, so a rate-limited run
failed samples instead of slowing down.
"""

from __future__ import annotations

import asyncio
import logging
import typing as t

from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from ragas.run_config import RunConfig

logger = logging.getLogger("ragas.llms.retry")

T = t.TypeVar("T")

# Class names that mean "try again later" across the openai, anthropic,
# litellm and mistralai SDKs, for errors that carry no HTTP status.
_TRANSIENT_NAMES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
        "Timeout",
        "TimeoutException",
        "ConnectError",
        "ReadTimeout",
        "RemoteProtocolError",
    }
)


def _status_code(exc: BaseException) -> t.Optional[int]:
    for owner in (exc, getattr(exc, "response", None)):
        code = getattr(owner, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _causes(exc: BaseException) -> t.Iterator[BaseException]:
    """``exc`` and what it wraps (instructor wraps provider errors)."""
    seen: t.Set[int] = set()
    current: t.Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient(exc: BaseException) -> bool:
    """Whether retrying ``exc`` later could succeed.

    HTTP 408, 409, 429 and 5xx, timeouts, and connection failures. A 400
    (bad parameters, a rejected model) never is, so it surfaces at once
    rather than after minutes of backoff.
    """
    for e in _causes(exc):
        code = _status_code(e)
        if code is not None:
            return code in (408, 409, 429) or code >= 500
        if type(e).__name__ in _TRANSIENT_NAMES:
            return True
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
            return True
    return False


def _should_retry(run_config: RunConfig) -> t.Callable[[BaseException], bool]:
    # The default exception_types, (Exception,), would retry a 400 ten times;
    # a caller who narrows it gets exactly what they asked for.
    if run_config.exception_types in ((Exception,), Exception):
        return is_transient
    types = run_config.exception_types
    return lambda exc: isinstance(exc, types)


def _policy(run_config: RunConfig) -> t.Dict[str, t.Any]:
    return dict(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(max(1, run_config.max_retries)),
        retry=retry_if_exception(_should_retry(run_config)),
        reraise=True,
        before_sleep=before_sleep_log(
            logger, logging.WARNING if run_config.log_tenacity else logging.DEBUG
        ),
    )


def call_with_retry(run_config: RunConfig, fn: t.Callable[..., T], *args: t.Any) -> T:
    return Retrying(**_policy(run_config))(fn, *args)


async def acall_with_retry(
    run_config: RunConfig, fn: t.Callable[..., t.Awaitable[T]], *args: t.Any
) -> T:
    return await AsyncRetrying(**_policy(run_config))(fn, *args)
