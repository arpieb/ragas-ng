"""Run a metric's independent per-item LLM calls concurrently (#58).

ContextPrecision judged each retrieved context, NoiseSensitivity checked
statements against each context, and TopicAdherence checked each topic one
call at a time. Those calls don't depend on each other, but awaiting them in
sequence made a sample's latency N round-trips. Under the per-call timeout of a
scoring run that turned samples with many contexts into timeouts, so they
scored NaN.
"""

from __future__ import annotations

import asyncio
import typing as t

T = t.TypeVar("T")

# Upper bound on in-flight judge calls *within one sample*. A scoring run already
# bounds how many samples run at once (max_workers); this stops a
# single sample with many contexts from bursting past provider rate limits.
MAX_CONCURRENT_CALLS_PER_SAMPLE = 8


async def gather_bounded(
    coros: t.Iterable[t.Awaitable[T]],
    limit: int = MAX_CONCURRENT_CALLS_PER_SAMPLE,
) -> t.List[T]:
    """``asyncio.gather`` with at most ``limit`` running at once; order kept."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(coro: t.Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(run(c) for c in coros)))
