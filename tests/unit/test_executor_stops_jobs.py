"""The executor stops its jobs when it stops (#207).

``Executor`` (behind ``evaluate()``) and ``run_async_tasks`` ran jobs through
``async_utils.as_completed``. Three problems:

1. ``raise_exceptions=True`` re-raised the first error but left the other jobs
   running in the background -- for ``evaluate()``, LLM calls still being made
   and paid for after the error.
2. ``cancel()`` only took effect when the next job finished.
3. The cancel path left a coroutine un-awaited (``RuntimeWarning``).

They now run through ``concurrent_results``, which cancels and awaits what is
still running whenever the caller stops.
"""

import asyncio
import gc
import time
import warnings

import pytest

from ragas.async_utils import as_completed, run_async_tasks
from ragas.executor import Executor
from ragas.run_config import RunConfig


class Boom(Exception):
    pass


def _unawaited_during(run) -> list:
    """The "never awaited" warnings raised while ``run()`` runs.

    Collects garbage first, so a coroutine another test leaked warns before
    the window opens rather than inside it (CI on #209 caught one from
    test_async_evaluation's mocked run_sync).
    """
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run()
        gc.collect()
    return [str(w.message) for w in caught if "never awaited" in str(w.message)]


def _jobs(log, slow=0.3):
    async def job(i):
        if i == 0:
            await asyncio.sleep(0.01)
            raise Boom("job 0")
        log.append(("started", i))
        await asyncio.sleep(slow)
        log.append(("finished", i))
        return i

    return job


@pytest.mark.parametrize("batch_size", [None, 4])
def test_raise_exceptions_stops_the_other_jobs(batch_size):
    log = []
    job = _jobs(log)

    async def main():
        executor = Executor(
            raise_exceptions=True, show_progress=False, batch_size=batch_size
        )
        for i in range(4):
            executor.submit(job, i)
        with pytest.raises(Boom):
            await executor.aresults()
        await asyncio.sleep(0.5)  # anything left running would finish now

    asyncio.run(main())
    assert ("finished", 1) not in log  # was: jobs 1-3 finished after the raise
    assert not [e for e in log if e[0] == "finished"]


def test_raise_exceptions_starts_no_job_after_the_failure():
    # max_workers=1: the failing job must stop the rest before giving up its
    # worker slot, even when the jobs never await
    started = []

    async def job(i):
        started.append(i)
        if i == 0:
            raise Boom()
        return i

    executor = Executor(
        raise_exceptions=True,
        show_progress=False,
        run_config=RunConfig(max_workers=1),
    )
    for i in range(5):
        executor.submit(job, i)
    with pytest.raises(Boom):
        executor.results()
    assert started == [0]


@pytest.mark.parametrize("batch_size", [None, 3])
def test_cancel_takes_effect_while_jobs_are_still_waiting(batch_size):
    log = []

    async def job(i):
        log.append(("started", i))
        await asyncio.sleep(5)
        log.append(("finished", i))
        return i

    async def main():
        executor = Executor(show_progress=False, batch_size=batch_size)
        for i in range(3):
            executor.submit(job, i)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            executor.cancel()

        canceller = asyncio.create_task(cancel_soon())
        start = time.monotonic()
        results = await executor.aresults()
        await canceller
        return time.monotonic() - start, results

    elapsed, results = asyncio.run(main())
    assert elapsed < 1  # was: waited for the next job to finish (5s)
    assert results == []
    assert not [e for e in log if e[0] == "finished"]


def test_cancelling_leaves_no_coroutine_unawaited():
    async def job(i):
        await asyncio.sleep(0.2)
        return i

    async def main():
        executor = Executor(show_progress=False)
        for i in range(3):
            executor.submit(job, i)

        async def cancel_soon():
            await asyncio.sleep(0.02)
            executor.cancel()

        asyncio.create_task(cancel_soon())
        await executor.aresults()

    assert not _unawaited_during(lambda: asyncio.run(main()))


def test_as_completed_cancel_path_leaves_no_coroutine_unawaited():
    # the public helper is kept; its cancel path no longer leaks a waiter
    async def job(i):
        await asyncio.sleep(0.01 * i)
        return i

    async def main():
        seen = []
        for future in as_completed(
            [job(i) for i in range(3)], cancel_check=lambda: bool(seen)
        ):
            seen.append(await future)
        await asyncio.sleep(0)
        assert seen == [0]

    assert not _unawaited_during(lambda: asyncio.run(main()))


@pytest.mark.parametrize("batch_size", [None, 2])
def test_run_async_tasks_fail_fast_stops_the_other_tasks(batch_size):
    log = []
    job = _jobs(log)
    with pytest.raises(Boom):
        run_async_tasks(
            [job(i) for i in range(4)],
            batch_size=batch_size,
            show_progress=False,
            fail_fast=True,
        )
    time.sleep(0.5)  # the shared loop keeps running; nothing may finish now
    assert not [e for e in log if e[0] == "finished"]


def test_run_async_tasks_without_fail_fast_still_runs_everything():
    log = []
    job = _jobs(log, slow=0.01)
    with pytest.raises(Boom):  # raised at the end, as before
        run_async_tasks([job(i) for i in range(4)], show_progress=False)
    assert sorted(i for kind, i in log if kind == "finished") == [1, 2, 3]
