"""arun() limits concurrency, runs sync functions in threads, keeps dataset
order and logs failures (#117).

Before #117 every row started at once, a sync function ran the rows one at a
time on the event loop, results were stored in completion order and a failed
row was only print()ed.
"""

import asyncio
import logging
import threading

import pytest

from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.experiment import DEFAULT_MAX_WORKERS, experiment


def _dataset(n):
    return Dataset(
        name="rows", backend=InMemoryBackend(), data=[{"i": i} for i in range(n)]
    )


class _Peak:
    def __init__(self):
        self.running = 0
        self.peak = 0

    async def track(self):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0)  # let the other started rows run up to here
        await asyncio.sleep(0)
        self.running -= 1


@pytest.mark.asyncio
async def test_arun_max_workers_bounds_concurrency():
    peak = _Peak()

    @experiment()
    async def run(row):
        await peak.track()
        return row

    await run.arun(_dataset(10), name="x", max_workers=2)
    assert peak.peak == 2


@pytest.mark.asyncio
async def test_default_limit_is_16():
    assert DEFAULT_MAX_WORKERS == 16
    peak = _Peak()

    @experiment()
    async def run(row):
        await peak.track()
        return row

    await run.arun(_dataset(40), name="x")
    assert peak.peak == 16


@pytest.mark.asyncio
async def test_decorator_max_workers_and_unlimited():
    peak = _Peak()

    @experiment(max_workers=3)
    async def run(row):
        await peak.track()
        return row

    await run.arun(_dataset(10), name="x")
    assert peak.peak == 3

    peak.peak = 0
    await run.arun(_dataset(30), name="y", max_workers=-1)
    assert peak.peak == 30


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "4"])
def test_invalid_max_workers_raises(bad):
    with pytest.raises(ValueError, match="max_workers"):
        experiment(max_workers=bad)


@pytest.mark.asyncio
async def test_sync_function_rows_overlap_in_threads():
    # Both rows must be inside the function at once to pass the barrier.
    # Run inline on the event loop (before #117), the first row would wait
    # alone and the barrier would time out.
    barrier = threading.Barrier(2, timeout=5)

    @experiment()
    def run(row):
        barrier.wait()
        return {"i": row["i"], "thread": threading.get_ident()}

    result = await run.arun(_dataset(2), name="x", max_workers=2)
    assert len(result) == 2
    assert result[0]["thread"] != result[1]["thread"]


@pytest.mark.asyncio
async def test_results_are_in_dataset_order():
    # row 0 finishes last, row 4 first
    events = [asyncio.Event() for _ in range(5)]

    @experiment()
    async def run(row):
        i = row["i"]
        if i < 4:
            await events[i + 1].wait()
        events[i].set()
        return {"i": i}

    result = await run.arun(_dataset(5), name="x", max_workers=-1)
    assert [r["i"] for r in result] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_failures_are_logged_with_index_and_summary(caplog):
    @experiment()
    async def run(row):
        if row["i"] in (1, 3):
            raise RuntimeError(f"boom {row['i']}")
        return {"i": row["i"]}

    with caplog.at_level(logging.WARNING, logger="ragas.experiment"):
        result = await run.arun(_dataset(5), name="exp")

    assert [r["i"] for r in result] == [0, 2, 4]
    messages = [r.getMessage() for r in caplog.records]
    assert any("row 1 failed" in m and "boom 1" in m for m in messages)
    assert any("row 3 failed" in m and "boom 3" in m for m in messages)
    assert any("2 of 5 rows failed" in m for m in messages)
