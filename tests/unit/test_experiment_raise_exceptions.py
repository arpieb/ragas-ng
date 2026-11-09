"""arun(raise_exceptions=True) stops on the first failed row (#170).

By default (#117) a row whose function raises is logged and left out, and the
run goes on. With ``raise_exceptions`` the first failure cancels the remaining
rows, nothing is saved, and the row's exception is re-raised -- like
``evaluate(raise_exceptions=True)``.
"""

import asyncio
import time

import pytest

from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.experiment import experiment


def _dataset(n, backend=None):
    return Dataset(
        name="rows",
        backend=backend or InMemoryBackend(),
        data=[{"i": i} for i in range(n)],
    )


class Boom(Exception):
    pass


@pytest.mark.asyncio
async def test_the_first_failure_is_raised_with_its_row():
    @experiment()
    async def run(row):
        if row["i"] == 2:
            raise Boom("row two")
        return {"i": row["i"]}

    with pytest.raises(Boom, match="row two") as e:
        await run.arun(_dataset(4), name="exp", raise_exceptions=True)
    assert any(
        "row 2 of experiment 'exp'" in n and "nothing was saved" in n
        for n in e.value.__notes__
    )


@pytest.mark.asyncio
async def test_nothing_is_saved():
    backend = InMemoryBackend()

    @experiment()
    async def run(row):
        if row["i"] == 3:
            raise Boom()
        return {"i": row["i"]}  # rows 0-2 finish first

    with pytest.raises(Boom):
        await run.arun(
            _dataset(4, backend), name="exp", max_workers=1, raise_exceptions=True
        )
    assert backend.list_experiments() == []


@pytest.mark.asyncio
async def test_rows_not_yet_started_never_run():
    started = []

    @experiment()
    async def run(row):
        started.append(row["i"])
        if row["i"] == 0:
            raise Boom()
        return {"i": row["i"]}

    with pytest.raises(Boom):
        await run.arun(_dataset(5), name="exp", max_workers=1, raise_exceptions=True)
    assert started == [0]


@pytest.mark.asyncio
async def test_running_rows_are_cancelled():
    cancelled = []

    @experiment()
    async def run(row):
        if row["i"] == 0:
            await asyncio.sleep(0)
            raise Boom()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(row["i"])
            raise
        return {"i": row["i"]}

    start = time.monotonic()
    with pytest.raises(Boom):
        await run.arun(_dataset(3), name="exp", max_workers=-1, raise_exceptions=True)
    await asyncio.sleep(0)  # let the cancellations land
    assert time.monotonic() - start < 5
    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_sync_function_raises_too():
    @experiment()
    def run(row):
        if row["i"] == 1:
            raise Boom("sync")
        return {"i": row["i"]}

    with pytest.raises(Boom, match="sync"):
        await run.arun(_dataset(3), name="exp", raise_exceptions=True)


@pytest.mark.asyncio
async def test_decorator_default_and_arun_override():
    @experiment(raise_exceptions=True)
    async def run(row):
        if row["i"] == 1:
            raise Boom()
        return {"i": row["i"]}

    with pytest.raises(Boom):
        await run.arun(_dataset(3), name="a")
    result = await run.arun(_dataset(3), name="b", raise_exceptions=False)
    assert [r["i"] for r in result] == [0, 2]


@pytest.mark.asyncio
async def test_default_still_logs_and_continues():
    @experiment()
    async def run(row):
        if row["i"] == 1:
            raise Boom()
        return {"i": row["i"]}

    result = await run.arun(_dataset(3), name="exp")
    assert [r["i"] for r in result] == [0, 2]


@pytest.mark.asyncio
async def test_raise_exceptions_is_not_passed_to_the_function():
    seen = {}

    @experiment()
    async def run(row, **kwargs):
        seen.update(kwargs)
        return {"i": row["i"]}

    await run.arun(
        _dataset(1), name="exp", raise_exceptions=True, save_partial=True, other=1
    )
    assert seen == {"other": 1}


# --- save_partial: keep the rows that finished --------------------------------


def _fails_at(n):
    @experiment()
    async def run(row):
        if row["i"] == n:
            raise Boom()
        return {"i": row["i"]}

    return run


@pytest.mark.asyncio
async def test_save_partial_saves_the_finished_rows_then_raises():
    backend = InMemoryBackend()
    with pytest.raises(Boom) as e:
        await _fails_at(3).arun(
            _dataset(5, backend),
            name="exp",
            max_workers=1,
            raise_exceptions=True,
            save_partial=True,
        )
    assert [r["i"] for r in backend.load_experiment("exp")] == [0, 1, 2]
    assert any("saved the 3 of 5 rows that finished" in n for n in e.value.__notes__)


@pytest.mark.asyncio
async def test_save_partial_on_the_decorator_with_a_per_run_override():
    @experiment(raise_exceptions=True, save_partial=True)
    async def run(row):
        if row["i"] == 2:
            raise Boom()
        return {"i": row["i"]}

    backend = InMemoryBackend()
    with pytest.raises(Boom):
        await run.arun(_dataset(3, backend), name="kept", max_workers=1)
    with pytest.raises(Boom):
        await run.arun(
            _dataset(3, backend), name="dropped", max_workers=1, save_partial=False
        )
    assert backend.list_experiments() == ["kept"]


@pytest.mark.asyncio
async def test_save_partial_alone_changes_nothing():
    # without raise_exceptions every row that didn't fail is saved anyway
    result = await _fails_at(1).arun(_dataset(3), name="exp", save_partial=True)
    assert [r["i"] for r in result] == [0, 2]
