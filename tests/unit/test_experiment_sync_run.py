"""experiment.run(): arun() from synchronous code (#172).

Experiments could only be run with ``await fn.arun(...)``, or by wrapping that
in ``asyncio.run`` in a script. ``evaluate()`` and ``metric.score()`` have sync
forms; ``run()`` is the experiments' one. Like them it goes through
``ragas.async_utils.run_sync``, so it also works inside a running event loop
(Jupyter).
"""

import asyncio

import pytest

from ragas.backends import InMemoryBackend
from ragas.dataset import Dataset
from ragas.experiment import Experiment, experiment


def _dataset(n, backend=None):
    return Dataset(
        name="rows",
        backend=backend or InMemoryBackend(),
        data=[{"i": i} for i in range(n)],
    )


@experiment()
async def double(row, factor=2):
    await asyncio.sleep(0)
    return {"i": row["i"], "out": row["i"] * factor}


def test_run_returns_the_saved_experiment():
    backend = InMemoryBackend()
    result = double.run(_dataset(3, backend), name="exp")
    assert isinstance(result, Experiment)
    assert [r["out"] for r in result] == [0, 2, 4]
    assert [r["out"] for r in backend.load_experiment("exp")] == [0, 2, 4]


def test_run_matches_arun():
    sync = double.run(_dataset(4), name="a")
    via_async = asyncio.run(double.arun(_dataset(4), name="b"))
    assert list(sync) == list(via_async)


def test_run_passes_arguments_like_arun():
    seen = {}

    @experiment()
    async def run_with(row, prefix, *, suffix="", **extra):
        seen.update(extra)
        return {"text": f"{prefix}{row['i']}{suffix}"}

    result = run_with.run(
        _dataset(2), "exp", None, "p-", suffix="-s", max_workers=1, other=1
    )
    assert [r["text"] for r in result] == ["p-0-s", "p-1-s"]
    assert seen == {"other": 1}  # max_workers is run()'s, as it is arun()'s


def test_a_sync_experiment_function():
    @experiment()
    def plain(row):
        return {"i": row["i"]}

    assert [r["i"] for r in plain.run(_dataset(3), name="exp")] == [0, 1, 2]


def test_raise_exceptions_and_save_partial_reach_arun():
    class Boom(Exception):
        pass

    @experiment()
    async def flaky(row):
        if row["i"] == 2:
            raise Boom()
        return {"i": row["i"]}

    backend = InMemoryBackend()
    with pytest.raises(Boom):
        flaky.run(
            _dataset(3, backend),
            name="exp",
            max_workers=1,
            raise_exceptions=True,
            save_partial=True,
        )
    assert [r["i"] for r in backend.load_experiment("exp")] == [0, 1]
    # without raise_exceptions: the failed row is left out, as with arun
    assert [r["i"] for r in flaky.run(_dataset(3), name="ok")] == [0, 1]


@pytest.mark.asyncio
async def test_run_works_inside_a_running_event_loop():
    # as in Jupyter: sync code called while the caller's loop is running
    result = double.run(_dataset(2), name="exp")
    assert [r["out"] for r in result] == [0, 2]
