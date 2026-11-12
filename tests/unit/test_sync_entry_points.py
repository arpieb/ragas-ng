"""Sync entry points share one event loop and need no nest_asyncio (#80).

``metric.score()``, ``batch_score()``, the ``@*_metric`` decorators'
``score()``, ``async_to_sync``, ``Executor.results()`` and ``evaluate()``
(now ``metrics_experiment(...).run()``) used ``asyncio.run()`` (a new loop per call) or nest_asyncio (patching the
caller's loop). With an async client, whose connections are bound to the
loop that opened them, the second ``score()`` failed with "bound to a
different event loop"; inside Jupyter ``score()`` refused to run at all.
They all run on ``run_sync``'s shared background loop now.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import typing as t

import pytest

from ragas.llms.base import InstructorLLM


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
        self.calls += 1
        return self.calls


def entity_recall(pool: LoopBound):
    from ragas.metrics.collections import ContextEntityRecall

    class Completions:
        async def create(self, response_model, **kwargs):
            await pool.use()
            return response_model(entities=["Paris"])

    client = type("C", (), {})()
    client.chat = type("Chat", (), {"completions": Completions()})()
    return ContextEntityRecall(
        llm=InstructorLLM(client=client, model="m", provider="openai")
    )


SAMPLE = dict(reference="Paris", retrieved_contexts=["Paris"])


def test_repeated_score_with_an_async_client():
    metric = entity_recall(LoopBound())
    for _ in range(3):
        assert metric.score(**SAMPLE).value == pytest.approx(1.0)


def test_repeated_batch_score_with_an_async_client():
    metric = entity_recall(LoopBound())
    for _ in range(2):
        results = metric.batch_score([SAMPLE, SAMPLE])
        assert [r.value for r in results] == pytest.approx([1.0, 1.0])


@pytest.mark.asyncio
async def test_score_works_inside_a_running_loop():
    # The Jupyter case: it used to raise "Cannot call sync score() from an
    # async context".
    metric = entity_recall(LoopBound())
    assert metric.score(**SAMPLE).value == pytest.approx(1.0)
    assert metric.score(**SAMPLE).value == pytest.approx(1.0)


def test_decorated_metric_sync_score_shares_one_loop():
    from ragas.metrics import numeric_metric

    pool = LoopBound()

    @numeric_metric(name="calls", allowed_values=(0, 100))
    async def calls(x: int) -> float:
        return float(await pool.use())

    assert calls.score(x=1).value == 1.0
    assert calls.score(x=1).value == 2.0


def test_async_to_sync_shares_one_loop():
    from ragas.async_utils import async_to_sync

    pool = LoopBound()
    use = async_to_sync(pool.use)
    assert [use(), use()] == [1, 2]


def test_executor_results_share_one_loop():
    from ragas.executor import Executor

    pool = LoopBound()
    for _ in range(2):
        executor = Executor(show_progress=False)
        executor.submit(pool.use)
        executor.results()
    assert pool.calls == 2


def test_repeated_experiment_run_with_an_async_client():
    from ragas import Dataset, SingleTurnSample, metrics_experiment, summarize
    from ragas.backends import InMemoryBackend

    run = metrics_experiment([entity_recall(LoopBound())])
    dataset = Dataset(
        "rows",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[SingleTurnSample(reference="Paris", retrieved_contexts=["Paris"])],
    )
    for _ in range(2):
        exp = run.run(dataset)
        assert summarize(exp)["context_entity_recall"] == {
            "mean": 1.0,
            "count": 1,
            "missing": 0,
        }


def test_nest_asyncio_is_never_imported():
    script = textwrap.dedent(
        """
        import asyncio, sys
        import ragas
        from ragas.async_utils import run
        from ragas.metrics.collections import BleuScore

        async def main():
            # sync APIs from inside a running loop, the old nest_asyncio case
            m = BleuScore()
            m.score(reference="a b c", response="a b c")
            async def one():
                return 1
            return run(one)

        assert asyncio.run(main()) == 1
        assert "nest_asyncio" not in sys.modules, "nest_asyncio was imported"
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
