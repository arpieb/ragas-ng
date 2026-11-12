"""Scoring runs leave the caller's metrics and LLMs untouched (#76).

The legacy path used to fill in ``metric.llm`` / ``metric.embeddings``, set its RunConfig
and token collector on the metric's LLM, and reset it all in ``finally``.
Two runs sharing a metric or an LLM -- threads, a web server, a notebook --
then used each other's LLM, lost theirs mid-run ("NoneType has no attribute
generate"), or counted each other's tokens. (These tests ran on evaluate()
until #257; they now run on metrics_experiment.)
"""

from __future__ import annotations

import asyncio
import math
import threading
import typing as t

import pytest

from ragas import Dataset, SingleTurnSample, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.cost import TokenUsage, TokenUsageCollector
from ragas.llms.base import InstructorLLM
from ragas.run_config import RunConfig


class Gate:
    """Holds the first ``parties`` LLM calls until all have arrived.

    Each run's first call blocks before the run can make another, so the
    first two calls come from two runs: both are then in flight at once.
    """

    def __init__(self, parties: int = 2):
        self.parties = parties
        self.arrived = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(parties, timeout=5)

    async def wait(self):
        with self.lock:
            self.arrived += 1
            hold = self.arrived <= self.parties
        if hold:
            try:
                await asyncio.to_thread(self.barrier.wait)
            except threading.BrokenBarrierError:
                pass


def entity_llm(sees: t.Sequence[str], gate: t.Optional[Gate] = None):
    """A fake LLM that extracts the tokens in ``sees`` found in the text."""
    seen_configs: t.List[RunConfig] = []

    class Completions:
        async def _answer(self, response_model, messages):
            if gate is not None:
                await gate.wait()
            text = messages[-1]["content"].rsplit("Now perform", 1)[-1]
            seen_configs.append(llm.run_config)
            return response_model(entities=[w for w in sees if w in text])

        async def create(self, response_model, messages, **kwargs):
            return await self._answer(response_model, messages)

        async def create_with_completion(self, response_model, messages, **kwargs):
            result = await self._answer(response_model, messages)
            return result, {"usage": 1}

    client = type("C", (), {})()
    client.chat = type("Chat", (), {"completions": Completions()})()
    llm = InstructorLLM(client=client, model="m", provider="openai")
    llm.seen_configs = seen_configs  # type: ignore[attr-defined]
    return llm


def dataset(n: int = 1) -> Dataset:
    return Dataset(
        "rows",
        InMemoryBackend(),
        data_model=SingleTurnSample,
        data=[
            SingleTurnSample(reference="ALPHA BETA", retrieved_contexts=["ALPHA"])
            for _ in range(n)
        ],
    )


def shared_metric(llm):
    from ragas.metrics.collections import ContextEntityRecall

    return ContextEntityRecall(llm=llm)


def run(results: t.Dict[str, t.Any], key: str, *, dataset, metrics, **kwargs):
    results[key] = metrics_experiment(metrics, **kwargs).run(dataset)


def in_parallel(*jobs: t.Callable[[], None]):
    threads = [threading.Thread(target=job) for job in jobs]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)
    assert not any(th.is_alive() for th in threads)


def test_concurrent_runs_on_a_shared_llm_count_their_own_tokens():
    gate = Gate()
    llm = entity_llm(["ALPHA", "BETA"], gate)
    results: t.Dict[str, t.Any] = {}

    def parser(raw):
        return TokenUsage(input_tokens=raw["usage"], output_tokens=0, model="m")

    one, three = TokenUsageCollector(parser), TokenUsageCollector(parser)
    in_parallel(
        lambda: run(
            results,
            "one",
            dataset=dataset(1),
            metrics=[shared_metric(llm)],
            usage=one,
        ),
        lambda: run(
            results,
            "three",
            dataset=dataset(3),
            metrics=[shared_metric(llm)],
            usage=three,
        ),
    )

    # two extraction calls per sample
    assert one.total_tokens().input_tokens == 2
    assert three.total_tokens().input_tokens == 6
    assert llm.usage_collector is None


def test_the_runs_config_applies_inside_the_run_and_does_not_stick():
    llm = entity_llm(["ALPHA", "BETA"])
    own = RunConfig(max_retries=4)
    llm.set_run_config(own)
    rc = RunConfig(max_retries=2, timeout=30)
    results: t.Dict[str, t.Any] = {}

    run(
        results,
        "r",
        dataset=dataset(),
        metrics=[shared_metric(llm)],
        run_config=rc,
    )

    assert llm.seen_configs and all(c is rc for c in llm.seen_configs)
    assert llm.run_config is own


def test_without_a_run_config_the_llms_own_applies():
    llm = entity_llm(["ALPHA", "BETA"])
    own = RunConfig(max_retries=4)
    llm.set_run_config(own)
    results: t.Dict[str, t.Any] = {}

    run(results, "r", dataset=dataset(), metrics=[shared_metric(llm)])

    assert all(c is own for c in llm.seen_configs)


def test_metric_objects_are_not_modified():
    llm = entity_llm(["ALPHA", "BETA"])
    metric = shared_metric(llm)
    before = dict(metric.__dict__)
    results: t.Dict[str, t.Any] = {}
    run(results, "r", dataset=dataset(), metrics=[metric], llm=entity_llm(["BETA"]))
    assert metric.llm is llm
    assert {k: v for k, v in metric.__dict__.items() if not k.startswith("_")} == {
        k: v for k, v in before.items() if not k.startswith("_")
    }


# --- ContextEntityRecall epsilon ------------------------------------------------


@pytest.mark.asyncio
async def test_entity_recall_perfect_match_is_exactly_one():
    from ragas.metrics.collections import ContextEntityRecall

    llm = entity_llm(["ALPHA"])
    result = await ContextEntityRecall(llm=llm).ascore(
        reference="ALPHA", retrieved_contexts=["ALPHA"]
    )
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_entity_recall_without_reference_entities_is_nan():
    from ragas.metrics.collections import ContextEntityRecall

    llm = entity_llm(["ALPHA"])
    result = await ContextEntityRecall(llm=llm).ascore(
        reference="nothing here", retrieved_contexts=["ALPHA"]
    )
    assert math.isnan(result.value)
