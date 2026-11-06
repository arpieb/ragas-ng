"""evaluate() leaves the caller's metrics and LLMs untouched (#76).

It used to fill in ``metric.llm`` / ``metric.embeddings``, set its RunConfig
and token collector on the metric's LLM, and reset it all in ``finally``.
Two runs sharing a metric or an LLM -- threads, a web server, a notebook --
then used each other's LLM, lost theirs mid-run ("NoneType has no attribute
generate"), or counted each other's tokens.
"""

from __future__ import annotations

import asyncio
import math
import threading
import typing as t
import warnings

import pytest

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.cost import TokenUsage
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


def dataset(n: int = 1) -> EvaluationDataset:
    return EvaluationDataset(
        samples=[SingleTurnSample(reference="ALPHA BETA", retrieved_contexts=["ALPHA"])]
        * n
    )


def shared_metric():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import ContextEntityRecall

        return ContextEntityRecall()


def run(results: t.Dict[str, t.Any], key: str, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results[key] = evaluate(show_progress=False, **kwargs)


def in_parallel(*jobs: t.Callable[[], None]):
    threads = [threading.Thread(target=job) for job in jobs]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=30)
    assert not any(th.is_alive() for th in threads)


def test_concurrent_runs_on_a_shared_metric_use_their_own_llm():
    gate = Gate()
    metric = shared_metric()
    reads_all = entity_llm(["ALPHA", "BETA"], gate)  # 1 of 2 entities -> 0.5
    reads_alpha = entity_llm(["ALPHA"], gate)  # 1 of 1 -> 1.0
    results: t.Dict[str, t.Any] = {}

    in_parallel(
        lambda: run(results, "a", dataset=dataset(), metrics=[metric], llm=reads_all),
        lambda: run(results, "b", dataset=dataset(), metrics=[metric], llm=reads_alpha),
    )

    assert results["a"]["context_entity_recall"] == pytest.approx([0.5])
    assert results["b"]["context_entity_recall"] == pytest.approx([1.0])
    assert metric.llm is None


def test_concurrent_runs_on_a_shared_llm_count_their_own_tokens():
    gate = Gate()
    llm = entity_llm(["ALPHA", "BETA"], gate)
    results: t.Dict[str, t.Any] = {}

    def parser(raw):
        return TokenUsage(input_tokens=raw["usage"], output_tokens=0, model="m")

    in_parallel(
        lambda: run(
            results,
            "one",
            dataset=dataset(1),
            metrics=[shared_metric()],
            llm=llm,
            token_usage_parser=parser,
        ),
        lambda: run(
            results,
            "three",
            dataset=dataset(3),
            metrics=[shared_metric()],
            llm=llm,
            token_usage_parser=parser,
        ),
    )

    # two extraction calls per sample
    assert results["one"].total_tokens().input_tokens == 2
    assert results["three"].total_tokens().input_tokens == 6
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
        metrics=[shared_metric()],
        llm=llm,
        run_config=rc,
    )

    assert llm.seen_configs and all(c is rc for c in llm.seen_configs)
    assert llm.run_config is own


def test_without_a_run_config_the_llms_own_applies():
    llm = entity_llm(["ALPHA", "BETA"])
    own = RunConfig(max_retries=4)
    llm.set_run_config(own)
    results: t.Dict[str, t.Any] = {}

    run(results, "r", dataset=dataset(), metrics=[shared_metric()], llm=llm)

    assert all(c is own for c in llm.seen_configs)


def test_return_executor_jobs_keep_the_runs_config():
    llm = entity_llm(["ALPHA", "BETA"])
    rc = RunConfig(max_retries=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        executor = evaluate(
            dataset(),
            metrics=[shared_metric()],
            llm=llm,
            run_config=rc,
            show_progress=False,
            return_executor=True,
        )
    # results() runs here, after evaluate() returned
    assert executor.results() == pytest.approx([0.5])  # type: ignore[union-attr]
    assert llm.seen_configs and all(c is rc for c in llm.seen_configs)


def test_metric_objects_are_not_modified():
    metric = shared_metric()
    llm = entity_llm(["ALPHA", "BETA"])
    before = dict(metric.__dict__)
    results: t.Dict[str, t.Any] = {}
    run(results, "r", dataset=dataset(), metrics=[metric], llm=llm)
    assert metric.llm is None
    assert {k: v for k, v in metric.__dict__.items() if not k.startswith("_")} == {
        k: v for k, v in before.items() if not k.startswith("_")
    }


def test_default_metrics_are_fresh_hand_off_instances():
    from ragas.evaluation import _default_metrics
    from ragas.metrics._legacy_handoff import _CollectionsHandoff

    first, second = _default_metrics(), _default_metrics()
    assert [m.name for m in first] == [
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "context_recall",
    ]
    assert all(isinstance(m, _CollectionsHandoff) for m in first)
    assert all(a is not b for a, b in zip(first, second))


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
