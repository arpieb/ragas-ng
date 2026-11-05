"""Per-context judge calls run concurrently and keep their order (#58)."""

from __future__ import annotations

import asyncio
import random
import re
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics.collections import (
    ContextPrecision,
    NoiseSensitivity,
    TopicAdherence,
    _concurrency,
)
from ragas.metrics.collections._concurrency import gather_bounded
from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput
from ragas.metrics.collections.noise_sensitivity import util as ns
from ragas.metrics.collections.topic_adherence import util as ta


class InFlight:
    def __init__(self):
        self.now = 0
        self.peak = 0

    async def __aenter__(self):
        self.now += 1
        self.peak = max(self.peak, self.now)

    async def __aexit__(self, *exc):
        self.now -= 1


# --- gather_bounded ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_bounded_keeps_order_when_completion_order_differs():
    async def job(i):
        await asyncio.sleep(0.001 * (5 - i))  # later items finish first
        return i

    assert await gather_bounded(job(i) for i in range(5)) == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_gather_bounded_caps_in_flight_calls():
    flight = InFlight()

    async def job():
        async with flight:
            await asyncio.sleep(0.002)

    await gather_bounded((job() for _ in range(20)), limit=3)
    assert flight.peak == 3


# --- metrics ------------------------------------------------------------------------


def delayed_llm(reply: t.Callable[[str, type], t.Any], flight: InFlight):
    """LLM whose calls overlap only if the metric issues them concurrently.

    Delays are random so completion order differs from submission order.
    """
    llm = MagicMock(spec=InstructorLLM)
    rng = random.Random(0)

    async def agenerate(prompt, model, **kwargs):
        async with flight:
            await asyncio.sleep(rng.uniform(0.001, 0.005))
            return reply(prompt, model)

    llm.agenerate = agenerate
    return llm


def ctx_tag(prompt: str) -> str:
    return re.findall(r"CTX\d+", prompt)[-1]


@pytest.mark.asyncio
async def test_context_precision_is_concurrent_and_keeps_rank_order():
    flight = InFlight()
    relevant = {"CTX0": 0, "CTX1": 1, "CTX2": 0, "CTX3": 1}
    llm = delayed_llm(
        lambda p, m: ContextPrecisionOutput(reason="r", verdict=relevant[ctx_tag(p)]),
        flight,
    )
    result = await ContextPrecision(llm=llm).ascore(
        user_input="q", reference="ref", retrieved_contexts=list(relevant)
    )
    assert flight.peak > 1
    # verdicts [0,1,0,1] in rank order -> AP = (1/2 + 2/4) / 2 = 0.5
    assert result.value == pytest.approx(0.5)


def noise_llm(flight: InFlight, seed: int):
    def verdict(statement: str, ctx: str) -> int:
        # fixed per (seed, statement, context), independent of call order
        return random.Random(f"{seed}-{statement}-{ctx}").randint(0, 1)

    def reply(prompt, model):
        if model is ns.StatementGeneratorOutput:
            who = "gt" if "REFTEXT" in prompt.split("Now perform")[-1] else "ans"
            return ns.StatementGeneratorOutput(
                statements=[f"{who}{i}" for i in range(3)]
            )
        tail = prompt.split("Now perform")[-1]
        ctx = re.findall(r"CTX\d+|REFTEXT", tail)[0]
        stmts = re.findall(r"(?:gt|ans)\d", tail)
        return ns.StatementFaithfulnessOutput(
            statements=[
                ns.StatementFaithfulnessAnswer(
                    statement=s,
                    reason="r",
                    verdict=verdict(s, ctx),
                )
                for s in stmts
            ]
        )

    return delayed_llm(reply, flight)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(5))
async def test_noise_sensitivity_concurrent_matches_sequential(monkeypatch, seed):
    contexts = [f"CTX{i}" for i in range(6)]
    kwargs = dict(
        user_input="q",
        response="answer",
        reference="REFTEXT",
        retrieved_contexts=contexts,
    )

    flight = InFlight()
    concurrent = await NoiseSensitivity(llm=noise_llm(flight, seed)).ascore(**kwargs)
    assert flight.peak > 1

    monkeypatch.setattr(_concurrency, "MAX_CONCURRENT_CALLS_PER_SAMPLE", 1)
    import ragas.metrics.collections.noise_sensitivity.metric as nsm

    async def sequential(coros, limit=1):
        return [await c for c in coros]

    monkeypatch.setattr(nsm, "gather_bounded", sequential)
    flight2 = InFlight()
    serial = await NoiseSensitivity(llm=noise_llm(flight2, seed)).ascore(**kwargs)
    assert flight2.peak == 2  # only the two decompositions overlap

    assert concurrent.value == pytest.approx(serial.value, nan_ok=True)


@pytest.mark.asyncio
async def test_topic_adherence_refusal_checks_are_concurrent_and_ordered():
    flight = InFlight()
    refused = {"alpha": True, "beta": False, "gamma": True}

    def reply(prompt, model):
        if model is ta.TopicExtractionOutput:
            return ta.TopicExtractionOutput(topics=list(refused))
        if model is ta.TopicRefusedOutput:
            topic = next(k for k in refused if k in prompt.split("Now perform")[-1])
            return ta.TopicRefusedOutput(refused_to_answer=refused[topic])
        return ta.TopicClassificationOutput(classifications=[True, True, True])

    metric = TopicAdherence(llm=delayed_llm(reply, flight), mode="precision")
    answered = await metric._check_topics_answered("conv", list(refused))
    assert flight.peak > 1
    assert answered.tolist() == [False, True, False]
    result = await metric.ascore(
        user_input=[HumanMessage(content="hi"), AIMessage(content="hello")],
        reference_topics=["x"],
    )
    assert result.value == pytest.approx(1.0)
