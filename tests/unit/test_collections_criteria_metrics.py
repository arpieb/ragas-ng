"""Collections ports of AspectCritic and SimpleCriteriaScore.

Both take a free-text criterion and ask an LLM to judge against it -- the most
open-ended metrics ragas ships.

The one deliberate behaviour change is ``strictness``. The legacy metrics
document it as "the number of times self consistency checks is made, final
judgement is made using majority vote", but both took exactly one sample and
passed a single-element list into the vote, so it never did anything. These
ports implement it, matching both the docstring and collections AnswerRelevancy,
which already does ``for _ in range(self.strictness)``. At the default
``strictness=1`` the ports are equivalent to legacy.
"""

from __future__ import annotations

import typing as t
from unittest.mock import MagicMock

import pytest

from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample, evaluate
from ragas.llms.base import InstructorLLM
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics.collections import AspectCritic, SimpleCriteriaScore
from ragas.metrics.collections.aspect_critic.util import AspectCriticOutput
from ragas.metrics.collections.simple_criteria.util import SimpleCriteriaOutput

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def scripted_llm(outputs: t.Sequence[t.Any]):
    """An LLM that returns the given outputs in order, then repeats the last."""
    llm = MagicMock(spec=InstructorLLM)
    calls = {"n": 0}

    async def agenerate(prompt, model, **kwargs):
        i = min(calls["n"], len(outputs) - 1)
        calls["n"] += 1
        return outputs[i]

    llm.agenerate = agenerate
    llm.calls = calls
    return llm


def critic(verdict: int, reason: str = "because"):
    return AspectCriticOutput(reason=reason, verdict=verdict)


def criteria(score: int, reason: str = "because"):
    return SimpleCriteriaOutput(reason=reason, score=score)


class TestAspectCritic:
    @pytest.mark.asyncio
    async def test_returns_the_verdict_and_its_reason(self):
        llm = scripted_llm([critic(1, "clearly harmful")])
        metric = AspectCritic(llm=llm, name="harmfulness", definition="Is it harmful?")
        result = await metric.ascore(user_input="q", response="r")
        assert result.value == 1.0
        assert result.reason == "clearly harmful"

    def test_definition_is_injected_into_the_instruction(self):
        metric = AspectCritic(
            llm=scripted_llm([critic(0)]),
            name="x",
            definition="Does it spread fake information?",
        )
        assert "Does it spread fake information?" in metric.prompt.instruction

    def test_definition_setter_updates_the_instruction(self):
        """The legacy metric exposed this; scripts mutate it between runs."""
        metric = AspectCritic(llm=scripted_llm([critic(0)]), name="x", definition="A?")
        metric.definition = "B?"
        assert metric.definition == "B?"
        assert "B?" in metric.prompt.instruction
        assert "A?" not in metric.prompt.instruction

    @pytest.mark.asyncio
    async def test_strictness_takes_a_majority_vote(self):
        """The behaviour the legacy docstring promised but never delivered."""
        llm = scripted_llm([critic(1, "yes"), critic(0, "no"), critic(0, "also no")])
        metric = AspectCritic(llm=llm, name="x", definition="?", strictness=3)
        result = await metric.ascore(user_input="q", response="r")
        assert llm.calls["n"] == 3
        assert result.value == 0.0
        assert result.reason == "no"  # first judgement in the winning majority

    @pytest.mark.asyncio
    async def test_strictness_one_makes_a_single_call(self):
        """The default must stay equivalent to legacy."""
        llm = scripted_llm([critic(1)])
        metric = AspectCritic(llm=llm, name="x", definition="?")
        await metric.ascore(user_input="q", response="r")
        assert llm.calls["n"] == 1

    @pytest.mark.parametrize("given,expected", [(1, 1), (2, 3), (3, 3), (4, 5)])
    def test_even_strictness_is_rounded_up_to_avoid_ties(self, given, expected):
        metric = AspectCritic(
            llm=scripted_llm([critic(1)]), name="x", definition="?", strictness=given
        )
        assert metric.strictness == expected


class TestSimpleCriteriaScore:
    @pytest.mark.asyncio
    async def test_returns_the_score_and_its_reason(self):
        llm = scripted_llm([criteria(4, "close to the reference")])
        metric = SimpleCriteriaScore(
            llm=llm, name="cg", definition="Score 0 to 5 by similarity."
        )
        result = await metric.ascore(user_input="q", response="r", reference="r")
        assert result.value == 4.0
        assert result.reason == "close to the reference"

    @pytest.mark.asyncio
    async def test_strictness_takes_a_majority_vote(self):
        llm = scripted_llm([criteria(5), criteria(2), criteria(2)])
        metric = SimpleCriteriaScore(llm=llm, name="cg", definition="?", strictness=3)
        result = await metric.ascore(user_input="q", response="r")
        assert llm.calls["n"] == 3
        assert result.value == 2.0

    def test_score_range_is_configurable(self):
        """Legacy imposed no range; the definition decides what is valid."""
        metric = SimpleCriteriaScore(
            llm=scripted_llm([criteria(7)]),
            name="cg",
            definition="Score 0 to 10.",
            allowed_values=(0.0, 10.0),
        )
        assert metric.allowed_values == (0.0, 10.0)


class TestEquivalenceWithLegacyAtDefaultStrictness:
    """Only `strictness` changes; the judgement itself must not move."""

    @pytest.mark.asyncio
    async def test_aspect_critic_matches_legacy(self):
        import ragas.metrics as legacy_module

        for verdict in (0, 1):
            llm = scripted_llm([critic(verdict)])
            new = AspectCritic(llm=llm, name="x", definition="Is it harmful?")
            new_result = await new.ascore(user_input="q", response="r")

            old = legacy_module.AspectCritic(name="x", definition="Is it harmful?")
            old_score = old._compute_score([critic(verdict)])

            assert new_result.value == float(old_score)

    @pytest.mark.asyncio
    async def test_simple_criteria_matches_legacy(self):
        import ragas.metrics as legacy_module

        for score in (0, 3, 5):
            llm = scripted_llm([criteria(score)])
            new = SimpleCriteriaScore(llm=llm, name="cg", definition="Score 0 to 5.")
            new_result = await new.ascore(user_input="q", response="r")

            old = legacy_module.SimpleCriteriaScore(
                name="cg", definition="Score 0 to 5."
            )
            old_score = old._compute_score([criteria(score)])

            assert new_result.value == float(old_score)


class TestRunsThroughEvaluate:
    def test_single_turn(self):
        llm = scripted_llm([critic(1)])
        dataset = EvaluationDataset(
            samples=[SingleTurnSample(user_input="q", response="r")]
        )
        result = evaluate(
            dataset,
            metrics=[AspectCritic(llm=llm, name="maliciousness", definition="?")],
        )
        assert result["maliciousness"] == [1.0]

    def test_multi_turn_conversation_is_rendered_for_the_prompt(self):
        """Legacy AspectCritic is a MultiTurnMetric; the port must stay one.

        The conversation reaches the metric as text, the way the legacy
        implementation did with MultiTurnSample.pretty_repr().
        """
        seen = {}
        llm = MagicMock(spec=InstructorLLM)

        async def agenerate(prompt, model, **kwargs):
            seen["prompt"] = prompt
            return critic(1)

        llm.agenerate = agenerate

        dataset = EvaluationDataset(
            samples=[
                MultiTurnSample(
                    user_input=[
                        HumanMessage(content="is this safe?"),
                        AIMessage(content="yes it is"),
                    ]
                )
            ]
        )
        result = evaluate(
            dataset, metrics=[AspectCritic(llm=llm, name="safety", definition="?")]
        )
        assert result["safety"] == [1.0]
        assert "Human: is this safe?" in seen["prompt"]
        assert "AI: yes it is" in seen["prompt"]

    def test_simple_criteria_single_turn(self):
        llm = scripted_llm([criteria(3)])
        dataset = EvaluationDataset(
            samples=[SingleTurnSample(user_input="q", response="r")]
        )
        result = evaluate(
            dataset,
            metrics=[SimpleCriteriaScore(llm=llm, name="cg", definition="?")],
        )
        assert result["cg"] == [3.0]
