"""Legacy-implementation fixes, reached when the short-import hand-off falls
back (legacy-only LLM wrapper, customised prompts, legacy-only options), and
by direct imports from the private modules."""

from __future__ import annotations

import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import BaseRagasLLM


def capturing_legacy_llm(reply: str = "4"):
    """BaseRagasLLM whose agenerate_text records prompts and returns `reply`."""
    llm = MagicMock(spec=BaseRagasLLM)
    llm.prompts: t.List[str] = []

    async def agenerate_text(prompt, **kwargs):
        llm.prompts.append(prompt.to_string())
        gen = MagicMock()
        gen.text = reply
        result = MagicMock()
        result.generations = [[gen]]
        return result

    llm.agenerate_text = agenerate_text
    return llm


@pytest.mark.asyncio
async def test_answer_accuracy_judge_two_swaps_roles():
    from ragas.metrics._nv_metrics import AnswerAccuracy

    llm = capturing_legacy_llm()
    metric = AnswerAccuracy(llm=llm)
    score = await metric._single_turn_ascore(
        SingleTurnSample(user_input="Q?", response="RESPONSE", reference="REFERENCE"),
        None,
    )
    assert score == 1.0
    judge1, judge2 = llm.prompts
    assert "User Answer: RESPONSE" in judge1 and "Reference Answer: REFERENCE" in judge1
    # judge 2: the reference is rated *as* the user answer (#63)
    assert "User Answer: REFERENCE" in judge2 and "Reference Answer: RESPONSE" in judge2


@pytest.mark.asyncio
async def test_instance_rubrics_without_user_input_has_no_literal_none():
    from ragas.metrics._instance_specific_rubrics import InstanceRubrics

    metric = InstanceRubrics(llm=MagicMock(spec=BaseRagasLLM))
    captured = {}

    async def generate(data, llm, callbacks=None):
        captured["input"] = data
        out = MagicMock()
        out.score = 3
        return out

    metric.single_turn_prompt.generate = generate  # type: ignore[method-assign]
    await metric._ascore(
        {
            "retrieved_contexts": ["ctx one", "ctx two"],
            "response": "r",
            "rubrics": {"score1_description": "bad"},
        },
        None,
    )
    assert "None" not in captured["input"].user_input
    assert captured["input"].user_input == "Answer using context: ctx one\nctx two"


@pytest.mark.parametrize(
    "cls_name", ["AgentGoalAccuracyWithReference", "AgentGoalAccuracyWithoutReference"]
)
def test_agent_goal_accuracy_output_type_is_binary(cls_name):
    # train() refuses to run without an output_type (#66)
    import ragas.metrics._goal_accuracy as ga
    from ragas.metrics.base import MetricOutputType

    assert getattr(ga, cls_name)().output_type == MetricOutputType.BINARY
