"""Collections AnswerCorrectness factuality edge cases (#136)."""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    StatementGeneratorOutput,
    StatementsWithReason,
)


def scripted_llm(response_statements: t.List[str], reference_statements: t.List[str]):
    llm = MagicMock(spec=InstructorLLM)
    llm.calls = []

    async def agenerate(prompt, model, **kwargs):
        llm.calls.append(model.__name__)
        if model is StatementGeneratorOutput:
            tail = prompt.split("Now perform")[-1]
            stmts = response_statements if "RESPONSE" in tail else reference_statements
            return StatementGeneratorOutput(statements=stmts)
        # one TP, one FP, one FN
        s = StatementsWithReason(statement="s", reason="r")
        return ClassificationWithReason(TP=[s], FP=[s], FN=[s])

    llm.agenerate = agenerate
    return llm


async def score(resp_stmts, ref_stmts):
    llm = scripted_llm(resp_stmts, ref_stmts)
    metric = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])
    result = await metric.ascore(user_input="q", response="RESPONSE", reference="REF")
    return result.value, llm.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resp, ref", [([], ["Einstein was born in 1879."]), (["x"], []), ([], [])]
)
async def test_no_statements_is_undefined_not_a_perfect_match(resp, ref):
    value, calls = await score(resp, ref)
    # was 1.0 ("assume perfect match"), so evasive answers got full credit
    assert math.isnan(value)
    assert "ClassificationWithReason" not in calls  # nothing to classify


@pytest.mark.asyncio
async def test_factuality_is_fbeta_of_the_classification():
    value, _ = await score(["a"], ["b"])
    assert value == pytest.approx(0.5)  # TP=FP=FN=1 -> P=R=0.5 -> F1=0.5
