"""Collections SummaryScore: QA-score arithmetic and its degenerate cases."""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import SummaryScore
from ragas.metrics.collections.summary_score.util import (
    AnswersGenerated,
    ExtractedKeyphrases,
    QuestionsGenerated,
)


def scripted_llm(questions: t.List[str], answers: t.List[str]):
    llm = MagicMock(spec=InstructorLLM)
    llm.calls = []

    async def agenerate(prompt, model):
        llm.calls.append(model.__name__)
        if model is ExtractedKeyphrases:
            return ExtractedKeyphrases(keyphrases=["k1", "k2"])
        if model is QuestionsGenerated:
            return QuestionsGenerated(questions=questions)
        if model is AnswersGenerated:
            return AnswersGenerated(answers=answers)
        raise AssertionError(model)

    llm.agenerate = agenerate
    return llm


async def qa_score(questions, answers):
    llm = scripted_llm(questions, answers)
    metric = SummaryScore(llm=llm, length_penalty=False)
    result = await metric.ascore(
        reference_contexts=["A long source text."], response="A summary."
    )
    return result.value, llm.calls


@pytest.mark.asyncio
async def test_qa_score_is_fraction_of_questions_answered():
    value, _ = await qa_score(["q1", "q2", "q3", "q4"], ["1", "0", "1", " 1 "])
    assert value == 0.75


@pytest.mark.asyncio
async def test_no_questions_is_nan_and_skips_the_answer_call():
    # Before #60 this raised ZeroDivisionError.
    value, calls = await qa_score([], [])
    assert math.isnan(value)
    assert "AnswersGenerated" not in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("answers", [[], ["1"], ["1", "1", "1"]])
async def test_answer_count_mismatch_is_nan(answers):
    # Two questions asked. Scoring the subset that came back would inflate
    # (or deflate) the score, and [] used to raise ZeroDivisionError.
    value, _ = await qa_score(["q1", "q2"], answers)
    assert math.isnan(value)


@pytest.mark.asyncio
async def test_length_penalty_propagates_nan():
    llm = scripted_llm([], [])
    metric = SummaryScore(llm=llm, length_penalty=True)
    result = await metric.ascore(
        reference_contexts=["A long source text."], response="A summary."
    )
    assert math.isnan(result.value)
