"""Collections ContextRecall classifies a fixed list of reference sentences (#52).

The judge used to split the reference itself and the score was the fraction
of whatever classifications it returned, so a dropped unsupported sentence
raised recall. The reference is now split in code, and the judge must return
one classification per sentence.
"""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import ContextRecall
from ragas.metrics.collections._text import split_sentences
from ragas.metrics.collections.context_recall.util import (
    ContextRecallClassification,
    ContextRecallOutput,
)

REFERENCE = (
    "The Eiffel Tower is located in Paris. It was completed in 1889. It is 330 m tall."
)
SENTENCES = split_sentences(REFERENCE)


def judge(*replies: t.List[t.Tuple[str, int]]):
    llm = MagicMock(spec=InstructorLLM)
    llm.prompts = []
    it = iter(replies)

    async def agenerate(prompt, model, **kwargs):
        llm.prompts.append(prompt)
        return ContextRecallOutput(
            classifications=[
                ContextRecallClassification(statement=s, reason="r", attributed=a)
                for s, a in next(it)
            ]
        )

    llm.agenerate = agenerate
    return llm


async def recall(llm):
    return (
        await ContextRecall(llm=llm).ascore(
            user_input="q", retrieved_contexts=["ctx"], reference=REFERENCE
        )
    ).value


def test_reference_is_split_into_three_sentences():
    assert SENTENCES == [
        "The Eiffel Tower is located in Paris.",
        "It was completed in 1889.",
        "It is 330 m tall.",
    ]


@pytest.mark.asyncio
async def test_prompt_lists_the_sentences_to_classify():
    llm = judge([(s, 1) for s in SENTENCES])
    assert await recall(llm) == 1.0
    tail = llm.prompts[0].split("Now perform")[-1]
    assert all(s in tail for s in SENTENCES)
    assert '"statements"' in tail


@pytest.mark.asyncio
async def test_score_is_over_all_sentences():
    llm = judge([(SENTENCES[0], 1), (SENTENCES[1], 0), (SENTENCES[2], 0)])
    assert await recall(llm) == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_dropped_unsupported_sentence_is_not_scored_over_the_subset():
    # judge keeps returning only the supported sentence. Pre-fix: 1/1 = 1.0
    dropped = [(SENTENCES[0], 1)]
    assert math.isnan(await recall(judge(dropped, dropped)))


@pytest.mark.asyncio
async def test_retry_recovers_a_complete_answer():
    full = [(SENTENCES[0], 1), (SENTENCES[1], 1), (SENTENCES[2], 0)]
    assert await recall(judge([(SENTENCES[0], 1)], full)) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Dr. Smith met Mr. Jones in the U.S. last year. They talked.", 2),
        ('He said "hello." Then he left!', 2),
        ("今天天气很好。我们去公园吧！", 2),
        ("Prices rose 3.5 percent in 2020. Inflation followed.", 2),
        ("Кошка спит. Собака лает.", 2),
        ("England", 1),
        ("   ", 0),
    ],
)
def test_split_sentences(text, expected):
    out = split_sentences(text)
    assert len(out) == expected
    assert "".join(out).replace(" ", "") == text.replace(" ", "")
