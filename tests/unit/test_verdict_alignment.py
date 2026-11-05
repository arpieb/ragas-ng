"""One verdict per submitted item, or no score (#52).

Metrics that send the judge a list and score the fraction marked supported used
to divide by however many verdicts came back. A judge that dropped an
unsupported statement *raised* the score.
"""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics.collections import Faithfulness, NoiseSensitivity, TopicAdherence
from ragas.metrics.collections._verdicts import align_verdicts, judge_each
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
    StatementGeneratorOutput,
)
from ragas.metrics.collections.noise_sensitivity import util as ns
from ragas.metrics.collections.topic_adherence import util as ta


class V(t.NamedTuple):
    statement: str
    verdict: int


# --- align_verdicts ------------------------------------------------------------


def test_same_count_is_taken_in_order_even_if_paraphrased():
    got = align_verdicts(
        ["a", "b"], [V("A paraphrase", 1), V("other", 0)], lambda v: v.statement
    )
    assert [v.verdict for v in got] == [1, 0]


def test_missing_verdict_is_not_scored_over_the_subset():
    assert (
        align_verdicts(["a", "b", "c"], [V("a", 1), V("b", 1)], lambda v: v.statement)
        is None
    )


def test_extra_and_reordered_verdicts_are_matched_by_text():
    got = align_verdicts(
        ["Paris is in France.", "The sky is green"],
        [V("the sky is green.", 0), V("invented", 1), V("Paris is  in France", 1)],
        lambda v: v.statement,
    )
    assert [v.verdict for v in got] == [1, 0]


def test_without_text_only_count_matters():
    assert align_verdicts(["a", "b"], [True], None) is None
    assert align_verdicts(["a", "b"], [True, False], None) == [True, False]


@pytest.mark.asyncio
async def test_judge_each_retries_once_then_succeeds():
    replies = iter([[V("a", 1)], [V("a", 1), V("b", 0)]])
    calls = []

    async def ask():
        calls.append(1)
        return next(replies)

    got = await judge_each(["a", "b"], ask, text_of=lambda v: v.statement)
    assert [v.verdict for v in got] == [1, 0] and len(calls) == 2


@pytest.mark.asyncio
async def test_judge_each_gives_up_after_one_retry(caplog):
    calls = []

    async def ask():
        calls.append(1)
        return [V("a", 1)]

    got = await judge_each(["a", "b"], ask, text_of=lambda v: v.statement, metric="m")
    assert got is None and len(calls) == 2
    assert "m: judge returned 1 verdicts for 2 items" in caplog.text


# --- metrics --------------------------------------------------------------------


def scripted_llm(handlers: t.Dict[type, t.Callable[[str], t.Any]]):
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(prompt, model):
        return handlers[model](prompt)

    llm.agenerate = agenerate
    return llm


def faithfulness_llm(nli_replies: t.List[t.List[t.Tuple[str, int]]]):
    replies = iter(nli_replies)
    return scripted_llm(
        {
            StatementGeneratorOutput: lambda p: StatementGeneratorOutput(
                statements=["s1", "s2", "s3"]
            ),
            NLIStatementOutput: lambda p: NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(statement=s, reason="r", verdict=v)
                    for s, v in next(replies)
                ]
            ),
        }
    )


async def faithfulness(nli_replies):
    metric = Faithfulness(llm=faithfulness_llm(nli_replies))
    return (
        await metric.ascore(user_input="q", response="r", retrieved_contexts=["c"])
    ).value


@pytest.mark.asyncio
async def test_faithfulness_dropped_unsupported_statement_is_nan_not_inflated():
    # s3 (unsupported) dropped both times. Pre-fix: 2/2 = 1.0.
    dropped = [("s1", 1), ("s2", 1)]
    assert math.isnan(await faithfulness([dropped, dropped]))


@pytest.mark.asyncio
async def test_faithfulness_recovers_on_retry():
    full = [("s1", 1), ("s2", 1), ("s3", 0)]
    assert await faithfulness([[("s1", 1)], full]) == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_faithfulness_matches_reordered_extra_verdicts_by_text():
    reply = [("s3", 0), ("bonus", 1), ("s1", 1), ("s2", 1)]
    assert await faithfulness([reply]) == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_noise_sensitivity_mismatch_is_nan_not_a_ragged_matrix():
    def nli(prompt):
        # always one verdict short
        return ns.StatementFaithfulnessOutput(
            statements=[
                ns.StatementFaithfulnessAnswer(statement="s1", reason="r", verdict=1)
            ]
        )

    llm = scripted_llm(
        {
            ns.StatementGeneratorOutput: lambda p: ns.StatementGeneratorOutput(
                statements=["s1", "s2"]
            ),
            ns.StatementFaithfulnessOutput: nli,
        }
    )
    result = await NoiseSensitivity(llm=llm).ascore(
        user_input="q", response="r", reference="ref", retrieved_contexts=["c1", "c2"]
    )
    assert math.isnan(result.value)


@pytest.mark.asyncio
async def test_topic_adherence_missing_classification_is_nan_not_padded_false():
    llm = scripted_llm(
        {
            ta.TopicExtractionOutput: lambda p: ta.TopicExtractionOutput(
                topics=["t1", "t2"]
            ),
            ta.TopicRefusedOutput: lambda p: ta.TopicRefusedOutput(
                refused_to_answer=False
            ),
            ta.TopicClassificationOutput: lambda p: ta.TopicClassificationOutput(
                classifications=[True]
            ),
        }
    )
    result = await TopicAdherence(llm=llm).ascore(
        user_input=[HumanMessage(content="hi"), AIMessage(content="hello")],
        reference_topics=["t"],
    )
    assert math.isnan(result.value)
