"""Collections FactualCorrectness scoring arithmetic.

The LLM is scripted, so these tests pin how verdicts become a score rather
than what a model would say.

precision = fraction of RESPONSE claims supported by the reference
recall    = fraction of REFERENCE claims supported by the response
f-beta    = combination of those two

Before #48 recall reused the precision-direction verdicts as its numerator,
and f1 fed that mixture into ``fbeta_score``.
"""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections.factual_correctness.util import (
    ClaimDecompositionOutput,
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
)
from ragas.metrics.utils import fbeta_from_precision_recall

RESPONSE = "RESPONSE-TEXT"
REFERENCE = "REFERENCE-TEXT"


def claims_llm(response_verdicts: t.Sequence[int], reference_verdicts: t.Sequence[int]):
    """Script an LLM from per-claim verdicts for each direction.

    ``response_verdicts`` are the verdicts for the response's claims checked
    against the reference (the precision direction); ``reference_verdicts`` are
    the reference's claims checked against the response (the recall direction).
    """
    llm = MagicMock(spec=InstructorLLM)
    llm.calls = []

    def claims(prefix, n):
        return [f"{prefix} claim {i}" for i in range(n)]

    async def agenerate(prompt: str, model):
        llm.calls.append(model.__name__)
        if model is ClaimDecompositionOutput:
            # The decomposition prompt contains the text being decomposed
            # after the few-shot examples, so look at the tail.
            tail = prompt[prompt.rindex("Now perform") :]
            if RESPONSE in tail:
                return ClaimDecompositionOutput(
                    claims=claims("resp", len(response_verdicts))
                )
            return ClaimDecompositionOutput(
                claims=claims("ref", len(reference_verdicts))
            )
        if model is NLIStatementOutput:
            tail = prompt[prompt.rindex("Now perform") :]
            verdicts = response_verdicts if "resp claim" in tail else reference_verdicts
            prefix = "resp" if "resp claim" in tail else "ref"
            return NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(statement=c, reason="r", verdict=v)
                    for c, v in zip(claims(prefix, len(verdicts)), verdicts)
                ]
            )
        raise AssertionError(model)

    llm.agenerate = agenerate
    return llm


async def score(mode, response_verdicts, reference_verdicts, beta=1.0):
    llm = claims_llm(response_verdicts, reference_verdicts)
    metric = FactualCorrectness(llm=llm, mode=mode, beta=beta)
    result = await metric.ascore(response=RESPONSE, reference=REFERENCE)
    return result.value, llm.calls


@pytest.mark.asyncio
async def test_precision_is_fraction_of_response_claims_supported():
    value, calls = await score("precision", [1, 1, 1, 0], [0, 0])
    assert value == 0.75
    # precision only needs one direction
    assert calls == ["ClaimDecompositionOutput", "NLIStatementOutput"]


@pytest.mark.asyncio
async def test_recall_is_fraction_of_reference_claims_supported():
    # Response: 4 claims, all supported (precision 1.0).
    # Reference: 4 claims, only 1 covered by the response (recall 0.25).
    # Pre-fix this returned tp_precision / (tp_precision + fn_recall) = 4/7.
    value, calls = await score("recall", [1, 1, 1, 1], [1, 0, 0, 0])
    assert value == 0.25
    assert calls == ["ClaimDecompositionOutput", "NLIStatementOutput"]


@pytest.mark.asyncio
async def test_recall_does_not_depend_on_the_response_claim_count():
    # Same reference coverage, very different response verbosity.
    terse, _ = await score("recall", [1], [1, 1, 0, 0])
    verbose, _ = await score("recall", [1] * 20, [1, 1, 0, 0])
    assert terse == verbose == 0.5


@pytest.mark.asyncio
async def test_f1_combines_independent_precision_and_recall():
    # precision 1.0, recall 0.25 -> F1 = 2*1*0.25/1.25 = 0.4
    value, _ = await score("f1", [1, 1, 1, 1], [1, 0, 0, 0])
    assert value == 0.4


@pytest.mark.asyncio
async def test_fbeta_weights_recall_when_beta_above_one():
    # precision 1.0, recall 0.5, beta 2 -> 5*0.5/(4*1+0.5) = 0.5556
    value, _ = await score("f1", [1, 1], [1, 0], beta=2.0)
    assert value == 0.56


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, response_verdicts, reference_verdicts",
    [
        ("precision", [], [1]),  # response makes no claims
        ("recall", [1], []),  # reference makes no claims
        ("f1", [], [1, 0]),
        ("f1", [1], []),
    ],
)
async def test_no_claims_is_undefined_not_zero(
    mode, response_verdicts, reference_verdicts
):
    value, _ = await score(mode, response_verdicts, reference_verdicts)
    assert math.isnan(value)


@pytest.mark.asyncio
async def test_all_claims_unsupported_is_zero():
    value, _ = await score("f1", [0, 0], [0, 0])
    assert value == 0.0


@pytest.mark.parametrize(
    "p, r, beta, expected",
    [
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
        (0.5, 1.0, 1.0, 2 / 3),
        (1.0, 0.5, 0.5, 1.25 * 0.5 / (0.25 + 0.5)),
    ],
)
def test_fbeta_from_precision_recall(p, r, beta, expected):
    assert fbeta_from_precision_recall(p, r, beta) == pytest.approx(expected)


def test_fbeta_from_precision_recall_propagates_nan():
    assert math.isnan(fbeta_from_precision_recall(float("nan"), 1.0))
    assert math.isnan(fbeta_from_precision_recall(1.0, float("nan")))
