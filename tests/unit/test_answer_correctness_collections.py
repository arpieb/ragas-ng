"""Collections AnswerCorrectness factuality edge cases (#136)."""

from __future__ import annotations

import math
import typing as t
from unittest.mock import MagicMock

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    StatementsWithReason,
)
from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput


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


def fixed_embeddings(vectors: t.Dict[str, t.List[float]]):
    emb = MagicMock(spec=BaseRagasEmbedding)

    async def aembed_text(text, **kw):
        return vectors[text]

    emb.aembed_text = aembed_text
    return emb


@pytest.mark.asyncio
async def test_negative_cosine_is_clipped_so_the_score_stays_in_range():
    # Opposite embeddings: cosine -1. Unclipped, the similarity term was -1 and
    # 0.5 * 0.5 + 0.5 * -1 = -0.25. Clipped like SemanticSimilarity (#107) it is
    # 0, giving 0.5 * 0.5 = 0.25.
    emb = fixed_embeddings({"RESPONSE": [1.0, 0.0], "REF": [-1.0, 0.0]})
    metric = AnswerCorrectness(
        llm=scripted_llm(["a"], ["b"]), embeddings=emb, weights=[0.5, 0.5]
    )
    result = await metric.ascore(user_input="q", response="RESPONSE", reference="REF")
    assert result.value == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_positive_cosine_is_unchanged():
    emb = fixed_embeddings({"RESPONSE": [1.0, 0.0], "REF": [1.0, 1.0]})
    metric = AnswerCorrectness(
        llm=scripted_llm(["a"], ["b"]), embeddings=emb, weights=[0.5, 0.5]
    )
    result = await metric.ascore(user_input="q", response="RESPONSE", reference="REF")
    assert result.value == pytest.approx(0.5 * 0.5 + 0.5 * math.sqrt(0.5))


@pytest.mark.parametrize("weights", [[1.0, 0.0], [0.75, 0.25]])
def test_non_embedding_object_is_rejected(weights):
    # _validate_embeddings was `pass`, so any object was accepted
    with pytest.raises(ValueError, match="modern embeddings"):
        AnswerCorrectness(
            llm=scripted_llm([], []),
            embeddings=t.cast(t.Any, object()),
            weights=weights,
        )


def test_no_embeddings_is_fine_for_pure_factuality():
    metric = AnswerCorrectness(llm=scripted_llm([], []), weights=[1.0, 0.0])
    assert metric.embeddings is None
