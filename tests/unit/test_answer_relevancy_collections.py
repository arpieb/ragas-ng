"""Collections AnswerRelevancy score range."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy
from ragas.metrics.collections.answer_relevancy.util import AnswerRelevanceOutput


def make_metric(question_vec, generated_vec, noncommittal=0):
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(prompt, model, **kwargs):
        return AnswerRelevanceOutput(question="generated?", noncommittal=noncommittal)

    llm.agenerate = agenerate

    emb = MagicMock(spec=BaseRagasEmbedding)

    async def aembed_text(text, **kw):
        return list(question_vec)

    async def aembed_texts(texts, **kw):
        return [list(generated_vec) for _ in texts]

    emb.aembed_text = aembed_text
    emb.aembed_texts = aembed_texts
    return AnswerRelevancy(llm=llm, embeddings=emb, strictness=3)


@pytest.mark.asyncio
async def test_opposed_embeddings_clip_to_zero():
    # cosine = -1 before clipping; the metric is documented as 0-1 (#56)
    metric = make_metric([1.0, 0.0], [-1.0, 0.0])
    result = await metric.ascore(user_input="q", response="r")
    assert result.value == 0.0


@pytest.mark.asyncio
async def test_identical_embeddings_score_one():
    metric = make_metric([0.3, 0.4], [0.3, 0.4])
    result = await metric.ascore(user_input="q", response="r")
    assert result.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_partial_similarity_is_cosine():
    metric = make_metric([1.0, 0.0], [1.0, 1.0])
    result = await metric.ascore(user_input="q", response="r")
    assert result.value == pytest.approx(1 / np.sqrt(2))


@pytest.mark.asyncio
async def test_noncommittal_scores_zero():
    metric = make_metric([1.0, 0.0], [1.0, 0.0], noncommittal=1)
    result = await metric.ascore(user_input="q", response="r")
    assert result.value == 0.0
