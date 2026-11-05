"""Collections SemanticSimilarity: score range and async embedding (#107)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import SemanticSimilarity


def embeddings(vectors):
    emb = MagicMock(spec=BaseRagasEmbedding)

    def sync(*a, **k):
        raise AssertionError("sync embed_text blocks the event loop")

    async def aembed_text(text, **k):
        return vectors[text]

    emb.embed_text = sync
    emb.aembed_text = aembed_text
    return emb


@pytest.mark.asyncio
async def test_opposed_embeddings_clip_to_zero():
    metric = SemanticSimilarity(
        embeddings=embeddings({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
    )
    assert (await metric.ascore(reference="a", response="b")).value == 0.0


@pytest.mark.asyncio
async def test_cosine_in_range_is_unchanged():
    metric = SemanticSimilarity(
        embeddings=embeddings({"a": [1.0, 0.0], "b": [1.0, 1.0]})
    )
    assert (await metric.ascore(reference="a", response="b")).value == pytest.approx(
        2**-0.5
    )


@pytest.mark.asyncio
async def test_threshold_gives_binary_score():
    vecs = {"a": [1.0, 0.0], "b": [1.0, 1.0]}
    assert (
        await SemanticSimilarity(embeddings=embeddings(vecs), threshold=0.5).ascore(
            reference="a", response="b"
        )
    ).value == 1.0
    assert (
        await SemanticSimilarity(embeddings=embeddings(vecs), threshold=0.9).ascore(
            reference="a", response="b"
        )
    ).value == 0.0
