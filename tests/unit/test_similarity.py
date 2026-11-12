"""ragas.similarity: one cosine with a zero-vector and clipping policy (#217)."""

import math
import warnings

import numpy as np
import pytest

from ragas.similarity import cosine, cosine_matrix


def test_cosine_of_parallel_orthogonal_and_opposite_vectors():
    assert cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 0], [-1, 0]) == -1.0


def test_rounding_never_leaves_minus_one_to_one():
    v = np.array([0.1] * 1536)
    assert cosine(v, v) <= 1.0
    assert cosine(v, -v) >= -1.0


def test_zero_vector_is_zero_not_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero RuntimeWarning
        assert cosine([0, 0], [1, 1]) == 0.0
        assert cosine([0, 0], [0, 0]) == 0.0
        m = cosine_matrix([[0, 0], [1, 0]], [[1, 0], [0, 0]])
    assert m.tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_matrix_shape_and_values():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    b = np.array([[3.0, 0.0], [1.0, 1.0]])
    m = cosine_matrix(a, b)
    assert m.shape == (3, 2)
    assert m[2, 1] == pytest.approx(1.0)
    assert m[1, 0] == 0.0
    assert m[0, 1] == pytest.approx(1 / math.sqrt(2))


def test_no_epsilon_so_identical_vectors_score_exactly_one():
    assert cosine([3.0, 4.0], [3.0, 4.0]) == 1.0


@pytest.mark.asyncio
async def test_semantic_similarity_zero_embedding_scores_zero():
    from unittest.mock import AsyncMock, MagicMock

    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.metrics.collections import SemanticSimilarity

    emb = MagicMock(spec=BaseRagasEmbedding)
    emb.aembed_text = AsyncMock(side_effect=[[0.0, 0.0], [1.0, 0.0]])
    result = await SemanticSimilarity(embeddings=emb).ascore(
        reference="a", response="b"
    )
    assert result.value == 0.0  # was NaN
