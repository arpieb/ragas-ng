"""Semantic Similarity metric."""

import typing as t

import numpy as np

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.similarity import cosine

if t.TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbedding


class SemanticSimilarity(BaseMetric):
    """
    Evaluate semantic similarity between reference and response using embeddings.

    Scores the semantic similarity of ground truth with generated answer using
    cosine similarity of embeddings. Based on the SAS paper:
    https://arxiv.org/pdf/2108.06130.pdf

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.embeddings.base import embedding_factory
        >>> from ragas.metrics.collections import SemanticSimilarity
        >>>
        >>> # Setup embeddings
        >>> client = AsyncOpenAI()
        >>> embeddings = embedding_factory("openai", model="text-embedding-ada-002", client=client, interface="modern")
        >>>
        >>> # Create metric instance
        >>> metric = SemanticSimilarity(embeddings=embeddings)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     reference="Paris is the capital of France.",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> # Batch evaluation
        >>> results = await metric.abatch_score([
        ...     {"reference": "Text 1", "response": "Response 1"},
        ...     {"reference": "Text 2", "response": "Response 2"},
        ... ])

    Attributes:
        embeddings: Modern embeddings model with embed_text() method
        name: The metric name
        threshold: Optional threshold for binary classification
        allowed_values: Score range (0.0 to 1.0)
    """

    required_components = ("embeddings",)

    embeddings: "BaseRagasEmbedding"

    def __init__(
        self,
        embeddings: "BaseRagasEmbedding",
        name: str = "semantic_similarity",
        threshold: t.Optional[float] = None,
        **kwargs,
    ):
        """Initialize SemanticSimilarity metric with required embeddings."""
        self.threshold = threshold

        super().__init__(embeddings=embeddings, name=name, **kwargs)

    async def ascore(self, reference: str, response: str) -> MetricResult:
        """
        Calculate semantic similarity score asynchronously.

        Components are guaranteed to be validated and non-None by the base class.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with similarity score (0.0-1.0)
        """
        reference = reference or " "
        response = response or " "

        # Async API: the sync embed_text blocked the event loop inside ascore.
        embedding_1 = np.array(await self.embeddings.aembed_text(reference))
        embedding_2 = np.array(await self.embeddings.aembed_text(response))

        similarity = np.array([cosine(embedding_1, embedding_2)])
        # Cosine similarity spans [-1, 1]; the metric is defined on [0, 1], and
        # unrelated texts can come out slightly negative with some embedding
        # models (#107). Same treatment as AnswerRelevancy (#56).
        score = np.clip(similarity.flatten(), 0.0, 1.0)

        assert isinstance(score, np.ndarray), "Expects ndarray"
        if self.threshold:
            score = score >= self.threshold

        return MetricResult(value=float(score.item()))
