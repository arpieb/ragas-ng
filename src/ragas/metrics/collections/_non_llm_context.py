"""Context precision/recall judged by string distance rather than an LLM.

These compare retrieved contexts against reference contexts with a string
distance measure, so they need no LLM -- useful when reference contexts are
available verbatim and an LLM judgement would be both slower and less
reproducible.
"""

import typing as t

import numpy as np

from ragas.metrics.collections._average_precision import average_precision
from ragas.metrics.collections._string import DistanceMeasure, NonLLMStringSimilarity
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


async def _best_match(
    similarity: NonLLMStringSimilarity, target: str, candidates: t.List[str]
) -> float:
    """Highest similarity between ``target`` and any candidate."""
    scores = [
        (await similarity.ascore(reference=target, response=candidate)).value
        for candidate in candidates
    ]
    return max(scores) if scores else 0.0


class NonLLMContextPrecisionWithReference(BaseMetric):
    """
    Context precision judged by string distance against reference contexts.

    Each retrieved context is matched against its closest reference context; a
    match at or above ``threshold`` counts as relevant. The verdicts are then
    scored with average precision, so relevant contexts ranked earlier score
    higher.

    Usage:
        >>> from ragas.metrics.collections import NonLLMContextPrecisionWithReference
        >>>
        >>> metric = NonLLMContextPrecisionWithReference()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_contexts=["Paris is the capital of France."],
        ...     reference_contexts=["Paris is the capital of France."],
        ... )
        >>> print(result.value)  # 1.0

    Attributes:
        name: The metric name
        distance_measure: Which string distance to use (default Levenshtein)
        threshold: Similarity at or above which a context counts as relevant
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "non_llm_context_precision_with_reference",
        distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN,
        threshold: float = 0.5,
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)
        self.threshold = threshold
        self.distance_measure = distance_measure
        self._similarity = NonLLMStringSimilarity(distance_measure=distance_measure)

    async def ascore(
        self,
        retrieved_contexts: t.List[str],
        reference_contexts: t.List[str],
    ) -> MetricResult:
        """
        Args:
            retrieved_contexts: Contexts the retriever returned, in rank order
            reference_contexts: Contexts known to be relevant

        Returns:
            MetricResult with the average precision of the retrieved ranking.
        """
        scores = [
            await _best_match(self._similarity, rc, reference_contexts)
            for rc in retrieved_contexts
        ]
        # A context at or above the threshold counts (same rule as recall, #64).
        verdicts = [1 if score >= self.threshold else 0 for score in scores]
        return MetricResult(value=average_precision(verdicts))


class NonLLMContextRecall(BaseMetric):
    """
    Context recall judged by string distance against reference contexts.

    Each reference context is matched against its closest retrieved context; a
    match at or above ``threshold`` counts as recalled. The score is the proportion of
    reference contexts recalled.

    Usage:
        >>> from ragas.metrics.collections import NonLLMContextRecall
        >>>
        >>> metric = NonLLMContextRecall()
        >>>
        >>> result = await metric.ascore(
        ...     retrieved_contexts=["Paris is the capital of France."],
        ...     reference_contexts=["Paris is the capital of France."],
        ... )
        >>> print(result.value)  # 1.0

    Attributes:
        name: The metric name
        distance_measure: Which string distance to use (default Levenshtein)
        threshold: Similarity at or above which a context counts as recalled
        allowed_values: Score range (0.0 to 1.0)
    """

    def __init__(
        self,
        name: str = "non_llm_context_recall",
        distance_measure: DistanceMeasure = DistanceMeasure.LEVENSHTEIN,
        threshold: float = 0.5,
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)
        self.threshold = threshold
        self.distance_measure = distance_measure
        self._similarity = NonLLMStringSimilarity(distance_measure=distance_measure)

    async def ascore(
        self,
        retrieved_contexts: t.List[str],
        reference_contexts: t.List[str],
    ) -> MetricResult:
        """
        Args:
            retrieved_contexts: Contexts the retriever returned
            reference_contexts: Contexts known to be relevant

        Returns:
            MetricResult with the proportion of reference contexts recalled,
            or NaN when there are no reference contexts.
        """
        scores = [
            await _best_match(self._similarity, rc, retrieved_contexts)
            for rc in reference_contexts
        ]
        # `>=`, matching NonLLMContextPrecisionWithReference. Legacy used `>`
        # here, so an exact match at threshold=1.0 counted for precision but not
        # for recall (#64).
        verdicts = [1 if score >= self.threshold else 0 for score in scores]
        if not verdicts:
            return MetricResult(value=float(np.nan))
        return MetricResult(value=sum(verdicts) / len(verdicts))
