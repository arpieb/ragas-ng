"""Context Relevance metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t
from typing import List

from ragas.metrics.collections._dual_judge import DualJudgeMetric
from ragas.metrics.collections._inputs import require
from ragas.metrics.result import MetricResult

from .util import (
    ContextRelevanceInput,
    ContextRelevanceJudge1Prompt,
    ContextRelevanceJudge2Prompt,
    ContextRelevanceOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ContextRelevance(DualJudgeMetric):
    """
    Context Relevance metric using dual-judge evaluation.

    Evaluates whether the retrieved contexts are pertinent to the user input
    using a dual-judge system. This metric averages two distinct judge prompts
    to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct context relevance evaluation
    2. Judge 2: Alternative perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (not relevant), 1 (partially relevant), 2 (fully relevant)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextRelevance
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextRelevance(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="When was Einstein born?",
        ...     retrieved_contexts=["Albert Einstein was born March 14, 1879."]
        ... )
        >>> print(f"Context Relevance: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for dual-judge evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        max_retries: Attempts per judge; a retry after an off-scale rating is
            a fresh sample, not a resend of the same request
    """

    judge_prompts = (ContextRelevanceJudge1Prompt, ContextRelevanceJudge2Prompt)
    output_model = ContextRelevanceOutput
    ratings = (0, 1, 2)

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_relevance",
        max_retries: int = 5,
        **kwargs,
    ):
        """
        Args:
            llm: Modern instructor-based LLM for dual-judge evaluation
            name: The metric name
            max_retries: Attempts per judge (see ``DualJudgeMetric``)
        """
        super().__init__(llm=llm, name=name, max_retries=max_retries, **kwargs)

    async def ascore(
        self, user_input: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context relevance score using dual-judge evaluation.

        Args:
            user_input: The original question
            retrieved_contexts: The retrieved contexts to evaluate for relevance

        Returns:
            MetricResult with context relevance score (0.0-1.0, higher is better)
        """
        require(user_input=user_input, retrieved_contexts=retrieved_contexts)

        context_str = "\n".join(retrieved_contexts)
        question, context = user_input.strip(), context_str.strip()
        # Empty input, or a context that only repeats (part of) the question.
        if not question or not context or context in question:
            return MetricResult(value=0.0)

        judge_input = ContextRelevanceInput(user_input=user_input, context=context_str)
        return await self._judge_both(judge_input, judge_input)
