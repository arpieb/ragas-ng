"""Response Groundedness metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t
from typing import List

from ragas.metrics.collections._dual_judge import DualJudgeMetric
from ragas.metrics.collections._inputs import require
from ragas.metrics.result import MetricResult

from .util import (
    ResponseGroundednessInput,
    ResponseGroundednessJudge1Prompt,
    ResponseGroundednessJudge2Prompt,
    ResponseGroundednessOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ResponseGroundedness(DualJudgeMetric):
    """
    Response Groundedness metric using dual-judge evaluation.

    Evaluates how well grounded a response is in the retrieved contexts
    using a dual-judge system. This metric averages two distinct judge prompts
    to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct groundedness evaluation with structured instructions
    2. Judge 2: Alternative perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (not grounded), 1 (partially grounded), 2 (fully grounded)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ResponseGroundedness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ResponseGroundedness(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     response="Einstein was born in Germany in 1879.",
        ...     retrieved_contexts=["Albert Einstein was born in Ulm, Germany on March 14, 1879."]
        ... )
        >>> print(f"Response Groundedness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for dual-judge evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        max_retries: Attempts per judge; a retry after an off-scale rating is
            a fresh sample, not a resend of the same request
    """

    judge_prompts = (ResponseGroundednessJudge1Prompt, ResponseGroundednessJudge2Prompt)
    output_model = ResponseGroundednessOutput
    ratings = (0, 1, 2)

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "response_groundedness",
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
        self, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate response groundedness score using dual-judge evaluation.

        Args:
            response: The response to evaluate for groundedness
            retrieved_contexts: The retrieved contexts to check groundedness against

        Returns:
            MetricResult with response groundedness score (0.0-1.0, higher is better)
        """
        require(response=response, retrieved_contexts=retrieved_contexts)

        context_str = "\n".join(retrieved_contexts)
        if not response.strip() or not context_str.strip():
            return MetricResult(value=0.0)

        judge_input = ResponseGroundednessInput(response=response, context=context_str)
        return await self._judge_both(judge_input, judge_input)
