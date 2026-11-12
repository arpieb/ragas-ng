"""Answer Accuracy metric v2 - Modern implementation with dual-judge evaluation."""

import typing as t

from ragas.metrics.collections._dual_judge import DualJudgeMetric
from ragas.metrics.collections._inputs import require
from ragas.metrics.result import MetricResult

from .util import (
    AnswerAccuracyInput,
    AnswerAccuracyJudge1Prompt,
    AnswerAccuracyJudge2Prompt,
    AnswerAccuracyOutput,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class AnswerAccuracy(DualJudgeMetric):
    """
    Answer Accuracy metric using dual-judge evaluation.

    Measures answer accuracy compared to ground truth using a dual-judge system.
    This metric averages two distinct judge prompts to ensure robust evaluation.

    The metric uses NVIDIA's proven dual-judge approach:
    1. Judge 1: Direct User Answer vs Reference Answer comparison
    2. Judge 2: Swapped perspective for fairness
    3. Average both judges for final score

    Rating scale: 0 (no match), 2 (partial match), 4 (exact match)
    Final score: Average of both judges converted to 0.0-1.0 scale

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import AnswerAccuracy
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = AnswerAccuracy(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="When was Einstein born?",
        ...     response="Albert Einstein was born in 1879.",
        ...     reference="Albert Einstein was born in 1879."
        ... )
        >>> print(f"Answer Accuracy: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for dual-judge evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
        max_retries: Attempts per judge; a retry after an off-scale rating is
            a fresh sample, not a resend of the same request
    """

    judge_prompts = (AnswerAccuracyJudge1Prompt, AnswerAccuracyJudge2Prompt)
    output_model = AnswerAccuracyOutput
    ratings = (0, 2, 4)

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "answer_accuracy",
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
        self, user_input: str, response: str, reference: str
    ) -> MetricResult:
        """
        Calculate answer accuracy score using dual-judge evaluation.

        Args:
            user_input: The original question
            response: The user's answer to evaluate
            reference: The ground truth reference answer

        Returns:
            MetricResult with answer accuracy score (0.0-1.0, higher is better)
        """
        require(user_input=user_input, response=response, reference=reference)

        # Judge 2 sees the answers swapped, for fairness.
        return await self._judge_both(
            AnswerAccuracyInput(
                query=user_input, user_answer=response, reference_answer=reference
            ),
            AnswerAccuracyInput(
                query=user_input, user_answer=reference, reference_answer=response
            ),
        )
