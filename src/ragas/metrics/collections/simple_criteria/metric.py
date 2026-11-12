"""SimpleCriteriaScore metric - Modern collections implementation."""

import typing as t

from ragas.metrics.base import MetricOutputType
from ragas.metrics.collections._criteria import CriteriaJudgeMetric

from .util import BASE_INSTRUCTION, SimpleCriteriaPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class SimpleCriteriaScore(CriteriaJudgeMetric):
    """
    Score a submission against a criterion you define, numerically.

    The sibling of :class:`~ragas.metrics.collections.AspectCritic`: same
    free-text criterion, but the LLM returns a score rather than a yes/no
    verdict. State the range you want inside the definition -- the metric does
    not impose one.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms import llm_factory
        >>> from ragas.metrics.collections import SimpleCriteriaScore
        >>>
        >>> llm = llm_factory("<model>", client=AsyncOpenAI())
        >>>
        >>> metric = SimpleCriteriaScore(
        ...     llm=llm,
        ...     name="course_grained_score",
        ...     definition="Score 0 to 5 by similarity to the reference.",
        ...     allowed_values=(0.0, 5.0),
        ... )
        >>> result = await metric.ascore(
        ...     user_input="Where is the Eiffel Tower?",
        ...     response="The Eiffel Tower is in Paris.",
        ...     reference="The Eiffel Tower is located in Paris.",
        ... )
        >>> print(result.value)  # 5.0

    Attributes:
        llm: Modern instructor-based LLM for evaluation
        definition: The criterion to score against
        strictness: How many independent scores to take a majority vote over
        name: The metric name
        allowed_values: Score range; widen it to match your definition
    """

    output_type = MetricOutputType.DISCRETE
    prompt_type = SimpleCriteriaPrompt
    base_instruction = BASE_INSTRUCTION
    vote_field = "score"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        definition: str,
        name: str = "simple_criteria_score",
        strictness: int = 1,
        allowed_values: t.Tuple[float, float] = (0.0, 5.0),
        **kwargs,
    ):
        """
        Args:
            llm: Modern instructor-based LLM
            definition: The criterion to score against
            name: The metric name
            strictness: Number of independent scores to majority-vote over.
                Forced odd to avoid ties. Note this differs from the legacy
                ``SimpleCriteriaScore``, where the parameter was accepted but
                never took more than one sample, so the vote was a no-op.
            allowed_values: The score range your definition asks for. Defaults
                to 0-5; the legacy metric imposed no range at all.
        """
        super().__init__(
            llm=llm,
            definition=definition,
            name=name,
            strictness=strictness,
            allowed_values=allowed_values,
            **kwargs,
        )
