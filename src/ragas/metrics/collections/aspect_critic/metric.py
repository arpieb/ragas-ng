"""AspectCritic metric - Modern collections implementation."""

import typing as t

from ragas.metrics.base import MetricOutputType
from ragas.metrics.collections._criteria import CriteriaJudgeMetric

from .util import BASE_INSTRUCTION, AspectCriticPrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class AspectCritic(CriteriaJudgeMetric):
    """
    Judge a submission against a criterion you define, in binary terms.

    You supply the criterion as free text; the LLM answers yes (1) or no (0).
    This is the most open-ended metric in ragas -- anything you can phrase as a
    yes/no question about a response can be scored with it.

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms import llm_factory
        >>> from ragas.metrics.collections import AspectCritic
        >>>
        >>> llm = llm_factory("<model>", client=AsyncOpenAI())
        >>>
        >>> metric = AspectCritic(
        ...     llm=llm,
        ...     name="maliciousness",
        ...     definition="Is the submission intended to harm, deceive, or exploit users?",
        ... )
        >>> result = await metric.ascore(
        ...     user_input="How do I bake bread?",
        ...     response="Mix flour, water, salt and yeast, then bake at 220C.",
        ... )
        >>> print(result.value)  # 0.0

    Multi-turn conversations are scored by passing the rendered exchange as
    ``user_input``; ``evaluate()`` does that rendering for you.

    Attributes:
        llm: Modern instructor-based LLM for evaluation
        definition: The criterion to judge against
        strictness: How many independent judgements to take a majority vote over
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    output_type = MetricOutputType.BINARY
    prompt_type = AspectCriticPrompt
    base_instruction = BASE_INSTRUCTION
    vote_field = "verdict"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        definition: str,
        name: str = "aspect_critic",
        strictness: int = 1,
        **kwargs,
    ):
        """
        Args:
            llm: Modern instructor-based LLM
            definition: The criterion to judge against
            name: The metric name
            strictness: Number of independent judgements to majority-vote over.
                Forced odd to avoid ties. Note this differs from the legacy
                ``AspectCritic``, where the parameter was accepted but never
                took more than one sample, so the vote was a no-op.
        """
        super().__init__(
            llm=llm,
            definition=definition,
            name=name,
            strictness=strictness,
            allowed_values=(0.0, 1.0),
            **kwargs,
        )
