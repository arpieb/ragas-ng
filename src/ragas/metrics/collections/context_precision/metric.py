"""Context Precision metrics v2 - Modern implementation with function-based prompts."""

import typing as t
from typing import List

from ragas.metrics.collections._average_precision import average_precision
from ragas.metrics.collections._concurrency import gather_bounded
from ragas.metrics.collections._inputs import require
from ragas.metrics.collections._reasons import count_phrase, itemized
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult

from .util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
    ContextPrecisionPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class _ContextPrecisionBase(BaseMetric):
    """Judge each retrieved context against an answer; score average precision.

    The with- and without-reference variants differ only in which answer the
    judge sees (the reference or the response).
    """

    required_components = ("llm",)

    llm: "InstructorBaseRagasLLM"

    def __init__(self, llm: "InstructorBaseRagasLLM", name: str, **kwargs):
        self.prompt = ContextPrecisionPrompt()
        super().__init__(llm=llm, name=name, **kwargs)

    async def _score(
        self, user_input: str, answer: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        # One independent judgement per context, run concurrently (#58).
        results = await gather_bounded(
            self.llm.agenerate(
                self.prompt.to_string(
                    ContextPrecisionInput(
                        question=user_input, context=context, answer=answer
                    )
                ),
                ContextPrecisionOutput,
            )
            for context in retrieved_contexts
        )
        score = average_precision([result.verdict for result in results])
        return MetricResult(
            value=float(score), reason=_ranking_reason(retrieved_contexts, results)
        )


class ContextPrecisionWithReference(_ContextPrecisionBase):
    """
    Modern v2 implementation of context precision with reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against a reference answer. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision_with_reference",
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
        """
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self, user_input: str, reference: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using reference.

        Args:
            user_input: The question being asked
            reference: The reference answer to compare against
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        require(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        return await self._score(user_input, reference, retrieved_contexts)


class ContextPrecisionWithoutReference(_ContextPrecisionBase):
    """
    Modern v2 implementation of context precision without reference.

    Evaluates whether retrieved contexts are useful for answering a question by comparing
    each context against the generated response. The metric calculates average precision
    based on the usefulness verdicts from an LLM.

    This implementation uses modern instructor LLMs with structured output.
    Only supports modern components - legacy wrappers are rejected with clear error messages.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecisionWithoutReference
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = ContextPrecisionWithoutReference(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France.", "Berlin is the capital of Germany."]
        ... )
        >>> print(f"Context Precision: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for context evaluation
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision_without_reference",
        **kwargs,
    ):
        """
        Initialize ContextPrecisionWithoutReference metric with required components.

        Args:
            llm: Modern instructor-based LLM for context evaluation
            name: The metric name
        """
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self, user_input: str, response: str, retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Calculate context precision score using response.

        Args:
            user_input: The question being asked
            response: The response that was generated
            retrieved_contexts: The retrieved contexts to evaluate

        Returns:
            MetricResult with context precision score (0.0-1.0, higher is better)
        """
        require(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        return await self._score(user_input, response, retrieved_contexts)


class ContextPrecision(ContextPrecisionWithReference):
    """
    Modern v2 wrapper for ContextPrecisionWithReference with shorter name.

    This is a simple wrapper that provides the legacy "context_precision" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextPrecision
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithReference)
        >>> metric = ContextPrecision(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     reference="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_precision",
        **kwargs,
    ):
        """Initialize ContextPrecision with the legacy default name."""
        super().__init__(llm, name=name, **kwargs)


class ContextUtilization(ContextPrecisionWithoutReference):
    """
    Modern v2 wrapper for ContextPrecisionWithoutReference with shorter name.

    This is a simple wrapper that provides the legacy "context_utilization" name
    while using the modern V2 implementation underneath.

    Usage:
        >>> import openai
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import ContextUtilization
        >>>
        >>> # Setup dependencies
        >>> client = openai.AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance (same as ContextPrecisionWithoutReference)
        >>> metric = ContextUtilization(llm=llm)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     user_input="What is the capital of France?",
        ...     response="Paris is the capital of France.",
        ...     retrieved_contexts=["Paris is the capital and largest city of France."]
        ... )
    """

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "context_utilization",
        **kwargs,
    ):
        """Initialize ContextUtilization with the legacy default name."""
        super().__init__(llm, name=name, **kwargs)


def _ranking_reason(contexts: List[str], results: List[ContextPrecisionOutput]) -> str:
    """Per-rank verdicts: precision depends on where useful contexts appear."""

    def preview(text: str, limit: int = 80) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "\u2026"

    useful = sum(1 for r in results if r.verdict)
    return itemized(
        f"{count_phrase(useful, len(results), 'retrieved context')} judged useful; "
        f"average precision rewards useful contexts ranked first.",
        (
            (f"#{rank} {preview(ctx)!r}", bool(r.verdict), r.reason)
            for rank, (ctx, r) in enumerate(zip(contexts, results), start=1)
        ),
        yes="useful",
        no="not useful",
    )
