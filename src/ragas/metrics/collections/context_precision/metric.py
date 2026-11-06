"""Context Precision metrics v2 - Modern implementation with function-based prompts."""

import typing as t
from typing import List

import numpy as np

from ragas.metrics.collections._average_precision import average_precision
from ragas.metrics.collections._concurrency import gather_bounded
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


class ContextPrecisionWithReference(BaseMetric):
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
        >>> llm = llm_factory("gpt-6-luna", client=client)
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

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

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
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.prompt = ContextPrecisionPrompt()  # Initialize prompt class once

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

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
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not reference:
            raise ValueError("reference cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        # One independent judgement per context, run concurrently (#58).
        results = await gather_bounded(
            self.llm.agenerate(
                self.prompt.to_string(
                    ContextPrecisionInput(
                        question=user_input, context=context, answer=reference
                    )
                ),
                ContextPrecisionOutput,
            )
            for context in retrieved_contexts
        )
        verdicts = [result.verdict for result in results]

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(
            value=float(score), reason=_ranking_reason(retrieved_contexts, results)
        )

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts."""
        score = average_precision(verdicts)

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


class ContextPrecisionWithoutReference(BaseMetric):
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
        >>> llm = llm_factory("gpt-6-luna", client=client)
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

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

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
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.prompt = ContextPrecisionPrompt()  # Initialize prompt class once

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

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
        # Input validation
        if not user_input:
            raise ValueError("user_input cannot be empty")
        if not response:
            raise ValueError("response cannot be empty")
        if not retrieved_contexts:
            raise ValueError("retrieved_contexts cannot be empty")

        # Evaluate each retrieved context
        # One independent judgement per context, run concurrently (#58).
        results = await gather_bounded(
            self.llm.agenerate(
                self.prompt.to_string(
                    ContextPrecisionInput(
                        question=user_input, context=context, answer=response
                    )
                ),
                ContextPrecisionOutput,
            )
            for context in retrieved_contexts
        )
        verdicts = [result.verdict for result in results]

        # Calculate average precision
        score = self._calculate_average_precision(verdicts)
        return MetricResult(
            value=float(score), reason=_ranking_reason(retrieved_contexts, results)
        )

    def _calculate_average_precision(self, verdicts: List[int]) -> float:
        """Calculate average precision from binary verdicts."""
        score = average_precision(verdicts)

        if np.isnan(score):
            # Match legacy warning behavior
            import logging

            logging.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )

        return score


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
        >>> llm = llm_factory("gpt-6-luna", client=client)
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
        **kwargs,
    ):
        """Initialize ContextPrecision with the legacy default name."""
        super().__init__(llm, name="context_precision", **kwargs)


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
        >>> llm = llm_factory("gpt-6-luna", client=client)
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
        **kwargs,
    ):
        """Initialize ContextUtilization with the legacy default name."""
        super().__init__(llm, name="context_utilization", **kwargs)


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
