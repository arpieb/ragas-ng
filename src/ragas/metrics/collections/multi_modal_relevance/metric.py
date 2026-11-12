"""MultiModalRelevance metric - Collections implementation for multimodal relevance evaluation."""

import typing as t
from typing import List

from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.collections.image_sources import ImageSourcePolicy
from ragas.metrics.result import MetricResult

from .util import (
    MULTIMODAL_RELEVANCE_INSTRUCTION,
    MultiModalRelevanceOutput,
    build_multimodal_relevance_message_content,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class MultiModalRelevance(BaseMetric):
    """
    MultiModalRelevance metric for evaluating response relevance against
    both visual and textual context.

    Measures whether a response appropriately addresses the user's question
    and is in line with the retrieved context, which can include both text
    and images.

    The metric returns a binary score:
    - 1.0 if the response is relevant to the question and contexts
    - 0.0 if the response is not relevant

    This implementation uses modern instructor LLMs with vision capabilities
    for multimodal evaluation.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import MultiModalRelevance
        >>>
        >>> # Setup dependencies (use a vision-capable model)
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)  # Vision-capable model
        >>>
        >>> # Create metric instance
        >>> metric = MultiModalRelevance(llm=llm)
        >>>
        >>> # Single evaluation with image context
        >>> result = await metric.ascore(
        ...     user_input="What type of vehicle is shown in the image?",
        ...     response="The image shows a Tesla Model X, which is an electric SUV.",
        ...     retrieved_contexts=["path/to/tesla_image.jpg", "Tesla makes electric vehicles."]
        ... )
        >>> print(f"Relevance Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM with vision capabilities
        name: The metric name
        allowed_values: Score range (0.0 or 1.0)

    Note:
        This metric requires an LLM that accepts image input to evaluate image
        contexts. Check the model's page on your provider's site.
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"
    image_policy: ImageSourcePolicy

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "multi_modal_relevance",
        image_policy: t.Optional[ImageSourcePolicy] = None,
        **kwargs,
    ):
        """
        Initialize MultiModalRelevance metric with required components.

        Args:
            llm: Modern instructor-based LLM with vision capabilities
            name: The metric name
            image_policy: Where context images may be loaded from. The default
                fetches public http(s) URLs and data URIs only, and reads no
                local files; pass ``ImageSourcePolicy(local_dir=...)`` to read
                images from a directory. See ``ImageSourcePolicy``.
        """
        self.llm = llm
        self.image_policy = image_policy or ImageSourcePolicy()
        super().__init__(name=name, **kwargs)

    async def ascore(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
    ) -> MetricResult:
        """
        Calculate multimodal relevance score.

        Args:
            user_input: The user's question or input
            response: The response to evaluate for relevance
            retrieved_contexts: List of retrieved contexts (text strings or
                              image paths/URLs/base64 data)

        Returns:
            MetricResult with relevance score (0.0 or 1.0)

        Raises:
            ValueError: If user_input, response, or retrieved_contexts is missing
        """
        # Input validation
        if not user_input:
            raise ValueError(
                "user_input is missing. Please provide a question to evaluate against."
            )
        if not response:
            raise ValueError(
                "response is missing. Please provide a response to evaluate."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please provide contexts to check against."
            )

        # Build multimodal message content
        message_content = build_multimodal_relevance_message_content(
            instruction=MULTIMODAL_RELEVANCE_INSTRUCTION,
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            image_policy=self.image_policy,
        )

        # Call the LLM with multimodal content
        result = await self._evaluate_relevance(message_content)

        # Return score based on relevance verdict
        score = 1.0 if result.relevant else 0.0
        return MetricResult(value=score, reason=result.reason)

    async def _evaluate_relevance(
        self,
        message_content: List[t.Dict[str, t.Any]],
    ) -> MultiModalRelevanceOutput:
        """
        Evaluate relevance using the LLM with multimodal content.

        Args:
            message_content: List of content blocks (text and images)

        Returns:
            MultiModalRelevanceOutput with verdict and reason
        """
        # One user message with text and image parts. agenerate sends it
        # through the LLM layer: retries, cache, rate limit, tracing, usage,
        # system_prompt and provider parameter mapping, with sync or async
        # clients (#214).
        messages = [{"role": "user", "content": message_content}]
        return await self.llm.agenerate(messages, MultiModalRelevanceOutput)
