"""MultiModalFaithfulness metric - Collections implementation for multimodal faithfulness evaluation."""

import typing as t
from typing import List

from ragas.metrics.collections._inputs import require
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.collections.image_sources import ImageSourcePolicy
from ragas.metrics.result import MetricResult

from .util import (
    MULTIMODAL_FAITHFULNESS_INSTRUCTION,
    MultiModalFaithfulnessOutput,
    build_multimodal_message_content,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class MultiModalFaithfulness(BaseMetric):
    """
    MultiModalFaithfulness metric for evaluating response faithfulness against
    both visual and textual context.

    Measures how factually consistent a response is with the retrieved context,
    which can include both text and images. A response is considered faithful
    if all its claims can be supported by the provided contexts.

    The metric returns a binary score:
    - 1.0 if the response is faithful to the contexts
    - 0.0 if the response is not faithful

    This implementation uses modern instructor LLMs with vision capabilities
    for multimodal evaluation.

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import MultiModalFaithfulness
        >>>
        >>> # Setup dependencies (use a vision-capable model)
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)  # Vision-capable model
        >>>
        >>> # Create metric instance
        >>> metric = MultiModalFaithfulness(llm=llm)
        >>>
        >>> # Single evaluation with image context
        >>> result = await metric.ascore(
        ...     response="The Tesla Model X is an electric SUV.",
        ...     retrieved_contexts=["path/to/tesla_image.jpg", "Tesla makes electric vehicles."]
        ... )
        >>> print(f"Faithfulness Score: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM with vision capabilities
        name: The metric name
        allowed_values: Score range (0.0 or 1.0)

    Note:
        This metric requires an LLM that accepts image input to evaluate image
        contexts. Check the model's page on your provider's site.
    """

    required_components = ("llm",)

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"
    image_policy: ImageSourcePolicy

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "multi_modal_faithfulness",
        image_policy: t.Optional[ImageSourcePolicy] = None,
        **kwargs,
    ):
        """
        Initialize MultiModalFaithfulness metric with required components.

        Args:
            llm: Modern instructor-based LLM with vision capabilities
            name: The metric name
            image_policy: Where context images may be loaded from. The default
                fetches public http(s) URLs and data URIs only, and reads no
                local files; pass ``ImageSourcePolicy(local_dir=...)`` to read
                images from a directory. See ``ImageSourcePolicy``.
        """
        self.image_policy = image_policy or ImageSourcePolicy()
        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self,
        response: str,
        retrieved_contexts: List[str],
    ) -> MetricResult:
        """
        Calculate multimodal faithfulness score.

        Args:
            response: The response to evaluate for faithfulness
            retrieved_contexts: List of retrieved contexts (text strings or
                              image paths/URLs/base64 data)

        Returns:
            MetricResult with faithfulness score (0.0 or 1.0)

        Raises:
            ValueError: If response or retrieved_contexts is missing
        """
        require(response=response, retrieved_contexts=retrieved_contexts)

        # Build multimodal message content
        message_content = build_multimodal_message_content(
            instruction=MULTIMODAL_FAITHFULNESS_INSTRUCTION,
            response=response,
            retrieved_contexts=retrieved_contexts,
            image_policy=self.image_policy,
        )

        # Call the LLM with multimodal content
        result = await self._evaluate_faithfulness(message_content)

        # Return score based on faithfulness verdict
        score = 1.0 if result.faithful else 0.0
        return MetricResult(value=score, reason=result.reason)

    async def _evaluate_faithfulness(
        self,
        message_content: List[t.Dict[str, t.Any]],
    ) -> MultiModalFaithfulnessOutput:
        """
        Evaluate faithfulness using the LLM with multimodal content.

        Args:
            message_content: List of content blocks (text and images)

        Returns:
            MultiModalFaithfulnessOutput with verdict and reason
        """
        # One user message with text and image parts. agenerate sends it
        # through the LLM layer: retries, cache, rate limit, tracing, usage,
        # system_prompt and provider parameter mapping, with sync or async
        # clients (#214).
        messages = [{"role": "user", "content": message_content}]
        return await self.llm.agenerate(messages, MultiModalFaithfulnessOutput)
