"""TopicAdherence metric - Modern collections implementation."""

import typing as t
from typing import List, Literal, Union

import numpy as np

from ragas.messages import AIMessage, HumanMessage, ToolMessage
from ragas.metrics.collections._concurrency import gather_bounded
from ragas.metrics.collections._verdicts import judge_each
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.utils import fbeta_from_precision_recall, precision_recall

from .util import (
    TopicClassificationInput,
    TopicClassificationOutput,
    TopicClassificationPrompt,
    TopicExtractionInput,
    TopicExtractionOutput,
    TopicExtractionPrompt,
    TopicRefusedInput,
    TopicRefusedOutput,
    TopicRefusedPrompt,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class TopicAdherence(BaseMetric):
    """
    Measures how well an AI system adheres to predefined topics during conversations.

    AI systems deployed in real-world applications are expected to stay within domains
    of interest. This metric evaluates the ability of the AI to only answer queries
    related to predefined topics and refuse queries outside those topics.

    The metric works by:
    1. Extracting topics discussed in the conversation
    2. Checking which topics the AI answered vs refused
    3. Classifying if each topic falls within the reference topics
    4. Computing precision, recall, or F1 based on these classifications

    Score interpretation:
    - Precision: Ratio of answered topics that are within reference topics
    - Recall: Ratio of reference-aligned topics that were answered (not refused)
    - F1: Harmonic mean of precision and recall

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import TopicAdherence
        >>> from ragas.messages import HumanMessage, AIMessage
        >>>
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> metric = TopicAdherence(llm=llm, mode="precision")
        >>>
        >>> result = await metric.ascore(
        ...     user_input=[
        ...         HumanMessage(content="Tell me about quantum physics"),
        ...         AIMessage(content="Quantum physics is a branch of physics..."),
        ...     ],
        ...     reference_topics=["Physics", "Science"],
        ... )
        >>> print(f"Topic Adherence: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for topic extraction and classification
        mode: Evaluation mode - "precision", "recall", or "f1" (default: "f1")
        name: The metric name
    """

    required_components = ("llm",)

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        mode: Literal["precision", "recall", "f1"] = "f1",
        name: str = "topic_adherence",
        **kwargs,
    ):
        self.mode = mode
        self.topic_extraction_prompt = TopicExtractionPrompt()
        self.topic_refused_prompt = TopicRefusedPrompt()
        self.topic_classification_prompt = TopicClassificationPrompt()

        super().__init__(llm=llm, name=name, **kwargs)

    async def ascore(
        self,
        user_input: List[Union[HumanMessage, AIMessage, ToolMessage]],
        reference_topics: List[str],
    ) -> MetricResult:
        """
        Calculate topic adherence score.

        Args:
            user_input: List of conversation messages
            reference_topics: List of allowed topics the AI should adhere to

        Returns:
            MetricResult with topic adherence score (0.0-1.0, higher is better)
        """
        if not isinstance(user_input, list):
            raise ValueError("user_input must be a list of messages")
        if not isinstance(reference_topics, list) or not reference_topics:
            raise ValueError("reference_topics must be a non-empty list of topics")

        # Format conversation as pretty string
        conversation = self._format_conversation(user_input)

        # Step 1: Extract topics from the conversation
        topics = await self._extract_topics(conversation)
        if not topics:
            return MetricResult(value=float("nan"))

        # Step 2: Check which topics the AI answered vs refused
        topic_answered = await self._check_topics_answered(conversation, topics)

        # Step 3: Classify topics against reference topics
        topic_classifications = await self._classify_topics(reference_topics, topics)
        if topic_classifications is None:
            return MetricResult(value=float("nan"))

        # Step 4: Compute score based on mode
        score = self._compute_score(topic_answered, topic_classifications)

        lines = [
            f"{len(topics)} topics discussed; {self.mode} of answering in-scope "
            f"topics (reference topics: {', '.join(reference_topics)})."
        ]
        for topic, answered, in_scope in zip(
            topics, topic_answered, topic_classifications
        ):
            lines.append(
                f"- {topic}: {'answered' if answered else 'refused'}, "
                f"{'in scope' if in_scope else 'out of scope'}"
            )
        return MetricResult(value=float(score), reason="\n".join(lines))

    def _format_conversation(
        self, messages: List[Union[HumanMessage, AIMessage, ToolMessage]]
    ) -> str:
        """Format messages into a readable conversation string."""
        lines = []
        for msg in messages:
            lines.append(msg.pretty_repr())
        return "\n".join(lines)

    async def _extract_topics(self, conversation: str) -> List[str]:
        """Extract topics from the conversation."""
        input_data = TopicExtractionInput(user_input=conversation)
        prompt_str = self.topic_extraction_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, TopicExtractionOutput)
        return result.topics

    async def _check_topics_answered(
        self, conversation: str, topics: List[str]
    ) -> np.ndarray:
        """Check which topics were answered (not refused) by the AI."""
        # One independent check per topic, run concurrently (#58).
        results = await gather_bounded(
            self.llm.agenerate(
                self.topic_refused_prompt.to_string(
                    TopicRefusedInput(user_input=conversation, topic=topic)
                ),
                TopicRefusedOutput,
            )
            for topic in topics
        )
        # Invert: answered = NOT refused
        return np.array([not r.refused_to_answer for r in results], dtype=bool)

    async def _classify_topics(
        self, reference_topics: List[str], topics: List[str]
    ) -> t.Optional[np.ndarray]:
        """Classify if each topic falls within reference topics."""
        input_data = TopicClassificationInput(
            reference_topics=reference_topics, topics=topics
        )
        prompt_str = self.topic_classification_prompt.to_string(input_data)

        async def ask() -> List[t.Any]:
            result = await self.llm.agenerate(prompt_str, TopicClassificationOutput)
            return list(result.classifications)

        # One classification per topic or no score. Padding missing ones with
        # False (or truncating extras) silently mislabelled topics (#52).
        aligned = await judge_each(
            topics, ask, what="classifications", metric=self.name
        )
        if aligned is None:
            return None
        return self._safe_bool_conversion(aligned)

    def _safe_bool_conversion(self, classifications: List) -> np.ndarray:
        """Safely convert classifications to boolean array."""
        arr = np.array(classifications)
        if arr.dtype == bool:
            return arr
        if arr.dtype in [int, np.int64, np.int32, np.int16, np.int8]:
            return arr.astype(bool)
        if arr.dtype.kind in ["U", "S", "O"]:
            bool_list = []
            for item in arr:
                if isinstance(item, bool):
                    bool_list.append(item)
                elif isinstance(item, (int, np.integer)):
                    bool_list.append(bool(item))
                elif isinstance(item, str):
                    bool_list.append(item.lower() in ["true", "1", "yes"])
                else:
                    bool_list.append(bool(item))
            return np.array(bool_list, dtype=bool)
        return arr.astype(bool)

    def _compute_score(
        self, topic_answered: np.ndarray, topic_classifications: np.ndarray
    ) -> float:
        """Compute precision, recall, or F1 score."""
        true_positives = np.sum(topic_answered & topic_classifications)
        false_positives = np.sum(topic_answered & ~topic_classifications)
        false_negatives = np.sum(~topic_answered & topic_classifications)

        precision, recall = precision_recall(
            int(true_positives), int(false_positives), int(false_negatives)
        )
        if self.mode == "precision":
            return precision
        if self.mode == "recall":
            return recall
        return fbeta_from_precision_recall(precision, recall)
