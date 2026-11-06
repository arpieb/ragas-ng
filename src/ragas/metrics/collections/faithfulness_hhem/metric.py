"""FaithfulnesswithHHEM metric - Modern collections implementation."""

import typing as t

from ragas.metrics.collections._reasons import count_phrase, itemized
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.result import MetricResult

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM

HHEM_MODEL = "vectara/hallucination_evaluation_model"


class FaithfulnesswithHHEM(Faithfulness):
    """
    Faithfulness scored by Vectara's HHEM cross-encoder instead of the LLM.

    Shares the first half of :class:`~ragas.metrics.collections.Faithfulness`:
    the LLM breaks the response into atomic statements. Each statement is then
    checked against the retrieved contexts by a local NLI model rather than by a
    second LLM call, which is cheaper and more reproducible at the cost of a
    model download.

    !!! warning "Runs third-party code"
        The model is loaded with ``trust_remote_code=True``, so importing it
        executes code published on the HuggingFace Hub. That is required by this
        model's architecture, and it is what the legacy metric always did, but
        it is worth knowing before adding this to a pipeline.

    Requires ``transformers``, which ragas does not install by default:

    ```bash
    pip install transformers
    ```

    Usage:
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms import llm_factory
        >>> from ragas.metrics.collections import FaithfulnesswithHHEM
        >>>
        >>> llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
        >>>
        >>> metric = FaithfulnesswithHHEM(llm=llm)
        >>> result = await metric.ascore(
        ...     user_input="Where is the Eiffel Tower?",
        ...     response="The Eiffel Tower is in Paris.",
        ...     retrieved_contexts=["The Eiffel Tower is located in Paris, France."],
        ... )
        >>> print(result.value)

    Unlike the legacy metric, the model is loaded on first use rather than at
    construction, so building the metric neither downloads nor requires anything.

    Attributes:
        llm: Modern instructor-based LLM, used for statement generation only
        device: Torch device to load the classifier onto
        batch_size: Statements scored per forward pass, to bound memory
        name: The metric name
        allowed_values: Score range (0.0 to 1.0)
    """

    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        name: str = "faithfulness_with_hhem",
        device: str = "cpu",
        batch_size: int = 10,
        classifier: t.Optional[t.Any] = None,
        **kwargs,
    ):
        """
        Args:
            llm: Modern instructor-based LLM, used only to generate statements
            name: The metric name
            device: Torch device for the classifier
            batch_size: Statements per forward pass
            classifier: A pre-built model. Supplying one skips the download --
                useful for tests and for pinning a local copy.
        """
        super().__init__(llm=llm, name=name, **kwargs)
        self.device = device
        self.batch_size = batch_size
        self._classifier = classifier

    @property
    def classifier(self) -> t.Any:
        """The NLI model, downloaded on first access."""
        if self._classifier is None:
            try:
                from transformers import (  # type: ignore[import-not-found]
                    AutoModelForSequenceClassification,
                )
            except ImportError as e:
                raise ImportError(
                    "FaithfulnesswithHHEM requires transformers. "
                    "Install it with `pip install transformers`."
                ) from e

            model = AutoModelForSequenceClassification.from_pretrained(
                HHEM_MODEL, trust_remote_code=True
            )
            model.to(self.device)
            self._classifier = model
        return self._classifier

    def _batches(
        self, pairs: t.List[t.Tuple[str, str]]
    ) -> t.Iterator[t.List[t.Tuple[str, str]]]:
        """Chunk the pairs so a long response cannot exhaust memory."""
        for start in range(0, len(pairs), self.batch_size):
            yield pairs[start : start + self.batch_size]

    async def ascore(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: t.List[str],
    ) -> MetricResult:
        """
        Args:
            user_input: The original question
            response: The response to check for faithfulness
            retrieved_contexts: The contexts the response should be grounded in

        Returns:
            MetricResult with the proportion of statements the NLI model finds
            entailed by the contexts, or NaN when no statements were generated.
        """
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not user_input:
            raise ValueError(
                "user_input is missing. Please add user_input to the test sample."
            )
        if not retrieved_contexts:
            raise ValueError(
                "retrieved_contexts is missing. Please add retrieved_contexts to "
                "the test sample."
            )

        statements = await self._create_statements(user_input, response)
        if not statements:
            return MetricResult(
                value=float("nan"),
                reason="No statements could be extracted from the response, so "
                "faithfulness is undefined.",
            )

        premise = "\n".join(retrieved_contexts)
        pairs = [(premise, statement) for statement in statements]

        scores: t.List[float] = []
        for batch in self._batches(pairs):
            predictions = self.classifier.predict(batch).cpu().detach().round()
            scores.extend(predictions.tolist())

        supported = sum(1 for sc in scores if sc)
        reason = itemized(
            f"{count_phrase(supported, len(scores), 'statement')} in the response "
            f"are entailed by the retrieved context (HHEM classifier).",
            ((st, bool(sc), None) for st, sc in zip(statements, scores)),
        )
        return MetricResult(value=sum(scores) / len(scores), reason=reason)
