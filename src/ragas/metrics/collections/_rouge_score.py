"""Rouge Score metric v2 - Class-based implementation with automatic validation."""

import typing as t

from ragas.metrics.collections._text import UnicodeTokenizer, is_ascii
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult


class RougeScore(BaseMetric):
    """
    Calculate ROUGE score between reference and response texts.

    This implementation provides automatic validation and pure async design
    without requiring LLM or embedding components.

    Usage:
        >>> from ragas.metrics.collections import RougeScore
        >>>
        >>> # Create metric instance (no LLM/embeddings needed)
        >>> metric = RougeScore(rouge_type="rougeL", mode="fmeasure")
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     reference="The capital of France is Paris.",
        ...     response="Paris is the capital of France."
        ... )
        >>> print(f"Score: {result.value}")
        >>>
        >>> # Batch evaluation
        >>> results = await metric.abatch_score([
        ...     {"reference": "Text 1", "response": "Response 1"},
        ...     {"reference": "Text 2", "response": "Response 2"},
        ... ])

    Attributes:
        name: The metric name
        rouge_type: Type of ROUGE metric ("rouge1" for unigrams, "rougeL" for LCS)
        mode: Scoring mode ("fmeasure", "precision", or "recall")
        allowed_values: Score range (0.0 to 1.0)

    Note: This metric doesn't define llm or embeddings fields, so no validation is performed.
    """

    def __init__(
        self,
        name: str = "rouge_score",
        rouge_type: t.Literal["rouge1", "rougeL"] = "rougeL",
        mode: t.Literal["fmeasure", "precision", "recall"] = "fmeasure",
        tokenizer: t.Union[t.Literal["auto", "default", "unicode"], t.Any] = "auto",
        use_stemmer: bool = True,
        **kwargs,
    ):
        """Initialize RougeScore metric.

        Args:
            tokenizer: ``"auto"`` (default) uses rouge_score's own tokenizer when
                both texts are ASCII, so English scores are unchanged, and
                ``"unicode"`` otherwise. ``"default"`` always uses rouge_score's,
                which drops every non-``[a-z0-9]`` character, so non-Latin text
                scores 0. ``"unicode"`` keeps words in any script and splits CJK
                into characters. Any object with a ``tokenize(text)`` method
                (e.g. a language-specific segmenter) is used as-is.
            use_stemmer: Porter-stem English tokens (as rouge_score does).
                Never applied to non-ASCII tokens.
        """
        super().__init__(name=name, **kwargs)
        self.rouge_type = rouge_type
        self.mode = mode
        self.tokenizer = tokenizer
        self.use_stemmer = use_stemmer

    def _tokenizer_for(self, reference: str, response: str) -> t.Any:
        """None means rouge_score's built-in tokenizer."""
        if self.tokenizer == "default":
            return None
        if self.tokenizer == "auto":
            if is_ascii(reference, response):
                return None
            return UnicodeTokenizer(self.use_stemmer)
        if self.tokenizer == "unicode":
            return UnicodeTokenizer(self.use_stemmer)
        return self.tokenizer

    async def ascore(
        self,
        reference: str,
        response: str,
    ) -> MetricResult:
        """
        Calculate ROUGE score asynchronously.

        Args:
            reference: The reference/ground truth text
            response: The response text to evaluate

        Returns:
            MetricResult with ROUGE score (0.0-1.0)
        """
        # Import and check dependencies
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge_score is required for ROUGE score calculation. "
                "Please install it using `pip install rouge_score`"
            )

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(
            [self.rouge_type],
            use_stemmer=self.use_stemmer,
            tokenizer=self._tokenizer_for(reference, response),
        )
        scores = scorer.score(reference, response)
        score_value = getattr(scores[self.rouge_type], self.mode)

        return MetricResult(value=float(score_value))
