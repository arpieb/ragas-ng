"""Factual Correctness metrics v2 - Modern implementation with multi-modal scoring."""

import typing as t
from typing import List

import numpy as np

from ragas.metrics.collections._reasons import (
    UNMATCHED_VERDICTS,
    count_phrase,
    itemized,
)
from ragas.metrics.collections._verdicts import judge_each
from ragas.metrics.collections.base import BaseMetric
from ragas.metrics.result import MetricResult
from ragas.metrics.utils import fbeta_from_precision_recall

from .util import (
    ClaimDecompositionInput,
    ClaimDecompositionOutput,
    ClaimDecompositionPrompt,
    NLIStatementInput,
    NLIStatementOutput,
    NLIStatementPrompt,
    StatementFaithfulnessAnswer,
)

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class FactualCorrectness(BaseMetric):
    """
    Modern v2 implementation of factual correctness evaluation.

    Evaluates the factual correctness of responses by comparing claims made in the response
    against a reference text. Uses claim decomposition and natural language inference (NLI)
    to verify claims in both directions.

    The metric supports three evaluation modes:
    - Precision: What fraction of response claims are supported by reference
    - Recall: What fraction of reference claims are covered by response
    - F1: Harmonic mean of precision and recall (with configurable beta)

    The metric also supports configurable claim decomposition:
    - Atomicity: "low" (fewer, broader claims) vs "high" (more, atomic claims)
    - Coverage: "low" (partial coverage) vs "high" (comprehensive coverage)

    Usage:
        >>> import instructor
        >>> from openai import AsyncOpenAI
        >>> from ragas.llms.base import llm_factory
        >>> from ragas.metrics.collections import FactualCorrectness
        >>>
        >>> # Setup dependencies
        >>> client = AsyncOpenAI()
        >>> llm = llm_factory("<model>", client=client)
        >>>
        >>> # Create metric instance
        >>> metric = FactualCorrectness(llm=llm, mode="f1", beta=1.0)
        >>>
        >>> # Single evaluation
        >>> result = await metric.ascore(
        ...     response="Einstein was born in Germany in 1879.",
        ...     reference="Albert Einstein was born in Ulm, Germany on March 14, 1879."
        ... )
        >>> print(f"Factual Correctness: {result.value}")

    Attributes:
        llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
        mode: Evaluation mode ("precision", "recall", or "f1")
        beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
        atomicity: Claim decomposition atomicity ("low" or "high")
        coverage: Claim decomposition coverage ("low" or "high")
        name: The metric name
        allowed_values: Score range (0.0 to 1.0, higher is better)
    """

    # Type hints for linter (attributes are set in __init__)
    llm: "InstructorBaseRagasLLM"

    def __init__(
        self,
        llm: "InstructorBaseRagasLLM",
        mode: t.Literal["precision", "recall", "f1"] = "f1",
        beta: float = 1.0,
        atomicity: t.Literal["low", "high"] = "low",
        coverage: t.Literal["low", "high"] = "low",
        name: str = "factual_correctness",
        **kwargs,
    ):
        """
        Initialize FactualCorrectness metric with required components.

        Args:
            llm: Modern instructor-based LLM for claim decomposition and NLI evaluation
            mode: Evaluation mode ("precision", "recall", or "f1")
            beta: Beta parameter for F1 score (>1 favors recall, <1 favors precision)
            atomicity: Claim decomposition atomicity ("low" or "high")
            coverage: Claim decomposition coverage ("low" or "high")
            name: The metric name
        """
        # Set attributes explicitly before calling super()
        self.llm = llm
        self.mode = mode
        self.beta = beta
        self.atomicity = atomicity
        self.coverage = coverage
        self.prompt = ClaimDecompositionPrompt()
        self.nli_prompt = NLIStatementPrompt()

        # Validate beta parameter
        if not isinstance(beta, (int, float)):
            raise ValueError(
                "Beta must be a float. A beta > 1 gives more weight to recall, while beta < 1 favors precision."
            )

        # Call super() for validation (without passing llm in kwargs)
        super().__init__(name=name, **kwargs)

    async def ascore(self, response: str, reference: str) -> MetricResult:
        """
        Calculate factual correctness score.

        Args:
            response: The response to evaluate for factual correctness
            reference: The reference text to check claims against

        Returns:
            MetricResult with factual correctness score (0.0-1.0, higher is better)
        """
        # Input validation
        if not response:
            raise ValueError(
                "response is missing. Please add response to the test sample."
            )
        if not reference:
            raise ValueError(
                "reference is missing. Please add reference to the test sample."
            )

        # Precision: what fraction of the RESPONSE's claims the reference
        # supports. Recall: what fraction of the REFERENCE's claims the response
        # supports. Each direction is decomposed and verified independently, and
        # only the directions the mode needs are computed.
        precision = recall = float("nan")
        sections: List[str] = []
        if self.mode != "recall":
            verdicts = await self._judge_direction(response, reference)
            precision = _supported_fraction(verdicts)
            sections.append(
                _direction_reason("response", "reference", "Precision", verdicts)
            )
        if self.mode != "precision":
            verdicts = await self._judge_direction(reference, response)
            recall = _supported_fraction(verdicts)
            sections.append(
                _direction_reason("reference", "response", "Recall", verdicts)
            )

        if self.mode == "precision":
            score = precision
        elif self.mode == "recall":
            score = recall
        else:  # f-beta
            score = fbeta_from_precision_recall(precision, recall, self.beta)

        reason = "\n\n".join(sections)
        # NaN means "no claims to judge" (e.g. a one-word answer or an
        # abstention), which is undefined, not a failure.
        if np.isnan(score):
            return MetricResult(value=float("nan"), reason=reason)
        return MetricResult(value=float(np.round(score, 2)), reason=reason)

    async def _decompose_claims(self, text: str) -> List[str]:
        """Break text into claims using configurable decomposition."""
        input_data = ClaimDecompositionInput(
            response=text, atomicity=self.atomicity, coverage=self.coverage
        )
        prompt_str = self.prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, ClaimDecompositionOutput)
        return result.claims

    async def _verify_claims(
        self, claims: List[str], reference: str
    ) -> NLIStatementOutput:
        """Verify claims against reference using NLI."""
        input_data = NLIStatementInput(context=reference, statements=claims)
        prompt_str = self.nli_prompt.to_string(input_data)
        result = await self.llm.agenerate(prompt_str, NLIStatementOutput)
        return result

    async def _judge_direction(
        self, text_to_decompose: str, reference_text: str
    ) -> t.Optional[List[StatementFaithfulnessAnswer]]:
        """Claims of one text, each with a verdict against the other text.

        ``[]`` if no claims were extracted; ``None`` if the judge's verdicts
        couldn't be matched to the claims (#52).
        """
        claims = await self._decompose_claims(text_to_decompose)
        if not claims:
            return []

        async def ask() -> List[StatementFaithfulnessAnswer]:
            return (await self._verify_claims(claims, reference_text)).statements

        aligned = await judge_each(
            claims, ask, text_of=lambda v: v.statement, metric=self.name
        )
        if aligned is None:
            return None
        # Report the claim as decomposed, whatever the judge echoed back.
        return [v.model_copy(update={"statement": c}) for c, v in zip(claims, aligned)]


def _supported_fraction(
    verdicts: t.Optional[List[StatementFaithfulnessAnswer]],
) -> float:
    """Fraction of claims judged supported; NaN for no claims or unmatched."""
    if not verdicts:
        return float("nan")
    return sum(1 for v in verdicts if v.verdict) / len(verdicts)


def _direction_reason(
    source: str,
    against: str,
    label: str,
    verdicts: t.Optional[List[StatementFaithfulnessAnswer]],
) -> str:
    """Explanation for one direction (response vs reference or the reverse)."""
    if verdicts is None:
        return f"{label}: " + UNMATCHED_VERDICTS.format(what=f"{source} claims")
    if not verdicts:
        return (
            f"{label}: no claims could be extracted from the {source}, so it is "
            f"undefined."
        )
    supported = sum(1 for v in verdicts if v.verdict)
    return itemized(
        f"{label}: {count_phrase(supported, len(verdicts), 'claim')} in the "
        f"{source} are supported by the {against}.",
        ((v.statement, bool(v.verdict), v.reason) for v in verdicts),
    )
