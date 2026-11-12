"""Factual Correctness prompt classes and models."""

import copy
import typing as t
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from ragas.prompt.metrics.base_prompt import BasePrompt

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM


class ClaimDecompositionInput(BaseModel):
    """Input for claim decomposition."""

    response: str = Field(..., description="The response text to decompose into claims")
    atomicity: str = Field(
        default="low", description="Atomicity level: 'low' or 'high'"
    )
    coverage: str = Field(default="low", description="Coverage level: 'low' or 'high'")


class ClaimDecompositionOutput(BaseModel):
    """Output from claim decomposition."""

    claims: List[str] = Field(..., description="Decomposed claims")


class ClaimDecompositionPrompt(
    BasePrompt[ClaimDecompositionInput, ClaimDecompositionOutput]
):
    """Prompt for decomposing text into claims with configurable atomicity and coverage."""

    input_model = ClaimDecompositionInput
    output_model = ClaimDecompositionOutput

    instruction = """Decompose and break down each of the input sentences into one or more standalone statements. Each statement should be a standalone claim that can be independently verified.
Follow the level of atomicity and coverage as shown in the examples."""

    # Store all example sets for different atomicity/coverage combinations
    _all_examples: Dict[
        Tuple[str, str], List[Tuple[ClaimDecompositionInput, ClaimDecompositionOutput]]
    ] = {
        ("low", "low"): [
            (
                ClaimDecompositionInput(
                    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                    atomicity="low",
                    coverage="low",
                ),
                ClaimDecompositionOutput(
                    claims=["Charles Babbage was a mathematician and philosopher."]
                ),
            ),
            (
                ClaimDecompositionInput(
                    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    atomicity="low",
                    coverage="low",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Albert Einstein was a German physicist.",
                        "Albert Einstein developed relativity and contributed to quantum mechanics.",
                    ]
                ),
            ),
        ],
        ("low", "high"): [
            (
                ClaimDecompositionInput(
                    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                    atomicity="low",
                    coverage="high",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Charles Babbage was a French mathematician, philosopher, and food critic."
                    ]
                ),
            ),
            (
                ClaimDecompositionInput(
                    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    atomicity="low",
                    coverage="high",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    ]
                ),
            ),
        ],
        ("high", "low"): [
            (
                ClaimDecompositionInput(
                    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                    atomicity="high",
                    coverage="low",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                    ]
                ),
            ),
            (
                ClaimDecompositionInput(
                    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    atomicity="high",
                    coverage="low",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                    ]
                ),
            ),
        ],
        ("high", "high"): [
            (
                ClaimDecompositionInput(
                    response="Charles Babbage was a French mathematician, philosopher, and food critic.",
                    atomicity="high",
                    coverage="high",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Charles Babbage was a mathematician.",
                        "Charles Babbage was a philosopher.",
                        "Charles Babbage was a food critic.",
                        "Charles Babbage was French.",
                    ]
                ),
            ),
            (
                ClaimDecompositionInput(
                    response="Albert Einstein was a German theoretical physicist. He developed the theory of relativity and also contributed to the development of quantum mechanics.",
                    atomicity="high",
                    coverage="high",
                ),
                ClaimDecompositionOutput(
                    claims=[
                        "Albert Einstein was a German theoretical physicist.",
                        "Albert Einstein developed the theory of relativity.",
                        "Albert Einstein contributed to the development of quantum mechanics.",
                    ]
                ),
            ),
        ],
    }

    # Default examples (low atomicity, low coverage)
    examples = _all_examples[("low", "low")]

    def to_string(self, input_data: ClaimDecompositionInput) -> str:
        """Generate prompt string with examples based on atomicity and coverage."""
        # Temporarily switch examples based on atomicity/coverage
        key = (input_data.atomicity, input_data.coverage)
        original_examples = self.examples
        self.examples = self._all_examples.get(key, self._all_examples[("low", "low")])

        try:
            # Use parent class implementation
            return super().to_string(input_data)
        finally:
            # Restore original examples
            self.examples = original_examples

    async def adapt(
        self,
        target_language: str,
        llm: "InstructorBaseRagasLLM",
        adapt_instruction: bool = False,
    ) -> "ClaimDecompositionPrompt":
        """
        Adapt the prompt to a new language by translating all example sets.

        Args:
            target_language: Target language (e.g., "spanish", "french", "hindi")
            llm: InstructorLLM instance for translation (must support agenerate)
            adapt_instruction: Whether to adapt instruction text (default: False)

        Returns:
            New prompt instance adapted to the target language
        """
        # Import here to avoid circular dependency
        from ragas.prompt.metrics.base_prompt import _translate_strings
        from ragas.prompt.utils import get_all_strings, update_strings

        # Create a new instance
        new_prompt = copy.deepcopy(self)
        new_prompt.language = target_language

        # Adapt all example sets
        adapted_examples = {}
        for key, examples in self._all_examples.items():
            # Extract strings from this example set
            strings = get_all_strings(examples)

            if strings:
                # Translate all strings
                translated = await _translate_strings(strings, target_language, llm)

                # Update examples with translated strings
                adapted_examples[key] = update_strings(
                    obj=examples,
                    old_strings=strings,
                    new_strings=translated,
                )
            else:
                adapted_examples[key] = examples

        new_prompt._all_examples = adapted_examples
        new_prompt.examples = adapted_examples[("low", "low")]

        # Translate instruction if requested
        if adapt_instruction:
            [translated_instruction] = await _translate_strings(
                [self.instruction], target_language, llm
            )
            new_prompt.instruction = translated_instruction

        return new_prompt
