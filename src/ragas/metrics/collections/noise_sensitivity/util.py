"""Noise Sensitivity prompt classes and models.

NoiseSensitivity breaks texts into statements and judges them against
contexts with the same two prompts as Faithfulness, so it uses Faithfulness's
prompt classes directly. These used to be separate classes whose
``to_string`` returned a hard-coded English string with the example baked in.
``adapt()`` translates a prompt's ``instruction`` and ``examples``, and these
had neither, so the metric could not be adapted to another language (#59).

The names below are kept as aliases for code that imported them.
"""

from ragas.metrics.collections.faithfulness.util import (
    NLIStatementInput as StatementFaithfulnessInput,
    NLIStatementOutput as StatementFaithfulnessOutput,
    NLIStatementPrompt as StatementFaithfulnessPrompt,
    StatementFaithfulnessAnswer,
    StatementGeneratorInput,
    StatementGeneratorOutput,
    StatementGeneratorPrompt,
)

__all__ = [
    "StatementFaithfulnessAnswer",
    "StatementFaithfulnessInput",
    "StatementFaithfulnessOutput",
    "StatementFaithfulnessPrompt",
    "StatementGeneratorInput",
    "StatementGeneratorOutput",
    "StatementGeneratorPrompt",
]
