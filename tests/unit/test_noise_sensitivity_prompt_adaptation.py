"""NoiseSensitivity prompts can be adapted to another language (#59).

They used to override ``to_string`` with a hard-coded English string and had
no ``examples``, so ``adapt()`` had nothing to translate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import NoiseSensitivity


def translator_llm():
    """Translates by tagging every string, so translated text is recognisable."""
    llm = MagicMock(spec=InstructorLLM)

    async def agenerate(prompt, model, **kwargs):
        import json
        import re

        payload = json.loads(re.search(r"\[.*\]", prompt, re.S).group(0))
        return model(statements=[f"[es] {s}" for s in payload])

    llm.agenerate = agenerate
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("attr", ["statement_prompt", "faithfulness_prompt"])
async def test_both_prompts_adapt_and_render_translated_examples(attr):
    metric = NoiseSensitivity(llm=MagicMock(spec=InstructorLLM))
    prompt = getattr(metric, attr)
    adapted = await prompt.adapt("spanish", translator_llm(), adapt_instruction=True)

    assert adapted.language == "spanish"
    sample_input = adapted.examples[0][0]
    rendered = adapted.to_string(sample_input)
    assert "[es] " in rendered
    assert adapted.instruction.startswith("[es] ")
    # the original is untouched
    assert "[es]" not in prompt.to_string(prompt.examples[0][0])


def test_statement_and_nli_prompts_are_shared():
    """One statements/NLI module serves all four metrics (#217)."""
    from ragas.metrics.collections import (
        AnswerCorrectness,
        FactualCorrectness,
        Faithfulness,
    )
    from ragas.metrics.collections.faithfulness import util as faith

    llm = MagicMock(spec=InstructorLLM)
    ns = NoiseSensitivity(llm=llm)
    assert type(ns.statement_prompt) is faith.StatementGeneratorPrompt
    assert type(ns.faithfulness_prompt) is faith.NLIStatementPrompt
    assert type(Faithfulness(llm=llm).nli_statement_prompt) is faith.NLIStatementPrompt
    assert type(FactualCorrectness(llm=llm).nli_prompt) is faith.NLIStatementPrompt
    ac = AnswerCorrectness(llm=llm, weights=[1.0, 0.0])
    assert type(ac.statement_generator_prompt) is faith.StatementGeneratorPrompt
    assert faith.StatementGeneratorPrompt().examples
    assert faith.NLIStatementPrompt().examples
