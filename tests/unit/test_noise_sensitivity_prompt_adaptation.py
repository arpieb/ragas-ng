"""NoiseSensitivity prompts can be adapted to another language (#59).

They used to override ``to_string`` with a hard-coded English string and had
no ``examples``, so ``adapt()`` had nothing to translate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import NoiseSensitivity
from ragas.metrics.collections.noise_sensitivity import util as ns


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


def test_prompts_are_the_faithfulness_prompts():
    from ragas.metrics.collections.faithfulness import util as faith

    assert ns.StatementGeneratorPrompt is faith.StatementGeneratorPrompt
    assert ns.StatementFaithfulnessPrompt is faith.NLIStatementPrompt
    assert ns.StatementGeneratorPrompt().examples
    assert ns.StatementFaithfulnessPrompt().examples
