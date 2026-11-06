"""The output-language line on adapted prompts (#106).

``adapt()`` sets ``prompt.language``. Before #106 nothing rendered it, so an
adapted query-generation prompt kept writing in the corpus's language. The
line is opt-in: prompts whose output must echo input values (persona names,
entities) would break if those were translated.
"""

import copy

import pytest

from ragas.prompt.utils import output_language_directive
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryAnswerGenerationPrompt as MultiHopQAPrompt,
)
from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryAnswerGenerationPrompt as SingleHopQAPrompt,
)

LINE = "Write every text value in the output in Spanish"


def _adapted(prompt_cls, language="spanish"):
    prompt = copy.deepcopy(prompt_cls())
    prompt.language = language
    return prompt


@pytest.mark.parametrize("prompt_cls", [SingleHopQAPrompt, MultiHopQAPrompt])
def test_query_generation_prompts_state_the_target_language(prompt_cls):
    assert LINE in _adapted(prompt_cls).to_string()


@pytest.mark.parametrize("prompt_cls", [SingleHopQAPrompt, MultiHopQAPrompt])
def test_english_prompts_are_unchanged(prompt_cls):
    prompt = prompt_cls()
    assert "Write every text value" not in prompt.to_string()
    assert _adapted(prompt_cls, "English").to_string() == prompt.to_string()


def test_matching_prompts_do_not_translate_their_output():
    # persona names in the output must equal the input names exactly
    assert (
        "Write every text value"
        not in _adapted(ThemesPersonasMatchingPrompt).to_string()
    )


@pytest.mark.parametrize(
    "language,expected",
    [(None, ""), ("", ""), ("english", ""), (" English ", "")],
)
def test_directive_is_empty_for_english(language, expected):
    assert output_language_directive(language) == expected


def test_directive_capitalises_the_language():
    assert output_language_directive("spanish").startswith(LINE)
