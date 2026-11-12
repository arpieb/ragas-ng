"""The seven LLM extractors on their two shared bases (#217)."""

from __future__ import annotations

import typing as t

import pytest

from ragas.testset.graph import Node
from ragas.testset.transforms.base import LLMBasedExtractor
from ragas.testset.transforms.extractors.llm_based import (
    Headlines,
    HeadlinesExtractor,
    Keyphrases,
    KeyphrasesExtractor,
    NERExtractor,
    NEROutput,
    StringIO,
    SummaryExtractor,
    ThemesAndConcepts,
    ThemesExtractor,
    TitleExtractor,
    TopicDescription,
    TopicDescriptionExtractor,
)
from tests.unit.testset_fakes import fake_llm


class _Prompt:
    """Answers every call with ``make(data)`` and records the inputs."""

    name = "fake"

    def __init__(self, make: t.Callable[[t.Any], t.Any]):
        self.make = make
        self.calls: t.List[t.Any] = []

    async def generate(self, llm, data, callbacks=None):
        self.calls.append(data)
        return self.make(data)


FIRST_CHUNK = [
    (SummaryExtractor, "summary", lambda d: StringIO(text="sum:" + d.text)),
    (TitleExtractor, "title", lambda d: StringIO(text="title:" + d.text)),
    (
        TopicDescriptionExtractor,
        "topic_description",
        lambda d: TopicDescription(description="topic:" + d.text),
    ),
]

PER_CHUNK = [
    (
        KeyphrasesExtractor,
        "keyphrases",
        5,
        None,
        lambda d: Keyphrases(keyphrases=[d.text]),
    ),
    (HeadlinesExtractor, "headlines", 5, None, lambda d: Headlines(headlines=[d.text])),
    (NERExtractor, "entities", 10, [], lambda d: NEROutput(entities=[d.text])),
    (ThemesExtractor, "themes", 10, [], lambda d: ThemesAndConcepts(output=[d.text])),
]


def _node(text: t.Optional[str]) -> Node:
    return Node(properties={} if text is None else {"page_content": text})


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, prop, make", FIRST_CHUNK)
async def test_first_chunk_extractors_prompt_once(cls, prop, make):
    prompt = _Prompt(make)
    extractor = cls(llm=fake_llm(), prompt=prompt, max_token_limit=2)

    name, value = await extractor.extract(_node("one two three four"))

    assert name == prop
    assert len(prompt.calls) == 1
    assert value.endswith(prompt.calls[0].text)
    assert await extractor.extract(_node(None)) == (prop, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, prop, default_max, missing, make", PER_CHUNK)
async def test_per_chunk_extractors_join_every_chunk(
    cls, prop, default_max, missing, make
):
    prompt = _Prompt(make)
    extractor = cls(llm=fake_llm(), prompt=prompt, max_token_limit=2)

    name, value = await extractor.extract(_node("one two three four"))

    assert name == prop
    assert len(prompt.calls) == 2
    assert value == [call.text for call in prompt.calls]
    assert {call.max_num for call in prompt.calls} == {default_max}
    assert await extractor.extract(_node(None)) == (prop, missing)


def test_per_chunk_limits_keep_their_field_names():
    prompt = _Prompt(lambda d: None)
    assert KeyphrasesExtractor(llm=fake_llm(), prompt=prompt, max_num=3).max_num == 3
    assert NERExtractor(llm=fake_llm(), max_num_entities=4).max_num_entities == 4
    assert ThemesExtractor(llm=fake_llm(), max_num_themes=2).max_num_themes == 2


def test_merge_if_possible_is_gone():
    with pytest.raises(TypeError):
        SummaryExtractor(llm=fake_llm(), merge_if_possible=False)  # type: ignore[call-arg]
    assert "merge_if_possible" not in LLMBasedExtractor.__dataclass_fields__
