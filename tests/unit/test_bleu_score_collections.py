"""Collections BleuScore scores the whole response against the whole reference."""

import pytest
from sacrebleu import corpus_bleu

from ragas.metrics.collections import BleuScore

REFERENCE = (
    "The cat sat on the mat. The dog ran in the park. Birds sang loudly all morning"
)


@pytest.mark.asyncio
async def test_identical_text_scores_one():
    result = await BleuScore().ascore(reference=REFERENCE, response=REFERENCE)
    assert result.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_sentences_after_the_first_are_scored():
    # First sentence right, the rest wrong. Before #50 this scored 1.0.
    response = (
        "The cat sat on the mat. A completely different thing happened. "
        "Nothing else matched here at all"
    )
    result = await BleuScore().ascore(reference=REFERENCE, response=response)
    assert result.value < 0.5


@pytest.mark.asyncio
async def test_matches_sacrebleu_on_a_single_segment():
    response = "The cat sat on a mat. The dog ran through the park."
    result = await BleuScore().ascore(reference=REFERENCE, response=response)
    expected = corpus_bleu([response], [[REFERENCE]]).score / 100
    assert result.value == pytest.approx(expected)


@pytest.mark.asyncio
async def test_explicit_tokenize_reaches_sacrebleu():
    # Identical Chinese text: forcing the space-based 13a tokenizer sees one
    # "word" per sentence and cannot form 4-grams; zh can (upstream #2101).
    text = "今天天气很好，我们去公园散步吧。"
    forced_13a = await BleuScore(kwargs={"tokenize": "13a"}).ascore(
        reference=text, response=text
    )
    zh = await BleuScore(kwargs={"tokenize": "zh"}).ascore(
        reference=text, response=text
    )
    assert zh.value == pytest.approx(1.0)
    assert forced_13a.value < zh.value


@pytest.mark.asyncio
async def test_chinese_defaults_to_zh_tokenizer():
    ref = "今天天气很好我们去公园散步"
    resp = "今天天气很好我们去公园跑步"
    auto = await BleuScore().ascore(reference=ref, response=resp)
    zh = await BleuScore(kwargs={"tokenize": "zh"}).ascore(reference=ref, response=resp)
    assert auto.value == pytest.approx(zh.value)
    assert auto.value > 0.5  # one character differs; 13a scored this 0.0 (#51)


@pytest.mark.asyncio
async def test_japanese_defaults_to_char_tokenizer():
    ref = "きょうはいいてんきですね"
    resp = "きょうはいいてんきでした"
    auto = await BleuScore().ascore(reference=ref, response=resp)
    char = await BleuScore(kwargs={"tokenize": "char"}).ascore(
        reference=ref, response=resp
    )
    assert auto.value == pytest.approx(char.value) and auto.value > 0.5


@pytest.mark.asyncio
async def test_english_default_unchanged():
    ref, resp = "The cat sat on the mat today", "The cat sat on a mat today"
    auto = await BleuScore().ascore(reference=ref, response=resp)
    expected = corpus_bleu([resp], [[ref]]).score / 100
    assert auto.value == pytest.approx(expected)
