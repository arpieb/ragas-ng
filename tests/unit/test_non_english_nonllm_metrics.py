"""Non-LLM metrics on text that isn't space-separated Latin (#51)."""

from __future__ import annotations

import importlib.util

import pytest

from ragas.metrics.collections import QuotedSpansAlignment
from ragas.metrics.collections._text import unicode_words
from ragas.metrics.collections.quoted_spans.util import extract_quoted_spans

# --- tokenizer ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The Cat sat.", ["the", "cat", "sat"]),
        ("Кошка сидела", ["кошка", "сидела"]),
        ("Η γάτα", ["η", "γάτα"]),
        ("très café", ["très", "café"]),
        ("今天很好", ["今", "天", "很", "好"]),
        ("東京は晴れ", ["東", "京", "は", "晴", "れ"]),
        ("안녕 하세요", ["안녕", "하세요"]),  # Hangul is space-delimited
        ("GPT-4 用户", ["gpt", "4", "用", "户"]),
    ],
)
def test_unicode_words(text, expected):
    assert unicode_words(text) == expected


# --- ROUGE ----------------------------------------------------------------------

requires_rouge = pytest.mark.skipif(
    importlib.util.find_spec("rouge_score") is None,
    reason="rouge_score not installed",
)


async def rouge(reference, response, **kw):
    from ragas.metrics.collections import RougeScore

    return (await RougeScore(**kw).ascore(reference=reference, response=response)).value


@requires_rouge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Кошка сидела на коврике",
        "Η γάτα κάθεται στο χαλί",
        "今天天气很好",
        "مرحبا بالعالم",
    ],
)
async def test_identical_non_latin_text_scores_one(text):
    # rouge_score's own tokenizer drops all of these characters: 0.0 before #51
    assert await rouge(text, text) == pytest.approx(1.0)


@requires_rouge
@pytest.mark.asyncio
async def test_partial_overlap_in_cyrillic_and_chinese():
    assert 0.5 < await rouge("Кошка сидела на коврике", "Кошка сидела на ковре") < 1.0
    assert (
        0.8
        < await rouge("今天天气很好我们去公园散步", "今天天气很好我们去公园跑步")
        < 1.0
    )


@requires_rouge
@pytest.mark.asyncio
async def test_accented_words_are_not_split():
    # "très" became "tr s" under rouge_score's ASCII tokenizer
    score = await rouge("le café est très chaud", "le café est tres chaud")
    assert score == pytest.approx(0.8)


@requires_rouge
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ref, resp",
    [
        ("The cat sat on the mat", "The cat sat on a mat"),
        ("running quickly towards the houses", "they ran quick to the house"),
    ],
)
async def test_ascii_text_is_bit_identical_to_rouge_score(ref, resp):
    from rouge_score import rouge_scorer

    expected = (
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        .score(ref, resp)["rougeL"]
        .fmeasure
    )
    assert await rouge(ref, resp) == expected


@requires_rouge
@pytest.mark.asyncio
async def test_default_tokenizer_can_still_be_forced():
    assert await rouge("Кошка", "Кошка", tokenizer="default") == 0.0


@requires_rouge
@pytest.mark.asyncio
async def test_custom_tokenizer_object():
    class Chars:
        def tokenize(self, text):
            return list(text.replace(" ", ""))

    assert await rouge("abc", "abd", tokenizer=Chars(), rouge_type="rouge1") == (
        pytest.approx(2 / 3)
    )


# --- quoted spans ---------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, span",
    [
        ("Il a dit «la vie est belle» hier", "la vie est belle"),
        ("Er sagte „das ist sehr gut“ dazu", "das ist sehr gut"),
        ("He wrote “the quick brown fox” here", "the quick brown fox"),
        ("他说「今天天气很好」", "今天天气很好"),
        ("『東京は晴れです』と言った", "東京は晴れです"),
    ],
)
def test_extracts_quotes_in_other_conventions(answer, span):
    assert extract_quoted_spans(answer, min_len=3) == [span]


@pytest.mark.asyncio
async def test_quoted_spans_alignment_on_chinese():
    metric = QuotedSpansAlignment()
    result = await metric.ascore(
        response="报告说「今天天气很好」",
        retrieved_contexts=["新闻：今天天气很好，适合散步。"],
    )
    assert result.value == 1.0
