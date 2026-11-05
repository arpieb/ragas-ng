"""Language-neutral tokenisation shared by the non-LLM metrics (#51).

The libraries these metrics wrap assume space-separated Latin text:

- ``rouge_score``'s tokenizer replaces everything outside ``[a-z0-9]`` with
  spaces, so Cyrillic, Greek, Arabic or CJK text tokenises to nothing and
  scores 0, even against itself.
- sacreBLEU's default ``13a`` tokenizer splits on spaces, so a Chinese sentence
  is a single "word" and partial matches score 0.
- ``str.split()`` counts a quoted Chinese sentence as one word.

Words here are runs of Unicode letters/digits; each CJK ideograph or kana is
its own token, the usual character-level treatment for scripts written
without spaces. Hangul is space-delimited and stays word-level.
"""

from __future__ import annotations

import re
import typing as t

# Han (CJK unified + extension A + compatibility), Hiragana, Katakana.
_CJK_CHARS = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_TOKEN_RE = re.compile(rf"[{_CJK_CHARS}]|[^\W_{_CJK_CHARS}]+")
_HAN_RE = re.compile("[㐀-䶿一-鿿豈-﫿]")
_KANA_RE = re.compile("[぀-ヿ]")


def unicode_words(text: str) -> t.List[str]:
    """Lower-cased word tokens in any script; one token per CJK character."""
    return _TOKEN_RE.findall(text.casefold())


def is_ascii(*texts: str) -> bool:
    return all(text.isascii() for text in texts)


def has_han(*texts: str) -> bool:
    return any(_HAN_RE.search(text) for text in texts)


def has_kana(*texts: str) -> bool:
    return any(_KANA_RE.search(text) for text in texts)


class UnicodeTokenizer:
    """``rouge_score``-compatible tokenizer (has ``tokenize(text)``).

    With ``use_stemmer``, ASCII tokens longer than 3 characters get the same
    Porter stemming ``rouge_score`` applies; other scripts are left as-is,
    since an English stemmer would only damage them.
    """

    def __init__(self, use_stemmer: bool = True):
        self._stemmer = None
        if use_stemmer:
            from nltk.stem import porter

            self._stemmer = porter.PorterStemmer("ORIGINAL_ALGORITHM")

    def tokenize(self, text: str) -> t.List[str]:
        tokens = unicode_words(text)
        if self._stemmer is None:
            return tokens
        return [
            self._stemmer.stem(tok) if tok.isascii() and len(tok) > 3 else tok
            for tok in tokens
        ]


# Abbreviations that end in "." but don't end a sentence.
_ABBREVIATIONS = (
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "no",
    "fig",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "a.m",
    "p.m",
)
# Sentence end: Latin terminal punctuation followed by whitespace and a capital
# letter, digit or opening quote/bracket; or a CJK full-width terminator.
_NEXT_SENTENCE = (
    r"(?=[\"'\u201c\u2018(\[]*[A-Z0-9\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF])"
)
_SENTENCE_END = re.compile(
    # terminal punctuation, optionally followed by one closing quote/bracket
    # that stays with the sentence it closes
    rf"(?<=[.!?][\"'\u201d\u2019)\]])\s+{_NEXT_SENTENCE}"
    rf"|(?<=[.!?])\s+{_NEXT_SENTENCE}"
    r"|(?<=[\u3002\uff01\uff1f])"
)


def split_sentences(text: str) -> t.List[str]:
    """Split ``text`` into sentences; never returns empty strings.

    Deliberately simple and deterministic: splits on ``. ! ?`` followed by
    whitespace and an upper-case letter or digit (Latin, Greek, Cyrillic), and
    after the CJK terminators ``。！？``. Doesn't split after common
    abbreviations ("Dr.", "e.g.", "U.S.").
    """
    pieces = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p and p.strip()]
    merged: t.List[str] = []
    for piece in pieces:
        if merged:
            last_word = merged[-1].rsplit(None, 1)[-1].rstrip(".").casefold()
            if merged[-1].endswith(".") and last_word in _ABBREVIATIONS:
                merged[-1] = f"{merged[-1]} {piece}"
                continue
        merged.append(piece)
    return merged
