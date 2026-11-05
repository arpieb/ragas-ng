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
