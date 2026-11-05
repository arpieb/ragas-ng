"""Quoted Spans utility functions."""

from __future__ import annotations

import re
import typing as t

from ragas.metrics.collections._text import unicode_words

# Opening and closing quote characters: straight and curly English quotes,
# low-9 quotes („ ‚), guillemets (« » ‹ ›), and CJK corner brackets (「」『』).
_QUOTES = (
    "\"'`\u00b4"
    "\u201c\u201d\u201e\u201f\u2018\u2019\u201a"
    "\u00ab\u00bb\u2039\u203a"
    "\u300c\u300d\u300e\u300f"
)
QUOTE_RE = re.compile(f"[{re.escape(_QUOTES)}](.*?)[{re.escape(_QUOTES)}]")


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and lower-casing."""
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_quoted_spans(answer: str, min_len: int = 3) -> t.List[str]:
    """
    Extract quoted spans from an answer.

    Args:
        answer: The model answer to search for quoted spans.
        min_len: Minimum number of words required for a span to be considered
            (each CJK character counts as one).
            Shorter spans are ignored to avoid spurious matches.

    Returns:
        A list of quoted spans (strings) that meet the minimum length requirement.
    """
    spans: t.List[str] = []
    for match in QUOTE_RE.finditer(answer):
        span = (match.group(1) or "").strip()
        # unicode_words counts each CJK character as a token: a quoted Chinese
        # sentence has no spaces, so split() saw one "word" and dropped it.
        if len(unicode_words(span)) >= min_len:
            spans.append(span)
    return spans


def count_matched_spans(
    spans: t.List[str],
    sources: t.List[str],
    casefold: bool = True,
) -> t.Tuple[int, int]:
    """
    Count how many spans appear in the sources.

    Args:
        spans: List of quoted spans to check.
        sources: List of source passages to search in.
        casefold: Whether to normalize text before matching.

    Returns:
        Tuple of (matched_count, total_count).
    """
    if not spans:
        return 0, 0

    joined_sources = " ".join(sources)
    normalized_sources = normalize_text(joined_sources) if casefold else joined_sources

    matched = 0
    for span in spans:
        span_norm = normalize_text(span) if casefold else span
        if span_norm and span_norm in normalized_sources:
            matched += 1

    return matched, len(spans)
