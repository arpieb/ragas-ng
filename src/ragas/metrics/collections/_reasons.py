"""Explanations attached to metric scores (``MetricResult.reason``, #116).

The judge prompts already return a reason per statement, context or claim;
the metrics used to compute a number from the verdicts and drop the rest.
These helpers turn the verdicts into a short, readable explanation.
"""

from __future__ import annotations

import typing as t

# Undefined-score explanations shared across metrics.
UNMATCHED_VERDICTS = (
    "The judge's verdicts could not be matched to the {what} it was given "
    "(even after a retry), so the score is undefined."
)


def itemized(
    summary: str,
    items: t.Iterable[t.Tuple[str, bool, t.Optional[str]]],
    yes: str = "supported",
    no: str = "not supported",
) -> str:
    """``summary`` followed by one line per item: ``- [label] text: why``."""
    lines = [summary]
    for text, ok, why in items:
        line = f"- [{yes if ok else no}] {text}"
        if why:
            line += f": {why}"
        lines.append(line)
    return "\n".join(lines)


def count_phrase(k: int, n: int, noun: str) -> str:
    """``"2 of 3 statements"``; ``"1 of 1 entity"`` / ``"0 of 2 entities"``."""
    if n == 1:
        plural = noun
    elif noun.endswith("y") and noun[-2:-1] not in "aeiou":
        plural = noun[:-1] + "ies"
    else:
        plural = noun + "s"
    return f"{k} of {n} {plural}"
