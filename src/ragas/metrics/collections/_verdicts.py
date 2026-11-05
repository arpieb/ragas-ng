"""Match a judge's per-item verdicts back to the items it was asked about.

Several metrics send the judge a list (statements, claims, topics) and score
the fraction it marks supported. They used to score over whatever list came
back. If the judge dropped, merged or invented items, the denominator silently
changed: a dropped unsupported statement *raised* faithfulness (#52).

``judge_each`` makes the contract explicit: one verdict per submitted item, or
no score. On a mismatch it first tries to line verdicts up by the statement
text the judge echoes back, then asks once more, then gives up (the metric
returns NaN, which aggregation reports as a missing row).
"""

from __future__ import annotations

import logging
import re
import typing as t

logger = logging.getLogger(__name__)

T = t.TypeVar("T")


def _normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().casefold())
    return text.rstrip(" .;:!?")


def align_verdicts(
    submitted: t.Sequence[str],
    returned: t.Sequence[T],
    text_of: t.Optional[t.Callable[[T], str]] = None,
) -> t.Optional[t.List[T]]:
    """One verdict per submitted item, in submitted order, or None.

    - Same count: taken in order (the judge is asked to keep the order, and
      paraphrased echoes are common, so text isn't required to match).
    - Different count: if ``text_of`` gives the echoed statement, each
      submitted item is matched to a returned verdict with the same normalised
      text. Every item must match exactly once; extra verdicts are ignored.
    - Otherwise None.
    """
    if len(returned) == len(submitted):
        return list(returned)
    if text_of is None:
        return None

    unused = list(returned)
    aligned: t.List[T] = []
    for item in submitted:
        key = _normalise(item)
        match = next((v for v in unused if _normalise(text_of(v)) == key), None)
        if match is None:
            return None
        unused.remove(match)
        aligned.append(match)
    return aligned


async def judge_each(
    submitted: t.Sequence[str],
    ask: t.Callable[[], t.Awaitable[t.Sequence[T]]],
    *,
    text_of: t.Optional[t.Callable[[T], str]] = None,
    what: str = "verdicts",
    metric: str = "metric",
) -> t.Optional[t.List[T]]:
    """Ask the judge, insisting on one verdict per submitted item.

    ``ask`` performs the LLM call and returns the per-item results. It is
    retried once on a mismatch. Returns None if the verdicts still can't be
    matched to the submitted items, and the caller should then score NaN.
    """
    returned: t.Sequence[T] = []
    for attempt in range(2):
        returned = await ask()
        aligned = align_verdicts(submitted, returned, text_of)
        if aligned is not None:
            return aligned
        if attempt == 0:
            logger.debug(
                f"{metric}: judge returned {len(returned)} {what} for "
                f"{len(submitted)} items; asking again."
            )
    logger.warning(
        f"{metric}: judge returned {len(returned)} {what} for {len(submitted)} "
        f"items and they could not be matched up; scoring NaN for this sample."
    )
    return None
