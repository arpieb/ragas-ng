"""Average precision over a ranked list of binary verdicts.

Shared by the context-precision metrics. It lived as two identical copies inside
``collections/context_precision/metric.py``; the non-LLM variant would have made
a third.
"""

import typing as t


def average_precision(verdicts: t.Sequence[int]) -> float:
    """Mean of the precision@k values taken at each relevant position.

    Rewards relevant contexts appearing early in the ranking. With no relevant
    contexts (all-zero or empty verdicts) the score is 0.0. A perfect ranking
    scores exactly 1.0; an unconditional ``+1e-10`` in the denominator used to
    cap it at 0.9999999999 (#57).
    """
    cumsum = 0
    numerator = 0.0
    for i, v in enumerate(verdicts):
        cumsum += v
        if v:
            numerator += cumsum / (i + 1)

    if cumsum == 0:
        return 0.0
    return numerator / cumsum
