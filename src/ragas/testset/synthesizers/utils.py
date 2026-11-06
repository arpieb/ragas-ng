import math
import typing as t


def calculate_split_values(
    probs: t.List[float], n: int
) -> t.Tuple[t.List[int], t.List[int]]:
    """Split ``n`` samples across scenarios in proportion to ``probs``.

    Uses largest-remainder rounding, so the shares always sum to exactly ``n``
    (rounding each share up overshot: 10 across three equal shares gave 12).
    Ties in the remainder go to the earlier scenario, so the split is
    deterministic. ``probs`` need not sum to 1; they are normalised.

    Returns the per-scenario counts and their cumulative offsets, e.g.
    ``([5, 3, 2], [0, 5, 8, 10])``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if any(p < 0 for p in probs):
        raise ValueError(f"probabilities must be non-negative, got {probs}")
    total = sum(probs)
    if not probs or total <= 0:
        raise ValueError(f"probabilities must sum to a positive value, got {probs}")

    exact = [n * p / total for p in probs]
    splits = [math.floor(x) for x in exact]
    leftover = n - sum(splits)
    # hand the remaining samples to the largest fractional parts
    by_remainder = sorted(range(len(probs)), key=lambda i: (-(exact[i] - splits[i]), i))
    for i in by_remainder[:leftover]:
        splits[i] += 1

    split_values = [0]
    for s in splits:
        split_values.append(split_values[-1] + s)
    return (splits, split_values)
