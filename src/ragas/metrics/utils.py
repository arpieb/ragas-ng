"""Precision, recall and F-beta with one edge-case policy for every metric.

Edge cases, where a denominator is zero:

- **Nothing expected and nothing produced** (``tp + fp + fn == 0``):
  precision, recall and F are all 1.0. Nothing was missed and nothing was
  wrong, e.g. an agent that correctly made no tool calls.
- **Nothing produced, something expected** (``tp + fp == 0``, ``fn > 0``):
  precision and recall are 0.0, so F is 0.0.
- **Nothing expected, something produced** (``tp + fn == 0``, ``fp > 0``):
  precision and recall are 0.0, so F is 0.0.
- Otherwise ``tp == 0`` gives precision = recall = F = 0.0.

No epsilon is added anywhere, so a perfect score is exactly 1.0.
"""

from __future__ import annotations

import math
import typing as t


def precision_recall(tp: int, fp: int, fn: int) -> t.Tuple[float, float]:
    """``(precision, recall)`` from counts, under the module's edge-case policy."""
    if tp + fp + fn == 0:
        return 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """F-beta from true positive, false positive and false negative counts.

    ``beta > 1`` weighs recall more, ``beta < 1`` precision. See the module
    docstring for the zero-denominator cases.
    """
    return fbeta_from_precision_recall(*precision_recall(tp, fp, fn), beta=beta)


def fbeta_from_precision_recall(
    precision: float, recall: float, beta: float = 1.0
) -> float:
    """F-beta from a precision and a recall that were measured separately.

    ``fbeta_score(tp, fp, fn)`` assumes one shared set of true positives, which
    does not hold when precision and recall are computed over different claim
    sets (response claims vs reference claims). NaN in either input propagates:
    with no claims on one side the score is undefined.
    """
    if math.isnan(precision) or math.isnan(recall):
        return float("nan")
    if precision == 0 and recall == 0:
        return 0.0
    beta_squared = beta**2
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
