import math


def fbeta_score(tp, fp, fn, beta=1.0):
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta


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
