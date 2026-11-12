"""Per-metric summaries and threshold checks, shared by every result type.

``ragas.summarize`` / ``ragas.check`` compute them from an experiment's
metric columns; the ``ThresholdError`` table is #169's.
"""

from __future__ import annotations

import math
import typing as t

from ragas.utils import safe_nanmean

Summary = t.Dict[str, t.Dict[str, float]]


def has_score(value: t.Any) -> bool:
    """A row counts towards a metric's mean: not None and not NaN."""
    if value is None:
        return False
    try:
        return not math.isnan(value)
    except TypeError:
        return True  # non-numeric outputs (e.g. discrete labels) are present


def summarize_columns(columns: t.Mapping[str, t.Sequence[t.Any]]) -> Summary:
    """``{metric: {"mean", "count", "missing"}}`` for columns of scores.

    ``mean`` skips rows without a score (None or NaN); ``count`` is how many
    rows it covers and ``missing`` how many it doesn't.
    """
    summary: Summary = {}
    for name, values in columns.items():
        scored = sum(has_score(v) for v in values)
        summary[name] = {
            "mean": safe_nanmean(list(values)),
            "count": scored,
            "missing": len(values) - scored,
        }
    return summary


def check_summary(
    summary: Summary,
    min_mean: t.Union[float, t.Mapping[str, float], None] = None,
    max_mean: t.Union[float, t.Mapping[str, float], None] = None,
    allow_missing: t.Optional[int] = 0,
) -> None:
    """Raise ``ThresholdError`` unless every metric in ``summary`` meets its bounds.

    See ``ragas.check`` for the arguments.
    """
    from ragas.exceptions import ThresholdError

    def per_metric(
        bound: t.Union[float, t.Mapping[str, float], None], arg: str
    ) -> t.Dict[str, float]:
        if bound is None:
            return {}
        if isinstance(bound, bool) or not isinstance(bound, (int, float, t.Mapping)):
            raise TypeError(
                f"check({arg}=...) takes a number or {{metric: bound}}, "
                f"not {type(bound).__name__}"
            )
        if not isinstance(bound, t.Mapping):
            return {name: float(bound) for name in summary}
        unknown = sorted(set(bound) - set(summary))
        if unknown:
            raise ValueError(
                f"check({arg}=...) names metrics not in this result: "
                f"{', '.join(unknown)}. Metrics: {', '.join(summary)}"
            )
        return {name: float(value) for name, value in bound.items()}

    lows = per_metric(min_mean, "min_mean")
    highs = per_metric(max_mean, "max_mean")
    if allow_missing is not None and allow_missing < 0:
        raise ValueError("check(allow_missing=...) must be >= 0 or None")

    failures: t.List[t.Tuple[str, str]] = []
    rows = []
    for name, stats in summary.items():
        low, high = lows.get(name), highs.get(name)
        mean, missing = stats["mean"], stats["missing"]
        problems = []
        scored = not math.isnan(mean)
        if low is not None and not (scored and mean >= low):
            problems.append(f"mean below {low:g}" if scored else "no scores")
        if high is not None and not (scored and mean <= high):
            problem = f"mean above {high:g}" if scored else "no scores"
            if problem not in problems:
                problems.append(problem)
        if allow_missing is not None and missing > allow_missing:
            rows_word = "row" if missing == 1 else "rows"
            problems.append(f"{missing} {rows_word} missing (allowed {allow_missing})")
        failures.extend((name, p) for p in problems)
        if low is None and high is None and allow_missing is None:
            continue
        bounds = (
            f"{low:g} .. {high:g}"
            if low is not None and high is not None
            else f">= {low:g}"
            if low is not None
            else f"<= {high:g}"
            if high is not None
            else ""
        )
        rows.append(
            (
                name,
                "nan" if not scored else f"{mean:.4f}",
                bounds,
                str(stats["count"]),
                str(missing),
                "; ".join(problems) or "ok",
            )
        )

    if not failures:
        return
    header = ("metric", "mean", "required", "count", "missing", "result")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(6)]
    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip()
        for row in [header, *rows]
    ]
    failed = len({name for name, _ in failures})
    message = (
        f"{failed} of {len(rows)} {'metric' if len(rows) == 1 else 'metrics'} "
        "failed their thresholds:\n" + "\n".join(lines)
    )
    raise ThresholdError(message, failures)
