"""Deprecated: Langfuse helpers (#97).

This module never fetched real traces in ragas-ng: ``sync_trace()`` returned a
placeholder trace and ``LangfuseTrace.filter()`` always returned ``[]``. It is
retired rather than ported to the Langfuse SDK, because both of its jobs are
covered without ragas-specific vendor code:

- **Send ragas runs to Langfuse:** ragas emits OpenTelemetry spans; export
  them to Langfuse's OTLP endpoint (or let the Langfuse SDK export them).
- **Score your app's traces:** fetch them with Langfuse's own SDK
  (``get_client().api.trace.get(trace_id)``), score with ragas metrics, and
  write the scores back with ``score_current_trace`` / ``create_score``.

See ``docs/howtos/integrations/_langfuse.md``. This module will be removed in a
future release.
"""

from __future__ import annotations

__all__ = ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]

import logging
import typing as t
import warnings
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

warnings.warn(
    "ragas.integrations.tracing.langfuse is deprecated and will be removed. "
    "Use the Langfuse SDK directly (`from langfuse import observe, "
    "get_client`); ragas runs reach Langfuse as OpenTelemetry spans. See the "
    "Langfuse integration guide.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

_RETIRED = (
    "{name} has been retired: in ragas-ng it never fetched real traces (it "
    "returned a placeholder). Fetch traces with the Langfuse SDK instead, e.g. "
    "`from langfuse import get_client; "
    "get_client().api.trace.get(trace_id)`, and read `trace.observations`. "
    "See the Langfuse integration guide in the ragas docs."
)


def observe(*args: t.Any, **kwargs: t.Any) -> t.Any:
    """Deprecated re-export of ``langfuse.observe``; import it from langfuse.

    Without langfuse installed it is a no-op decorator, as before.
    """
    try:
        from langfuse import (
            observe as langfuse_observe,  # type: ignore[import-not-found]
        )
    except ImportError:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func: t.Any) -> t.Any:
            return func

        return decorator
    return langfuse_observe(*args, **kwargs)


class LangfuseTrace:
    """Retired; raises. Use the Langfuse SDK's trace objects directly."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        raise NotImplementedError(_RETIRED.format(name="LangfuseTrace"))


async def sync_trace(*args: t.Any, **kwargs: t.Any) -> t.NoReturn:
    """Retired; raises. Use ``get_client().api.trace.get(trace_id)``."""
    raise NotImplementedError(_RETIRED.format(name="sync_trace()"))


def add_query_param(url: str, param_name: str, param_value: str) -> str:
    """Add a query parameter to a URL."""
    url_parts = list(urlparse(url))
    query_dict = dict(parse_qsl(url_parts[4]))
    query_dict[param_name] = param_value
    url_parts[4] = urlencode(query_dict)
    return urlunparse(url_parts)
