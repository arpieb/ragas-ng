"""tracing using mlflow"""

__all__ = ["MLflowTrace", "sync_trace"]

import os
import typing as t
from urllib.parse import quote, urlencode, urlsplit

if t.TYPE_CHECKING:
    from mlflow import get_last_active_trace_id, get_trace
    from mlflow.entities.span import Span
    from mlflow.entities.trace import Trace
else:
    try:
        from mlflow import get_last_active_trace_id, get_trace  # type: ignore
        from mlflow.entities.span import Span  # type: ignore
        from mlflow.entities.trace import Trace  # type: ignore

        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False

        # Define stub classes for type checking when imports fail
        class Span:  # type: ignore
            name: str = ""

        class Trace:  # type: ignore
            def __init__(self):  # type: ignore
                self.info = type(
                    "TraceInfo", (), {"request_id": "", "experiment_id": ""}
                )()

            def search_spans(self, name: str) -> t.List["Span"]:  # type: ignore
                return []

        # Bound indirectly: defining these under the imported names makes griffe
        # resolve the name to the (unresolvable) `mlflow` alias above and fail
        # the docs build, since mlflow is not installed for `mkdocs build`.
        def _no_last_active_trace_id() -> t.Optional[str]:
            return None

        def _no_trace(trace_id: str) -> t.Optional["Trace"]:
            return None

        get_last_active_trace_id = _no_last_active_trace_id  # type: ignore
        get_trace = _no_trace  # type: ignore


class MLflowTrace:
    def __init__(self, trace: "Trace"):
        self.trace = trace

    def get_url(self) -> str:
        """The MLflow UI URL of this trace, on the server in ``MLFLOW_HOST``.

        ``MLFLOW_HOST`` must be an http(s) URL with a host, optionally with a
        path prefix (``https://example.com/mlflow``). The experiment and trace
        IDs are percent-encoded, so they can't change the URL's structure.
        """
        server_url = os.getenv("MLFLOW_HOST")
        if not server_url:
            raise ValueError("MLFLOW_HOST environment variable is not set.")
        parsed = urlsplit(server_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                f"MLFLOW_HOST must be an http(s) URL with a host, got {server_url!r}."
            )
        if parsed.query or parsed.fragment:
            raise ValueError(
                f"MLFLOW_HOST must not have a query or fragment, got {server_url!r}."
            )
        trace_info = self.trace.info
        base = server_url.rstrip("/")
        experiment_id = quote(str(trace_info.experiment_id), safe="")
        query = urlencode(
            {
                "compareRunsMode": "TRACES",
                "selectedTraceId": str(trace_info.request_id),
            }
        )
        return f"{base}/#/experiments/{experiment_id}?{query}"

    def get_filter(self, span_name: str) -> t.List["Span"]:
        return self.trace.search_spans(name=span_name)


async def sync_trace() -> MLflowTrace:
    trace_id = get_last_active_trace_id()
    if trace_id is None:
        raise ValueError("No active trace found.")

    trace = get_trace(trace_id)
    if trace is None:
        raise ValueError("Trace not found.")

    return MLflowTrace(trace)
