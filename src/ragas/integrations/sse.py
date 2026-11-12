"""A minimal Server-Sent Events (SSE) client for JSON event streams.

Used by the AG-UI integration (``ragas.integrations.ag_ui.call_ag_ui_endpoint``)
but protocol-agnostic: it POSTs a JSON body and yields each event's ``data``
payload parsed as JSON. Requires ``httpx``.
"""

from __future__ import annotations

__all__ = ["post_sse_json"]

import json
import logging
import typing as t

logger = logging.getLogger(__name__)


def _data_payload(line: str) -> t.Optional[str]:
    """The payload of a ``data:`` line, or None for any other line."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :]
    # The spec strips one leading space after the colon.
    return payload[1:] if payload.startswith(" ") else payload


async def post_sse_json(
    url: str,
    body: t.Any,
    headers: t.Optional[t.Dict[str, str]] = None,
    timeout: float = 60.0,
) -> t.AsyncIterator[t.Any]:
    """POST ``body`` as JSON to ``url`` and yield each SSE event's data as JSON.

    Each event is expected on one ``data:`` line. Lines that are not ``data:``
    lines (comments, ``event:``/``id:`` fields, blank separators) are skipped,
    and so is a ``data:`` line that isn't JSON, with a warning.

    Raises ``ImportError`` without httpx, and ``httpx.HTTPStatusError`` for a
    non-2xx response.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "The SSE client requires httpx. Install it with: pip install httpx"
        ) from e

    request_headers = {"Accept": "text/event-stream", **(headers or {})}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream(
            "POST", url, json=body, headers=request_headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                payload = _data_payload(line.strip())
                if payload is None:
                    continue
                try:
                    yield json.loads(payload)
                except ValueError as e:
                    logger.warning(f"Failed to parse SSE event: {e}")
