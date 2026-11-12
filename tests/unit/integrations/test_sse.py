"""The SSE client used by the AG-UI integration (#217)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragas.integrations.sse import _data_payload, post_sse_json

pytest.importorskip("httpx")


@pytest.mark.parametrize(
    "line, payload",
    [
        ('data: {"a": 1}', '{"a": 1}'),
        ('data:{"a": 1}', '{"a": 1}'),
        ("event: message", None),
        (": comment", None),
        ("", None),
    ],
)
def test_data_payload(line, payload):
    assert _data_payload(line) == payload


def _client(lines):
    async def aiter_lines():
        for line in lines:
            yield line

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_lines = aiter_lines
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=response)
    stream.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_yields_json_payloads_and_skips_the_rest(caplog):
    client = _client(['data: {"n": 1}', "", "event: x", "data: not json", "data:[2]"])
    with patch("httpx.AsyncClient", return_value=client):
        got = [
            d
            async for d in post_sse_json(
                "http://agent", {"q": 1}, headers={"X-Key": "k"}
            )
        ]
    assert got == [{"n": 1}, [2]]
    assert "Failed to parse SSE event" in caplog.text
    _, kwargs = client.stream.call_args
    assert kwargs["json"] == {"q": 1}
    assert kwargs["headers"] == {"Accept": "text/event-stream", "X-Key": "k"}
