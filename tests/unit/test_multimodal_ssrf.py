"""SSRF guard for image URLs fetched by ImageTextPromptValue.

No network: DNS resolution and ``requests.get`` are stubbed, so these tests
pin which hosts the guard lets through and which it refuses.
"""

from __future__ import annotations

import socket
import typing as t

import pytest

from ragas.prompt import multi_modal_prompt as mmp
from ragas.prompt.multi_modal_prompt import ImageTextPromptValue

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def pv() -> ImageTextPromptValue:
    return ImageTextPromptValue(items=[])


@pytest.fixture
def fake_dns(monkeypatch):
    """Resolve hostnames from a table; IP literals resolve to themselves."""
    table: t.Dict[str, str] = {"public.example": PUBLIC_IP}

    def getaddrinfo(host, port, family=0, *args, **kwargs):
        ip = table.get(host) or str(host)
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(mmp.socket, "getaddrinfo", getaddrinfo)
    return table


class FakeResponse:
    def __init__(self, status: int = 200, location: t.Optional[str] = None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self.is_redirect = location is not None and 300 <= status < 400
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_get(monkeypatch):
    """Record requested URLs; reply per a url -> FakeResponse table."""
    calls: t.List[str] = []
    routes: t.Dict[str, FakeResponse] = {}

    def get(url, **kwargs):
        assert kwargs.get("allow_redirects") is False
        calls.append(url)
        return routes.get(url, FakeResponse())

    monkeypatch.setattr(mmp.requests, "get", get)
    return calls, routes


# --- host extraction -------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        # urlparse sees 1.1.1.1, requests connects to 127.0.0.1 (upstream #2754)
        "http://127.0.0.1:6666\\@1.1.1.1",
        "http://127.0.0.1\\@1.1.1.1/x.png",
        # userinfo is never needed for an image fetch and hides the host
        "http://user@127.0.0.1/x.png",
        "http://1.1.1.1@127.0.0.1/x.png",
        "ftp://public.example/x.png",
        "file:///etc/passwd",
    ],
)
def test_rejects_ambiguous_or_disallowed_urls(pv, url):
    assert pv._validated_request_host(url) is None


@pytest.mark.parametrize(
    "url, host",
    [
        ("https://public.example/img.png", "public.example"),
        ("http://PUBLIC.example:8080/a/b.png?x=1", "public.example"),
        ("http://[2606:4700::1111]/x.png", "2606:4700::1111"),
    ],
)
def test_returns_request_host_for_plain_urls(pv, url, host):
    assert pv._validated_request_host(url) == host


# --- IP policy --------------------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "0.0.0.0",
        "100.64.0.1",  # CGNAT: neither private nor reserved, but not global
        "224.0.0.1",  # multicast
        "::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:10.0.0.1",
        "fe80::1%eth0",  # link-local with zone id
    ],
)
def test_blocks_non_global_targets(pv, fake_dns, ip):
    fake_dns["internal.example"] = ip
    assert pv._is_safe_url_target("internal.example") is False


def test_allows_public_target(pv, fake_dns):
    assert pv._is_safe_url_target("public.example") is True


# --- redirects --------------------------------------------------------------


def test_no_request_is_made_for_the_backslash_bypass(pv, fake_dns, fake_get):
    calls, _ = fake_get
    assert pv._fetch_with_validated_redirects("http://127.0.0.1:6666\\@1.1.1.1") is None
    assert calls == []


def test_redirect_to_internal_host_is_blocked(pv, fake_dns, fake_get):
    calls, routes = fake_get
    first = FakeResponse(302, "http://169.254.169.254/latest/meta-data/")
    routes["https://public.example/img.png"] = first

    assert pv._fetch_with_validated_redirects("https://public.example/img.png") is None
    assert calls == ["https://public.example/img.png"]
    assert first.closed


def test_relative_redirect_to_public_host_is_followed(pv, fake_dns, fake_get):
    calls, routes = fake_get
    routes["https://public.example/a.png"] = FakeResponse(301, "/b.png")

    resp = pv._fetch_with_validated_redirects("https://public.example/a.png")
    assert resp is not None and not resp.is_redirect
    assert calls == ["https://public.example/a.png", "https://public.example/b.png"]


def test_redirect_loop_is_capped(pv, fake_dns, fake_get):
    calls, routes = fake_get
    routes["https://public.example/loop.png"] = FakeResponse(
        302, "https://public.example/loop.png"
    )
    assert pv._fetch_with_validated_redirects("https://public.example/loop.png") is None
    assert len(calls) == mmp.MAX_REDIRECTS + 1


def test_end_to_end_never_contacts_internal_host(monkeypatch, fake_dns):
    """to_messages() on the bypass URL must not issue any HTTP request."""

    def boom(*a, **k):
        raise AssertionError("requests.get must not be called")

    monkeypatch.setattr(mmp.requests, "get", boom)
    messages = ImageTextPromptValue(
        items=["http://127.0.0.1:6666\\@1.1.1.1/x.png"]
    ).to_messages()
    # falls back to treating the item as text
    assert len(messages) == 1
