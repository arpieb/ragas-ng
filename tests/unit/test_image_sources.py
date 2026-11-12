"""Where the multimodal metrics may load context images from (#229).

The collections ``MultiModalFaithfulness`` and ``MultiModalRelevance`` used
to open any local image path and fetch any http(s) URL, redirects included,
with none of the checks upstream added in vibrantlabsai/ragas#1991. These
tests need no network: DNS (``socket.getaddrinfo``) and ``requests.get`` are
patched.
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import typing as t
from io import BytesIO

import pytest
from PIL import Image

from ragas.metrics.collections import (
    ImageSourcePolicy,
    MultiModalFaithfulness,
    MultiModalRelevance,
    image_sources,
)
from ragas.metrics.collections.image_sources import process_image_to_base64


def _png(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "ok.png").write_bytes(PNG)
    (tmp_path / "secret.png").write_bytes(_png("blue"))
    return d


# --- Local files ---------------------------------------------------------


def test_local_files_are_not_read_by_default(image_dir, caplog):
    path = str(image_dir / "ok.png")
    with caplog.at_level(logging.WARNING, logger=image_sources.__name__):
        assert process_image_to_base64(path) is None
    assert "local_dir" in caplog.text


def test_local_dir_reads_relative_and_absolute_paths_inside_it(image_dir):
    policy = ImageSourcePolicy(local_dir=image_dir)
    for item in ("ok.png", str(image_dir / "ok.png")):
        result = process_image_to_base64(item, policy)
        assert result is not None
        assert result["mime_type"] == "image/png"
        assert base64.b64decode(result["encoded_data"]) == PNG


@pytest.mark.parametrize(
    "item",
    ["../secret.png", "sub/../../secret.png", "{root}/secret.png", "/etc/passwd"],
)
def test_paths_outside_local_dir_are_refused(image_dir, item):
    item = item.format(root=image_dir.parent)
    policy = ImageSourcePolicy(local_dir=image_dir)
    assert process_image_to_base64(item, policy) is None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_symlink_out_of_local_dir_is_refused(image_dir):
    link = image_dir / "link.png"
    try:
        link.symlink_to(image_dir.parent / "secret.png")
    except OSError:
        pytest.skip("cannot create symlinks here")
    policy = ImageSourcePolicy(local_dir=image_dir)
    assert process_image_to_base64("link.png", policy) is None


def test_a_symlinked_local_dir_works(image_dir, tmp_path):
    link = tmp_path / "linked"
    try:
        link.symlink_to(image_dir, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symlinks here")
    policy = ImageSourcePolicy(local_dir=link)
    assert process_image_to_base64("ok.png", policy) is not None
    assert process_image_to_base64(str(link / "ok.png"), policy) is not None


def test_non_images_and_oversized_files_are_refused(image_dir):
    (image_dir / "notes.png").write_text("not an image")
    assert process_image_to_base64("notes.png", ImageSourcePolicy(image_dir)) is None
    small = ImageSourcePolicy(local_dir=image_dir, max_bytes=len(PNG) - 1)
    assert process_image_to_base64("ok.png", small) is None


def test_plain_text_is_not_an_image_under_local_dir(image_dir):
    policy = ImageSourcePolicy(local_dir=image_dir)
    assert process_image_to_base64("The sky is blue.", policy) is None
    assert process_image_to_base64("nul\x00byte", policy) is None


def test_local_dir_must_be_a_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        ImageSourcePolicy(local_dir=tmp_path / "missing")
    with pytest.raises(ValueError, match="max_bytes"):
        ImageSourcePolicy(max_bytes=0)


# --- URLs ----------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        status: int = 200,
        body: bytes = PNG,
        headers: t.Optional[t.Dict[str, str]] = None,
    ):
        self.status_code = status
        self.body = body
        self.headers = headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308) and (
            "Location" in self.headers
        )
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int = 8192):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Net:
    """Patched DNS and HTTP: ``dns`` maps host to addresses, ``routes`` maps
    URL to response; ``requested`` records every GET."""

    def __init__(self, monkeypatch):
        self.dns: t.Dict[str, t.List[str]] = {}
        self.routes: t.Dict[str, FakeResponse] = {}
        self.requested: t.List[str] = []
        self.kwargs: t.List[t.Dict[str, t.Any]] = []
        monkeypatch.setattr(image_sources.socket, "getaddrinfo", self.getaddrinfo)
        monkeypatch.setattr(image_sources.requests, "get", self.get)

    def getaddrinfo(self, host, port, family=0, *args, **kwargs):
        if host not in self.dns:
            raise socket.gaierror(f"unknown host {host}")
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, 1, 6, "", (a, 0))
            for a in self.dns[host]
        ]

    def get(self, url, **kwargs):
        self.requested.append(url)
        self.kwargs.append(kwargs)
        return self.routes.get(url, FakeResponse(404, b""))


@pytest.fixture
def net(monkeypatch):
    return Net(monkeypatch)


def test_a_public_url_is_fetched_without_automatic_redirects(net):
    net.dns["img.example"] = ["93.184.216.34"]
    net.routes["https://img.example/a.png"] = FakeResponse()
    result = process_image_to_base64("https://img.example/a.png")
    assert result is not None and result["mime_type"] == "image/png"
    assert net.kwargs[0]["allow_redirects"] is False
    assert net.kwargs[0]["timeout"]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # shared (CGNAT) space
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fe80::1%eth0",
        "fd00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_non_public_addresses_are_refused(net, address):
    net.dns["internal.example"] = [address]
    assert process_image_to_base64("http://internal.example/a.png") is None
    assert net.requested == []


def test_one_private_address_among_public_ones_is_refused(net):
    net.dns["mixed.example"] = ["93.184.216.34", "10.0.0.1"]
    assert process_image_to_base64("http://mixed.example/a.png") is None
    assert net.requested == []


def test_ip_literal_urls_are_checked(net, monkeypatch):
    monkeypatch.setattr(
        image_sources.socket,
        "getaddrinfo",
        lambda host, *a, **k: [(socket.AF_INET, 1, 6, "", (host, 0))],
    )
    assert process_image_to_base64("http://169.254.169.254/latest/") is None
    assert net.requested == []


def test_unresolvable_hosts_are_refused(net):
    assert process_image_to_base64("http://nowhere.example/a.png") is None
    assert net.requested == []


def test_a_redirect_to_a_private_address_is_refused(net):
    net.dns["img.example"] = ["93.184.216.34"]
    net.dns["internal.example"] = ["10.0.0.1"]
    net.routes["http://img.example/a.png"] = FakeResponse(
        302, b"", {"Location": "http://internal.example/secret"}
    )
    assert process_image_to_base64("http://img.example/a.png") is None
    assert net.requested == ["http://img.example/a.png"]


def test_public_redirects_are_followed_and_relative_locations_resolved(net):
    net.dns["img.example"] = ["93.184.216.34"]
    net.dns["cdn.example"] = ["93.184.216.35"]
    net.routes["http://img.example/a.png"] = FakeResponse(
        301, b"", {"Location": "https://cdn.example/x/a.png"}
    )
    net.routes["https://cdn.example/x/a.png"] = FakeResponse(
        302, b"", {"Location": "../b.png"}
    )
    net.routes["https://cdn.example/b.png"] = FakeResponse()
    assert process_image_to_base64("http://img.example/a.png") is not None
    assert net.requested[-1] == "https://cdn.example/b.png"


def test_redirect_loops_stop(net):
    net.dns["img.example"] = ["93.184.216.34"]
    net.routes["http://img.example/a.png"] = FakeResponse(
        302, b"", {"Location": "/a.png"}
    )
    assert process_image_to_base64("http://img.example/a.png") is None
    assert len(net.requested) == image_sources.MAX_REDIRECTS + 1


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:6666\\@img.example/a.png",  # parsers disagree on host
        "http://user:pw@img.example/a.png",
        "ftp://img.example/a.png",
    ],
)
def test_ambiguous_or_credentialed_urls_are_refused(net, url):
    net.dns["img.example"] = ["93.184.216.34"]
    assert process_image_to_base64(url) is None
    assert net.requested == []


def test_allow_private_urls_opts_in(net):
    net.dns["intranet"] = ["10.0.0.7"]
    net.routes["http://intranet/a.png"] = FakeResponse()
    policy = ImageSourcePolicy(allow_private_urls=True)
    assert process_image_to_base64("http://intranet/a.png", policy) is not None


def test_oversized_and_non_image_downloads_are_refused(net):
    net.dns["img.example"] = ["93.184.216.34"]
    net.routes["http://img.example/big.png"] = FakeResponse(
        headers={"Content-Length": str(10**9)}
    )
    net.routes["http://img.example/stream.png"] = FakeResponse(body=PNG * 4)
    net.routes["http://img.example/page.png"] = FakeResponse(body=b"<html></html>")
    small = ImageSourcePolicy(max_bytes=len(PNG) * 2)
    assert process_image_to_base64("http://img.example/big.png") is None
    assert process_image_to_base64("http://img.example/stream.png", small) is None
    assert process_image_to_base64("http://img.example/page.png") is None


def test_data_uris_still_work_without_network(net):
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    result = process_image_to_base64(uri)
    assert result is not None and result["encoded_data"] in uri
    assert net.requested == []


# --- The metrics ---------------------------------------------------------


class CapturingLLM:
    """A real LLM whose ``agenerate`` records the messages and replies with a
    fixed verdict, so no request is made."""

    def __init__(self, monkeypatch):
        from ragas.llms import llm_factory

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        self.llm = llm_factory("gpt-6-luna")
        self.messages: t.List[t.Any] = []

        async def agenerate(messages, response_model, **kwargs):
            self.messages.append(messages)
            fields = response_model.model_fields
            return response_model(**{k: True for k in fields if k != "reason"})

        monkeypatch.setattr(self.llm, "agenerate", agenerate)


def _parts(llm: CapturingLLM) -> t.List[t.Dict[str, t.Any]]:
    return llm.messages[-1][0]["content"]


METRICS = [
    (MultiModalFaithfulness, dict(response="Red.")),
    (MultiModalRelevance, dict(user_input="Colour?", response="Red.")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs", METRICS)
async def test_metrics_send_local_paths_as_text_by_default(
    metric_cls, inputs, image_dir, monkeypatch
):
    llm = CapturingLLM(monkeypatch)
    metric = metric_cls(llm=llm.llm)
    assert metric.image_policy == ImageSourcePolicy()
    path = str(image_dir / "ok.png")
    await metric.ascore(retrieved_contexts=[path], **inputs)
    parts = _parts(llm)
    assert not [p for p in parts if p["type"] == "image_url"]
    assert any(path in p.get("text", "") for p in parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_cls, inputs", METRICS)
async def test_metrics_read_images_from_their_local_dir(
    metric_cls, inputs, image_dir, monkeypatch
):
    llm = CapturingLLM(monkeypatch)
    metric = metric_cls(
        llm=llm.llm, image_policy=ImageSourcePolicy(local_dir=image_dir)
    )
    await metric.ascore(retrieved_contexts=["ok.png", "../secret.png"], **inputs)
    images = [p for p in _parts(llm) if p["type"] == "image_url"]
    assert len(images) == 1
    assert base64.b64encode(PNG).decode() in images[0]["image_url"]["url"]
