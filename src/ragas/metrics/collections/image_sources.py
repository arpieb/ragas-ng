"""Loading the images that multimodal metrics send to the LLM.

A context string in ``retrieved_contexts`` may name an image: a base64 data
URI, an http(s) URL or a local file path. Evaluation data is not always
trusted, so loading is restricted by an :class:`ImageSourcePolicy`
(#229, porting the checks upstream added in vibrantlabsai/ragas#1991):

- Data URIs are always accepted.
- URLs are fetched only when the host resolves to public (globally routable)
  addresses. Redirects are followed by hand, and every hop is checked again.
  The download is capped in size and time, and the bytes must decode as an
  image.
- Local files are read only when the policy names a directory
  (``local_dir``), and only from inside it, after symlinks are resolved.

A context that is not loaded as an image is sent to the LLM as text.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import os
import re
import socket
import typing as t
from dataclasses import dataclass, field
from io import BytesIO
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image
from urllib3.util import parse_url as urllib3_parse_url

logger = logging.getLogger(__name__)

ALLOWED_URL_SCHEMES = {"http", "https"}
MAX_DOWNLOAD_SIZE_BYTES = 10 * 1024 * 1024
REQUESTS_TIMEOUT_SECONDS = 10
MAX_REDIRECTS = 5
DATA_URI_REGEX = re.compile(
    r"^data:(image\/(?:png|jpeg|gif|webp));base64,([a-zA-Z0-9+/=]+)$"
)
COMMON_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


class ImageData(t.TypedDict):
    mime_type: str
    encoded_data: str


@dataclass(frozen=True)
class ImageSourcePolicy:
    """Where a multimodal metric may load context images from.

    The default policy accepts data URIs and public http(s) URLs, and reads no
    local files.

    Args:
        local_dir: Directory local image paths are read from. Relative paths
            are resolved against it; absolute paths must lie inside it. Any
            path that resolves outside it, symlinks included, is refused.
            ``None`` (the default) reads no local files.
        allow_private_urls: Also fetch URLs whose host resolves to a private,
            loopback, link-local or otherwise non-public address. Only for
            data you trust, e.g. images on an internal server.
        max_bytes: Largest image, downloaded or read, in bytes.

    Note:
        The host is resolved for the check and again when connecting, so a
        hostname whose DNS answer changes in between (DNS rebinding) can
        still reach an internal address. For untrusted data, also restrict
        outbound network access where the evaluation runs.
    """

    local_dir: t.Optional[t.Union[str, os.PathLike[str]]] = None
    allow_private_urls: bool = False
    max_bytes: int = MAX_DOWNLOAD_SIZE_BYTES
    _resolved_dir: t.Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.local_dir is not None:
            resolved = os.path.realpath(os.fspath(self.local_dir))
            if not os.path.isdir(resolved):
                raise ValueError(
                    f"ImageSourcePolicy(local_dir={self.local_dir!r}): not a directory"
                )
            object.__setattr__(self, "_resolved_dir", resolved)
        if self.max_bytes <= 0:
            raise ValueError("ImageSourcePolicy(max_bytes=...) must be positive")


DEFAULT_IMAGE_POLICY = ImageSourcePolicy()


def is_image_path_or_url(item: str) -> bool:
    """Whether ``item`` looks like a data URI, an http(s) URL or an image path."""
    if not isinstance(item, str) or not item:
        return False
    if DATA_URI_REGEX.match(item):
        return True
    try:
        if urlparse(item).scheme in ALLOWED_URL_SCHEMES:
            return True
    except ValueError:
        pass
    return _has_image_extension(item)


def process_image_to_base64(
    item: str, policy: t.Optional[ImageSourcePolicy] = None
) -> t.Optional[ImageData]:
    """Load ``item`` as an image under ``policy``, or return None.

    None means ``item`` is not an image, or not one ``policy`` lets us load.
    """
    policy = policy or DEFAULT_IMAGE_POLICY
    if not isinstance(item, str) or not item:
        return None
    if DATA_URI_REGEX.match(item):
        return _try_process_base64_uri(item)
    try:
        is_url = urlparse(item).scheme in ALLOWED_URL_SCHEMES
    except ValueError:
        is_url = False
    if is_url:
        return _try_process_url(item, policy)
    return _try_process_local_file(item, policy)


def _has_image_extension(item: str) -> bool:
    return os.path.splitext(item)[1].lower() in COMMON_IMAGE_EXTENSIONS


def _try_process_base64_uri(item: str) -> t.Optional[ImageData]:
    match = DATA_URI_REGEX.match(item)
    if not match:
        return None
    try:
        base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None
    return {"mime_type": match.group(1), "encoded_data": match.group(2)}


def _encode_verified_image(data: bytes) -> t.Optional[ImageData]:
    """Encode ``data`` if Pillow decodes it as an image; the MIME type comes
    from the decoded format, not from a file name or a response header."""
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            img_format = img.format
    except (Image.UnidentifiedImageError, SyntaxError, OSError, ValueError):
        return None
    if not img_format:
        return None
    return {
        "mime_type": f"image/{img_format.lower()}",
        "encoded_data": base64.b64encode(data).decode("utf-8"),
    }


# --- URLs ---------------------------------------------------------------


def _validated_request_host(url: str) -> t.Optional[str]:
    """The host ``requests`` will connect to for ``url``, or None to refuse.

    ``urllib.parse`` and ``requests``/``urllib3`` disagree on malformed
    authorities (``http://127.0.0.1:6666\\@1.1.1.1`` is ``1.1.1.1`` to one and
    ``127.0.0.1`` to the other), so the host is read from the prepared
    request, and a URL the two disagree on, or one with credentials or a
    backslash in its authority, is refused.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            return None
        if "\\" in parsed.netloc or "@" in parsed.netloc:
            logger.warning(
                f"Image URL refused: credentials or a backslash in the authority of {url!r}"
            )
            return None
        prepared_url = requests.Request("GET", url).prepare().url or ""
        request_host = urllib3_parse_url(prepared_url).host
    except Exception as e:
        logger.warning(f"Image URL refused: could not parse {url!r}: {e}")
        return None
    if not request_host or not parsed.hostname:
        return None
    # urllib3 keeps IPv6 brackets; urlparse strips them.
    if request_host.strip("[]").lower() != parsed.hostname.lower():
        logger.warning(
            f"Image URL refused: URL parsers disagree on the host of {url!r}"
        )
        return None
    return parsed.hostname


def _is_public_host(hostname: str) -> bool:
    """Whether every address ``hostname`` resolves to is globally routable."""
    try:
        results = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except (socket.gaierror, UnicodeError, OSError) as e:
        logger.warning(f"Image URL refused: cannot resolve {hostname!r}: {e}")
        return False
    if not results:
        return False
    for *_, sockaddr in results:
        address = str(sockaddr[0])
        try:
            # Strip an IPv6 zone id ("fe80::1%eth0").
            ip = ipaddress.ip_address(address.split("%")[0])
        except ValueError:
            return False
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global excludes private, loopback, link-local, reserved,
        # unspecified and shared (CGNAT) space; multicast is checked
        # separately because some multicast ranges count as global.
        if not ip.is_global or ip.is_multicast:
            logger.warning(
                f"Image URL refused: {hostname!r} resolves to non-public address {address}"
            )
            return False
    return True


def _fetch(url: str, policy: ImageSourcePolicy) -> t.Optional[requests.Response]:
    """GET ``url``, following up to MAX_REDIRECTS redirects by hand so that
    every hop passes the host checks."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = _validated_request_host(current)
        if host is None:
            return None
        if not policy.allow_private_urls and not _is_public_host(host):
            return None
        response = requests.get(
            current,
            timeout=REQUESTS_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=False,
        )
        if not response.is_redirect:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            return None
        current = urljoin(current, location)
    logger.warning(
        f"Image URL refused: more than {MAX_REDIRECTS} redirects from {url!r}"
    )
    return None


def _try_process_url(url: str, policy: ImageSourcePolicy) -> t.Optional[ImageData]:
    try:
        response = _fetch(url, policy)
        if response is None:
            return None
        with response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > policy.max_bytes:
                logger.warning(
                    f"Image URL {url!r} is larger than {policy.max_bytes} bytes"
                )
                return None
            buffer = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if buffer.tell() + len(chunk) > policy.max_bytes:
                    logger.warning(
                        f"Image URL {url!r} is larger than {policy.max_bytes} bytes"
                    )
                    return None
                buffer.write(chunk)
        return _encode_verified_image(buffer.getvalue())
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.warning(f"Could not fetch image URL {url!r}: {e}")
        return None


# --- Local files --------------------------------------------------------


def _try_process_local_file(
    item: str, policy: ImageSourcePolicy
) -> t.Optional[ImageData]:
    base = policy._resolved_dir
    if base is None:
        if _has_image_extension(item):
            logger.warning(
                f"Context {item!r} looks like a local image path, but local image "
                "files are not read by default; it is sent as text. To read "
                "local images, pass image_policy=ImageSourcePolicy(local_dir=...) "
                "to the metric."
            )
        return None
    try:
        path = os.path.realpath(os.path.join(base, item))
        inside = os.path.commonpath([path, base]) == base
    except (TypeError, ValueError, OSError):
        # ValueError includes a NUL byte, or a path on another drive on Windows.
        return None
    if not inside:
        logger.warning(f"Image path {item!r} is outside local_dir {policy.local_dir!r}")
        return None
    try:
        if not os.path.isfile(path):
            return None
        if os.path.getsize(path) > policy.max_bytes:
            logger.warning(
                f"Image file {item!r} is larger than {policy.max_bytes} bytes"
            )
            return None
        with open(path, "rb") as f:
            data = f.read(policy.max_bytes + 1)
    except OSError as e:
        logger.warning(f"Could not read image file {item!r}: {e}")
        return None
    if len(data) > policy.max_bytes:
        return None
    return _encode_verified_image(data)


__all__ = [
    "ImageSourcePolicy",
    "ImageData",
    "is_image_path_or_url",
    "process_image_to_base64",
]
