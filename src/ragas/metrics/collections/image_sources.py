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

A context that is not an image reference (plain text) is sent to the LLM as
text. An image reference that is refused or fails to load raises
:class:`~ragas.exceptions.ImageLoadError` (#231), unless the policy says
``on_blocked="warn"``: then a warning is logged and the reference is sent as
text, so the judge never sees the image.
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

from ragas.exceptions import ImageLoadError

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


OnBlocked = t.Literal["raise", "warn"]


@dataclass(frozen=True)
class ImageSourcePolicy:
    """Where a multimodal metric may load context images from.

    The default policy accepts data URIs and public http(s) URLs, reads no
    local files, and raises :class:`~ragas.exceptions.ImageLoadError` for an
    image reference it refuses or cannot load.

    Args:
        local_dir: Directory local image paths are read from. Relative paths
            are resolved against it; absolute paths must lie inside it. Any
            path that resolves outside it, symlinks included, is refused.
            ``None`` (the default) reads no local files.
        allow_private_urls: Also fetch URLs whose host resolves to a private,
            loopback, link-local or otherwise non-public address. Only for
            data you trust, e.g. images on an internal server.
        max_bytes: Largest image, downloaded or read, in bytes.
        on_blocked: What to do with an image reference that is refused or
            fails to load. ``"raise"`` (the default) raises ``ImageLoadError``;
            ``"warn"`` logs a warning and sends the reference to the LLM as
            text, so the score is computed without the image.

    Note:
        The host is resolved for the check and again when connecting, so a
        hostname whose DNS answer changes in between (DNS rebinding) can
        still reach an internal address. For untrusted data, also restrict
        outbound network access where the evaluation runs.
    """

    local_dir: t.Optional[t.Union[str, os.PathLike[str]]] = None
    allow_private_urls: bool = False
    max_bytes: int = MAX_DOWNLOAD_SIZE_BYTES
    on_blocked: OnBlocked = "raise"
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
        if self.on_blocked not in ("raise", "warn"):
            raise ValueError(
                f"ImageSourcePolicy(on_blocked={self.on_blocked!r}): "
                "use 'raise' or 'warn'"
            )


DEFAULT_IMAGE_POLICY = ImageSourcePolicy()


def is_image_path_or_url(item: str) -> bool:
    """Whether ``item`` is an image reference: an image data URI, an http(s)
    URL, or a path with an image file extension."""
    if not isinstance(item, str) or not item:
        return False
    if item.startswith("data:image/") or DATA_URI_REGEX.match(item):
        return True
    if _is_url(item):
        return True
    return _has_image_extension(item)


def process_image_to_base64(
    item: str, policy: t.Optional[ImageSourcePolicy] = None
) -> t.Optional[ImageData]:
    """Load ``item`` as an image under ``policy``.

    Returns None when ``item`` is not an image reference (see
    :func:`is_image_path_or_url`), so it should be sent as text.

    Raises:
        ImageLoadError: ``item`` is an image reference that ``policy`` refuses
            or that fails to load, and ``policy.on_blocked`` is ``"raise"``.
            With ``"warn"``, a warning is logged and None is returned.
    """
    policy = policy or DEFAULT_IMAGE_POLICY
    if not is_image_path_or_url(item):
        return None
    try:
        return load_image(item, policy)
    except ImageLoadError as e:
        if policy.on_blocked == "raise":
            raise
        logger.warning(f"{e.describe()} Sending it to the LLM as text.")
        return None


def load_image(item: str, policy: ImageSourcePolicy) -> ImageData:
    """Load the image reference ``item``, or raise ``ImageLoadError``."""
    if item.startswith("data:"):
        return _load_data_uri(item)
    if _is_url(item):
        return _load_url(item, policy)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", item):
        raise ImageLoadError(item, "only http and https URLs are fetched")
    return _load_local_file(item, policy)


def _is_url(item: str) -> bool:
    try:
        return urlparse(item).scheme in ALLOWED_URL_SCHEMES
    except ValueError:
        return False


def _has_image_extension(item: str) -> bool:
    return os.path.splitext(item)[1].lower() in COMMON_IMAGE_EXTENSIONS


def _load_data_uri(item: str) -> ImageData:
    match = DATA_URI_REGEX.match(item)
    if not match:
        raise ImageLoadError(
            item,
            "not a base64 PNG, JPEG, GIF or WebP data URI",
        )
    try:
        base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageLoadError(item, f"invalid base64 data ({e})") from None
    return {"mime_type": match.group(1), "encoded_data": match.group(2)}


def _encode_verified_image(item: str, data: bytes) -> ImageData:
    """Encode ``data`` if Pillow decodes it as an image; the MIME type comes
    from the decoded format, not from a file name or a response header."""
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            img_format = img.format
    except (Image.UnidentifiedImageError, SyntaxError, OSError, ValueError):
        img_format = None
    if not img_format:
        raise ImageLoadError(item, "the content is not an image")
    return {
        "mime_type": f"image/{img_format.lower()}",
        "encoded_data": base64.b64encode(data).decode("utf-8"),
    }


def _too_large(item: str, policy: ImageSourcePolicy) -> ImageLoadError:
    return ImageLoadError(
        item,
        f"larger than {policy.max_bytes} bytes",
        "Raise ImageSourcePolicy(max_bytes=...) to allow it.",
    )


# --- URLs ---------------------------------------------------------------


def _validated_request_host(url: str) -> str:
    """The host ``requests`` will connect to for ``url``.

    ``urllib.parse`` and ``requests``/``urllib3`` disagree on malformed
    authorities (``http://127.0.0.1:6666\\@1.1.1.1`` is ``1.1.1.1`` to one and
    ``127.0.0.1`` to the other), so the host is read from the prepared
    request, and a URL the two disagree on, or one with credentials or a
    backslash in its authority, is refused.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            raise ImageLoadError(url, "only http and https URLs are fetched")
        if "\\" in parsed.netloc or "@" in parsed.netloc:
            raise ImageLoadError(
                url, "the URL has credentials or a backslash in its host part"
            )
        prepared_url = requests.Request("GET", url).prepare().url or ""
        request_host = urllib3_parse_url(prepared_url).host
    except ImageLoadError:
        raise
    except Exception as e:
        raise ImageLoadError(url, f"the URL could not be parsed ({e})") from None
    if not request_host or not parsed.hostname:
        raise ImageLoadError(url, "the URL has no host")
    # urllib3 keeps IPv6 brackets; urlparse strips them.
    if request_host.strip("[]").lower() != parsed.hostname.lower():
        raise ImageLoadError(url, "URL parsers disagree on its host")
    return parsed.hostname


def _check_public_host(url: str, hostname: str) -> None:
    """Raise unless every address ``hostname`` resolves to is globally
    routable."""
    try:
        results = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except (socket.gaierror, UnicodeError, OSError) as e:
        raise ImageLoadError(url, f"cannot resolve host {hostname!r} ({e})") from None
    if not results:
        raise ImageLoadError(url, f"cannot resolve host {hostname!r}")
    for *_, sockaddr in results:
        address = str(sockaddr[0])
        try:
            # Strip an IPv6 zone id ("fe80::1%eth0").
            ip = ipaddress.ip_address(address.split("%")[0])
        except ValueError:
            raise ImageLoadError(
                url, f"host {hostname!r} resolved to unparseable address {address!r}"
            ) from None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global excludes private, loopback, link-local, reserved,
        # unspecified and shared (CGNAT) space; multicast is checked
        # separately because some multicast ranges count as global.
        if not ip.is_global or ip.is_multicast:
            raise ImageLoadError(
                url,
                f"host {hostname!r} resolves to non-public address {address}",
                "If you trust the data, ImageSourcePolicy(allow_private_urls=True) "
                "allows it.",
            )


def _fetch(url: str, policy: ImageSourcePolicy) -> requests.Response:
    """GET ``url``, following up to MAX_REDIRECTS redirects by hand so that
    every hop passes the host checks."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        host = _validated_request_host(current)
        if not policy.allow_private_urls:
            _check_public_host(current, host)
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
            raise ImageLoadError(url, "a redirect had no Location header")
        current = urljoin(current, location)
    raise ImageLoadError(url, f"more than {MAX_REDIRECTS} redirects")


def _load_url(url: str, policy: ImageSourcePolicy) -> ImageData:
    try:
        with _fetch(url, policy) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > policy.max_bytes:
                raise _too_large(url, policy)
            buffer = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                if buffer.tell() + len(chunk) > policy.max_bytes:
                    raise _too_large(url, policy)
                buffer.write(chunk)
    except ImageLoadError as e:
        if e.source != url:
            # Raised for a redirect hop: report it against the original URL.
            raise ImageLoadError(
                url, f"{e.reason} (after a redirect)", e.hint
            ) from None
        raise
    except (requests.exceptions.RequestException, ValueError) as e:
        raise ImageLoadError(url, f"the download failed ({e})") from None
    return _encode_verified_image(url, buffer.getvalue())


# --- Local files --------------------------------------------------------


def _load_local_file(item: str, policy: ImageSourcePolicy) -> ImageData:
    base = policy._resolved_dir
    if base is None:
        raise ImageLoadError(
            item,
            "local image files are not read by default",
            "To read them, pass image_policy=ImageSourcePolicy(local_dir=...) "
            "to the metric.",
        )
    if "\x00" in item:
        # Checked here because os.path handles NUL bytes differently by
        # platform (it raises on POSIX, not always on Windows).
        raise ImageLoadError(item, "the path contains a NUL byte")
    try:
        path = os.path.realpath(os.path.join(base, item))
        inside = os.path.commonpath([path, base]) == base
    except (TypeError, ValueError, OSError):
        # ValueError includes a path on another drive on Windows.
        inside = False
    if not inside:
        raise ImageLoadError(item, f"the path is outside local_dir {base!r}")
    try:
        if not os.path.isfile(path):
            raise ImageLoadError(item, "no such file in local_dir")
        if os.path.getsize(path) > policy.max_bytes:
            raise _too_large(item, policy)
        with open(path, "rb") as f:
            data = f.read(policy.max_bytes + 1)
    except OSError as e:
        raise ImageLoadError(item, f"the file could not be read ({e})") from None
    if len(data) > policy.max_bytes:
        raise _too_large(item, policy)
    return _encode_verified_image(item, data)


__all__ = [
    "ImageLoadError",
    "ImageSourcePolicy",
    "ImageData",
    "is_image_path_or_url",
    "load_image",
    "process_image_to_base64",
]
