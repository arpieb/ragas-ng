"""Helpers shared by the chat-message converters (AG-UI, OpenAI-format/Swarm)."""

from __future__ import annotations

import json
import logging
import typing as t

logger = logging.getLogger(__name__)


def text_from_content(content: t.Any) -> str:
    """The text of a message's content.

    ``None`` is ``""``; a string is returned as is; a list of content parts
    (AG-UI ``InputContent`` objects or OpenAI-style dicts) keeps its text
    parts, joined by newlines. ragas messages are text-only, so image,
    audio, video, document and binary parts are skipped. ``str()`` on these
    gave ``"None"`` or a pydantic repr (#96).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        texts: t.List[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue
            if isinstance(part, dict):
                kind, text = part.get("type"), part.get("text")
            else:
                kind, text = getattr(part, "type", None), getattr(part, "text", None)
            if kind in (None, "text", "input_text") and isinstance(text, str):
                texts.append(text)
            else:
                logger.debug("Skipping non-text content part of type %r", kind)
        return "\n".join(texts)
    return str(content)


def parse_tool_arguments(arguments: t.Any) -> t.Dict[str, t.Any]:
    """Tool-call arguments as a dict.

    Chat formats carry arguments as a JSON string. A dict passes through;
    empty means no arguments; anything that isn't a JSON object is kept as
    ``{"raw_args": ...}`` rather than failing the whole conversion.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {"raw_args": arguments}
        return parsed if isinstance(parsed, dict) else {"raw_args": arguments}
    return {"raw_args": arguments}
