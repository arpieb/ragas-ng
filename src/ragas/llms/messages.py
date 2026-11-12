"""What ``generate``/``agenerate`` send: a prompt as OpenAI-format messages."""

from __future__ import annotations

import typing as t

# One chat message in OpenAI format: ``{"role": ..., "content": ...}``, where
# ``content`` is a string or a list of content parts (``{"type": "text", ...}``,
# ``{"type": "image_url", ...}``). LiteLLM translates these for each provider.
ChatMessage = t.Dict[str, t.Any]
# What ``generate``/``agenerate`` take: a prompt string, sent as one user
# message, or a list of chat messages sent as given (#214).
LLMInput = t.Union[str, t.Sequence[ChatMessage]]


def build_messages(
    prompt: LLMInput, system_prompt: t.Optional[str]
) -> t.List[ChatMessage]:
    """The messages sent for ``prompt``, with ``system_prompt`` prepended.

    A string becomes one user message. A list of messages is sent as given
    (copied, never mutated); ``system_prompt`` is prepended unless the list
    already has a system message, which then takes precedence.
    """
    if isinstance(prompt, str):
        messages: t.List[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    if not isinstance(prompt, t.Sequence) or isinstance(prompt, (bytes, bytearray)):
        raise TypeError(
            "prompt must be a string or a list of chat messages "
            f"({{'role': ..., 'content': ...}}), not {type(prompt).__name__}"
        )
    messages = []
    for message in prompt:
        if not isinstance(message, t.Mapping) or "role" not in message:
            raise TypeError(
                "each chat message must be a dict with 'role' and 'content', "
                f"got {message!r:.200}"
            )
        messages.append(dict(message))
    if not messages:
        raise ValueError("prompt is an empty list of chat messages")
    if system_prompt and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages
