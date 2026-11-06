"""Deprecated: ``llm_factory()`` has one path, LiteLLM (#182).

``get_adapter`` / ``auto_detect_adapter`` and the adapters stay importable for
one release. Both adapters now build their LLM with ``llm_factory()``, so they
behave identically and accept the same clients. Call ``llm_factory()``
directly.
"""

import typing as t
import warnings

from ragas.llms.adapters.instructor import InstructorAdapter
from ragas.llms.adapters.litellm import LiteLLMAdapter

ADAPTERS = {
    "instructor": InstructorAdapter(),
    "litellm": LiteLLMAdapter(),
}


def get_adapter(name: str) -> t.Any:
    """Deprecated. The adapter registered as ``name``; both call ``llm_factory()``."""
    if name not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {name}. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name]


def auto_detect_adapter(client: t.Any, provider: str) -> str:
    """Deprecated. Always ``"litellm"``: every call goes through LiteLLM (#182)."""
    warnings.warn(
        "auto_detect_adapter() is deprecated: llm_factory() has one path, "
        "LiteLLM (#182).",
        DeprecationWarning,
        stacklevel=2,
    )
    return "litellm"


__all__ = [
    "get_adapter",
    "auto_detect_adapter",
    "ADAPTERS",
]
