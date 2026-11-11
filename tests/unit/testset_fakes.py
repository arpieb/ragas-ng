"""Modern LLM and embedding fakes for the testset tests (#214).

Testset generation accepts only ``llm_factory()`` LLMs and
``BaseRagasEmbedding`` models. These pass that check without a network: a
spec'd mock for tests that never call the model, and ``llm_factory`` with
LiteLLM's ``mock_response`` for tests that do.
"""

from __future__ import annotations

import typing as t
from unittest.mock import MagicMock

import instructor

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorLLM


def fake_llm() -> t.Any:
    """An ``InstructorLLM`` stand-in that is never called."""
    return MagicMock(spec=InstructorLLM)


def fake_embedding() -> t.Any:
    """A ``BaseRagasEmbedding`` stand-in that is never called."""
    return MagicMock(spec=BaseRagasEmbedding)


def mock_response_llm(response: str) -> InstructorLLM:
    """A real ``llm_factory()`` LLM whose every completion is ``response``."""
    return t.cast(
        InstructorLLM,
        llm_factory("gpt-4o", mock_response=response, mode=instructor.Mode.JSON),
    )
