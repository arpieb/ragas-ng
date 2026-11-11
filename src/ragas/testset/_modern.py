"""The LLM, embedding and prompt types testset generation is built on (#214).

Testset generation takes an ``llm_factory()`` LLM
(``InstructorBaseRagasLLM``) and a modern embedding (``BaseRagasEmbedding``,
e.g. from ``embedding_factory()``), and its prompts are
``ragas.prompt.BasePrompt``. Anything else is rejected with a ``TypeError``
that says what to use instead.
"""

from __future__ import annotations

import typing as t

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

_T = t.TypeVar("_T")


def require_llm(llm: t.Any, owner: str) -> InstructorBaseRagasLLM:
    """``llm``, if it is an ``llm_factory()`` LLM; else a ``TypeError``."""
    if isinstance(llm, InstructorBaseRagasLLM):
        return llm
    raise TypeError(
        f"{owner} needs an LLM created with ragas.llms.llm_factory(), e.g. "
        f"llm_factory('gpt-4o-mini'); got {type(llm).__name__}. "
        "llm_factory() reaches most providers through LiteLLM."
    )


def require_embedding(embedding: t.Any, owner: str) -> BaseRagasEmbedding:
    """``embedding``, if it is a modern embedding; else a ``TypeError``."""
    if isinstance(embedding, BaseRagasEmbedding):
        return embedding
    raise TypeError(
        f"{owner} needs a modern embedding model (a BaseRagasEmbedding), e.g. "
        "ragas.embeddings.base.embedding_factory('openai', "
        "model='text-embedding-3-small') or ragas.embeddings.LiteLLMEmbeddings; "
        f"got {type(embedding).__name__}."
    )


def optional(
    check: t.Callable[[t.Any, str], _T], value: t.Any, owner: str
) -> t.Optional[_T]:
    """``check(value, owner)``, letting ``None`` through."""
    return None if value is None else check(value, owner)
