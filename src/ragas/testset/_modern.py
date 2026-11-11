"""The LLM, embedding and prompt types testset generation is built on (#214).

Testset generation takes an ``llm_factory()`` LLM
(``InstructorBaseRagasLLM``) and a modern embedding (``BaseRagasEmbedding``,
e.g. from ``embedding_factory()``), and its prompts are
``ragas.prompt.metrics.BasePrompt``. Legacy ``BaseRagasLLM`` /
``BaseRagasEmbeddings`` objects are rejected with a ``TypeError`` that says
what to use instead.
"""

from __future__ import annotations

import inspect
import typing as t

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.prompt.mixin import PromptMixin

_T = t.TypeVar("_T")


def require_llm(llm: t.Any, owner: str) -> InstructorBaseRagasLLM:
    """``llm``, if it is an ``llm_factory()`` LLM; else a ``TypeError``."""
    if isinstance(llm, InstructorBaseRagasLLM):
        return llm
    raise TypeError(
        f"{owner} needs an LLM created with ragas.llms.llm_factory(), e.g. "
        f"llm_factory('gpt-4o-mini'); got {type(llm).__name__}. Legacy "
        "BaseRagasLLM wrappers (LlamaIndexLLMWrapper, HaystackLLMWrapper, "
        "OCIGenAIWrapper, ...) are not supported: llm_factory() reaches those "
        "providers through LiteLLM."
    )


def require_embedding(embedding: t.Any, owner: str) -> BaseRagasEmbedding:
    """``embedding``, if it is a modern embedding; else a ``TypeError``."""
    if isinstance(embedding, BaseRagasEmbedding):
        return embedding
    raise TypeError(
        f"{owner} needs a modern embedding model (a BaseRagasEmbedding), e.g. "
        "ragas.embeddings.base.embedding_factory('openai', "
        "model='text-embedding-3-small') or ragas.embeddings.LiteLLMEmbeddings; "
        f"got {type(embedding).__name__}. Legacy BaseRagasEmbeddings wrappers "
        "(LlamaIndexEmbeddingsWrapper, HuggingfaceEmbeddings, ...) are not "
        "supported."
    )


def optional(
    check: t.Callable[[t.Any, str], _T], value: t.Any, owner: str
) -> t.Optional[_T]:
    """``check(value, owner)``, letting ``None`` through."""
    return None if value is None else check(value, owner)


class TestsetPromptMixin(PromptMixin):
    """``PromptMixin`` over ``ragas.prompt.metrics.BasePrompt``.

    ``PromptMixin`` finds and accepts only ``PydanticPrompt``; testset prompts
    are its base class, ``BasePrompt``. ``get_prompts``, ``adapt_prompts``,
    ``save_prompts`` and ``load_prompts`` all go through ``_get_prompts``.
    """

    def _get_prompts(self) -> t.Dict[str, BasePrompt]:  # type: ignore[override]
        return {
            key: value
            for key, value in inspect.getmembers(self)
            if isinstance(value, BasePrompt)
        }

    def set_prompts(self, **prompts: BasePrompt) -> None:
        """Replace prompts by name (see ``get_prompts()``).

        Raises
        ------
        ValueError
            If there is no prompt of that name, or the value is not a
            ``ragas.prompt.metrics.BasePrompt``.
        """
        name_to_var = {v.name: k for k, v in self._get_prompts().items()}
        for key, value in prompts.items():
            if key not in name_to_var:
                raise ValueError(
                    f"Prompt with name '{key}' does not exist. Use get_prompts() "
                    "to see available prompts."
                )
            if not isinstance(value, BasePrompt):
                raise ValueError(
                    f"Prompt with name '{key}' must be an instance of "
                    "'ragas.prompt.metrics.BasePrompt'"
                )
            setattr(self, name_to_var[key], value)
