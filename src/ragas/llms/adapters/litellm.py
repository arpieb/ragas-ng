import typing as t
import warnings

from ragas.llms.adapters.base import StructuredOutputAdapter


class LiteLLMAdapter(StructuredOutputAdapter):
    """Deprecated: builds the LLM with ``llm_factory()`` (#182)."""

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> t.Any:
        from ragas.llms.base import llm_factory

        warnings.warn(
            "LiteLLMAdapter is deprecated: call llm_factory() directly; every call goes "
            "through LiteLLM (#182).",
            DeprecationWarning,
            stacklevel=2,
        )
        return llm_factory(model, provider=provider, client=client, **kwargs)
