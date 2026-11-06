import typing as t

from ragas.llms.adapters.base import StructuredOutputAdapter
from ragas.llms.base import (
    InstructorLLM,
    InstructorModelArgs,
    _get_instructor_clients,
)


class InstructorAdapter(StructuredOutputAdapter):
    """
    Adapter using Instructor library for structured outputs.

    Supports: OpenAI, Anthropic, Azure, Groq, Mistral, Cohere, Google, etc.
    """

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> InstructorLLM:
        """
        Create InstructorLLM instance by patching client with Instructor.

        Args:
            client: Pre-initialized client
            model: Model name
            provider: Provider name
            **kwargs: Additional model arguments including optional 'mode'

        Returns:
            InstructorLLM instance

        Raises:
            ValueError: If client patching fails
        """
        cache = kwargs.pop("cache", None)
        mode = kwargs.pop("mode", None)

        try:
            patched_client, sync_client = _get_instructor_clients(
                client, provider, mode=mode
            )
        except Exception as e:
            raise ValueError(f"Failed to patch {provider} client with Instructor: {e}")

        return InstructorLLM(
            client=patched_client,
            model=model,
            provider=provider,
            model_args=InstructorModelArgs(),
            cache=cache,
            sync_client=sync_client,
            **kwargs,
        )
