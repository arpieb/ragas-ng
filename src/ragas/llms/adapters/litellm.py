import typing as t
import warnings

from ragas.llms.adapters.base import StructuredOutputAdapter

if t.TYPE_CHECKING:
    import instructor

    from ragas.llms.litellm_llm import LiteLLMStructuredLLM


def _is_instructor_client(client: t.Any) -> bool:
    return type(client).__name__ in ("Instructor", "AsyncInstructor")


def _patch_for_litellm(
    client: t.Any, mode: t.Optional["instructor.Mode"]
) -> t.Tuple[t.Any, t.Optional[t.Any]]:
    """Instructor client(s) for ``client``: ``(primary, sync_twin)``.

    ``LiteLLMStructuredLLM`` calls ``client.chat.completions.create(...,
    response_model=...)``, which only an instructor-patched client accepts.
    This adapter used to pass the caller's client through unpatched, so a
    raw ``OpenAI(base_url=<litellm proxy>)`` failed with "unexpected keyword
    argument 'response_model'" and ``mode=`` was sent to the model (#71).
    """
    import instructor

    from ragas.llms.base import _mode_kwargs

    if _is_instructor_client(client):
        if mode is not None:
            warnings.warn(
                "mode= is ignored for a client that is already patched with "
                "instructor; set the mode when patching it.",
                stacklevel=4,
            )
        return client, None

    if hasattr(client, "acompletion") and hasattr(client, "completion"):
        # The litellm module or a litellm.Router: async primary, sync twin.
        return (
            instructor.from_litellm(client.acompletion, **_mode_kwargs(mode)),
            instructor.from_litellm(client.completion, **_mode_kwargs(mode)),
        )

    chat = getattr(client, "chat", None)
    if hasattr(getattr(chat, "completions", None), "create"):
        # OpenAI-compatible client, e.g. pointed at a LiteLLM proxy.
        return (
            instructor.from_openai(
                client, mode=instructor.Mode.JSON if mode is None else mode
            ),
            None,
        )

    raise ValueError(
        f"The LiteLLM adapter can't use a {type(client).__name__} client. "
        "Pass an OpenAI-compatible client (e.g. one pointed at a LiteLLM "
        "proxy), a litellm.Router, or no client at all: "
        "llm_factory('gemini/gemini-2.0-flash') routes through LiteLLM and "
        "reads credentials from the environment."
    )


class LiteLLMAdapter(StructuredOutputAdapter):
    """
    Adapter using LiteLLM for structured outputs.

    Supports: All 100+ LiteLLM providers (Gemini, Ollama, vLLM, Groq, etc.)
    """

    def create_llm(
        self,
        client: t.Any,
        model: str,
        provider: str,
        **kwargs,
    ) -> "LiteLLMStructuredLLM":
        """
        Create LiteLLMStructuredLLM instance.

        Args:
            client: An OpenAI-compatible client, a litellm.Router, or an
                instructor-patched client
            model: Model name
            provider: Provider name
            **kwargs: Additional model arguments; ``mode`` selects the
                instructor mode and is not sent to the model

        Returns:
            LiteLLMStructuredLLM instance
        """
        from ragas.llms.litellm_llm import LiteLLMStructuredLLM

        cache = kwargs.pop("cache", None)
        mode = kwargs.pop("mode", None)
        if kwargs.get("sync_client") is None:
            client, kwargs["sync_client"] = _patch_for_litellm(client, mode)

        return LiteLLMStructuredLLM(
            client=client,
            model=model,
            provider=provider,
            cache=cache,
            **kwargs,
        )
