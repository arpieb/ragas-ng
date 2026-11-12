"""``llm_factory()``: an ``InstructorLLM`` over LiteLLM for every model (#182)."""

from __future__ import annotations

import functools
import typing as t
import warnings

import instructor

from ragas.cache import CacheInterface
from ragas.llms.provider_rules import is_anthropic_model

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.rate_limit import RateLimit


DEFAULT_LLM_MODEL = "gpt-6-luna"


def default_llm() -> InstructorBaseRagasLLM:
    """The evaluator LLM used when the caller does not supply one.

    Routed through LiteLLM rather than a hard-coded ``openai.OpenAI()`` client,
    so ragas does not import a specific vendor SDK just to pick a default.
    Credentials are resolved from the environment.
    """
    return llm_factory(DEFAULT_LLM_MODEL)


# Native SDK clients llm_factory() no longer accepts (#182): module prefix ->
# (LiteLLM model prefix, environment variable LiteLLM reads the key from).
_NATIVE_CLIENTS: t.Dict[str, t.Tuple[str, str]] = {
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "google": ("gemini", "GOOGLE_API_KEY or GEMINI_API_KEY"),
    "mistralai": ("mistral", "MISTRAL_API_KEY"),
    "groq": ("groq", "GROQ_API_KEY"),
    "cohere": ("cohere_chat", "COHERE_API_KEY"),
    "botocore": ("bedrock", "AWS_ACCESS_KEY_ID / AWS_REGION_NAME"),
    "boto3": ("bedrock", "AWS_ACCESS_KEY_ID / AWS_REGION_NAME"),
    "xai_sdk": ("xai", "XAI_API_KEY"),
    "instructor": ("<provider>", "the provider's API key"),
}

# instructor modes that only exist for native SDK clients, and the mode that
# does the same through LiteLLM.
_NATIVE_MODE_EQUIVALENT = {
    "ANTHROPIC_JSON": "JSON",
    "ANTHROPIC_TOOLS": "TOOLS",
    "ANTHROPIC_REASONING_TOOLS": "TOOLS",
    "ANTHROPIC_PARALLEL_TOOLS": "PARALLEL_TOOLS",
}
_NATIVE_MODE_PREFIXES = (
    "ANTHROPIC_",
    "GENAI_",
    "GEMINI_",
    "VERTEXAI_",
    "MISTRAL_",
    "BEDROCK_",
    "COHERE_",
    "WRITER_",
    "XAI_",
    "CEREBRAS_",
    "FIREWORKS_",
)


def _native_mode_error(mode: instructor.Mode) -> t.Optional[str]:
    name = getattr(mode, "name", str(mode))
    if not name.startswith(_NATIVE_MODE_PREFIXES):
        return None
    equivalent = _NATIVE_MODE_EQUIVALENT.get(name, "JSON")
    return (
        f"mode=instructor.Mode.{name} is for native SDK clients, which "
        "llm_factory() no longer uses: every call goes through LiteLLM (#182). "
        f"Use mode=instructor.Mode.{equivalent}."
    )


def _mode_kwargs(mode: t.Optional[instructor.Mode]) -> t.Dict[str, t.Any]:
    """``mode=`` only when the caller chose one.

    Provider factories validate the mode against what the provider supports,
    so forcing ``Mode.JSON`` on e.g. Anthropic would break it; without a
    caller's choice each provider keeps its own default.
    """
    return {} if mode is None else {"mode": mode}


def _is_openai_sdk_client(client: t.Any) -> t.Optional[str]:
    """``"openai"`` / ``"azure"`` for openai SDK clients, else None."""
    cls = type(client)
    if not (getattr(cls, "__module__", "") or "").startswith("openai"):
        return None
    name = cls.__name__
    if name in ("AzureOpenAI", "AsyncAzureOpenAI"):
        return "azure"
    if name in ("OpenAI", "AsyncOpenAI"):
        return "openai"
    return None


def _unsupported_client_error(client: t.Any, model: str) -> ValueError:
    cls = type(client)
    root = (getattr(cls, "__module__", "") or "").split(".")[0]
    prefix, env = _NATIVE_CLIENTS.get(root, ("<provider>", "the provider's API key"))
    bare = model.split("/", 1)[-1]
    if root == "instructor":
        what = "an instructor-patched client"
        extra = (
            " To use your own instructor client, construct "
            "ragas.llms.InstructorLLM(client=..., model=..., provider=...) "
            "directly."
        )
    else:
        what = f"{cls.__module__}.{cls.__name__} clients"
        extra = ""
    return ValueError(
        f"llm_factory() no longer accepts {what} (#182). Every call goes "
        f"through LiteLLM, addressed by model name: "
        f"llm_factory('{prefix}/{bare}'), which reads {env} from the "
        "environment (or pass api_key=... / api_base=...). OpenAI, AsyncOpenAI "
        "and Azure OpenAI clients are still accepted, and so is a "
        "litellm.Router. See https://docs.litellm.ai/docs/providers for the "
        f"model prefixes.{extra}"
    )


class _Transport(t.NamedTuple):
    """How an ``llm_factory()`` LLM reaches LiteLLM."""

    model: str
    # The completion function the primary client wraps; a coroutine
    # function when ``is_async``.
    completion: t.Callable[..., t.Any]
    is_async: bool
    # A sync completion function over the same credentials, if there is one.
    sync_completion: t.Optional[t.Callable[..., t.Any]]
    # Sent with every call.
    call_defaults: t.Dict[str, t.Any]
    # "openai" / "azure" for those SDK clients, "litellm" for a Router or the
    # litellm module, None without a client.
    kind: t.Optional[str]


def _litellm_transport(model: str, provider: str, client: t.Any) -> _Transport:
    """How ``llm_factory(model, provider, client)`` calls LiteLLM.

    An OpenAI/Azure client is passed to LiteLLM as a copy, because LiteLLM
    writes ``max_retries`` and ``organization`` onto the client object it is
    given.
    """
    import litellm

    if client is None:
        # LiteLLM routes on "provider/model"; "openai" needs no prefix.
        if provider.lower() not in ("openai", "litellm") and "/" not in model:
            model = f"{provider.lower()}/{model}"
        return _Transport(
            model, litellm.acompletion, True, litellm.completion, {}, None
        )

    if hasattr(client, "acompletion") and hasattr(client, "completion"):
        # the litellm module or a litellm.Router
        return _Transport(
            model, client.acompletion, True, client.completion, {}, "litellm"
        )

    kind = _is_openai_sdk_client(client)
    if kind is None:
        raise _unsupported_client_error(client, model)

    if not model.startswith(f"{kind}/"):
        model = f"{kind}/{model}"
    defaults = {"client": client.copy()}
    if type(client).__name__.startswith("Async"):
        return _Transport(model, litellm.acompletion, True, None, defaults, kind)
    return _Transport(model, litellm.completion, False, None, defaults, kind)


def llm_factory(
    model: str,
    provider: str = "openai",
    client: t.Optional[t.Any] = None,
    adapter: str = "auto",
    cache: t.Optional[CacheInterface] = None,
    mode: t.Optional[instructor.Mode] = None,
    rate_limit: t.Optional[RateLimit] = None,
    **kwargs: t.Any,
) -> InstructorBaseRagasLLM:
    """
    Create an LLM for structured output. Every call goes through LiteLLM.

    Address any LiteLLM-supported provider by model name; credentials come from
    the environment (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``,
    ``GEMINI_API_KEY``, ...) or from ``api_key=`` / ``api_base=``.

    Args:
        model: A LiteLLM model string: ``"gpt-6-luna"``,
            ``"anthropic/claude-sonnet-5-5"``, ``"gemini/gemini-3.6-flash"``,
            ``"ollama/qwen3.8:27b"``, ``"bedrock/<model id>"``, ...
            See https://docs.litellm.ai/docs/providers.
        provider: Prefixed onto a bare ``model`` when there is no client:
            ``llm_factory("llama3", provider="ollama")`` is
            ``"ollama/llama3"``. ``"openai"`` adds no prefix.
        client: Optional. An ``OpenAI`` / ``AsyncOpenAI`` (any
            OpenAI-compatible endpoint, via ``base_url``) or ``AzureOpenAI`` /
            ``AsyncAzureOpenAI`` client, passed to LiteLLM with its settings
            (a copy of it: LiteLLM changes the object it is given), or a
            ``litellm.Router``. Native SDK clients of other providers
            (``Anthropic()``, ``genai.Client()``, Mistral, Groq, Cohere,
            boto3) are no longer accepted; use the provider's model string
            instead (#182).
        adapter: Deprecated and ignored: there is one path (#182).
        cache: Optional cache backend for LLM responses.
        mode: instructor mode. Defaults to ``Mode.JSON`` for OpenAI/Azure
            clients and for Claude, otherwise instructor's default. Native
            modes (``Mode.ANTHROPIC_*``, ``Mode.GENAI_*``, ...) raise.
        rate_limit: Optional ``ragas.rate_limit.RateLimit``: requests (and
            tokens) per minute. Every request this LLM sends waits for it,
            including retries; pass the same object to LLMs that share an
            API key (#171).
        **kwargs: Model arguments (``temperature``, ``top_p``, ``max_tokens``,
            ``system_prompt``, ``reasoning``, ...) and LiteLLM call
            arguments (``api_key``, ``api_base``, ``extra_headers``, ...).

    Returns:
        An ``InstructorLLM`` with ``generate()`` and ``agenerate()``.

    Examples:
        llm = llm_factory("gpt-6-luna")  # OPENAI_API_KEY
        llm = llm_factory("anthropic/claude-sonnet-5-5")  # ANTHROPIC_API_KEY
        llm = llm_factory("gemini/gemini-3.6-flash")  # GEMINI_API_KEY
        llm = llm_factory("ollama/qwen3.8:27b")  # local Ollama

        # an OpenAI-compatible endpoint, with your own client settings
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        llm = llm_factory("qwen3.8:27b", client=client)
    """
    from ragas.llms.base import InstructorLLM

    if not model:
        raise ValueError("model parameter is required")
    if adapter != "auto":
        warnings.warn(
            f"llm_factory(adapter={adapter!r}) is deprecated and ignored: every "
            "call goes through LiteLLM (#182).",
            DeprecationWarning,
            stacklevel=2,
        )
    if mode is not None:
        message = _native_mode_error(mode)
        if message:
            raise ValueError(message)

    transport = _litellm_transport(model, provider, client)
    kind = transport.kind

    # Default mode: JSON for OpenAI/Azure clients (TOOLS mode returned {} for
    # Dict fields, vibrantlabsai/ragas#2490) and for Claude (instructor's
    # forced tool call fails on claude-sonnet-5-5, #184); otherwise
    # instructor's own default, as before.
    if mode is None and (
        kind in ("openai", "azure") or is_anthropic_model("", transport.model)
    ):
        mode = instructor.Mode.JSON

    completion, sync_completion = transport.completion, transport.sync_completion
    if transport.call_defaults:
        # bound per call; instructor's from_litellm(**kw) has its own `client`
        completion = functools.partial(completion, **transport.call_defaults)
        if sync_completion is not None:
            sync_completion = functools.partial(
                sync_completion, **transport.call_defaults
            )
    if rate_limit is not None:
        # paced below instructor, so its validation retries are counted too
        from ragas.rate_limit import limit_completion_calls

        completion = limit_completion_calls(completion, rate_limit)
        if sync_completion is not None:
            sync_completion = limit_completion_calls(sync_completion, rate_limit)
    llm = InstructorLLM(
        client=instructor.from_litellm(completion, **_mode_kwargs(mode)),
        model=transport.model,
        # "openai"/"azure" for those clients, "litellm" otherwise: evaluate()
        # picks default embeddings from it (_infer_embedding_provider_from_llm)
        provider=kind if kind in ("openai", "azure") else "litellm",
        cache=cache,
        sync_client=(
            instructor.from_litellm(sync_completion, **_mode_kwargs(mode))
            if sync_completion is not None
            else None
        ),
        is_async=transport.is_async,
        **kwargs,
    )
    llm._litellm = True
    llm.rate_limit = rate_limit
    if kind in ("openai", "azure"):
        llm.source_client = client
    return llm
