from __future__ import annotations

import asyncio
import inspect
import logging
import re
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import instructor
from pydantic import BaseModel

from ragas.async_utils import run_sync
from ragas.cache import CacheInterface, cache_identity, cacher
from ragas.exceptions import LLMDidNotFinishException
from ragas.llms._retry import acall_with_retry, call_with_retry
from ragas.llms.output import Generation, LLMResult
from ragas.run_config import (
    RunConfig,
    add_async_retry,
    scoped_run_config,
    scoped_usage_collector,
)

if t.TYPE_CHECKING:
    from llama_index.core.base.llms.base import BaseLLM

    from ragas.callbacks import Callbacks
    from ragas.prompt.value import PromptValue


logger = logging.getLogger(__name__)

# TypeVar for Instructor LLM response models
InstructorTypeVar = t.TypeVar("T", bound=BaseModel)  # type: ignore


@dataclass
class BaseRagasLLM(ABC):
    run_config: RunConfig = field(default_factory=RunConfig, repr=False)
    multiple_completion_supported: bool = field(default=False, repr=False)
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)

    def __post_init__(self):
        # If a cache_backend is provided, wrap the implementation methods at construction time.
        if self.cache is not None:
            self.generate_text = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.generate_text)
            self.agenerate_text = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.agenerate_text)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

    def get_temperature(self, n: int) -> float:
        """Return the temperature to use for completion based on n."""
        return 0.3 if n > 1 else 0.01

    @abstractmethod
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult: ...

    @abstractmethod
    def is_finished(self, response: LLMResult) -> bool:
        """Check if the LLM response is finished/complete."""
        ...

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Generate text using the given event loop."""

        if temperature is None:
            temperature = self.get_temperature(n)

        agenerate_text_with_retry = add_async_retry(
            self.agenerate_text, scoped_run_config() or self.run_config
        )
        result = await agenerate_text_with_retry(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

        # check there are no max_token issues
        if not self.is_finished(result):
            raise LLMDidNotFinishException()
        return result


class LlamaIndexLLMWrapper(BaseRagasLLM):
    """
    A Adaptor for LlamaIndex LLMs

    # TODO: Revisit deprecation warning
    # .. deprecated::
    #     LlamaIndexLLMWrapper is deprecated and will be removed in a future version.
    #     Use llm_factory instead:
    #     from openai import OpenAI
    #     from ragas.llms import llm_factory
    #     client = OpenAI(api_key="...")
    #     llm = llm_factory("gpt-6-luna", client=client)
    """

    def __init__(
        self,
        llm: BaseLLM,
        run_config: t.Optional[RunConfig] = None,
        cache: t.Optional[CacheInterface] = None,
        bypass_temperature: bool = False,
    ):
        import warnings

        warnings.warn(
            "LlamaIndexLLMWrapper is deprecated and will be removed in a future version. "
            "Use llm_factory instead: "
            "from openai import OpenAI; from ragas.llms import llm_factory; "
            "client = OpenAI(api_key='...'); llm = llm_factory('gpt-6-luna', client=client)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.llm = llm
        # Certain LLMs (e.g., OpenAI o1 series) do not support temperature
        self.bypass_temperature = bypass_temperature

        try:
            self._signature = type(self.llm).__name__.lower()
        except AttributeError:
            self._signature = ""

        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def check_args(
        self,
        n: int,
        temperature: float,
        stop: t.Optional[t.List[str]],
        callbacks: Callbacks,
    ) -> dict[str, t.Any]:
        if n != 1:
            logger.warning("n values greater than 1 not support for LlamaIndex LLMs")
        if temperature != 0.01:
            logger.info("temperature kwarg passed to LlamaIndex LLM")
        if stop is not None:
            logger.info("stop kwarg passed to LlamaIndex LLM")
        if callbacks is not None:
            logger.info(
                "callbacks not supported for LlamaIndex LLMs, ignoring callbacks"
            )
        if self._signature in ["anthropic", "bedrock"]:
            return {"temperature": temperature}
        else:
            return {
                "n": n,
                "temperature": temperature,
                "stop": stop,
            }

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        kwargs = self.check_args(n, temperature, stop, callbacks)
        li_response = self.llm.complete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)

        kwargs = self.check_args(n, temperature, stop, callbacks)

        if self.bypass_temperature:
            kwargs.pop("temperature", None)

        li_response = await self.llm.acomplete(prompt.to_string(), **kwargs)

        return LLMResult(generations=[[Generation(text=li_response.text)]])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm.__class__.__name__}(...))"


def _patch_client_for_provider(
    client: t.Any, provider: str, mode: t.Optional[instructor.Mode] = None
) -> t.Any:
    """
    Patch a client with Instructor for generic providers.

    Detects the client API style and uses the appropriate instructor patching method:
    - OpenAI-compatible (chat.completions.create): Uses instructor.from_openai()
    - Anthropic-compatible (messages.create): Uses instructor.AsyncInstructor/Instructor

    This allows OpenAI-compatible providers (DeepSeek, Groq, Mistral, etc.) to work
    correctly when using OpenAI SDK clients.
    """
    from instructor import Provider

    if mode is None:
        mode = instructor.Mode.JSON

    provider_map = {
        "anthropic": Provider.ANTHROPIC,
        "google": Provider.GENAI,
        "gemini": Provider.GENAI,
        "azure": Provider.OPENAI,
        "groq": Provider.GROQ,
        "mistral": Provider.MISTRAL,
        "cohere": Provider.COHERE,
        "xai": Provider.XAI,
        "bedrock": Provider.BEDROCK,
        "deepseek": Provider.DEEPSEEK,
    }

    provider_enum = provider_map.get(provider, Provider.OPENAI)

    if (
        hasattr(client, "chat")
        and client.chat is not None
        and hasattr(client.chat, "completions")
        and hasattr(client.chat.completions, "create")
    ):
        return instructor.from_openai(client, mode=mode)

    elif (
        hasattr(client, "messages")
        and client.messages is not None
        and hasattr(client.messages, "create")
    ):
        create_method = client.messages.create
        is_async = "Async" in client.__class__.__name__

        if is_async:
            return instructor.AsyncInstructor(
                client=client,
                create=create_method,
                provider=provider_enum,
                mode=mode,
            )
        else:
            return instructor.Instructor(
                client=client,
                create=create_method,
                provider=provider_enum,
                mode=mode,
            )
    else:
        raise ValueError(
            f"Unable to detect API style for {provider} client. "
            f"Client should have either 'chat.completions.create' (OpenAI-style) "
            f"or 'messages.create' (Anthropic-style) method."
        )


def _is_new_google_genai_client(client: t.Any) -> bool:
    """
    Detect if client is from the new google-genai SDK vs old google-generativeai.

    New SDK (google-genai):
        - Import: from google import genai / import google.genai
        - Client: genai.Client(api_key="...")
        - Module: google.genai.client.Client

    Old SDK (google-generativeai):
        - Import: import google.generativeai as genai
        - Client: genai.GenerativeModel("model-name")
        - Module: google.generativeai.generative_models.GenerativeModel

    Note: The old SDK is deprecated (support ends Aug 2025). The new SDK is recommended
    but has a known upstream instructor issue with safety settings. See:
    https://github.com/567-labs/instructor/issues/1658
    """
    client_module = getattr(client, "__module__", "") or ""
    client_class = client.__class__.__name__

    # New SDK: google.genai.client.Client or similar
    if "google.genai" in client_module and "generativeai" not in client_module:
        return True

    # Check class name as fallback (new SDK uses Client, old uses GenerativeModel)
    if client_class == "Client" and "genai" in client_module.lower():
        return True

    return False


def _get_instructor_clients(
    client: t.Any, provider: str, mode: t.Optional[instructor.Mode] = None
) -> t.Tuple[t.Any, t.Optional[t.Any]]:
    """Instructor client(s) for ``client``: ``(primary, sync_twin)``.

    A google-genai ``Client`` or ``mistralai.Mistral`` serves both sync and
    async calls, but instructor wraps it one way only. The primary is async, so collections metrics
    (which are all async) work; the sync twin serves ``generate()`` (#68).
    Other SDKs come as separate sync/async classes, so the caller's choice
    stands and there is no twin.
    """
    provider_lower = provider.lower()
    if provider_lower in ("google", "gemini") and _is_new_google_genai_client(client):
        return (
            instructor.from_genai(client, use_async=True, **_mode_kwargs(mode)),
            instructor.from_genai(client, **_mode_kwargs(mode)),
        )
    if provider_lower == "mistral" and _is_mistralai_client(client):
        # mistralai.Mistral has chat.complete / chat.complete_async, not the
        # OpenAI or Anthropic shapes the generic patcher looks for (#70).
        return (
            instructor.from_mistral(client, use_async=True, **_mode_kwargs(mode)),
            instructor.from_mistral(client, **_mode_kwargs(mode)),
        )
    return _get_instructor_client(client, provider, mode=mode), None


def _is_mistralai_client(client: t.Any) -> bool:
    """A ``mistralai.Mistral`` SDK client (``chat.complete``, no ``completions``)."""
    module = getattr(type(client), "__module__", "") or ""
    return module.split(".")[0] == "mistralai"


def _mode_kwargs(mode: t.Optional[instructor.Mode]) -> t.Dict[str, t.Any]:
    """``mode=`` only when the caller chose one.

    Provider factories validate the mode against what the provider supports,
    so forcing ``Mode.JSON`` on e.g. Anthropic would break it; without a
    caller's choice each provider keeps its own default.
    """
    return {} if mode is None else {"mode": mode}


def _get_instructor_client(
    client: t.Any, provider: str, mode: t.Optional[instructor.Mode] = None
) -> t.Any:
    """
    Get an instructor-patched client for the specified provider.

    Uses provider-specific methods when available, falls back to generic patcher.
    ``mode`` reaches every provider factory (#71).

    Note: For OpenAI-style clients, we use Mode.JSON by default instead of
    Mode.TOOLS because OpenAI's function calling (TOOLS mode) has issues with
    Dict type annotations in Pydantic models - it returns empty objects `{}`
    instead of proper structured data. Mode.JSON works correctly with all
    Pydantic types including Dict.
    See: https://github.com/vibrantlabsai/ragas/issues/2490

    Other provider factories (Anthropic, Google, Mistral, Perplexity) keep
    their own default unless ``mode`` is given.

    For Google/Gemini, supports both SDKs:
    - New SDK (google-genai): Uses instructor.from_genai()
    - Old SDK (google-generativeai): Uses instructor.from_gemini()
    """
    provider_lower = provider.lower()
    json_by_default = instructor.Mode.JSON if mode is None else mode

    if provider_lower == "openai":
        return instructor.from_openai(client, mode=json_by_default)
    elif provider_lower == "anthropic":
        return instructor.from_anthropic(client, **_mode_kwargs(mode))
    elif provider_lower in ("google", "gemini"):
        if _is_new_google_genai_client(client):
            return instructor.from_genai(client, **_mode_kwargs(mode))
        else:
            return instructor.from_gemini(client, **_mode_kwargs(mode))
    elif provider_lower == "litellm":
        return instructor.from_litellm(client, mode=json_by_default)
    elif provider_lower == "perplexity":
        return instructor.from_perplexity(client, **_mode_kwargs(mode))
    elif provider_lower == "mistral" and _is_mistralai_client(client):
        return instructor.from_mistral(client, **_mode_kwargs(mode))
    else:
        return _patch_client_for_provider(client, provider_lower, mode=mode)


DEFAULT_LLM_MODEL = "gpt-6-luna"


def default_llm() -> "InstructorBaseRagasLLM":
    """The evaluator LLM used when the caller does not supply one.

    Routed through LiteLLM rather than a hard-coded ``openai.OpenAI()`` client,
    so ragas does not import a specific vendor SDK just to pick a default.
    Credentials are resolved from the environment.
    """
    return llm_factory(DEFAULT_LLM_MODEL)


def llm_factory(
    model: str,
    provider: str = "openai",
    client: t.Optional[t.Any] = None,
    adapter: str = "auto",
    cache: t.Optional[CacheInterface] = None,
    mode: t.Optional[instructor.Mode] = None,
    **kwargs: t.Any,
) -> InstructorBaseRagasLLM:
    """
    Create an LLM instance for structured output generation with automatic adapter selection.

    Supports multiple LLM providers and structured output backends with unified interface
    for both sync and async operations. Returns instances with .generate() and .agenerate()
    methods that accept Pydantic models for structured outputs.

    Without a client, routes through LiteLLM and resolves credentials from the
    environment, so any LiteLLM-supported provider works with no vendor SDK.

    With a client, auto-detects the best adapter for your provider:
    - Google Gemini → uses LiteLLM adapter
    - Other providers → uses Instructor adapter (default)
    - Explicit control available via adapter parameter

    Args:
        model: Model name (e.g., "gpt-6-luna", "claude-sonnet-5-5", "gemini-3.6-flash").
               Without a client, any LiteLLM model string is accepted, such as
               "anthropic/claude-sonnet-5-5" or "ollama/llama3".
        provider: LLM provider (default: "openai").
                 Examples: openai, anthropic, google, groq, mistral, etc.
                 Without a client, a non-OpenAI provider is prefixed onto a bare
                 model name to form LiteLLM's "provider/model" string.
        client: Optional pre-initialized client instance. For OpenAI, can be
               OpenAI(...) or AsyncOpenAI(...). If omitted, the LLM is created
               through LiteLLM and the adapter argument is ignored.
        adapter: Structured output adapter to use (default: "auto").
                - "auto": Auto-detect based on provider/client (recommended)
                - "instructor": Use Instructor library
                - "litellm": Use LiteLLM (supports 100+ providers)
        cache: Optional cache backend for caching LLM responses.
               Pass DiskCacheBackend() for persistent caching across runs.
               Saves costs and speeds up repeated evaluations by 60x.
        mode: Instructor mode for structured outputs (default: Mode.JSON).
              Only applies when using instructor adapter.
              Options: Mode.JSON, Mode.MD_JSON, Mode.TOOLS, Mode.JSON_SCHEMA, etc.
              Use Mode.MD_JSON for backends that don't support response_format parameter.
        **kwargs: Additional model arguments (temperature, max_tokens, top_p, etc).
                 ``reasoning=True/False`` is not sent to the model: it marks the
                 model as an OpenAI reasoning model (or not) where the name
                 doesn't say, e.g. an Azure deployment. See ``is_reasoning_model``.

    Returns:
        InstructorBaseRagasLLM: Instance with generate() and agenerate() methods.

    Raises:
        ValueError: If model is empty, provider is unsupported, or adapter
                   initialization fails.

    Examples:
        # No client: routed through LiteLLM, credentials read from the environment
        llm = llm_factory("gpt-6-luna")
        llm = llm_factory("anthropic/claude-sonnet-5-5")
        llm = llm_factory("llama3", provider="ollama")  # -> "ollama/llama3"

        # With an explicit client
        from openai import OpenAI

        client = OpenAI(api_key="...")
        llm = llm_factory("gpt-6-luna", client=client)
        response = llm.generate(prompt, ResponseModel)

        # With caching (recommended for experiments)
        from ragas.cache import DiskCacheBackend
        cache = DiskCacheBackend()
        llm = llm_factory("gpt-6-luna", client=client, cache=cache)

        # Anthropic
        from anthropic import Anthropic
        client = Anthropic(api_key="...")
        llm = llm_factory("claude-sonnet-5-5", provider="anthropic", client=client)

        # Google Gemini, no SDK needed (reads GEMINI_API_KEY)
        llm = llm_factory("gemini/gemini-3.6-flash")

        # Explicit adapter selection
        llm = llm_factory("gpt-6-luna", client=client, adapter="litellm")

        # Custom instructor mode for backends without response_format support
        import instructor
        client = OpenAI(api_key="...", base_url="https://custom-backend")
        llm = llm_factory("custom-model", client=client, mode=instructor.Mode.MD_JSON)

        # Async
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key="...")
        llm = llm_factory("gpt-6-luna", client=client)
        response = await llm.agenerate(prompt, ResponseModel)
    """
    if not model:
        raise ValueError("model parameter is required")

    if client is None:
        # No client: route through LiteLLM, which resolves credentials from the
        # environment. This mirrors embedding_factory(), which has worked this way
        # since the LangChain adapters were removed -- the two factories used to
        # disagree, with this one raising where the other succeeded.
        try:
            import instructor as _instructor
            import litellm as _litellm
        except ImportError as e:  # pragma: no cover - litellm is a core dependency
            raise ImportError(
                "llm_factory() without an explicit client requires litellm. "
                "Install it, or pass a client: "
                "llm_factory('gpt-6-luna', client=OpenAI())"
            ) from e

        # Prefix the model with the provider when the caller named one and the
        # model string does not already carry it, since LiteLLM routes on the
        # "provider/model" form. "openai" needs no prefix.
        if provider.lower() not in ("openai", "litellm") and "/" not in model:
            model = f"{provider.lower()}/{model}"

        # Async primary, because every collections metric is async: a sync
        # client here made the default path fail with "Cannot use agenerate()
        # with a synchronous client" (#68). generate() uses the sync twin.
        client = _instructor.from_litellm(_litellm.acompletion, **_mode_kwargs(mode))
        kwargs["sync_client"] = _instructor.from_litellm(
            _litellm.completion, **_mode_kwargs(mode)
        )
        mode = None  # applied above; never sent to the model (#71)
        provider = "litellm"
        adapter = "litellm"

    provider_lower = provider.lower()

    # Auto-detect adapter if needed
    if adapter == "auto":
        from ragas.llms.adapters import auto_detect_adapter

        adapter = auto_detect_adapter(client, provider_lower)

    # Create LLM using selected adapter
    from ragas.llms.adapters import get_adapter

    try:
        adapter_instance = get_adapter(adapter)
        llm = adapter_instance.create_llm(
            client, model, provider_lower, cache=cache, mode=mode, **kwargs
        )
    except ValueError as e:
        # Re-raise ValueError from get_adapter for unknown adapter names
        # Also handle adapter initialization failures
        if "Unknown adapter" in str(e):
            raise
        # Adapter-specific failures get wrapped
        raise ValueError(
            f"Failed to initialize {provider} client with {adapter} adapter. "
            f"Ensure you've created a valid {provider} client.\n"
            f"Error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to initialize {provider} client with {adapter} adapter. "
            f"Ensure you've created a valid {provider} client.\n"
            f"Error: {str(e)}"
        )

    return llm


# Bedrock-style vendor prefix on OpenAI model IDs: "openai.gpt-6-sol",
# "us.openai.gpt-5" (cross-region inference profile).
_VENDOR_PREFIX = re.compile(r"^(?:[a-z]{2,6}\.)?openai\.")
_O_SERIES = re.compile(r"^o\d+(?:$|[-_.])")
_GPT_VERSION = re.compile(r"^gpt-(\d+)(?:\.\d+)?(?:$|[-_.])")


def _bare_model_id(model: str) -> str:
    """``model`` without routing prefixes: LiteLLM's ``provider/`` and
    Bedrock's ``[region.]openai.``."""
    bare = model.lower().rsplit("/", 1)[-1]
    return _VENDOR_PREFIX.sub("", bare)


def is_reasoning_model(model: str) -> bool:
    """Whether ``model`` is an OpenAI reasoning model (o-series, GPT-5+, codex-mini).

    These reject ``max_tokens`` (they take ``max_completion_tokens``), any
    temperature but 1, and ``top_p``. Matches dotted versions (gpt-5.1,
    gpt-5.2-mini), o10+, and IDs behind LiteLLM or Bedrock prefixes
    (``openai/gpt-5``, ``openai.gpt-6-sol``) (#67).

    Azure deployment names are arbitrary, so no name check can be complete;
    pass ``reasoning=True`` to ``llm_factory`` for such deployments.
    """
    bare = _bare_model_id(model)
    if _O_SERIES.match(bare) or bare.startswith("codex-mini"):
        return True
    m = _GPT_VERSION.match(bare)
    return bool(m and int(m.group(1)) >= 5)


def _is_anthropic_model(provider: str, model: str) -> bool:
    return provider.lower() == "anthropic" or "claude" in model.lower()


# Experimental LLM classes migrated from ragas.experimental.llms


class InstructorModelArgs(BaseModel):
    """Simple model arguments configuration for instructor LLMs

    These are defaults: where a provider rejects one (``top_p`` on Anthropic,
    ``max_tokens``/``temperature``/``top_p`` on OpenAI reasoning models) it is
    dropped or mapped, while a value the caller sets explicitly is sent.
    """

    temperature: float = 0.01
    top_p: float = 0.1
    max_tokens: int = 1024
    system_prompt: t.Optional[str] = None


class InstructorBaseRagasLLM(ABC):
    """Base class for LLMs using the Instructor library pattern.

    Provider calls are retried with exponential backoff on transient errors
    (rate limits, timeouts, 5xx) per ``run_config`` (#73): the RunConfig
    passed to ``evaluate()`` / ``TestsetGenerator`` for calls made inside that
    run, else this LLM's ``set_run_config``, else ``RunConfig()`` defaults.
    A run's config never sticks to the LLM (#76).
    """

    _run_config: t.Optional[RunConfig] = None

    @property
    def run_config(self) -> RunConfig:
        scoped = scoped_run_config()
        if scoped is not None:
            return scoped
        if self._run_config is None:
            self._run_config = RunConfig()
        return self._run_config

    @run_config.setter
    def run_config(self, run_config: RunConfig) -> None:
        self._run_config = run_config

    def set_run_config(self, run_config: RunConfig) -> None:
        """Retry/backoff settings for this LLM's provider calls."""
        self._run_config = run_config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.

        ``temperature`` overrides the configured temperature for this call, for
        metrics that deliberately draw several samples (``strictness``).
        ``sample_index`` does not reach the provider; it only gives each sample
        its own cache entry so a cache doesn't collapse N samples into one.
        """

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM.

        See ``generate`` for ``temperature`` and ``sample_index``.
        """

    @staticmethod
    def _with_sampling(
        model_args: t.Dict[str, t.Any], temperature: t.Optional[float]
    ) -> t.Dict[str, t.Any]:
        """``model_args`` with a per-call temperature applied.

        A sampled call also drops ``top_p``: the deterministic default (0.1)
        restricts sampling to the top 10% of probability mass, under which a
        higher temperature barely changes the output.
        """
        if temperature is None:
            return model_args
        args = dict(model_args)
        args["temperature"] = temperature
        args.pop("top_p", None)
        return args


class InstructorLLM(InstructorBaseRagasLLM):
    """LLM wrapper using the Instructor library for structured outputs."""

    def __init__(
        self,
        client: t.Any,
        model: str,
        provider: str,
        model_args: t.Optional[InstructorModelArgs] = None,
        cache: t.Optional[CacheInterface] = None,
        sync_client: t.Optional[t.Any] = None,
        reasoning: t.Optional[bool] = None,
        **kwargs,
    ):
        self.client = client
        # Sync counterpart of an async ``client`` over the same credentials,
        # for providers whose SDK object serves both (LiteLLM, google-genai).
        # generate() then calls it directly rather than hopping onto a
        # thread with a fresh event loop.
        self.sync_client = sync_client
        self.model = model
        self.provider = provider

        # Use deterministic defaults if no model_args provided
        if model_args is None:
            model_args = InstructorModelArgs()

        # None: detect from the model ID. True/False: the caller knows better,
        # e.g. an Azure deployment whose name says nothing about the model.
        self.reasoning = reasoning
        # Arguments the caller chose, as opposed to InstructorModelArgs
        # defaults: a default may be dropped where a provider rejects it, a
        # caller's choice is sent as given.
        self._explicit_args = set(model_args.model_fields_set) | set(kwargs)

        # Convert to dict and merge with any additional kwargs
        self.model_args = {**model_args.model_dump(), **kwargs}

        # Extract system_prompt separately (not passed to LLM API)
        self.system_prompt = self.model_args.pop("system_prompt", None)

        # A collector you attach yourself. evaluate()/TestsetGenerator no longer
        # set this: their collector is run-scoped (see _active_usage_collector).
        self.usage_collector: t.Optional[t.Any] = None

        self.cache = cache

        # Check if client is async-capable at initialization
        self.is_async = self._check_client_async()

        if self.cache is not None:
            self.generate = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.generate)  # type: ignore
            self.agenerate = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(self.agenerate)  # type: ignore

    def _map_provider_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None
    ) -> t.Dict[str, t.Any]:
        """Route to provider-specific parameter mapping.

        ``args`` defaults to ``self.model_args``; pass a per-call variant (e.g.
        from ``_with_sampling``) to map that instead.

        - Google: Wraps parameters in generation_config and renames max_tokens
        - OpenAI reasoning models (any provider, e.g. OpenAI, Azure, Bedrock):
          see ``_map_openai_params``
        - Anthropic: see ``_map_anthropic_params``
        - Others: pass through unchanged
        """
        if args is None:
            args = self.model_args

        if self.provider.lower() == "google":
            return self._map_google_params(args)
        if _is_anthropic_model(self.provider, self.model):
            return self._map_anthropic_params(args)
        return self._map_openai_params(args)

    def _map_anthropic_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None
    ) -> t.Dict[str, t.Any]:
        """Current Claude models reject ``temperature`` and ``top_p`` together (#69).

        The default ``top_p`` is never sent. A caller who sets ``top_p``
        themselves gets it, and the *default* temperature is dropped instead.
        """
        mapped = dict(self.model_args if args is None else args)
        if "top_p" not in self._explicit_args:
            mapped.pop("top_p", None)
        elif "temperature" not in self._explicit_args:
            mapped.pop("temperature", None)
        return mapped

    def _map_openai_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None
    ) -> t.Dict[str, t.Any]:
        """Map parameters for OpenAI reasoning models, which have special constraints.

        Applies to any provider serving them (OpenAI, Azure, Bedrock, ...);
        detection is ``is_reasoning_model``, overridden by ``reasoning=``.
        Reasoning models (o-series, GPT-5+, codex-mini):
        1. take max_completion_tokens instead of max_tokens. The default cap
           is not sent, since it also counts the hidden reasoning tokens; a
           caller's ``max_tokens`` is sent as max_completion_tokens.
        2. only accept temperature 1.0
        3. reject top_p

        Other models are passed through unchanged.

        Note on Azure deployments: deployment names are arbitrary, so pass
        ``reasoning=True`` for a reasoning deployment whose name doesn't say so.
        """
        mapped_args = (self.model_args if args is None else args).copy()

        requires_max_completion_tokens = (
            is_reasoning_model(self.model) if self.reasoning is None else self.reasoning
        )
        if not requires_max_completion_tokens:
            return mapped_args

        if "max_tokens" in mapped_args:
            max_tokens = mapped_args.pop("max_tokens")
            # max_completion_tokens also counts the hidden reasoning tokens, so
            # the 1024 default truncated structured output. Only a caller's
            # own cap is sent; otherwise the provider default applies.
            if "max_tokens" in self._explicit_args:
                mapped_args["max_completion_tokens"] = max_tokens

        # Reasoning models only accept temperature 1 and reject top_p.
        mapped_args["temperature"] = 1.0
        mapped_args.pop("top_p", None)

        return mapped_args

    def _map_google_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None
    ) -> t.Dict[str, t.Any]:
        """Map parameters for Google Gemini models.

        Google models require parameters to be wrapped in a generation_config dict,
        and max_tokens is renamed to max_output_tokens.
        """
        google_kwargs = {}
        generation_config_keys = {"temperature", "max_tokens", "top_p", "top_k"}
        generation_config = {}

        for key, value in (self.model_args if args is None else args).items():
            if key in generation_config_keys:
                if key == "max_tokens":
                    generation_config["max_output_tokens"] = value
                else:
                    generation_config[key] = value
            else:
                google_kwargs[key] = value

        if generation_config:
            google_kwargs["generation_config"] = generation_config

        return google_kwargs

    def _check_client_async(self) -> bool:
        """Determine if the client is async-capable.

        Handles multiple cases:
        1. Instructor-wrapped AsyncInstructor clients (OpenAI/Anthropic/etc)
        2. Instructor-wrapped Instructor clients that wrap async underlying clients
        3. Direct async clients with chat.completions.create
        4. Instructor-wrapped clients where the underlying client is in a closure
        """
        try:
            # Check if this is an AsyncInstructor wrapper
            if self.client.__class__.__name__ == "AsyncInstructor":
                return True

            # Check if this is a sync Instructor wrapper that wraps an async client
            if hasattr(self.client, "client"):
                underlying = self.client.client
                # For OpenAI/Anthropic async clients
                if hasattr(underlying, "chat") and hasattr(
                    underlying.chat, "completions"
                ):
                    if hasattr(underlying.chat.completions, "create"):
                        if inspect.iscoroutinefunction(
                            underlying.chat.completions.create
                        ):
                            return True

            # Check if this is an async client by checking for a coroutine method
            if hasattr(self.client, "chat") and hasattr(
                self.client.chat, "completions"
            ):
                if hasattr(self.client.chat.completions, "create"):
                    return inspect.iscoroutinefunction(
                        self.client.chat.completions.create
                    )

            # For instructor-wrapped clients, also check the closure of create_fn
            # This handles cases where the underlying client is stored in a closure
            if (
                hasattr(self.client, "create_fn")
                and hasattr(self.client.create_fn, "__closure__")
                and self.client.create_fn.__closure__
            ):
                for cell in self.client.create_fn.__closure__:
                    try:
                        obj = cell.cell_contents
                        # Check if the closure object is an async client
                        if hasattr(obj, "chat") and hasattr(obj.chat, "completions"):
                            if hasattr(obj.chat.completions, "create"):
                                if inspect.iscoroutinefunction(
                                    obj.chat.completions.create
                                ):
                                    return True
                        # Also check for acompletion (e.g., litellm Router)
                        if hasattr(obj, "acompletion"):
                            if inspect.iscoroutinefunction(obj.acompletion):
                                return True
                    except (ValueError, AttributeError):
                        # cell_contents might not be accessible
                        pass

            return False
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(
        self, coro: t.Coroutine[t.Any, t.Any, t.Any]
    ) -> t.Any:
        """Run ``coro`` from sync code on ragas' shared background loop.

        See ``ragas.async_utils.run_sync``: one loop for every call, so the
        async client stays bound to it (#74).
        """
        return run_sync(coro)

    def _active_usage_collector(self) -> t.Optional[t.Any]:
        """The enclosing run's collector, else one attached to this LLM."""
        return scoped_usage_collector() or self.usage_collector

    def _messages(self, prompt: str) -> t.List[t.Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _target(self, client: t.Any) -> t.Any:
        return client if self.provider.lower() == "google" else client.chat.completions

    def _call_sync(
        self,
        client: t.Any,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        """One blocking structured-output call on a sync instructor client."""
        provider_kwargs = self._map_provider_params(
            self._with_sampling(self.model_args, temperature)
        )
        target = self._target(client)
        collector = self._active_usage_collector()
        if collector is not None:
            # create_with_completion also returns the provider's raw response,
            # which is the only place token counts are available. Gated so the
            # default path is byte-identical to before.
            result, raw = target.create_with_completion(
                model=self.model,
                messages=self._messages(prompt),
                response_model=response_model,
                **provider_kwargs,
            )
            collector.record(raw)
            return result
        return target.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **provider_kwargs,
        )

    def generate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        See ``InstructorBaseRagasLLM.generate`` for ``temperature``/``sample_index``.
        """
        rc = self.run_config
        if self.sync_client is not None:
            return call_with_retry(
                rc,
                self._call_sync,
                self.sync_client,
                prompt,
                response_model,
                temperature,
            )
        if self.is_async:
            return self._run_async_in_current_loop(
                acall_with_retry(
                    rc, self._call_async, prompt, response_model, temperature
                )
            )
        return call_with_retry(
            rc, self._call_sync, self.client, prompt, response_model, temperature
        )

    async def _call_async(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        provider_kwargs = self._map_provider_params(
            self._with_sampling(self.model_args, temperature)
        )
        target = self._target(self.client)
        collector = self._active_usage_collector()
        if collector is not None:
            result, raw = await target.create_with_completion(
                model=self.model,
                messages=self._messages(prompt),
                response_model=response_model,
                **provider_kwargs,
            )
            collector.record(raw)
            return result
        return await target.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **provider_kwargs,
        )

    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Asynchronously generate a response using the configured LLM.

        A synchronous client is run on a worker thread, so every collections
        metric (all of which are async) works with ``OpenAI()``,
        ``Anthropic()`` and other sync clients (#68).
        See ``InstructorBaseRagasLLM.generate`` for ``temperature``/``sample_index``.
        """
        rc = self.run_config
        if not self.is_async:
            return await asyncio.to_thread(
                call_with_retry,
                rc,
                self._call_sync,
                self.client,
                prompt,
                response_model,
                temperature,
            )
        return await acall_with_retry(
            rc, self._call_async, prompt, response_model, temperature
        )

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        # Show important model arguments
        important_args = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
        for arg in important_args:
            if arg in self.model_args:
                config_parts.append(f"{arg}={self.model_args[arg]}")

        # Show count of other args if there are any
        other_args = len([k for k in self.model_args.keys() if k not in important_args])
        if other_args > 0:
            config_parts.append(f"+{other_args} more")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM."""
        client_info = self._get_client_info()
        key_config = self._get_key_config()

        base_repr = f"InstructorLLM(provider='{self.provider}', model='{self.model}', client={client_info}"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
