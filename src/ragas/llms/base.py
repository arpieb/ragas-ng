from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import re
import typing as t
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import instructor
from pydantic import BaseModel

from ragas.async_utils import run_sync
from ragas.cache import CacheInterface, cache_identity, cacher
from ragas.callbacks import ChainType, clip_for_span, current_group, traced
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
    from ragas.rate_limit import RateLimit


logger = logging.getLogger(__name__)

# TypeVar for Instructor LLM response models
InstructorTypeVar = t.TypeVar("T", bound=BaseModel)  # type: ignore

# One chat message in OpenAI format: ``{"role": ..., "content": ...}``, where
# ``content`` is a string or a list of content parts (``{"type": "text", ...}``,
# ``{"type": "image_url", ...}``). LiteLLM translates these for each provider.
ChatMessage = t.Dict[str, t.Any]
# What ``generate``/``agenerate`` take: a prompt string, sent as one user
# message, or a list of chat messages sent as given (#214).
LLMInput = t.Union[str, t.Sequence[ChatMessage]]


def _build_messages(
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


def _content_for_span(content: t.Any) -> str:
    """A message's content as a span attribute: text as is, parts as JSON."""
    if isinstance(content, str):
        return clip_for_span(content)
    try:
        text = json.dumps(content, default=repr)
    except Exception:
        text = repr(content)
    return clip_for_span(text)


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
            "client = OpenAI(api_key='...'); "
            f"llm = llm_factory('{DEFAULT_LLM_MODEL}', client=client)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cache=cache)
        self.llm = llm
        # Certain LLMs (e.g., OpenAI reasoning models) do not support temperature
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


def _mode_kwargs(mode: t.Optional[instructor.Mode]) -> t.Dict[str, t.Any]:
    """``mode=`` only when the caller chose one.

    Provider factories validate the mode against what the provider supports,
    so forcing ``Mode.JSON`` on e.g. Anthropic would break it; without a
    caller's choice each provider keeps its own default.
    """
    return {} if mode is None else {"mode": mode}


DEFAULT_LLM_MODEL = "gpt-6-luna"


def default_llm() -> "InstructorBaseRagasLLM":
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


def _native_mode_error(mode: instructor.Mode) -> t.Optional[str]:
    name = getattr(mode, "name", str(mode))
    prefixes = (
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
    if not name.startswith(prefixes):
        return None
    equivalent = _NATIVE_MODE_EQUIVALENT.get(name, "JSON")
    return (
        f"mode=instructor.Mode.{name} is for native SDK clients, which "
        "llm_factory() no longer uses: every call goes through LiteLLM (#182). "
        f"Use mode=instructor.Mode.{equivalent}."
    )


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


def _litellm_transport(
    model: str, provider: str, client: t.Any
) -> t.Tuple[str, t.Any, t.Optional[t.Any], t.Dict[str, t.Any], t.Optional[str]]:
    """``(litellm_model, async_fn, sync_fn, call_defaults, client_kind)``.

    ``call_defaults`` are sent with every call: an OpenAI/Azure client is
    passed to LiteLLM as a copy, because LiteLLM writes ``max_retries`` and
    ``organization`` onto the client object it is given.
    """
    import litellm

    if client is None:
        # LiteLLM routes on "provider/model"; "openai" needs no prefix.
        if provider.lower() not in ("openai", "litellm") and "/" not in model:
            model = f"{provider.lower()}/{model}"
        return model, litellm.acompletion, litellm.completion, {}, None

    if hasattr(client, "acompletion") and hasattr(client, "completion"):
        # the litellm module or a litellm.Router
        return model, client.acompletion, client.completion, {}, "litellm"

    kind = _is_openai_sdk_client(client)
    if kind is None:
        raise _unsupported_client_error(client, model)

    if not model.startswith(f"{kind}/"):
        model = f"{kind}/{model}"
    copy = client.copy()
    is_async = type(client).__name__.startswith("Async")
    defaults = {"client": copy}
    if is_async:
        return model, litellm.acompletion, None, defaults, kind
    return model, litellm.completion, None, defaults, kind


def llm_factory(
    model: str,
    provider: str = "openai",
    client: t.Optional[t.Any] = None,
    adapter: str = "auto",
    cache: t.Optional[CacheInterface] = None,
    mode: t.Optional[instructor.Mode] = None,
    rate_limit: t.Optional["RateLimit"] = None,
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

    try:
        import instructor as _instructor
    except ImportError as e:  # pragma: no cover - a core dependency
        raise ImportError("llm_factory() requires instructor.") from e

    litellm_model, async_fn, sync_fn, call_defaults, kind = _litellm_transport(
        model, provider, client
    )

    # Default mode: JSON for OpenAI/Azure clients (TOOLS mode returned {} for
    # Dict fields, vibrantlabsai/ragas#2490) and for Claude (instructor's
    # forced tool call fails on claude-sonnet-5-5, #184); otherwise
    # instructor's own default, as before.
    if mode is None and (
        kind in ("openai", "azure") or _is_anthropic_model("", litellm_model)
    ):
        mode = _instructor.Mode.JSON

    if call_defaults:
        # bound per call; instructor's from_litellm(**kw) has its own `client`
        async_fn = functools.partial(async_fn, **call_defaults)
        if sync_fn is not None:
            sync_fn = functools.partial(sync_fn, **call_defaults)
    if rate_limit is not None:
        # paced below instructor, so its validation retries are counted too
        from ragas.rate_limit import limit_completion_calls

        async_fn = limit_completion_calls(async_fn, rate_limit)
        if sync_fn is not None:
            sync_fn = limit_completion_calls(sync_fn, rate_limit)
    primary = _instructor.from_litellm(async_fn, **_mode_kwargs(mode))
    sync_twin = (
        _instructor.from_litellm(sync_fn, **_mode_kwargs(mode))
        if sync_fn is not None
        else None
    )
    llm = InstructorLLM(
        client=primary,
        model=litellm_model,
        # "openai"/"azure" for those clients, "litellm" otherwise: evaluate()
        # picks default embeddings from it (_infer_embedding_provider_from_llm)
        provider=kind if kind in ("openai", "azure") else "litellm",
        cache=cache,
        sync_client=sync_twin,
        **kwargs,
    )
    llm._litellm = True
    llm.rate_limit = rate_limit
    if kind in ("openai", "azure"):
        llm.source_client = client
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
    (``openai/gpt-5.6-sol``, ``openai.gpt-6-sol``) (#67).

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

    These are defaults: where a provider rejects one (``temperature`` and
    ``top_p`` on Anthropic, ``max_tokens``/``temperature``/``top_p`` on OpenAI
    reasoning models) it is dropped or mapped, while a value the caller sets
    explicitly is sent.
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
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.

        ``prompt`` is a string, sent as one user message, or a list of
        OpenAI-format chat messages (content may be a list of text and
        ``image_url`` parts) sent as given; ``system_prompt`` is prepended to
        either unless the messages carry their own system message. Both go
        through the same retries, cache, rate limit, tracing and usage
        recording (#214).

        ``temperature`` overrides the configured temperature for this call, for
        metrics that deliberately draw several samples (``strictness``).
        ``sample_index`` does not reach the provider; it only gives each sample
        its own cache entry so a cache doesn't collapse N samples into one.
        """

    @abstractmethod
    async def agenerate(
        self,
        prompt: LLMInput,
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


# OpenInference LLM attributes (#196); spelled out, as in ragas.callbacks.
_LLM_MODEL_NAME = "llm.model_name"
_LLM_PROVIDER = "llm.provider"
_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_LLM_TOKEN_COUNT = "llm.token_count"


def _llm_span() -> t.Any:
    """The recording span of the LLM call in progress, if any."""
    group = current_group()
    manager = getattr(group, "parent_run_manager", None)
    span = getattr(manager, "span", None)
    if span is None or not span.is_recording():
        return None
    meta = getattr(group, "metadata", {})
    return span if meta.get("type") is ChainType.LLM else None


def _record_usage_on_span(span: t.Any, raw: t.Any) -> None:
    """Token counts from an OpenAI-shaped completion (what LiteLLM returns)."""
    try:
        from ragas.cost import get_token_usage_for_openai

        usage = get_token_usage_for_openai(raw)
        span.set_attribute(f"{_LLM_TOKEN_COUNT}.prompt", int(usage.input_tokens))
        span.set_attribute(f"{_LLM_TOKEN_COUNT}.completion", int(usage.output_tokens))
        span.set_attribute(
            f"{_LLM_TOKEN_COUNT}.total",
            int(usage.input_tokens) + int(usage.output_tokens),
        )
    except Exception:
        logger.debug("Could not record token usage on the LLM span", exc_info=True)


class InstructorLLM(InstructorBaseRagasLLM):
    """LLM wrapper using the Instructor library for structured outputs.

    ``llm_factory()`` builds these over LiteLLM (#182). You can also construct
    one directly around your own instructor client (e.g.
    ``instructor.from_provider(...)``); the provider rules below still apply,
    but LiteLLM's ``drop_params`` and the LiteLLM-only defaults don't.
    """

    # Set by llm_factory(): calls go through LiteLLM.
    _litellm: bool = False
    # The OpenAI/Azure client the caller passed to llm_factory(), if any.
    # evaluate() builds default embeddings from it (same base_url, proxy,
    # credentials); ``client`` is the instructor wrapper around LiteLLM.
    source_client: t.Optional[t.Any] = None
    # The RateLimit llm_factory() paced this LLM's requests with, if any.
    rate_limit: t.Optional["RateLimit"] = None

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

        self._wrap_with_cache()

    def _wrap_with_cache(self) -> None:
        """Route ``generate``/``agenerate`` through ``cache``, if set.

        The wrappers are bound to this object, so a copy gets its own
        (see ``__copy__``): with the original's, the copy's calls would run
        on the original, with its settings.
        """
        if self.cache is None:
            return
        cls = type(self)
        self.generate = cacher(  # type: ignore
            cache_backend=self.cache, identity=lambda: cache_identity(self)
        )(cls.generate.__get__(self, cls))
        self.agenerate = cacher(  # type: ignore
            cache_backend=self.cache, identity=lambda: cache_identity(self)
        )(cls.agenerate.__get__(self, cls))

    # Shared by copies, never copied: what connects to the provider, and the
    # limits and stores meant to be shared. A copied LLM goes to the same
    # provider, under the same rate limit, with the same cache (#211).
    _SHARED_ON_COPY = frozenset(
        {
            "client",
            "sync_client",
            "source_client",
            "cache",
            "rate_limit",
            "usage_collector",
        }
    )

    def _copied(self, memo: t.Optional[t.Dict[int, t.Any]]) -> "InstructorLLM":
        from ragas._copying import copy_sharing

        new = copy_sharing(
            self,
            shared=self._SHARED_ON_COPY,
            rebuilt={"generate", "agenerate"},
            memo=memo,
        )
        new._wrap_with_cache()
        return new

    def __copy__(self) -> "InstructorLLM":
        return self._copied(None)

    def __deepcopy__(self, memo: t.Dict[int, t.Any]) -> "InstructorLLM":
        # instructor's clients can't be deep-copied: their __getattr__
        # recurses on an object that copy made without __init__ (#211)
        return self._copied(memo)

    def _map_provider_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None, sampled: bool = False
    ) -> t.Dict[str, t.Any]:
        """Route to provider-specific parameter mapping.

        ``args`` defaults to ``self.model_args``; pass a per-call variant (e.g.
        from ``_with_sampling``) to map that instead, with ``sampled=True`` when
        it carries a per-call temperature ragas chose.

        - Google: Wraps parameters in generation_config and renames max_tokens
        - OpenAI reasoning models (any provider, e.g. OpenAI, Azure, Bedrock):
          see ``_map_openai_params``
        - Anthropic: see ``_map_anthropic_params``
        - Others: pass through unchanged
        """
        if args is None:
            args = self.model_args

        if self.provider.lower() == "google" and not self._litellm:
            mapped = self._map_google_params(args)
        elif _is_anthropic_model(self.provider, self.model):
            mapped = self._map_anthropic_params(args, sampled=sampled)
        else:
            mapped = self._map_openai_params(args)
        if self._litellm:
            mapped = self._map_litellm_defaults(mapped, sampled=sampled)
        return mapped

    def _map_litellm_defaults(
        self, mapped: t.Dict[str, t.Any], sampled: bool
    ) -> t.Dict[str, t.Any]:
        """LiteLLM-only rules (#182).

        - ``max_tokens``' default (1024) isn't sent: it truncated long and
          reasoning outputs, and LiteLLM supplies providers' required caps.
        - When the call carries a sampling value ragas chose (the
          ``temperature`` / ``top_p`` defaults, or a sampled temperature),
          ``drop_params=True`` lets LiteLLM drop it for models that reject
          it, rather than failing the call. A call carrying only values the
          caller set gets no ``drop_params``, so a rejection surfaces.
        """
        mapped = dict(mapped)
        if "max_tokens" not in self._explicit_args:
            mapped.pop("max_tokens", None)
        ragas_chosen = sampled or any(
            p in mapped and p not in self._explicit_args
            for p in ("temperature", "top_p")
        )
        if ragas_chosen:
            mapped["drop_params"] = True
        return mapped

    def _map_anthropic_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None, sampled: bool = False
    ) -> t.Dict[str, t.Any]:
        """Claude gets no sampling parameters unless the caller set them (#177).

        Current Claude models reject ``temperature`` (other than 1) and
        ``top_p``, and anthropic SDK 2.6 removed both from
        ``messages.create()``, so sending ragas' defaults failed every call.
        The defaults are never sent, and neither is a per-call temperature
        ragas chose for sampling (``strictness``, ``n > 1``): those calls use
        the provider's default sampling. A value the caller set is sent as
        given, so a provider that rejects it says so.
        """
        mapped = dict(self.model_args if args is None else args)
        for param in ("temperature", "top_p"):
            if sampled or param not in self._explicit_args:
                mapped.pop(param, None)
        if sampled and not getattr(self, "_warned_claude_sampling", False):
            self._warned_claude_sampling = True
            logger.warning(
                "%s doesn't accept a sampling temperature; sampled calls "
                "(strictness, n > 1) use the provider's default sampling.",
                self.model,
            )
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

    @staticmethod
    def _record_raw(
        collector: t.Any, span: t.Any, raw: t.Any, provider_kwargs: t.Dict[str, t.Any]
    ) -> None:
        if collector is not None:
            collector.record(raw)
        if span is not None:
            _record_usage_on_span(span, raw)
            try:
                span.set_attribute(
                    _LLM_INVOCATION_PARAMETERS,
                    json.dumps(provider_kwargs, default=repr, sort_keys=True),
                )
            except Exception:
                pass

    @contextlib.contextmanager
    def _traced_call(
        self, prompt: LLMInput, response_model: t.Type[BaseModel]
    ) -> t.Iterator[t.Any]:
        """One LLM run: a span (kind LLM) and a run-tree node (#196).

        Nested under the current run -- the metric, prompt or experiment row --
        so evaluate()'s ``result.traces`` gets each call's prompt and output.
        """
        name = getattr(response_model, "__name__", "llm")
        with traced(name, {"prompt": prompt}, ChainType.LLM) as run:
            span = run.span
            if span is not None and span.is_recording():
                span.set_attribute(_LLM_MODEL_NAME, self.model)
                span.set_attribute(_LLM_PROVIDER, self.provider)
                for i, message in enumerate(self._messages(prompt)):
                    prefix = f"llm.input_messages.{i}.message"
                    span.set_attribute(f"{prefix}.role", message["role"])
                    span.set_attribute(
                        f"{prefix}.content",
                        _content_for_span(message.get("content")),
                    )
            yield run

    @staticmethod
    def _end_call(run: t.Any, result: t.Any) -> None:
        run.end({"output": result})
        span = run.span
        if span is not None and span.is_recording():
            try:
                content = (
                    result.model_dump_json()
                    if isinstance(result, BaseModel)
                    else str(result)
                )
            except Exception:
                content = repr(result)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute(
                "llm.output_messages.0.message.content", clip_for_span(content)
            )

    def _active_usage_collector(self) -> t.Optional[t.Any]:
        """The enclosing run's collector, else one attached to this LLM."""
        return scoped_usage_collector() or self.usage_collector

    def _messages(self, prompt: LLMInput) -> t.List[ChatMessage]:
        return _build_messages(prompt, self.system_prompt)

    def _target(self, client: t.Any) -> t.Any:
        return client if self.provider.lower() == "google" else client.chat.completions

    def _call_sync(
        self,
        client: t.Any,
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        """One blocking structured-output call on a sync instructor client."""
        provider_kwargs = self._map_provider_params(
            self._with_sampling(self.model_args, temperature),
            sampled=temperature is not None,
        )
        target = self._target(client)
        collector = self._active_usage_collector()
        span = _llm_span()
        if collector is not None or span is not None:
            # create_with_completion also returns the provider's raw response,
            # which is the only place token counts are available. Gated so the
            # default path (no collector, no tracing SDK) is byte-identical.
            result, raw = target.create_with_completion(
                model=self.model,
                messages=self._messages(prompt),
                response_model=response_model,
                **provider_kwargs,
            )
            self._record_raw(collector, span, raw, provider_kwargs)
            return result
        return target.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **provider_kwargs,
        )

    def generate(
        self,
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float] = None,
        sample_index: t.Optional[int] = None,
    ) -> InstructorTypeVar:
        """Generate a response using the configured LLM.

        For async clients, this will run the async method in the appropriate event loop.
        See ``InstructorBaseRagasLLM.generate`` for ``temperature``/``sample_index``.
        """
        rc = self.run_config
        with self._traced_call(prompt, response_model) as run:
            if self.sync_client is not None:
                result = call_with_retry(
                    rc,
                    self._call_sync,
                    self.sync_client,
                    prompt,
                    response_model,
                    temperature,
                )
            elif self.is_async:
                result = self._run_async_in_current_loop(
                    acall_with_retry(
                        rc, self._call_async, prompt, response_model, temperature
                    )
                )
            else:
                result = call_with_retry(
                    rc,
                    self._call_sync,
                    self.client,
                    prompt,
                    response_model,
                    temperature,
                )
            self._end_call(run, result)
            return result

    async def _call_async(
        self,
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        provider_kwargs = self._map_provider_params(
            self._with_sampling(self.model_args, temperature),
            sampled=temperature is not None,
        )
        target = self._target(self.client)
        collector = self._active_usage_collector()
        span = _llm_span()
        if collector is not None or span is not None:
            result, raw = await target.create_with_completion(
                model=self.model,
                messages=self._messages(prompt),
                response_model=response_model,
                **provider_kwargs,
            )
            self._record_raw(collector, span, raw, provider_kwargs)
            return result
        return await target.create(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **provider_kwargs,
        )

    async def agenerate(
        self,
        prompt: LLMInput,
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
        with self._traced_call(prompt, response_model) as run:
            if not self.is_async:
                result = await asyncio.to_thread(
                    call_with_retry,
                    rc,
                    self._call_sync,
                    self.client,
                    prompt,
                    response_model,
                    temperature,
                )
            else:
                result = await acall_with_retry(
                    rc, self._call_async, prompt, response_model, temperature
                )
            self._end_call(run, result)
            return result

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
