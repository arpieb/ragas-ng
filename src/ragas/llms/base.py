"""The ragas LLM interface and ``InstructorLLM``, its implementation.

``llm_factory()`` (``ragas.llms.factory``) builds an ``InstructorLLM`` over
LiteLLM. The model arguments each provider takes are in
``ragas.llms.provider_rules``, tracing and token usage in
``ragas.llms.tracing``, and caching and copying in ``ragas._copying``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragas._copying import CachedCopyable
from ragas.async_utils import run_sync
from ragas.cache import CacheInterface
from ragas.llms._retry import acall_with_retry, call_with_retry
from ragas.llms.factory import DEFAULT_LLM_MODEL, default_llm, llm_factory
from ragas.llms.messages import ChatMessage, LLMInput, build_messages
from ragas.llms.provider_rules import map_provider_params
from ragas.llms.tracing import end_call, llm_span, record_raw, traced_call
from ragas.run_config import (
    RunConfig,
    scoped_run_config,
    scoped_usage_collector,
)

if t.TYPE_CHECKING:
    from ragas.rate_limit import RateLimit

__all__ = [
    "DEFAULT_LLM_MODEL",
    "InstructorBaseRagasLLM",
    "InstructorLLM",
    "InstructorModelArgs",
    "InstructorTypeVar",
    "default_llm",
    "llm_factory",
]

logger = logging.getLogger(__name__)

# TypeVar for Instructor LLM response models
InstructorTypeVar = t.TypeVar("T", bound=BaseModel)  # type: ignore


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


def _has_async_create(client: t.Any) -> bool:
    """Whether ``client.chat.completions.create`` is a coroutine function.

    True for instructor's ``AsyncInstructor`` (``instructor.from_provider(...,
    async_client=True)``, ``from_litellm(litellm.acompletion)``).
    """
    completions = getattr(getattr(client, "chat", None), "completions", None)
    return inspect.iscoroutinefunction(getattr(completions, "create", None))


class InstructorLLM(CachedCopyable, InstructorBaseRagasLLM):
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
    # Set once ClaudeSampling has warned that sampled calls aren't sampled.
    _warned_claude_sampling: bool = False

    def __init__(
        self,
        client: t.Any,
        model: str,
        provider: str,
        model_args: t.Optional[InstructorModelArgs] = None,
        cache: t.Optional[CacheInterface] = None,
        sync_client: t.Optional[t.Any] = None,
        reasoning: t.Optional[bool] = None,
        is_async: t.Optional[bool] = None,
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

        # Whether ``client`` is async: llm_factory() knows; for a client of
        # your own, its ``chat.completions.create`` says.
        self.is_async = _has_async_create(client) if is_async is None else is_async

        self._wrap_with_cache()

    _CACHED_METHODS = ("generate", "agenerate")
    # A copied LLM goes to the same provider, under the same rate limit, with
    # the same cache (#211).
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

    def _map_provider_params(
        self, args: t.Optional[t.Dict[str, t.Any]] = None, sampled: bool = False
    ) -> t.Dict[str, t.Any]:
        """The model arguments as this LLM's provider takes them.

        ``args`` defaults to ``self.model_args``; pass a per-call variant (e.g.
        from ``_with_sampling``) to map that instead, with ``sampled=True`` when
        it carries a per-call temperature ragas chose. The rules are
        ``ragas.llms.provider_rules.PROVIDER_RULES``.
        """
        return map_provider_params(
            self, self.model_args if args is None else args, sampled=sampled
        )

    def _active_usage_collector(self) -> t.Optional[t.Any]:
        """The enclosing run's collector, else one attached to this LLM."""
        return scoped_usage_collector() or self.usage_collector

    def _messages(self, prompt: LLMInput) -> t.List[ChatMessage]:
        return build_messages(prompt, self.system_prompt)

    def _prepare_call(
        self,
        prompt: LLMInput,
        response_model: t.Type[BaseModel],
        temperature: t.Optional[float],
    ) -> t.Tuple[t.Dict[str, t.Any], t.Optional[t.Callable[[t.Any], None]]]:
        """``(request, record)`` for one structured-output call.

        ``request`` is the keyword arguments for instructor's ``create``.
        ``record`` is set when a usage collector or a recording span wants
        the provider's raw response: the call then uses
        ``create_with_completion``, the only place token counts are
        available, and passes the raw response to ``record``. Gated so the
        default path (no collector, no tracing SDK) is a plain ``create``.
        """
        provider_kwargs = self._map_provider_params(
            self._with_sampling(self.model_args, temperature),
            sampled=temperature is not None,
        )
        request = dict(
            model=self.model,
            messages=self._messages(prompt),
            response_model=response_model,
            **provider_kwargs,
        )
        collector = self._active_usage_collector()
        span = llm_span()
        if collector is None and span is None:
            return request, None
        return request, functools.partial(
            record_raw, collector, span, provider_kwargs=provider_kwargs
        )

    def _call_sync(
        self,
        client: t.Any,
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        """One blocking structured-output call on a sync instructor client."""
        request, record = self._prepare_call(prompt, response_model, temperature)
        completions = client.chat.completions
        if record is None:
            return completions.create(**request)
        result, raw = completions.create_with_completion(**request)
        record(raw)
        return result

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
        with traced_call(
            prompt,
            lambda: self._messages(prompt),
            response_model,
            self.model,
            self.provider,
        ) as run:
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
                result = run_sync(
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
            end_call(run, result)
            return result

    async def _call_async(
        self,
        prompt: LLMInput,
        response_model: t.Type[InstructorTypeVar],
        temperature: t.Optional[float],
    ) -> InstructorTypeVar:
        """One structured-output call on the async instructor client."""
        request, record = self._prepare_call(prompt, response_model, temperature)
        completions = self.client.chat.completions
        if record is None:
            return await completions.create(**request)
        result, raw = await completions.create_with_completion(**request)
        record(raw)
        return result

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
        with traced_call(
            prompt,
            lambda: self._messages(prompt),
            response_model,
            self.model,
            self.provider,
        ) as run:
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
            end_call(run, result)
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
