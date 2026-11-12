"""Per-provider rules for the model arguments an ``InstructorLLM`` sends.

``InstructorModelArgs`` are defaults: where a provider rejects one it is
dropped or mapped, while a value the caller set explicitly is sent.
``map_provider_params`` applies the first rule in ``PROVIDER_RULES`` that
matches the LLM, then the LiteLLM-only defaults for ``llm_factory()`` LLMs.
"""

from __future__ import annotations

import logging
import re
import typing as t
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorLLM

logger = logging.getLogger(__name__)

ModelArgs = t.Dict[str, t.Any]

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


def is_anthropic_model(provider: str, model: str) -> bool:
    return provider.lower() == "anthropic" or "claude" in model.lower()


class ProviderRule(ABC):
    """How one kind of provider or model takes the model arguments."""

    @abstractmethod
    def applies(self, llm: InstructorLLM) -> bool:
        """Whether this rule maps ``llm``'s arguments."""

    @abstractmethod
    def map(self, llm: InstructorLLM, args: ModelArgs, sampled: bool) -> ModelArgs:
        """``args`` as this provider takes them, as a new dict.

        ``sampled`` is True when ``args`` carry a per-call temperature ragas
        chose (``strictness``, ``n > 1``).
        """


class GoogleGenerationConfig(ProviderRule):
    """Google Gemini through instructor's own client, not LiteLLM.

    Sampling parameters go in a ``generation_config`` dict, and
    ``max_tokens`` is renamed to ``max_output_tokens``.
    """

    _GENERATION_CONFIG_KEYS = frozenset({"temperature", "max_tokens", "top_p", "top_k"})

    def applies(self, llm: InstructorLLM) -> bool:
        return llm.provider.lower() == "google" and not llm._litellm

    def map(self, llm: InstructorLLM, args: ModelArgs, sampled: bool) -> ModelArgs:
        google_kwargs: ModelArgs = {}
        generation_config: ModelArgs = {}
        for key, value in args.items():
            if key not in self._GENERATION_CONFIG_KEYS:
                google_kwargs[key] = value
            elif key == "max_tokens":
                generation_config["max_output_tokens"] = value
            else:
                generation_config[key] = value
        if generation_config:
            google_kwargs["generation_config"] = generation_config
        return google_kwargs


class ClaudeSampling(ProviderRule):
    """Claude gets no sampling parameters unless the caller set them (#177).

    Current Claude models reject ``temperature`` (other than 1) and
    ``top_p``, and anthropic SDK 2.6 removed both from
    ``messages.create()``, so sending ragas' defaults failed every call.
    The defaults are never sent, and neither is a per-call temperature
    ragas chose for sampling (``strictness``, ``n > 1``): those calls use
    the provider's default sampling. A value the caller set is sent as
    given, so a provider that rejects it says so.
    """

    def applies(self, llm: InstructorLLM) -> bool:
        return is_anthropic_model(llm.provider, llm.model)

    def map(self, llm: InstructorLLM, args: ModelArgs, sampled: bool) -> ModelArgs:
        mapped = dict(args)
        for param in ("temperature", "top_p"):
            if sampled or param not in llm._explicit_args:
                mapped.pop(param, None)
        if sampled and not llm._warned_claude_sampling:
            llm._warned_claude_sampling = True
            logger.warning(
                "%s doesn't accept a sampling temperature; sampled calls "
                "(strictness, n > 1) use the provider's default sampling.",
                llm.model,
            )
        return mapped


class OpenAIReasoning(ProviderRule):
    """OpenAI reasoning models (o-series, GPT-5+, codex-mini), on any provider
    serving them (OpenAI, Azure, Bedrock, ...).

    Detection is ``is_reasoning_model``, overridden by ``reasoning=``: Azure
    deployment names are arbitrary, so pass ``reasoning=True`` for a
    reasoning deployment whose name doesn't say so. These models

    1. take ``max_completion_tokens`` instead of ``max_tokens``. The default
       cap is not sent, since it also counts the hidden reasoning tokens; a
       caller's ``max_tokens`` is sent as ``max_completion_tokens``.
    2. only accept temperature 1.0
    3. reject ``top_p``
    """

    def applies(self, llm: InstructorLLM) -> bool:
        if llm.reasoning is not None:
            return llm.reasoning
        return is_reasoning_model(llm.model)

    def map(self, llm: InstructorLLM, args: ModelArgs, sampled: bool) -> ModelArgs:
        mapped = dict(args)
        if "max_tokens" in mapped:
            max_tokens = mapped.pop("max_tokens")
            # max_completion_tokens also counts the hidden reasoning tokens, so
            # the 1024 default truncated structured output. Only a caller's
            # own cap is sent; otherwise the provider default applies.
            if "max_tokens" in llm._explicit_args:
                mapped["max_completion_tokens"] = max_tokens
        mapped["temperature"] = 1.0
        mapped.pop("top_p", None)
        return mapped


# The first rule that applies maps the arguments; with none, they are sent
# unchanged.
PROVIDER_RULES: t.Tuple[ProviderRule, ...] = (
    GoogleGenerationConfig(),
    ClaudeSampling(),
    OpenAIReasoning(),
)


def litellm_defaults(llm: InstructorLLM, mapped: ModelArgs, sampled: bool) -> ModelArgs:
    """LiteLLM-only rules (#182), applied after the provider's rule.

    - ``max_tokens``' default (1024) isn't sent: it truncated long and
      reasoning outputs, and LiteLLM supplies providers' required caps.
    - When the call carries a sampling value ragas chose (the
      ``temperature`` / ``top_p`` defaults, or a sampled temperature),
      ``drop_params=True`` lets LiteLLM drop it for models that reject
      it, rather than failing the call. A call carrying only values the
      caller set gets no ``drop_params``, so a rejection surfaces.
    """
    mapped = dict(mapped)
    if "max_tokens" not in llm._explicit_args:
        mapped.pop("max_tokens", None)
    ragas_chosen = sampled or any(
        p in mapped and p not in llm._explicit_args for p in ("temperature", "top_p")
    )
    if ragas_chosen:
        mapped["drop_params"] = True
    return mapped


def map_provider_params(
    llm: InstructorLLM, args: ModelArgs, sampled: bool = False
) -> ModelArgs:
    """``args`` as ``llm``'s provider takes them (a new dict)."""
    rule = next((r for r in PROVIDER_RULES if r.applies(llm)), None)
    mapped = dict(args) if rule is None else rule.map(llm, args, sampled)
    if llm._litellm:
        mapped = litellm_defaults(llm, mapped, sampled)
    return mapped
