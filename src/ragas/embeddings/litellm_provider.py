"""LiteLLM embeddings implementation for universal provider support."""

import typing as t

from ragas.cache import CacheInterface

if t.TYPE_CHECKING:
    from ragas.rate_limit import RateLimit

from .base import BaseRagasEmbedding
from .utils import batch_texts, get_optimal_batch_size, safe_import, validate_texts


def _litellm_provider(litellm: t.Any, model: str) -> str:
    """The provider LiteLLM routes ``model`` to, e.g. "openai" or "vertex_ai".

    Reads the prefix ("cohere/embed-english-v3.0") and knows bare names
    ("text-embedding-3-small" is OpenAI). An unknown model gives "litellm",
    which get_optimal_batch_size treats as unknown.

    LiteLLM prints a "Provider List" banner when it can't resolve a model; that
    is noise for a batch-size lookup, so it is suppressed for this one call.
    """
    suppress = getattr(litellm, "suppress_debug_info", False)
    litellm.suppress_debug_info = True
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:
        return "litellm"
    finally:
        litellm.suppress_debug_info = suppress


class LiteLLMEmbeddings(BaseRagasEmbedding):
    """Universal embedding interface using LiteLLM.

    Supports 100+ models across OpenAI, Azure, Google, Cohere, Anthropic, and more.
    Provides intelligent batching and provider-specific optimizations.

    ``rate_limit`` (a ``ragas.rate_limit.RateLimit``) paces every embedding
    request: one per call, or one per batch (#171).
    """

    PROVIDER_NAME = "litellm"
    REQUIRES_MODEL = True

    def __init__(
        self,
        model: str,
        api_key: t.Optional[str] = None,
        api_base: t.Optional[str] = None,
        api_version: t.Optional[str] = None,
        timeout: int = 600,
        max_retries: int = 3,
        batch_size: t.Optional[int] = None,
        cache: t.Optional[CacheInterface] = None,
        rate_limit: t.Optional["RateLimit"] = None,
        **litellm_params: t.Any,
    ):
        super().__init__(cache=cache)
        self.litellm = safe_import("litellm", "litellm")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        # The table is keyed by the real provider ("openai", "cohere", ...),
        # which LiteLLM reads from the model string. Passing "litellm" made
        # every model get the fallback of 10 (#216).
        self._default_batch_size = get_optimal_batch_size(
            _litellm_provider(self.litellm, model), model
        )
        self.batch_size = batch_size or self._default_batch_size
        self.litellm_params = litellm_params
        self.rate_limit = rate_limit

    def _tokens(self, texts: t.Sequence[str]) -> int:
        if self.rate_limit is None or not self.rate_limit.tpm:
            return 0
        from ragas.rate_limit import estimate_embedding_tokens

        return estimate_embedding_tokens(self.model, texts)

    def _wait(self, texts: t.Sequence[str]) -> None:
        if self.rate_limit is not None:
            self.rate_limit.acquire(self._tokens(texts))

    async def _await(self, texts: t.Sequence[str]) -> None:
        if self.rate_limit is not None:
            await self.rate_limit.aacquire(self._tokens(texts))

    def _prepare_kwargs(self, **kwargs: t.Any) -> t.Dict[str, t.Any]:
        """Prepare kwargs for LiteLLM call."""
        call_kwargs = {
            "model": self.model,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            **self.litellm_params,
            **kwargs,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_version:
            call_kwargs["api_version"] = self.api_version

        return call_kwargs

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using LiteLLM."""
        call_kwargs = self._prepare_kwargs(**kwargs)
        self._wait([text])
        response = self.litellm.embedding(input=[text], **call_kwargs)
        return response.data[0]["embedding"]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using LiteLLM."""
        call_kwargs = self._prepare_kwargs(**kwargs)
        await self._await([text])
        response = await self.litellm.aembedding(input=[text], **call_kwargs)
        return response.data[0]["embedding"]

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using LiteLLM with intelligent batching."""
        texts = validate_texts(texts)
        if not texts:
            return []

        embeddings = []
        batches = batch_texts(texts, self.batch_size)

        for batch in batches:
            call_kwargs = self._prepare_kwargs(**kwargs)
            self._wait(batch)
            response = self.litellm.embedding(input=batch, **call_kwargs)
            embeddings.extend([item["embedding"] for item in response.data])

        return embeddings

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using LiteLLM with intelligent batching."""
        texts = validate_texts(texts)
        if not texts:
            return []

        embeddings = []
        batches = batch_texts(texts, self.batch_size)

        for batch in batches:
            call_kwargs = self._prepare_kwargs(**kwargs)
            await self._await(batch)
            response = await self.litellm.aembedding(input=batch, **call_kwargs)
            embeddings.extend([item["embedding"] for item in response.data])

        return embeddings

    def _get_key_config(self) -> str:
        """Get key configuration parameters as a string."""
        config_parts = []

        if self.api_base:
            config_parts.append(f"api_base='{self.api_base}'")

        if self.batch_size != self._default_batch_size:  # unless the provider's
            config_parts.append(f"batch_size={self.batch_size}")

        if self.timeout != 600:  # Only show if different from default
            config_parts.append(f"timeout={self.timeout}")

        if self.max_retries != 3:  # Only show if different from default
            config_parts.append(f"max_retries={self.max_retries}")

        # Show count of other litellm params if there are any
        if self.litellm_params:
            config_parts.append(f"+{len(self.litellm_params)} litellm_params")

        return ", ".join(config_parts)

    def __repr__(self) -> str:
        """Return a detailed string representation of the LiteLLM embeddings."""
        key_config = self._get_key_config()

        base_repr = f"LiteLLMEmbeddings(provider='litellm', model='{self.model}'"

        if key_config:
            base_repr += f", {key_config}"

        base_repr += ")"
        return base_repr

    __str__ = __repr__
