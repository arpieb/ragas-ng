from __future__ import annotations

import asyncio
import inspect
import typing as t
from abc import ABC, abstractmethod

from ragas._copying import CachedCopyable
from ragas.cache import CacheInterface
from ragas.embeddings.utils import run_async_in_current_loop, validate_texts
from ragas.run_config import (
    RunConfig,
)


class BaseRagasEmbedding(CachedCopyable, ABC):
    """Modern abstract base class for Ragas embedding implementations.

    This class provides a consistent interface for embedding text using various
    providers. Implementations should provide both sync and async methods for
    embedding single texts, with batch methods automatically provided.
    """

    def __init__(self, cache: t.Optional[CacheInterface] = None):
        """Initialize embedding with optional caching.

        Args:
            cache: Optional cache backend for caching embeddings.
                Use DiskCacheBackend() for persistent caching.
        """
        self.cache = cache
        self._wrap_with_cache()

    _CACHED_METHODS = ("embed_text", "aembed_text")
    # Provider clients, the cache and the rate limit. Modules
    # (LiteLLMEmbeddings.litellm) are shared too (#211).
    _SHARED_ON_COPY = frozenset({"client", "cache", "rate_limit"})

    @abstractmethod
    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    @abstractmethod
    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text.

        Args:
            text: The text to embed
            **kwargs: Additional arguments for the embedding call

        Returns:
            List of floats representing the embedding
        """
        pass

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts.

        Default implementation processes texts individually. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        return [self.embed_text(text, **kwargs) for text in texts]

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts.

        Default implementation processes texts concurrently. Override for
        batch optimization.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments for the embedding calls

        Returns:
            List of embeddings, one for each input text
        """
        texts = validate_texts(texts)
        tasks = [self.aembed_text(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)

    def _check_client_async(
        self, client: t.Any, method_path: str = "embeddings.create"
    ) -> bool:
        """Check if a client supports async operations.

        Args:
            client: The client to check
            method_path: Dot-separated path to the method to check

        Returns:
            True if the client supports async operations
        """
        try:
            obj = client
            for attr in method_path.split("."):
                obj = getattr(obj, attr)
            return inspect.iscoroutinefunction(obj)
        except (AttributeError, TypeError):
            return False

    def _run_async_in_current_loop(self, coro):
        """Run an async coroutine in the current event loop if possible.

        This handles Jupyter environments correctly by using a separate thread
        when a running event loop is detected.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        return run_async_in_current_loop(coro)

    @classmethod
    def _from_factory(
        cls,
        model: t.Optional[str] = None,
        client: t.Optional[t.Any] = None,
        **kwargs: t.Any,
    ) -> "BaseRagasEmbedding":
        """Create an embedding instance from factory parameters with validation.

        This base implementation handles common validation patterns. Individual
        providers can override this for custom initialization logic.
        """
        # Validate client requirement
        if getattr(cls, "REQUIRES_CLIENT", False) and not client:
            provider_name = getattr(cls, "PROVIDER_NAME", cls.__name__)
            raise ValueError(f"{provider_name} provider requires a client instance")

        # Validate model requirement
        if getattr(cls, "REQUIRES_MODEL", False) and not model:
            provider_name = getattr(cls, "PROVIDER_NAME", cls.__name__)
            raise ValueError(f"{provider_name} provider requires a model name")

        # Use default model if available and not provided
        if not model:
            model = getattr(cls, "DEFAULT_MODEL", None)

        # Construct instance - let providers handle their own parameters
        # Build constructor arguments based on provider requirements
        init_kwargs = kwargs.copy()
        if model is not None:
            init_kwargs["model"] = model
        if getattr(cls, "REQUIRES_CLIENT", False) and client is not None:
            init_kwargs["client"] = client

        return cls(**init_kwargs)


def embedding_factory(
    provider: str = "openai",
    model: t.Optional[str] = None,
    run_config: t.Optional[RunConfig] = None,
    client: t.Optional[t.Any] = None,
    interface: str = "auto",
    base_url: t.Optional[str] = None,
    cache: t.Optional[CacheInterface] = None,
    **kwargs: t.Any,
) -> BaseRagasEmbedding:
    """
    Create and return an embeddings instance.

    Parameters
    ----------
    provider : str, optional
        Provider name or provider/model string (e.g., "openai", "openai/text-embedding-3-small").
        For backward compatibility, also accepts model names directly.
        Default is "openai".
    model : str, optional
        The embedding model name. If not provided, uses provider defaults.
        For legacy calls, defaults to "text-embedding-ada-002".
    run_config : RunConfig, optional
        Configuration for the run, by default None.
    client : Any, optional
        Pre-initialized client for modern providers. When provided, uses modern interface.
    interface : str, optional
        Interface type: "legacy", "modern", or "auto" (default).
        "auto" detects based on parameters.
    base_url : str, optional
        Base URL for the API, by default None.
    cache : CacheInterface, optional
        Optional cache backend for caching embeddings.
        Use DiskCacheBackend() for persistent caching across runs.
        Saves costs and speeds up repeated embedding calls.
    **kwargs : Any
        Additional provider-specific arguments.

    Returns
    -------
    BaseRagasEmbedding
        An instance of the requested embedding provider.

    Examples
    --------
    # Through LiteLLM (credentials from the environment)
    embedder = embedding_factory()
    embedder = embedding_factory("text-embedding-ada-002")

    # With a client
    embedder = embedding_factory("openai", "text-embedding-3-small", client=openai_client)
    embedder = embedding_factory("huggingface", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = embedding_factory("google", client=vertex_client, project_id="my-project")

    # With caching
    from ragas.cache import DiskCacheBackend
    cache = DiskCacheBackend()
    embedder = embedding_factory("openai", client=openai_client, cache=cache)
    """
    if interface == "legacy":
        raise ValueError(
            "The legacy embedding interface has been removed along with LangChain. "
            "Either pass a client -- embedding_factory('openai', "
            "model='text-embedding-3-small', client=openai_client) -- or omit it to "
            "use litellm, which resolves credentials from the environment."
        )

    # OpenAI is the one provider that REQUIRES_CLIENT, so a client-less call to it
    # has no modern home. Route it through litellm, which resolves credentials from
    # the environment and is provider-neutral. This replaces the old LangChain path.
    # google/huggingface/litellm already construct fine without a client.
    if client is None and (
        _looks_like_model_name(provider) or provider.lower() == "openai"
    ):
        from ragas.embeddings.litellm_provider import LiteLLMEmbeddings

        model_name = (
            provider
            if _looks_like_model_name(provider)
            else (model or DEFAULT_OPENAI_EMBEDDING_MODEL)
        )
        if base_url is not None:
            kwargs["api_base"] = base_url
        result = LiteLLMEmbeddings(model=model_name, cache=cache, **kwargs)

        return result

    # Modern interface - pass base_url and cache through kwargs for modern providers
    if base_url is not None:
        kwargs["base_url"] = base_url
    if cache is not None:
        kwargs["cache"] = cache
    result = _create_modern_embedding(provider, model, client, **kwargs)

    return result


# embedding_factory("litellm") without a model
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Model name patterns for backward compatibility detection
_LEGACY_MODEL_PATTERNS = {"text-embedding", "ada", "davinci", "gpt", "curie", "babbage"}


def _looks_like_model_name(name: str) -> bool:
    """Check if a string looks like an OpenAI model name rather than a provider name."""
    return any(pattern in name.lower() for pattern in _LEGACY_MODEL_PATTERNS)


def _get_provider_registry() -> t.Dict[str, t.Type[BaseRagasEmbedding]]:
    """Auto-discover available provider classes and build a registry.

    Returns:
        Dictionary mapping provider names to their classes.
    """
    from .google_provider import GoogleEmbeddings
    from .huggingface_provider import HuggingFaceEmbeddings
    from .litellm_provider import LiteLLMEmbeddings
    from .openai_provider import OpenAIEmbeddings

    providers = [
        OpenAIEmbeddings,
        GoogleEmbeddings,
        LiteLLMEmbeddings,
        HuggingFaceEmbeddings,
    ]

    return {
        cls.PROVIDER_NAME: cls for cls in providers if hasattr(cls, "PROVIDER_NAME")
    }


def _create_modern_embedding(
    provider: str, model: t.Optional[str], client: t.Optional[t.Any], **kwargs: t.Any
) -> BaseRagasEmbedding:
    """Create a modern embedding instance based on the provider."""
    cache = kwargs.pop("cache", None)

    # Handle provider/model string format
    if "/" in provider and model is None:
        provider_name, model_name = provider.split("/", 1)
        provider = provider_name
        model = model_name

    # Get provider registry and find the class
    registry = _get_provider_registry()
    provider_cls = registry.get(provider.lower())

    if not provider_cls:
        available = ", ".join(registry.keys())
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: {available}"
        )

    # Let the provider class validate and construct itself
    return provider_cls._from_factory(model=model, client=client, cache=cache, **kwargs)


def modern_embedding_factory(
    provider: str,
    model: t.Optional[str] = None,
    client: t.Optional[t.Any] = None,
    **kwargs: t.Any,
) -> BaseRagasEmbedding:
    """
    Factory function to create a modern embedding instance based on the provider.

    DEPRECATED: Use embedding_factory() with interface="modern" or client parameter instead.
    This function is kept for backward compatibility and will be removed in a future version.

    Args:
        provider (str): The name of the embedding provider or provider/model string.
        model (str, optional): The model name to use for embeddings.
        client (Any, optional): Pre-initialized client for the provider.
        **kwargs: Additional arguments for the provider.

    Returns:
        BaseRagasEmbedding: An instance of the specified embedding provider.
    """
    result = embedding_factory(
        provider=provider, model=model, client=client, interface="modern", **kwargs
    )
    # Type narrowing: modern interface always returns BaseRagasEmbedding
    assert isinstance(result, BaseRagasEmbedding), (
        "Modern interface should always return BaseRagasEmbedding"
    )
    return result
