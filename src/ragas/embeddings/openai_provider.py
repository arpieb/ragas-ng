import asyncio
import typing as t

from ragas.cache import CacheInterface

from .base import BaseRagasEmbedding
from .utils import validate_texts


class OpenAIEmbeddings(BaseRagasEmbedding):
    """OpenAI embeddings implementation with batch optimization.

    Supports both sync and async OpenAI clients with automatic detection.
    Provides optimized batch processing for better performance.
    """

    PROVIDER_NAME = "openai"
    REQUIRES_CLIENT = True
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        client: t.Any,
        model: str = "text-embedding-3-small",
        cache: t.Optional[CacheInterface] = None,
    ):
        super().__init__(cache=cache)
        self.client = client
        self.model = model
        self.is_async = self._check_client_async(client)

    def _request(
        self, input: t.Union[str, t.List[str]], kwargs: t.Dict[str, t.Any]
    ) -> t.Dict[str, t.Any]:
        """The keyword arguments for ``client.embeddings.create``."""
        return dict(input=input, model=self.model, **kwargs)

    @staticmethod
    def _vectors(response: t.Any) -> t.List[t.List[float]]:
        return [item.embedding for item in response.data]

    def embed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Embed a single text using OpenAI.

        For async clients, this will run the async method in the appropriate event loop.
        """
        if self.is_async:
            return self._run_async_in_current_loop(self.aembed_text(text, **kwargs))
        response = self.client.embeddings.create(**self._request(text, kwargs))
        return self._vectors(response)[0]

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        """Asynchronously embed a single text using OpenAI."""
        if not self.is_async:
            # A sync client runs on a worker thread, so the async collections
            # metrics work with OpenAI() too (#68).
            return await asyncio.to_thread(self.embed_text, text, **kwargs)
        response = await self.client.embeddings.create(**self._request(text, kwargs))
        return self._vectors(response)[0]

    def embed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        """Embed multiple texts using OpenAI's batch API for optimization."""
        texts = validate_texts(texts)
        if not texts:
            return []
        if self.is_async:
            return self._run_async_in_current_loop(self.aembed_texts(texts, **kwargs))
        # OpenAI supports batch embedding natively
        response = self.client.embeddings.create(**self._request(texts, kwargs))
        return self._vectors(response)

    async def aembed_texts(
        self, texts: t.List[str], **kwargs: t.Any
    ) -> t.List[t.List[float]]:
        """Asynchronously embed multiple texts using OpenAI's batch API."""
        texts = validate_texts(texts)
        if not texts:
            return []
        if not self.is_async:
            # A sync client runs on a worker thread, so the async collections
            # metrics work with OpenAI() too (#68).
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)
        response = await self.client.embeddings.create(**self._request(texts, kwargs))
        return self._vectors(response)

    def _get_client_info(self) -> str:
        """Get client type and async status information."""
        client_type = self.client.__class__.__name__
        async_status = "async" if self.is_async else "sync"
        return f"<{client_type}:{async_status}>"

    def __repr__(self) -> str:
        """Return a detailed string representation of the OpenAI embeddings."""
        client_info = self._get_client_info()
        return f"OpenAIEmbeddings(provider='openai', model='{self.model}', client={client_info})"

    __str__ = __repr__
