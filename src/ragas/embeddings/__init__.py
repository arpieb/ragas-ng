from ragas.embeddings.base import (
    BaseRagasEmbedding,
    embedding_factory as _embedding_factory,
)
from ragas.embeddings.google_provider import GoogleEmbeddings
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.embeddings.litellm_provider import LiteLLMEmbeddings
from ragas.embeddings.openai_provider import OpenAIEmbeddings

# Utilities
from ragas.embeddings.utils import batch_texts, get_optimal_batch_size, validate_texts


def embedding_factory(*args, **kwargs):
    """Deprecated: Use embedding_factory from base module directly."""
    import warnings

    warnings.warn(
        "Importing embedding_factory from ragas.embeddings is deprecated. "
        "Import directly from ragas.embeddings.base or use modern providers: "
        "from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings",
        DeprecationWarning,
        stacklevel=2,
    )
    return _embedding_factory(*args, **kwargs)


__all__ = [
    "BaseRagasEmbedding",
    "embedding_factory",
    # Backward compatibility alias
    "RagasBaseEmbedding",
    "OpenAIEmbeddings",
    "GoogleEmbeddings",
    "LiteLLMEmbeddings",
    "HuggingFaceEmbeddings",
    # Utilities
    "validate_texts",
    "batch_texts",
    "get_optimal_batch_size",
]

# Backward compatibility alias
RagasBaseEmbedding = BaseRagasEmbedding
