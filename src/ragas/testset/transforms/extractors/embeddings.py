import typing as t
from dataclasses import dataclass, field

from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
from ragas.testset._modern import require_embedding
from ragas.testset.graph import Node
from ragas.testset.transforms.base import Extractor


def _default_embedding() -> BaseRagasEmbedding:
    """``embedding_factory()``'s default: OpenAI's model through LiteLLM."""
    return require_embedding(embedding_factory(), "EmbeddingExtractor")


@dataclass
class EmbeddingExtractor(Extractor):
    """
    A class for extracting embeddings from nodes in a knowledge graph.

    Attributes
    ----------
    property_name : str
        The name of the property to store the embedding
    embed_property_name : str
        The name of the property containing the text to embed
    embedding_model : BaseRagasEmbedding
        The embedding model used for generating embeddings, such as the output
        of ``embedding_factory()``
    """

    property_name: str = "embedding"
    embed_property_name: str = "page_content"
    embedding_model: BaseRagasEmbedding = field(default_factory=_default_embedding)

    def __post_init__(self):
        super().__post_init__()
        self.embedding_model = require_embedding(
            self.embedding_model, type(self).__name__
        )

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        """
        Extracts the embedding for a given node.

        Raises
        ------
        ValueError
            If the property to be embedded is not a string.
        """
        text = node.get_property(self.embed_property_name)
        if not isinstance(text, str):
            raise ValueError(
                f"node.property('{self.embed_property_name}') must be a string, found '{type(text)}'"
            )
        # Every BaseRagasEmbedding implements aembed_text, moving a sync
        # client to a worker thread itself.
        embedding = await self.embedding_model.aembed_text(text)
        return self.property_name, embedding
