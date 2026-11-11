from llama_index.core import download_loader
from openai import OpenAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.testset.document import Document
from ragas.testset.synthesizers.generate import TestsetGenerator

generator_llm = llm_factory("gpt-4o", client=OpenAI())
embeddings = embedding_factory()

generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)


def get_documents():
    SemanticScholarReader = download_loader("SemanticScholarReader")
    loader = SemanticScholarReader()
    # Narrow down the search space
    query_space = "large language models"
    # Increase the limit to obtain more documents
    documents = loader.load_data(query=query_space, limit=10)

    return documents


IGNORE_ASYNCIO = False
# os.environ["PYTHONASYNCIODEBUG"] = "1"

if __name__ == "__main__":
    documents = get_documents()
    generator.generate_with_docs(
        documents=documents,
        testset_size=50,
        document_adapter=lambda d: Document(page_content=d.text, metadata=d.metadata),
    )
