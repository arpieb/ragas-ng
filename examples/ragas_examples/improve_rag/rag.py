"""
RAG implementation supporting both naive and agentic modes.

Usage:
    retriever = BM25Retriever()                        # create retriever
    rag = RAG(llm_client, retriever)                   # naive mode (default)
    rag = RAG(llm_client, retriever, mode="agentic")   # agentic mode
    result = await rag.query("What is...?")            # returns: {answer, retrieved_documents, num_retrieved}
"""

import logging
import os
from typing import Any, Dict, Optional

import mlflow

# Suppress MLflow warnings when server is not running
logging.getLogger("mlflow.tracing.export.mlflow_v3").setLevel(logging.ERROR)
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from ragas.testset.document import Document

import datasets

# Configure logger
logger = logging.getLogger(__name__)


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Walks the separators from coarsest to finest and splits on the first one
    that gets a piece under ``chunk_size``, falling back to a hard character
    cut. This replaces langchain's RecursiveCharacterTextSplitter -- ragas no
    longer depends on langchain, and neither should its examples.
    """
    separators = ["\n\n", "\n", ".", " ", ""]

    def split(piece: str, seps: list[str]) -> list[str]:
        if len(piece) <= chunk_size:
            return [piece] if piece.strip() else []
        if not seps:
            return [piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size)]

        sep, rest = seps[0], seps[1:]
        parts = piece.split(sep) if sep else list(piece)

        chunks, current = [], ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part if len(part) <= chunk_size else ""
                if not current:
                    chunks.extend(split(part, rest))
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    chunks = split(text, separators)
    if chunk_overlap <= 0 or len(chunks) < 2:
        return chunks

    # Re-introduce overlap by prefixing each chunk with the tail of the previous.
    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        overlapped.append(previous[-chunk_overlap:] + chunk)
    return overlapped


# The model the app under test calls. Override with RAGAS_EXAMPLE_APP_MODEL.
DEFAULT_MODEL = os.getenv("RAGAS_EXAMPLE_APP_MODEL", "gpt-6-luna")

class BM25Retriever:
    """Simple BM25-based retriever for document search."""

    def __init__(self, dataset_name="m-ric/huggingface_doc", default_k=3):
        self.default_k = default_k
        self.chunks: list[Document] = []
        self.retriever = self._build_retriever(dataset_name)

    def _build_retriever(self, dataset_name: str) -> BM25Okapi:
        """Build a BM25 index over chunks of the HuggingFace docs."""
        knowledge_base = datasets.load_dataset(dataset_name, split="train")
        
        # Create documents
        source_documents = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"].split("/")[1]},
            )
            for row in knowledge_base
        ]
        
        # Split documents
        all_chunks = [
            Document(page_content=piece, metadata=dict(document.metadata))
            for document in source_documents
            for piece in split_text(document.page_content)
        ]
        
        # Simple deduplication
        unique_chunks = []
        seen_content = set()
        for chunk in all_chunks:
            if chunk.page_content not in seen_content:
                seen_content.add(chunk.page_content)
                unique_chunks.append(chunk)
        
        self.chunks = unique_chunks
        # langchain's BM25Retriever was a thin wrapper over this same library,
        # with whitespace tokenisation.
        return BM25Okapi([chunk.page_content.split() for chunk in unique_chunks])

    def retrieve(self, query: str, top_k: int = None) -> list[Document]:
        """Retrieve documents for a given query."""
        if top_k is None:
            top_k = self.default_k
        scores = self.retriever.get_scores(query.split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k]]


class RAG:
    """RAG system that can operate in naive or agentic mode."""

    @staticmethod
    def _check_mlflow_server(uri: str = "http://127.0.0.1:5000", timeout: float = 0.5) -> bool:
        """Check if MLflow server is running."""
        import urllib.request
        import urllib.error
        try:
            urllib.request.urlopen(uri, timeout=timeout)
            return True
        except (urllib.error.URLError, OSError):
            return False

    def __init__(self, llm_client: AsyncOpenAI, retriever: BM25Retriever, mode="naive", system_prompt=None, model=DEFAULT_MODEL, default_k=3):
        # Enable MLflow autolog for OpenAI API calls (optional - only if server is running)
        self._mlflow_enabled = False
        if os.environ.get("MLFLOW_TRACKING_URI") or self._check_mlflow_server():
            try:
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
                mlflow.openai.autolog()
                self._mlflow_enabled = True
            except Exception:
                pass
        
        self.llm_client = llm_client
        self.retriever = retriever
        self.mode = mode.lower()
        self.model = model
        self.default_k = default_k
        self.system_prompt = system_prompt or "Answer only based on documents. Be concise.\n\nQuestion: {query}\nDocuments:\n{context}\nAnswer:"
        self._agent = None
        
        if self.mode == "agentic":
            self._setup_agent()

    def _setup_agent(self):
        """Setup agent for agentic mode."""
        try:
            from agents import Agent, function_tool
        except ImportError:
            raise ImportError("agents package required for agentic mode")

        @function_tool
        def retrieve(query: str) -> str:
            """Search Hugging Face docs for technical info, APIs, commands, and examples.
            Use exact terms (e.g., "from_pretrained", "ESPnet upload", "torchrun"). 
            Try 2-3 targeted searches: specific terms → tool names → alternatives."""
            docs = self.retriever.retrieve(query, self.default_k)
            if not docs:
                return f"No documents found for '{query}'. Try different search terms or break down the query into smaller parts."
            return "\n\n".join([f"Doc {i}: {doc.page_content}" for i, doc in enumerate(docs, 1)])

        self._agent = Agent(
            name="RAG Assistant",
            model=self.model,
            instructions="Search with exact terms first (commands, APIs, tool names). Try 2-3 different searches if needed. Only answer from retrieved documents. Preserve exact syntax and technical details.",
            tools=[retrieve]
        )

    async def _naive_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle naive mode: retrieve once, then generate."""
        # Retrieve documents
        docs = self.retriever.retrieve(question, top_k)
        
        if not docs:
            return {"answer": "No relevant documents found.", "retrieved_documents": [], "num_retrieved": 0}
        
        # Generate response
        context = "\n\n".join([f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(docs, 1)])
        prompt = self.system_prompt.format(query=question, context=context)
        
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Get the active trace ID (only if MLflow is enabled)
        trace_id = mlflow.get_last_active_trace_id() if self._mlflow_enabled else None

        return {
            "answer": response.choices[0].message.content.strip(),
            "retrieved_documents": [{"content": doc.page_content, "metadata": doc.metadata, "document_id": i} for i, doc in enumerate(docs)],
            "num_retrieved": len(docs),
            "mlflow_trace_id": trace_id
        }

    async def _agentic_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle agentic mode: agent controls retrieval strategy."""
        try:
            from agents import Runner
        except ImportError:
            raise ImportError("agents package required for agentic mode")

        # Let agent handle the retrieval and reasoning
        result = await Runner.run(self._agent, input=question)

        # Get the active trace ID (only if MLflow is enabled)
        trace_id = mlflow.get_last_active_trace_id() if self._mlflow_enabled else None

        # In agentic mode, the agent controls retrieval internally
        # so we don't return specific retrieved documents
        return {
            "answer": result.final_output,
            "retrieved_documents": [],  # Agent handles retrieval internally
            "num_retrieved": 0,  # Cannot determine exact count from agent execution
            "mlflow_trace_id": trace_id
        }

    async def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        if top_k is None:
            top_k = self.default_k

        try:
            if self.mode == "naive":
                return await self._naive_query(question, top_k)
            elif self.mode == "agentic":
                return await self._agentic_query(question, top_k)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            # Try to get trace ID even in error cases
            trace_id = mlflow.get_last_active_trace_id() if self._mlflow_enabled else None
            return {
                "answer": f"Error: {str(e)}",
                "retrieved_documents": [],
                "num_retrieved": 0,
                "mlflow_trace_id": trace_id
            }


# Demo
async def main():
    import os
    import pathlib

    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    
    # Load .env from root
    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Suppress HTTP request logs from OpenAI/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Test with a question that failed in previous evaluation
    query = "What command is used to upload an ESPnet model to a Hugging Face repository?"
    
    logger.info("RAG DEMO")
    logger.info("=" * 40)
    
    # Create retriever (shared by both modes)
    logger.info("Creating BM25 retriever...")
    retriever = BM25Retriever()
    
    # Test naive mode
    logger.info("NAIVE MODE:")
    rag = RAG(openai_client, retriever)
    result = await rag.query(query)
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"MLflow Trace ID: {result.get('mlflow_trace_id', 'N/A')}")
    
    
    # Test agentic mode
    logger.info("AGENTIC MODE:")
    try:
        rag = RAG(openai_client, retriever, mode="agentic")
        result = await rag.query(query)
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"MLflow Trace ID: {result.get('mlflow_trace_id', 'N/A')}")
    except ImportError:
        logger.warning("Agentic mode unavailable (agents package missing)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
