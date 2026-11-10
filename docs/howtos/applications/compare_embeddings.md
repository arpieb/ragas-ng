# Compare Embeddings for a Retriever

The retriever is usually the limiting factor in a RAG system, and the embedding
model is the biggest lever you have over it. Swapping embeddings is cheap to try
and hard to reason about in the abstract — the only reliable way to choose is to
measure both on *your* corpus.

This guide builds one testset, runs two embedding models through an identical
retrieval pipeline, and scores them with the same retrieval metrics.

<figure markdown="span">
![Compare Embeddings](../../_static/imgs/compare-embeddings.jpeg){width="600"}
<figcaption>Compare Embeddings</figcaption>
</figure>

!!! important "Change one thing at a time"
    Chunk size, chunk overlap and `top_k` all affect retrieval scores as much as
    the embedding model does. Hold them fixed across both runs, and reuse the
    *same* testset — otherwise you are not measuring the embeddings.

## Load your documents

Any object exposing `page_content` and `metadata` works, so LangChain
loaders are fine; LlamaIndex documents need a `document_adapter` (see
[LlamaIndex](../integrations/llamaindex.md)). Here we build ragas `Document`
objects directly, which needs no extra dependency.

```python
from pathlib import Path

from ragas.testset.document import Document

docs = [
    Document(page_content=p.read_text(), metadata={"source": str(p)})
    for p in Path("./my-corpus").rglob("*.md")
]
```

## Generate a testset

Ragas generates evaluation data from your own corpus, so the questions reflect
what your documents actually contain. See
[testset generation](../../getstarted/rag_testset_generation.md) for the details,
or [customizing test data generation](../customizations/testgenerator/index.md)
if you want to bring your own dataset instead.

```python
from openai import OpenAI

from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator

openai_client = OpenAI()
generator_llm = llm_factory("gpt-6-luna", client=openai_client)
generator_embeddings = embedding_factory(
    "openai", model="text-embedding-3-small", client=openai_client
)

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
testset = generator.generate_with_docs(docs, testset_size=50)
```

Generate this **once** and reuse it for every embedding model you compare.

## Build a retriever you can swap embeddings into

The pipeline below is deliberately minimal — an in-memory cosine-similarity
index over fixed-size chunks — so that the embedding model is the only thing
that varies between runs. Substitute your own vector store if you would rather
measure the stack you actually ship.

```python
import numpy as np


def chunk(docs, size=800, overlap=100):
    chunks = []
    for doc in docs:
        text = doc.page_content
        for start in range(0, len(text), size - overlap):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(piece)
    return chunks


class EmbeddingIndex:
    """In-memory cosine-similarity index over a fixed set of chunks."""

    def __init__(self, chunks, embedding):
        self.chunks = chunks
        self.embedding = embedding
        matrix = np.asarray(embedding.embed_texts(chunks), dtype=np.float32)
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def retrieve(self, query, top_k=3):
        q = np.asarray(self.embedding.embed_text(query), dtype=np.float32)
        q = q / np.linalg.norm(q)
        scores = self.matrix @ q
        return [self.chunks[i] for i in np.argsort(-scores)[:top_k]]


chunks = chunk(docs)
```

## Choose the metrics

These two isolate the retriever: neither one looks at a generated answer, so a
change in score is attributable to retrieval alone.

- **Context precision** — of the contexts retrieved, how many were relevant.
- **Context recall** — of the information needed to answer, how much was retrieved.

```python
from openai import AsyncOpenAI

from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

evaluator_llm = llm_factory("gpt-6-luna", client=AsyncOpenAI())
metrics = [
    ContextPrecisionWithReference(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]
```

## Score each embedding model

Both metrics take the same three fields, so one loop scores every question
against whichever embedding model you hand it:

```python
import asyncio

import pandas as pd


async def score(embedding, top_k=3):
    index = EmbeddingIndex(chunks, embedding)
    rows = []
    for sample in testset.samples:
        question = sample.eval_sample.user_input
        reference = sample.eval_sample.reference
        contexts = index.retrieve(question, top_k)
        scores = await asyncio.gather(
            *(
                metric.ascore(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference,
                )
                for metric in metrics
            )
        )
        rows.append({m.name: s.value for m, s in zip(metrics, scores)})
    return pd.DataFrame(rows)
```

Now run it for each contender. `embedding_factory` reaches both hosted and local
models, so you are not restricted to one vendor:

```python
results = {
    "text-embedding-3-small": await score(
        embedding_factory(
            "openai", model="text-embedding-3-small", client=openai_client
        )
    ),
    "bge-small-en-v1.5": await score(
        embedding_factory("huggingface", model="BAAI/bge-small-en-v1.5")
    ),
}
```

!!! note "Running outside a notebook"
    `await` at the top level works in Jupyter. In a script, wrap the calls in an
    `async def main()` and run it with `asyncio.run(main())`. Every metric also
    has a synchronous `.score()` if you would rather avoid async entirely.

The HuggingFace provider runs locally and needs `sentence-transformers`
installed. To reach a provider through LiteLLM instead, omit the client —
`embedding_factory()` resolves credentials from the environment. See
[customizing models](../customizations/customize_models.md) for the provider
options, including Azure and Vertex AI.

## Compare the scores

```python
summary = pd.DataFrame({name: df.mean() for name, df in results.items()}).T
summary
```

|  | context_precision_with_reference | context_recall |
|---|---|---|
| text-embedding-3-small | 0.7421 | 0.8033 |
| bge-small-en-v1.5 | 0.7108 | 0.8194 |

!!! note "Illustrative numbers"
    The values above are an example of the shape of the output, not a benchmark
    result. Which model wins depends entirely on your corpus — that is the whole
    reason to run this.

Read the two metrics together. A model can win on recall while losing on
precision, which usually means it is retrieving more broadly: often a good trade
at a larger `top_k`, and a bad one when your generator is sensitive to
distracting context.

Each `results[...]` entry is a per-question DataFrame, so the rows where the two
models disagree are one filter away:

```python
a, b = results["text-embedding-3-small"], results["bge-small-en-v1.5"]
gap = (a["context_recall"] - b["context_recall"]).abs()
gap.sort_values(ascending=False).head()
```

Those questions tell you what kind of query each model is failing on, which no
aggregate score will.
