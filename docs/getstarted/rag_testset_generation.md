# Testset Generation for RAG

This simple guide will help you generate a testset for evaluating your RAG pipeline using your own documents.

## Quickstart
Let's walk through a quick example of generating a testset for a RAG pipeline. Following that we will explore the main components of the testset generation pipeline.

### Load Sample Documents

For the sake of this tutorial we will use sample documents from this [repository](https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown). You can replace this with your own documents.

```bash
git clone https://huggingface.co/datasets/vibrantlabsai/Sample_Docs_Markdown
```

### Load documents

Now we will load the documents from the sample dataset. Any object exposing `page_content` and `metadata` works, so LangChain and LlamaIndex loaders are both fine -- here we build ragas `Document` objects directly, with no extra dependency.

```shell
# no extra dependency needed
```

```python
from pathlib import Path

from ragas.testset.document import Document

path = "Sample_Docs_Markdown/"
docs = [
    Document(page_content=p.read_text(), metadata={"source": str(p)})
    for p in Path(path).rglob("*.md")
]
```

### Choose your LLM

You may choose to use any [LLM of your choice](./../howtos/customizations/customize_models.md)
--8<--
choose_generator_llm.md
--8<--

### Generate Testset

Now we will run the test generation using the loaded documents and the LLM setup. If you have used `llama_index` to load documents, please use `generate_with_llama_index_docs` method instead.

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_docs(docs, testset_size=10)
```

`testset_size` is an upper bound. The samples are split across the query
synthesizers in proportion to their weights and add up to exactly
`testset_size`. If the knowledge graph can't supply that many scenarios, or
some samples fail with `raise_exceptions=False`, you get fewer samples and a
warning that says why.

### Analyzing the testset

Once you have generated a testset, you would want to view it and select the queries you see fit to include in your final testset. You can export the testset to a pandas DataFrame and do various analysis on it.

```python
dataset.to_pandas()
```

Output
![testset](./testset_output.png)

Each sample records where its reference contexts came from. The
`reference_contexts_metadata` column holds the metadata of the source
document for each reference context, in the same order, e.g.
`[{"source": "docs/intro.md"}]`. The column stays in place through
`to_evaluation_dataset()` and appears next to the scores in
`result.to_pandas()`, so you can group scores by source file. Dates and other
values that JSON can't hold are stored as strings.

## A Deeper Look

Now that we have a seen how to generate a testset, let's take a closer look at the main components of the testset generation pipeline and how you can quickly customize it.

At the core there are 2 main operations that are performed to generate a testset.

1. **KnowledgeGraph Creation**: We first create a [KnowledgeGraph][ragas.testset.graph.KnowledgeGraph] using the documents you provide and use various [Transformations][ragas.testset.transforms.base.BaseGraphTransformation] to enrich the knowledge graph with additional information that we can use to generate the testset. You can learn more about this from the [core concepts section](../concepts/test_data_generation/rag.md#knowledge-graph-creation).
2. **Testset Generation**: We use the [KnowledgeGraph][ragas.testset.graph.KnowledgeGraph] to generate a set of [scenarios][ragas.testset.synthesizers.base.BaseScenario]. These scenarios are used to generate the [testset][ragas.testset.synthesizers.generate.Testset]. You can learn more about this from the [core concepts section](../concepts/test_data_generation/rag.md#scenario-generation).

Now let's see an example of how these components work together to generate a testset.

### KnowledgeGraph Creation

Let's first create a [KnowledgeGraph][ragas.testset.graph.KnowledgeGraph] using the documents we loaded earlier.

```python
from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph()
```
Output
```
KnowledgeGraph(nodes: 0, relationships: 0)
```

and then add the documents to the knowledge graph.

```python
from ragas.testset.graph import Node, NodeType

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
    )
```
Output
```
KnowledgeGraph(nodes: 10, relationships: 0)
```

Now we will enrich the knowledge graph with additional information using [Transformations][ragas.testset.transforms.base.BaseGraphTransformation]. Here we will use [default_transforms][ragas.testset.transforms.default_transforms] to create a set of default transformations to apply with an LLM and Embedding Model of your choice.
But you can mix and match transforms or build your own as needed.

```python
from ragas.testset.transforms import default_transforms, apply_transforms


# define your LLM and Embedding Model
# here we are using the same LLM and Embedding Model that we used to generate the testset
transformer_llm = generator_llm
embedding_model = generator_embeddings

trans = default_transforms(
    documents=docs, llm=transformer_llm, embedding_model=embedding_model
)
apply_transforms(kg, trans)
```

`apply_transforms` changes `kg` in place and stops at the first failure. If a
step fails, it raises that error without running the remaining steps, and `kg`
keeps whatever the completed steps added. To retry, start again from a fresh
knowledge graph.

`default_transforms` picks its steps by document length:

- If at least a quarter of the documents are over 500 tokens, documents are
  split into chunks by their headlines.
- Otherwise, any document over 100 tokens is enough for the steps that work on
  whole documents.
- If every document is 100 tokens or shorter, it raises an error. If your
  input is already chunked, use `generator.generate_with_chunks(...)`, which
  skips splitting.

Now we have a knowledge graph with additional information. You can save the knowledge graph too.

```python
kg.save("knowledge_graph.json")
loaded_kg = KnowledgeGraph.load("knowledge_graph.json")
loaded_kg
```

Output
```
KnowledgeGraph(nodes: 48, relationships: 605)
```

### Testset Generation

Now we will use the `loaded_kg` to create the [TestsetGenerator][ragas.testset.synthesizers.generate.TestsetGenerator].

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm, embedding_model=embedding_model, knowledge_graph=loaded_kg
)
```

We can also define the distribution of queries we would like to generate. Here lets use the default distribution.

```python
from ragas.testset.synthesizers import default_query_distribution

query_distribution = default_query_distribution(generator_llm)
```

Output
```
[
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]
```

Now we can generate the testset.

```python
testset = generator.generate(testset_size=10, query_distribution=query_distribution)
testset.to_pandas()
```
Output
![testset](./testset_output.png)
