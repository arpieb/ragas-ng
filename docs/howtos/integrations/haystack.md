# Haystack Integration

--8<--
earlier_run_outputs.md
--8<--

Haystack is an LLM orchestration framework to build customizable, production-ready LLM applications.

The underlying concept of Haystack is that all individual tasks, such as storing documents, retrieving relevant data, and generating responses, are handled by modular components like Document Stores, Retrievers, and Generators, which are seamlessly connected and orchestrated using Pipelines.

## Overview

In this tutorial, we will build a RAG pipeline using Haystack and evaluate it with Ragas. We’ll start by setting up the various components of the RAG pipeline and connect them to form the complete pipeline. Then we run it, collect the question, the retrieved documents and the answer, and score them with Ragas metrics. Later in the tutorial, we will explore how to evaluate with custom-defined metrics in Ragas.

The `ragas-haystack` package's `RagasEvaluator` component is built for upstream Ragas' metric classes, so this guide scores the pipeline's outputs with Ragas directly instead.

## Installing Dependencies


```python
%pip install ragas-ng haystack-ai
```

#### Getting the data


```python
dataset = [
    "OpenAI is one of the most recognized names in the large language model space, known for its GPT series of models. These models excel at generating human-like text and performing tasks like creative writing, answering questions, and summarizing content. GPT-4, their latest release, has set benchmarks in understanding context and delivering detailed responses.",
    "Anthropic is well-known for its Claude series of language models, designed with a strong focus on safety and ethical AI behavior. Claude is particularly praised for its ability to follow complex instructions and generate text that aligns closely with user intent.",
    "DeepMind, a division of Google, is recognized for its cutting-edge Gemini models, which are integrated into various Google products like Bard and Workspace tools. These models are renowned for their conversational abilities and their capacity to handle complex, multi-turn dialogues.",
    "Meta AI is best known for its LLaMA (Large Language Model Meta AI) series, which has been made open-source for researchers and developers. LLaMA models are praised for their ability to support innovation and experimentation due to their accessibility and strong performance.",
    "Meta AI with it's LLaMA models aims to democratize AI development by making high-quality models available for free, fostering collaboration across industries. Their open-source approach has been a game-changer for researchers without access to expensive resources.",
    "Microsoft’s Azure AI platform is famous for integrating OpenAI’s GPT models, enabling businesses to use these advanced models in a scalable and secure cloud environment. Azure AI powers applications like Copilot in Office 365, helping users draft emails, generate summaries, and more.",
    "Amazon’s Bedrock platform is recognized for providing access to various language models, including its own models and third-party ones like Anthropic’s Claude and AI21’s Jurassic. Bedrock is especially valued for its flexibility, allowing users to choose models based on their specific needs.",
    "Cohere is well-known for its language models tailored for business use, excelling in tasks like search, summarization, and customer support. Their models are recognized for being efficient, cost-effective, and easy to integrate into workflows.",
    "AI21 Labs is famous for its Jurassic series of language models, which are highly versatile and capable of handling tasks like content creation and code generation. The Jurassic models stand out for their natural language understanding and ability to generate detailed and coherent responses.",
    "In the rapidly advancing field of artificial intelligence, several companies have made significant contributions with their large language models. Notable players include OpenAI, known for its GPT Series (including GPT-4); Anthropic, which offers the Claude Series; Google DeepMind with its Gemini Models; Meta AI, recognized for its LLaMA Series; Microsoft Azure AI, which integrates OpenAI’s GPT Models; Amazon AWS (Bedrock), providing access to various models including Claude (Anthropic) and Jurassic (AI21 Labs); Cohere, which offers its own models tailored for business use; and AI21 Labs, known for its Jurassic Series. These companies are shaping the landscape of AI by providing powerful models with diverse capabilities.",
]
```

## Initialize components for RAG pipeline

#### Initializing the DocumentStore


```python
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()
docs = [Document(content=doc) for doc in dataset]
```

#### Initialize the Document and Text Embedder


```python
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder

document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
```

Now we have our document store and the document embedder, using them we will fill populate out vector datastore.


```python
docs_with_embeddings = document_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])
```

#### Initialize the Retriever


```python
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store, top_k=2)
```

#### Define a Template Prompt


```python
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)
```

#### Initialize a ChatGenerator


```python
from haystack.components.generators.chat import OpenAIChatGenerator

chat_generator = OpenAIChatGenerator(model="gpt-6-luna")
```

## Building and Assembling the Pipeline

#### Creating the Pipeline


```python
from haystack import Pipeline

rag_pipeline = Pipeline()
```

#### Adding the components


```python
from haystack.components.builders import AnswerBuilder

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())
```
#### Connecting the components

```python
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
```

## Running the Pipeline


```python
question = "What makes Meta AI’s LLaMA models stand out?"

reference = "Meta AI’s LLaMA models stand out for being open-source, supporting innovation and experimentation due to their accessibility and strong performance."


result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)

answer = result["answer_builder"]["answers"][0]
print(answer.data)
```
Output
```
Meta AI's LLaMA models stand out due to their open-source nature, which allows researchers and developers easy access to high-quality language models without the need for expensive resources. This accessibility fosters innovation and experimentation, enabling collaboration across various industries. Moreover, the strong performance of the LLaMA models further enhances their appeal, making them valuable tools for advancing AI development.
```

## Evaluating with Ragas

Ragas metrics score the pipeline's inputs and outputs directly. Each metric names what it needs:

- **AnswerRelevancy**: the **query** and the **response**.
- **ContextPrecision**: the **query**, the **retrieved documents**, and the **reference**.
- **Faithfulness**: the **query**, the **retrieved documents**, and the **response**.

The evaluator LLM and embeddings come from Ragas' own factories, not from Haystack: `llm_factory()` takes any [LiteLLM model string](../llm-adapters.md).

```python
from ragas import Dataset, metrics_experiment, summarize
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LiteLLMEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

evaluator_llm = llm_factory("gpt-6-luna")
evaluator_embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")

dataset = Dataset(
    "haystack_eval",
    "inmemory",
    data_model=SingleTurnSample,
    data=[
        SingleTurnSample(
            user_input=question,
            retrieved_contexts=[doc.content for doc in answer.documents],
            response=answer.data,
            reference=reference,
        )
    ],
)

run = metrics_experiment(
    [
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
    ]
)
exp = run.run(dataset, name="haystack")
print(summarize(exp))
```
Output
```
Running experiment: 100%|██████████| 1/1 [00:14<00:00, 14.16s/it]

{'answer_relevancy': {'mean': 0.9782, 'count': 1, 'missing': 0}, 'context_precision': {'mean': 1.0, 'count': 1, 'missing': 0}, 'faithfulness': {'mean': 1.0, 'count': 1, 'missing': 0}}
```

To score many questions, run the pipeline once per question and add one row per run to the dataset.

## Advance Usage

Instead of using the default ragas metrics, you can change them to fit your needs or even create your own custom metrics. To learn more about how to customize ragas metrics, check out the [docs](../customizations/index.md).

In the example below, we will define two custom Ragas metrics:

1. **sports_relevance**: evaluates whether a question and its response are related to sports.
2. **answer_quality**: measures how well the response answers the user's question, on a rubric.


```python
from ragas.metrics.collections import AspectCritic, DomainSpecificRubrics

sports_relevance = AspectCritic(
    name="sports_relevance_metric",
    definition="Were the question and response related to sports?",
    llm=evaluator_llm,
)

rubrics = {
    "score1_description": "The response does not answer the user input.",
    "score2_description": "The response partially answers the user input.",
    "score3_description": "The response fully answer the user input",
}
answer_quality = DomainSpecificRubrics(llm=evaluator_llm, rubrics=rubrics)

sample = {
    "user_input": "Which is the most popular global sport?",
    "retrieved_contexts": [
        "Football is undoubtedly the world's most popular sport with"
        " major events like the FIFA World Cup and sports personalities"
        " like Ronaldo and Messi, drawing a followership of more than 4"
        " billion people."
    ],
    "response": "Football is the most popular sport with around 4 billion"
    " followers worldwide",
}

for metric in (sports_relevance, answer_quality):
    result = await metric.ascore(**sample)
    print(metric.name, result.value)
```
Output
```
sports_relevance_metric 1.0
domain_specific_rubrics 3.0
```
