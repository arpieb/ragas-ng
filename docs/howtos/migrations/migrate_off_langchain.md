# Migrating off LangChain

ragas no longer depends on LangChain. `import ragas` loads zero langchain
modules, and no langchain package is installed as a dependency.

This page covers every breaking change, what to replace it with, and the one
change that fails **silently**.

## Read this first

!!! danger "Environment-driven tracing stops silently"
    Setting `LANGCHAIN_TRACING_V2` used to make ragas runs appear in LangSmith
    automatically, because LangChain's callback manager read that variable.
    Nothing reads it now.

    Every other change on this page raises an error. This one does not — your
    evaluations keep working and simply stop being traced. If you rely on
    ambient tracing, see [Tracing](#tracing) below.

## Why

Four reasons, in the order they bite:

1. **Upstream instability.** langchain 1.x removed a module ragas imported at
   module scope, which broke `import ragas` outright for anyone installing
   fresh. ragas was pinned to a dying major version to avoid it.
2. **A vendor client was mandatory.** `langchain-core` depends on `langsmith`
   unconditionally, so every ragas install shipped a commercial SaaS telemetry
   client and a second HTTP stack.
3. **Provider lock-in.** The documented path ran through LangChain's OpenAI
   integration.
4. **Observability was hostage to one framework.** Tracing required LangChain's
   callback system.

## LLMs

`LangchainLLMWrapper` is removed. Use `llm_factory`.

=== "Before"
    ```python
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-6-luna"))
    ```

=== "After"
    ```python
    from openai import OpenAI
    from ragas.llms import llm_factory

    evaluator_llm = llm_factory("gpt-6-luna", client=OpenAI())
    ```

Passing a client is optional. Omit it and `llm_factory` routes through LiteLLM,
resolving credentials from the environment — the same way `embedding_factory()`
does:

```python
from ragas.llms import llm_factory

evaluator_llm = llm_factory("ollama/llama3")  # LiteLLM
evaluator_llm = llm_factory("claude-sonnet-5-5", provider="anthropic")
```

That covers 100+ providers, including Bedrock, Vertex AI, Groq, Mistral and
local models. See [choosing an evaluator LLM](../../extra/components/choose_evaluator_llm.md).

## Embeddings

`LangchainEmbeddingsWrapper` is removed. Use `embedding_factory`.

=== "Before"
    ```python
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    ```

=== "After"
    ```python
    from openai import OpenAI
    from ragas.embeddings import embedding_factory

    embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=OpenAI()
    )
    ```

Unlike `llm_factory`, `embedding_factory()` **works with no arguments** — it
routes to LiteLLM and resolves credentials from the environment:

```python
embeddings = embedding_factory()  # LiteLLM, provider-neutral
```

`embedding_factory(interface="legacy")` now raises; the legacy interface was the
LangChain path.

### Custom embedding subclasses

`BaseRagasEmbeddings` no longer subclasses `langchain_core.embeddings.Embeddings`.
Implement **either** pair — the other is bridged automatically:

```python
from ragas.embeddings import BaseRagasEmbeddings


class MyEmbeddings(BaseRagasEmbeddings):
    def embed_query(self, text): ...
    def embed_documents(self, texts): ...

    # aembed_query / aembed_documents are provided, via an executor
```

Overriding the higher-level `embed_text`/`embed_texts` is also accepted. A
subclass implementing none of these now raises `TypeError` at class-definition
time rather than recursing at call time.

## Tracing

`evaluate(callbacks=...)` is **removed**. It accepted LangChain callback
handlers, so there was no faithful translation.

ragas now emits **OpenTelemetry** spans, using OpenInference semantic
conventions. Langfuse, Phoenix/Arize, Opik, MLflow, Jaeger and Datadog all
ingest these with no ragas-specific integration code.

ragas depends on `opentelemetry-api` only, so spans are a no-op until you
configure an SDK:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

# ragas spans now flow to your collector
```

`EvaluationResult.traces` is **unchanged** and still populated without any OTel
setup — it comes from an in-process run tree, not from spans.

### Removed integrations

`ragas.integrations.langchain` (`EvaluatorChain`), `ragas.integrations.langsmith`,
`ragas.integrations.langgraph` and `ragas.integrations.opik` are removed. Opik
went too, because its tracer subclassed LangChain's.

For Opik, Langfuse and Phoenix, use their **OpenTelemetry** ingestion instead —
which no longer requires anything ragas-specific. The
[Langfuse guide](../integrations/langfuse.md) shows both the plain OTLP
exporter and the Langfuse SDK set-up (which needs ragas spans allowed through
its export filter).

## Cost and token accounting

`TokenUsageParser` now receives the provider's **raw completion object** rather
than a LangChain `LLMResult`/`ChatResult`. `CostCallbackHandler` is renamed
`TokenUsageCollector`; the old name is removed.

=== "Before"
    ```python
    def my_parser(llm_result):
        return TokenUsage(
            input_tokens=llm_result.llm_output["token_usage"]["prompt_tokens"],
            output_tokens=llm_result.llm_output["token_usage"]["completion_tokens"],
        )
    ```

=== "After"
    ```python
    def my_parser(completion):
        return TokenUsage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
            model=completion.model,
        )
    ```

The built-in parsers — `get_token_usage_for_openai`, `..._anthropic`,
`..._bedrock`, `..._azure_ai` — are updated and need no change on your side.

!!! note "This feature was broken before"
    `token_usage_parser` was driven by an `on_llm_end` callback that only
    LangChain LLMs ever fired. Once the LangChain adapters were removed it did
    nothing at all. It now works again.

## Testset generation

`TestsetGenerator.from_langchain()` is removed — construct the generator
directly:

```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
```

`generate_with_langchain_docs()` is renamed **`generate_with_docs()`**. The old
name is removed.

**Your documents do not need to change.** The parameter is typed as a structural
protocol requiring only `page_content` and `metadata`, which LangChain
`Document` objects satisfy — so LangChain loaders keep working:

```python
from langchain_community.document_loaders import DirectoryLoader

docs = DirectoryLoader("./docs", glob="**/*.md").load()
testset = generator.generate_with_docs(docs, testset_size=10)
```

ragas also ships a plain `Document` if you would rather not install LangChain:

```python
from ragas.testset.document import Document

docs = [Document(page_content=text, metadata={"source": path})]
```

## Prompt and LLM result types

A custom `BaseRagasLLM` and the result and prompt types it used (`ragas.llms.output`, `ragas.prompt.value`) were removed later, with the rest of the legacy layer ([#215](https://github.com/arpieb/ragas-ng/issues/215)). To use a model ragas doesn't route to, give `llm_factory()` a LiteLLM model string or a `litellm.Router`; chat messages are `ragas.messages.Message`, which carries an explicit `role` instead of encoding it in the class name the way `HumanMessage`/`AIMessage`/`SystemMessage` did.

## Summary

| Removed | Replacement |
|---|---|
| `LangchainLLMWrapper` | `llm_factory(model, client=...)` |
| `LangchainEmbeddingsWrapper` | `embedding_factory(...)` |
| `evaluate(callbacks=...)` | OpenTelemetry |
| `ragas.integrations.{langchain,langsmith,langgraph,opik}` | OTel ingestion |
| `TestsetGenerator.from_langchain()` | `TestsetGenerator(...)` |
| `generate_with_langchain_docs()` | `generate_with_docs()` |
| `CostCallbackHandler` | `TokenUsageCollector` |
| `LANGCHAIN_TRACING_V2` auto-tracing | configure an OTel SDK |

!!! note "`openai` is no longer a declared dependency"
    ragas used to list `openai` in its own dependencies, and hard-coded an
    `openai.OpenAI()` client as the default whenever you did not pass an LLM.
    Both are gone — the default now routes through LiteLLM, which resolves
    credentials from the environment.

    In practice `openai` is still installed, because LiteLLM and instructor both
    depend on it. But if your own code does `import openai`, declare it yourself
    rather than relying on ragas to supply it.
