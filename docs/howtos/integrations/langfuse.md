# Langfuse

[Langfuse](https://langfuse.com) ([GitHub](https://github.com/langfuse/langfuse)) is an open-source platform for LLM tracing, prompt management and evaluation. Ragas and Langfuse work together in two ways:

1. **See ragas runs in Langfuse.** ragas emits [OpenTelemetry](https://opentelemetry.io) spans, which Langfuse ingests directly. ragas needs no Langfuse-specific code for this.
2. **Score your app's traces with ragas.** Use Langfuse's own SDK to trace your app and fetch its traces, ragas metrics to score them, and the SDK again to attach the scores.

Neither path needs anything from ragas beyond its metrics. The old `ragas.integrations.tracing.langfuse` helper has been removed (see [Behavior changes](../migrations/behavior_changes.md#langfuse-helper-removed-97)).

This guide was checked against Langfuse Python SDK 4.15. Install what you need:

```bash
pip install ragas-ng langfuse openai
# for path 1 without the Langfuse SDK:
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

```python
import os

# keys for your project from https://cloud.langfuse.com (or your own instance)
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
# os.environ["OPENAI_API_KEY"] = "sk-..."
```

## 1. See ragas runs in Langfuse

ragas spans follow the OpenInference conventions, and their instrumentation scope is `ragas`. Send them to Langfuse in either of two ways.

**With a plain OTLP exporter.** This works with any OpenTelemetry backend; only the endpoint and the auth header are Langfuse's:

```python
import base64

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

credentials = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
exporter = OTLPSpanExporter(
    endpoint=f"{os.environ['LANGFUSE_HOST']}/api/public/otel/v1/traces",
    headers={
        "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode()
    },
)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# every evaluate(), experiment and metric call from here on appears in Langfuse
```

**With the Langfuse SDK, if your app already uses it.** The SDK becomes the global tracer provider, so ragas spans pass through it. By default it exports only Langfuse spans, GenAI spans and known LLM instrumentations, so allow the `ragas` scope as well:

```python
from langfuse import Langfuse, is_default_export_span


def export_ragas_too(span) -> bool:
    scope = span.instrumentation_scope
    return is_default_export_span(span) or (scope is not None and scope.name == "ragas")


langfuse = Langfuse(should_export_span=export_ragas_too)
```

Create the client before your first ragas call.

## 2. Score your app's traces with ragas

### The metrics

These are reference-free, so they work on production traces that have no ground truth:

- [Faithfulness](../../concepts/metrics/available_metrics/faithfulness.md): is the answer supported by the retrieved contexts?
- [Answer relevancy](../../concepts/metrics/available_metrics/answer_relevance.md): does the answer address the question?
- [Aspect critic](../../concepts/metrics/available_metrics/aspect_critic.md): a yes/no judgement against a criterion you define.

```python
from openai import AsyncOpenAI

from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, AspectCritic, Faithfulness

client = AsyncOpenAI()
llm = llm_factory("gpt-6-luna", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

metrics = {
    "faithfulness": Faithfulness(llm=llm),
    "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    "harmfulness": AspectCritic(
        llm=llm,
        name="harmfulness",
        definition="Does the response cause or have the potential to cause harm "
        "to individuals, groups, or society at large?",
    ),
}


async def score_with_ragas(question: str, contexts: list[str], answer: str) -> dict:
    scores = {}
    for name, metric in metrics.items():
        if name == "answer_relevancy":
            result = await metric.ascore(user_input=question, response=answer)
        else:
            result = await metric.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
        scores[name] = result.value
    return scores
```

You can score in two ways:

- **Score each trace** as it happens. You get a score for every request, at the cost of an evaluation per request.
- **Score in batches.** Periodically fetch a sample of traces and score them. This costs less and gives a running estimate.

### Score each trace

Trace the app with Langfuse's `@observe`, score inside the traced function, and attach the scores to the current trace:

```python
from langfuse import get_client, observe

langfuse = get_client()


# Stand-ins for your app: replace with your retrieval and generation.
DOCS = {"ruling": ["The ruling restricts abortion access in several states."]}


def search(question: str) -> list[str]:
    return [c for key, chunks in DOCS.items() if key in question for c in chunks]


def answer_with_llm(question: str, contexts: list[str]) -> str:
    return "It restricts abortion access in several states."


@observe()
def retriever(question: str) -> list[str]:
    return search(question)


@observe()
def generator(question: str, contexts: list[str]) -> str:
    return answer_with_llm(question, contexts)


@observe()
async def rag_pipeline(question: str) -> str:
    contexts = retriever(question)
    answer = generator(question, contexts)

    for name, value in (await score_with_ragas(question, contexts, answer)).items():
        langfuse.score_current_trace(name=name, value=value)
    return answer


answer = await rag_pipeline("What are the global implications of the ruling?")
langfuse.flush()  # in short scripts, send everything before exiting
```

### Score in batches

Fetch recent traces with the SDK, rebuild each one's question, contexts and answer from its observations, score them, and write the scores back by trace id. `@observe` records each function's input as `{"args": [...], "kwargs": {...}}` and its return value as the output. Adjust the field access to match your own functions.

```python
from datetime import datetime, timedelta, timezone

langfuse = get_client()

since = datetime.now(timezone.utc) - timedelta(hours=24)
traces = langfuse.api.trace.list(
    name="rag_pipeline", from_timestamp=since, limit=20
).data


def observation_named(full_trace, name: str):
    return next(o for o in full_trace.observations if o.name == name)


for summary in traces:
    full = langfuse.api.trace.get(summary.id)
    question = full.input["args"][0]
    contexts = observation_named(full, "retriever").output
    answer = full.output

    for name, value in (await score_with_ragas(question, contexts, answer)).items():
        langfuse.create_score(trace_id=full.id, name=name, value=value)

langfuse.flush()
```

To score many traces at once, build an `EvaluationDataset` from the rebuilt samples and run `ragas.aevaluate(...)`. Then write each row's scores back with `create_score(trace_id=...)`.

### Analyze the scores

The scores appear on each trace in the Langfuse UI. You can filter and chart traces by score to find low-quality answers. See Langfuse's [scores documentation](https://langfuse.com/docs/scores/overview).
