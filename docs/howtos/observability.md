# Observability

ragas emits [OpenTelemetry](https://opentelemetry.io) spans, so any backend that ingests OTLP can show what a ragas run did: Phoenix, LangSmith, Langfuse, Opik, MLflow, Jaeger, Datadog and others. ragas has no backend-specific tracing code. You configure an OpenTelemetry SDK and exporter, and ragas spans go wherever it sends them.

ragas depends only on `opentelemetry-api`. Until you configure an SDK, its spans are no-ops and cost nothing.

## What ragas traces

Spans use the instrumentation scope `ragas` and follow the [OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) conventions, so backends that understand those show inputs, outputs and span kinds. An `evaluate()` run produces this tree:

| Span | `openinference.span.kind` | `ragas.chain.type` |
|---|---|---|
| `ragas evaluation` | `CHAIN` | `evaluation` |
| `row 0`, `row 1`, ... | `CHAIN` | `row` |
| one per metric, named after it (e.g. `faithfulness`) | `EVALUATOR` | `metric` |

Each span carries `input.value` and `output.value` as JSON (cut to 4,096 characters) and `ragas.run.id`. A metric span's input is the row's fields and its output is the score, for example `{"output": 1.0}`. Testset generation is traced too.

!!! note "What is not traced yet"
    - **The LLM calls inside a metric.** With an `llm_factory()` LLM, metrics score through their collections implementation, which emits no spans of its own. A metric span therefore has no children. (Metrics that fall back to the legacy implementation, for example with customised prompts, also emit one `LLM` span per ragas prompt.)
    - **Collections metrics called directly** with `score()` / `ascore()`, and **`@experiment` runs**: no spans.

    To see the LLM calls themselves, trace the LLM client, for example with an OpenInference instrumentor for OpenAI or LiteLLM.

## Send spans to a backend

Install the SDK and an OTLP exporter:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

Then set a global tracer provider before your first ragas call. Only the endpoint and headers depend on the backend:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

exporter = OTLPSpanExporter(
    endpoint="http://localhost:6006/v1/traces",  # your backend's OTLP/HTTP traces endpoint
    headers={},  # auth headers, if the backend needs them
)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# every evaluate() call from here on is exported
```

In short scripts, call `provider.force_flush()` before exiting so the last batch is sent.

### Phoenix (Arize)

Phoenix accepts OTLP at `http://localhost:6006/v1/traces` when you run it locally (`phoenix serve`). The [Phoenix guide](integrations/_arize.md) shows the setup with `phoenix.otel`, and how to write ragas scores back onto your app's spans.

### LangSmith

LangSmith ingests OTLP at `https://api.smith.langchain.com/otel/v1/traces` (EU: `https://eu.api.smith.langchain.com/otel/v1/traces`). It needs the API key in an `x-api-key` header, and the project in a `Langsmith-Project` header:

```python
import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="https://api.smith.langchain.com/otel/v1/traces",
    headers={
        "x-api-key": os.environ["LANGSMITH_API_KEY"],
        "Langsmith-Project": "ragas-evals",
    },
)
```

Use this exporter in the setup above. LangSmith maps OpenInference attributes, so ragas spans show as runs with their inputs, outputs and scores. See LangSmith's [OpenTelemetry guide](https://docs.langchain.com/langsmith/trace-with-opentelemetry) for agent-based workspaces and self-hosted endpoints.

### Langfuse

See the [Langfuse guide](integrations/_langfuse.md), which also covers scoring Langfuse traces with ragas metrics.

## Without a backend: `result.traces`

`EvaluationResult.traces` holds the same run tree in process, with no OpenTelemetry setup: one entry per row, with each metric's score. For metrics on the legacy implementation it also holds the input and output of every ragas prompt the metric ran. Metrics scored through collections, the default with `llm_factory()`, record their score only.

```python
result = evaluate(dataset, metrics=[...], llm=evaluator_llm)

result.traces[0]  # the first row, e.g. {'harmless': 1.0, 'faithfulness': 1.0}
```

## Coming from LangChain callbacks

`evaluate(callbacks=...)` has been removed, along with the LangSmith and Opik tracers that hooked into it. Their replacement is the OpenTelemetry setup above. See [Migrating off LangChain](migrations/migrate_off_langchain.md#tracing).
