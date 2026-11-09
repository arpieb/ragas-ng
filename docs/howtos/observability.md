# Observability

ragas emits [OpenTelemetry](https://opentelemetry.io) spans, so any backend that ingests OTLP can show what a ragas run did: Phoenix, LangSmith, Langfuse, Opik, MLflow, Jaeger, Datadog and others. ragas has no backend-specific tracing code. You configure an OpenTelemetry SDK and exporter, and ragas spans go wherever it sends them.

ragas depends only on `opentelemetry-api`. Until you configure an SDK, its spans are no-ops and cost nothing.

## What ragas traces

Spans use the instrumentation scope `ragas` and follow the [OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) conventions, so backends that understand those show inputs, outputs, span kinds, models and token counts.

| Span | `openinference.span.kind` | `ragas.chain.type` | Emitted by |
|---|---|---|---|
| `ragas evaluation` | `CHAIN` | `evaluation` | `evaluate()` |
| `experiment <name>` | `CHAIN` | `experiment` | `@experiment` `arun()` |
| `row 0`, `row 1`, ... | `CHAIN` | `row` | each row of either |
| one per metric call, named after the metric (e.g. `faithfulness`) | `EVALUATOR` | `metric` | `evaluate()`, and any metric's `score()` / `ascore()` |
| one per legacy ragas prompt, named after it | `LLM` | `ragas_prompt` | legacy prompts (testset generation, metrics on the legacy implementation) |
| one per call to an `llm_factory()` LLM, named after its response model (e.g. `AspectCriticOutput`) | `LLM` | `llm` | `InstructorLLM.generate()` / `agenerate()` |

Spans nest the way the work does. An `evaluate()` run is `ragas evaluation` → `row N` → metric → LLM call. An experiment is `experiment <name>` → `row N`, with any metric scored inside your experiment function under its row. A metric scored on its own is a root span, or a child of your application's active span if there is one. Concurrent rows and calls each get their own branch.

Each span carries `input.value` and `output.value` as JSON and `ragas.run.id`:

- **metric spans:** the input is the metric's arguments, and the output is its score plus the reason, for example `{"output": 1.0, "reason": "..."}`;
- **LLM spans:** the input is the prompt, and the output is the validated response.

LLM spans also carry:

- `llm.model_name` and `llm.provider`;
- `llm.input_messages.*` and `llm.output_messages.*`;
- `llm.invocation_parameters`;
- token counts as `llm.token_count.prompt` / `completion` / `total`.

A cache hit makes no LLM call, so it has no LLM span.

### What leaves your process

LLM spans carry the judge's prompt, which holds your evaluation data: the question, the retrieved contexts, the response and any reference. By default, every text attribute (`input.value`, `output.value` and each message's content) is cut to its first 4,096 characters, ending in `...[truncated]`. That keeps spans under backend size limits, but what's left is still your data: if it can't go to your tracing service at all, don't configure an exporter for these runs.

To send prompts and outputs in full, opt in:

```bash
export RAGAS_TRACE_FULL_CONTENT=true
```

It is read on each span, so setting `os.environ["RAGAS_TRACE_FULL_CONTENT"] = "true"` in a running notebook works too. `result.traces` stays in your process and is never truncated.

LLMs that don't come from `llm_factory()` (the Haystack, OCI and LlamaIndex wrappers) emit no LLM spans of their own; their prompts still have prompt spans.

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

`EvaluationResult.traces` holds the same run tree in process, with no OpenTelemetry setup: one entry per row, keyed by metric name. Each metric's entry holds the input and output of every LLM call or ragas prompt it ran, and `.scores` holds the row's scores:

```python
result = evaluate(dataset, metrics=[...], llm=evaluator_llm)

result.traces[0].scores  # {'answers': 1.0}
dict(result.traces[0]["answers"])
# {'AspectCriticOutput': {'input': {'prompt': '...'},
#                         'output': AspectCriticOutput(reason='...', verdict=1)}}
```

A metric that makes the same call several times gets numbered keys: `name`, `name (2)`, and so on.

## Coming from LangChain callbacks

`evaluate(callbacks=...)` has been removed, along with the LangSmith and Opik tracers that hooked into it. Their replacement is the OpenTelemetry setup above. See [Migrating off LangChain](migrations/migrate_off_langchain.md#tracing).
