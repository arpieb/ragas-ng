# Tracing and logging evaluations

ragas emits [OpenTelemetry](https://opentelemetry.io) spans for experiment runs (`@experiment` and `metrics_experiment`), and for metrics scored on their own: one trace per run, with spans for each row, each metric (holding its score and reason) and each LLM call (holding the prompt, the response, the model and token counts). Any OpenTelemetry backend can collect them, including Phoenix, LangSmith, Langfuse, Opik, MLflow and Jaeger. ragas needs no backend-specific code.

Spans are no-ops until you configure an OpenTelemetry SDK. Install it with an OTLP exporter:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

Point the exporter at your backend before the first ragas call:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
trace.set_tracer_provider(provider)
```

Then score as usual. The trace appears in your backend:

```python
from datasets import load_dataset  # pip install datasets
from openai import OpenAI

from ragas import Dataset, metrics_experiment
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall

df = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")["eval"].to_pandas()
df = df.astype(object).where(df.notna(), None)  # NaN -> None
dataset = Dataset.from_pandas(df, "amnesty_qa", "inmemory", data_model=SingleTurnSample)

evaluator_llm = llm_factory("gpt-6-luna", client=OpenAI())
metrics_experiment([ContextRecall(llm=evaluator_llm)]).run(dataset, name="traced")
```

See [Observability](../../observability.md) for the span attributes and the LangSmith and Phoenix endpoints.

!!! note "Coming from `evaluate(callbacks=...)`"
    The `callbacks` argument, and LangChain tracers such as `LangChainTracer`, are no longer supported. ragas does not depend on LangChain. The OpenTelemetry setup above replaces them.
