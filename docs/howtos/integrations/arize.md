# Phoenix (Arize)

[Phoenix](https://github.com/Arize-ai/phoenix) is Arize's open-source platform for LLM tracing and evaluation. Ragas and Phoenix work together in two ways:

1. **See ragas runs in Phoenix.** ragas emits [OpenTelemetry](https://opentelemetry.io) spans with OpenInference conventions, which Phoenix ingests directly. ragas needs no Phoenix-specific code for this.
2. **Score your app's traces with ragas.** Fetch spans from Phoenix, score them with ragas metrics, and write the scores back as span annotations.

This guide was checked against Phoenix 21.0, `arize-phoenix-otel` 0.17 and `arize-phoenix-client` 3.5. Install what you need:

```bash
pip install ragas arize-phoenix-otel arize-phoenix-client openai
# to run Phoenix locally (UI and collector on http://localhost:6006):
pip install arize-phoenix
phoenix serve
```

## 1. See ragas runs in Phoenix

`phoenix.otel.register()` creates an OpenTelemetry tracer provider that exports to Phoenix, and makes it the global default, which is where ragas sends its spans. For Phoenix Cloud, use your space's endpoint and set `PHOENIX_API_KEY`.

```python
from phoenix.otel import register

tracer_provider = register(
    project_name="ragas-demo",
    endpoint="http://localhost:6006/v1/traces",
)
```

Then run an evaluation as usual:

```python
from openai import OpenAI

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import AspectCritic

evaluator_llm = llm_factory("gpt-6-luna", client=OpenAI())

dataset = EvaluationDataset.from_list(
    [
        {
            "user_input": "What is the capital of France?",
            "response": "Paris is the capital of France.",
        }
    ]
)
critic = AspectCritic(
    name="harmless", definition="Is the response harmless?", llm=evaluator_llm
)

evaluate(dataset, metrics=[critic])
```

The `ragas-demo` project in Phoenix now shows a `ragas evaluation` trace, with one `row` span per sample and one `EVALUATOR` span per metric holding its score. See [Observability](../observability.md#what-ragas-traces) for what each span contains.

## 2. Score your app's traces with ragas

If your app is traced into Phoenix, you can score its spans with ragas and attach the scores to them. The example below uses spans whose `input.value` is the question and whose `output.value` is the answer. That is what OpenInference instrumentors and Phoenix's own decorators record for a top-level chain. Adjust the column names to match your spans.

Fetch the spans to score:

```python
from phoenix.client import Client

px = Client(base_url="http://localhost:6006")

spans = px.spans.get_spans_dataframe(project_identifier="my-app")
app_spans = spans[spans["name"] == "answer"]  # your app's top-level span
```

Score each one with a [collections metric](../../concepts/metrics/available_metrics/aspect_critic.md):

```python
import asyncio

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import AspectCritic

critic = AspectCritic(
    name="answers_question",
    definition="Does the response answer the question?",
    llm=llm_factory("gpt-6-luna", client=AsyncOpenAI()),
)


async def score_spans(df):
    return await asyncio.gather(
        *(
            critic.ascore(
                user_input=row["attributes.input.value"],
                response=row["attributes.output.value"],
            )
            for _, row in df.iterrows()
        )
    )


results = asyncio.run(
    score_spans(app_spans)
)  # in a notebook: await score_spans(app_spans)
```

Write the scores back as annotations. The dataframe's index must be the span IDs, which `get_spans_dataframe()` already uses:

```python
import pandas as pd

annotations = pd.DataFrame(
    {
        "score": [r.value for r in results],
        "explanation": [r.reason for r in results],
    },
    index=app_spans.index,
)

px.spans.log_span_annotations_dataframe(
    dataframe=annotations,
    annotation_name="answers_question",
    annotator_kind="LLM",
)
```

Phoenix stores annotations asynchronously, so they can take a moment to appear. Each scored span then shows an `answers_question` annotation with the score and the judge's reason. Read them back with `px.spans.get_span_annotations_dataframe(spans_dataframe=app_spans, project_identifier="my-app")`.
