# From `evaluate()` to experiments

`evaluate()`, `aevaluate()`, `EvaluationResult` ([#257](https://github.com/arpieb/ragas-ng/issues/257)) and `EvaluationDataset` ([#258](https://github.com/arpieb/ragas-ng/issues/258)) are removed, and Hugging Face `datasets` is no longer a dependency: `pip install datasets` to load Hub data, as below. Scoring a dataset is now an [experiment](../../concepts/experimentation.md):

- the rows live in a `Dataset` of `SingleTurnSample` / `MultiTurnSample` objects;
- `metrics_experiment(metrics)` scores them, as `evaluate()` did;
- the result is an `Experiment`, saved through a [backend](../customizations/storage_backends.md), with a `<metric>` and a `<metric>_reason` column per metric;
- `summarize()` and `check()` give the per-metric means and the CI gate.

The field mapping, the up-front column checks and the scores are the same as `evaluate()`'s. This page has a before/after for each removed entry point.

## The basic call

Before:

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics.collections import Faithfulness, FactualCorrectness

dataset = EvaluationDataset.from_list(rows)
result = evaluate(dataset, metrics=[Faithfulness(llm=llm), FactualCorrectness(llm=llm)])
print(result)  # {'faithfulness': 0.86, 'factual_correctness': 0.71}
```

After:

```python
from ragas import Dataset, metrics_experiment, summarize
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import Faithfulness, FactualCorrectness

dataset = Dataset(
    "rag_runs",
    "inmemory",
    data_model=SingleTurnSample,
    data=[SingleTurnSample(**row) for row in rows],
)
run = metrics_experiment([Faithfulness(llm=llm), FactualCorrectness(llm=llm)])
exp = run.run(dataset, name="baseline")  # await run.arun(...) in async code
summarize(exp)
# {'faithfulness': {'mean': 0.86, 'count': 50, 'missing': 0}, ...}
```

The metrics object is built once and can run on any number of datasets. Each run is saved under its `name` (generated if you leave it out).

## `evaluate()` / `aevaluate()` arguments

| `evaluate(...)` | With `metrics_experiment` |
|---|---|
| `aevaluate(dataset, metrics)` | `await metrics_experiment(metrics).arun(dataset)` |
| `evaluate(dataset, metrics)` | `metrics_experiment(metrics).run(dataset)` |
| `metrics=None` (the four default metrics) | No default: list the metrics, e.g. `[AnswerRelevancy(llm=llm, embeddings=emb), ContextPrecision(llm=llm), Faithfulness(llm=llm), ContextRecall(llm=llm)]` |
| `llm=`, `embeddings=` | Collections metrics take their models at construction. `metrics_experiment(metrics, llm=..., embeddings=...)` is only for metrics that take them per call, such as `DiscreteMetric`. There are no default OpenAI models. |
| `column_map={"user_input": "question"}` | `metrics_experiment(metrics, column_map={"user_input": "question"})`: the same `{sample_field: dataset_column}` direction |
| `run_config=RunConfig(...)` | `metrics_experiment(metrics, run_config=RunConfig(...))`. Its `timeout` bounds each metric call. |
| `token_usage_parser=parser` | `usage = TokenUsageCollector(parser)`, then `metrics_experiment(metrics, usage=usage)`; read `usage.total_tokens()` / `usage.total_cost(...)` after the run |
| `raise_exceptions=True` | `metrics_experiment(metrics, raise_exceptions=True)`: the first failing metric stops the run. Add `save_partial=True` to keep the rows that finished. |
| `raise_exceptions=False` (default) | The default. A metric that raises scores NaN on that row, the error is logged, and the row is kept. |
| `batch_size=n` | `max_workers=n`: how many rows run at once (default 16). A row's metrics run concurrently. |
| `show_progress=False` | Not configurable: the experiment shows one progress bar per run. |
| `experiment_name="x"` | `run.run(dataset, name="x")` |
| `return_executor=True`, then `executor.cancel()` | Run `arun` as a task and cancel it: `task = asyncio.create_task(run.arun(dataset))`, then `task.cancel()`. Running rows are cancelled and nothing is saved. See [Cancelling a run](#cancelling-a-run). |
| `callbacks=` (removed earlier) | OpenTelemetry spans: see [Observability](../observability.md) |

### Cancelling a run

Before:

```python
executor = evaluate(dataset, metrics, return_executor=True)
# ... from another thread
executor.cancel()
```

After:

```python
import asyncio

task = asyncio.create_task(run.arun(dataset, name="long_run"))
await asyncio.sleep(60)
task.cancel()  # rows still running are cancelled; nothing is saved
```

To bound a run instead, set `run_config=RunConfig(timeout=...)`: each metric call that takes longer raises `TimeoutError` and scores NaN.

## `EvaluationResult` → `Experiment`

`exp` below is the `Experiment` a run returns. It's a list of row dicts, also saved through its backend.

| `EvaluationResult` | `Experiment` |
|---|---|
| `print(result)` (the means) | `summarize(exp)`: `{"metric": {"mean", "count", "missing"}}` |
| `result.summary()` | `summarize(exp)` |
| `result.check(min_mean=..., max_mean=..., allow_missing=...)` | `check(exp, min_mean=..., max_mean=..., allow_missing=...)`: the same `ThresholdError` |
| `result["faithfulness"]` | `[row["faithfulness"] for row in exp]` |
| `result.scores` | `[{m: row[m] for m in names} for row in exp]` |
| `result.reasons["faithfulness"]` | `[row["faithfulness_reason"] for row in exp]` |
| `result.to_pandas()` | `exp.to_pandas()`: the dataset columns plus each `<metric>` and `<metric>_reason` |
| `result.to_pandas(include_reasons=False)` | `exp.to_pandas().drop(columns=[f"{m}_reason" for m in names])` |
| `result.total_tokens()` / `result.total_cost(...)` | `usage.total_tokens()` / `usage.total_cost(...)` on the `TokenUsageCollector` you passed as `usage=` |
| `result.traces` / `result.ragas_traces` | OpenTelemetry spans: one per experiment, row, metric and LLM call. See [Observability](../observability.md). |
| `result.dataset` | The `Dataset` you ran on; the experiment rows also hold its columns |

`summarize()` and `check()` also work on an experiment loaded later:

```python
from ragas import Experiment, check

exp = Experiment.load("nightly", "local/jsonl", root_dir="./data")
check(exp, min_mean={"faithfulness": 0.8})
```

## `EvaluationDataset` → `Dataset`

| `EvaluationDataset` | `Dataset` |
|---|---|
| `EvaluationDataset(samples=samples)` | `Dataset("name", "inmemory", data_model=SingleTurnSample, data=samples)` |
| `EvaluationDataset.from_list(rows)` | `Dataset("name", "inmemory", data_model=SingleTurnSample, data=[SingleTurnSample(**r) for r in rows])` (`MultiTurnSample` for conversations) |
| `EvaluationDataset.from_dict({"user_input": [...], ...})` | `pd.DataFrame(columns)`, then `from_pandas` below |
| `EvaluationDataset.from_pandas(df)` | `Dataset.from_pandas(df, "name", "inmemory", data_model=SingleTurnSample)` |
| `dataset.to_pandas()` | `dataset.to_pandas()` |
| `EvaluationDataset.from_hf_dataset(hf)` | `Dataset.from_pandas(hf.to_pandas(), "name", "inmemory", data_model=SingleTurnSample)` |
| `dataset.to_hf_dataset()` | `datasets.Dataset.from_pandas(dataset.to_pandas())` (install `datasets` yourself) |
| `dataset.to_csv(path)` / `to_jsonl(path)` | Save through a backend: `Dataset("name", "local/jsonl", root_dir="./data", ...)`, then `dataset.save()` |
| `EvaluationDataset.from_jsonl(path)` | `Dataset.load("name", "local/jsonl", root_dir="./data", data_model=SingleTurnSample)` |
| `dataset.to_list()` | `[s.model_dump() for s in dataset]` |
| `dataset[0]`, `len(dataset)`, iteration | The same |
| `dataset[:10]` (an `EvaluationDataset`) | `dataset[:10]` is a list of samples; wrap it in a new `Dataset` to run on it |
| `dataset.features()` | `sorted({f for s in dataset for f in s.get_features()})` |
| `dataset.is_multi_turn()` | `dataset.data_model is MultiTurnSample` |

Rows may also be plain dicts, or a `Dataset` without a `data_model`: `metrics_experiment` turns each dict into a sample, multi-turn when `user_input` is a list of messages. All rows must be one sample type.

!!! warning "Prefer JSONL to CSV for datasets you'll score"
    CSV loads a missing text value as `""`, and `metrics_experiment` scores `""` as a value. A row whose `reference` was missing is scored against an empty reference instead of raising. See [What each backend keeps](../customizations/storage_backends.md#what-each-backend-keeps).

### From a Hugging Face dataset

`datasets` isn't installed with ragas any more; `pip install datasets` first.

`from_pandas` builds each sample from a DataFrame row, so missing values must be `None`, not NaN:

```python
from datasets import load_dataset  # pip install datasets

from ragas import Dataset
from ragas.dataset_schema import SingleTurnSample

hf = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")["eval"]
df = hf.to_pandas()
df = df.astype(object).where(df.notna(), None)  # NaN -> None
dataset = Dataset.from_pandas(df, "amnesty_qa", "inmemory", data_model=SingleTurnSample)
```

Old column names (`question`, `answer`, `contexts`, `ground_truth`) were renamed by `evaluate()`. Rename them yourself, or pass `column_map={"user_input": "question", "response": "answer", "retrieved_contexts": "contexts", "reference": "ground_truth"}` and build the `Dataset` without a `data_model`.

## Testset and integrations

| Before | After |
|---|---|
| `testset.to_evaluation_dataset()` | `testset.to_dataset("name")` (in memory; `testset.to_dataset("name", "local/jsonl", root_dir="./data")` to save) |
| `testset.to_hf_dataset()` / `Testset.from_hf_dataset(hf)` | `testset.to_pandas()`, `testset.to_list()` / `Testset.from_list(rows)` |
| `ragas.integrations.llama_index.evaluate(engine, dataset, metrics)` | `evaluate_query_engine(engine, dataset, metrics)`; returns the `Experiment` |
| griptape / r2r `transform_to_ragas_dataset(...)` → `EvaluationDataset` | The same call returns a `Dataset`; pass it to `metrics_experiment(metrics).run(...)` |
| `ragas.integrations.amazon_bedrock` traces → `evaluate(...)` | Build the samples with `convert_to_ragas_messages()` / `extract_kb_trace()` as before, put them in a `Dataset`, then `metrics_experiment` |

## Prompt optimization

`metric.train()` and the optimizers never needed `evaluate()`. Their annotations moved to `ragas.optimizers.annotations` ([#254](https://github.com/arpieb/ragas-ng/issues/254)).

## In CI

Before:

```python
result = evaluate(dataset, metrics)
result.check(min_mean={"faithfulness": 0.8}, allow_missing=1)
```

After:

```python
exp = metrics_experiment(metrics).run(dataset, name="ci")
check(exp, min_mean={"faithfulness": 0.8}, allow_missing=1)
```

## Your own experiment instead

When the rows need work before scoring, such as calling your app, write the experiment yourself and score inside it. See [Experiments](../../concepts/experimentation.md):

```python
from ragas import experiment


@experiment()
async def run(row):
    response = await my_app(row["user_input"])
    score = await faithfulness.ascore(
        user_input=row["user_input"],
        response=response,
        retrieved_contexts=row["retrieved_contexts"],
    )
    return {**row, "response": response, "faithfulness": score.value}
```
