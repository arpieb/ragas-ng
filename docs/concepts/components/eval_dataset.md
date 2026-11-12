# Evaluation Dataset

An evaluation dataset is a homogeneous collection of [data samples](eval_sample.md) designed to assess the performance and capabilities of an AI application. In Ragas, it's a [`Dataset`](../../howtos/customizations/storage_backends.md) whose rows are samples: `Dataset(..., data_model=SingleTurnSample)` or `data_model=MultiTurnSample`. You score it with [`metrics_experiment`](../experimentation.md#scoring-a-dataset-in-one-call-metrics_experiment).

- [Overview](#overview)
- [Creating a dataset from SingleTurnSamples](#creating-a-dataset-from-singleturnsamples)
- [Loading a dataset from Hugging Face Datasets](#loading-a-dataset-from-hugging-face-datasets)
- [Saving and loading](#saving-and-loading)

## Overview

### Structure of an evaluation dataset

An evaluation dataset consists of:

- **Samples**: [SingleTurnSample](eval_sample.md#singleturnsample) or [MultiTurnSample](eval_sample.md#multiturnsample) instances. Each sample represents a unique interaction or scenario.
- **One sample type**: all samples are single-turn, or all are multi-turn. The `data_model` names the type, and `metrics_experiment` rejects a dataset that mixes them.
- **A backend**: where the rows are saved, such as `inmemory`, `local/jsonl` or `local/csv`. See [Storage backends](../../howtos/customizations/storage_backends.md).


### Guidelines for Curating an Effective Evaluation Dataset

- **Define Clear Objectives**: Identify the specific aspects of the AI application that you want to evaluate and the scenarios you want to test. Collect data samples that reflect these objectives.

- **Collect Representative Data**: Ensure that the dataset covers a diverse range of scenarios, user inputs, and expected responses to provide a comprehensive evaluation of the AI application. This can be achieved by collecting data from various sources or [generating synthetic data](./../../howtos/customizations/index.md#testset-generation).

- **Quality and Size**: Aim for a dataset that is large enough to provide meaningful insights but not so large that it becomes unwieldy. Ensure that the data is of high quality and accurately reflects the real-world scenarios you want to evaluate.


## Creating a dataset from SingleTurnSamples

**Step 1:** Import the classes.

```python
from ragas import Dataset, SingleTurnSample
```

**Step 2:** Create the samples.

```python
# Sample 1
sample1 = SingleTurnSample(
    user_input="What is the capital of Germany?",
    retrieved_contexts=["Berlin is the capital and largest city of Germany."],
    response="The capital of Germany is Berlin.",
    reference="Berlin",
)

# Sample 2
sample2 = SingleTurnSample(
    user_input="Who wrote 'Pride and Prejudice'?",
    retrieved_contexts=["'Pride and Prejudice' is a novel by Jane Austen."],
    response="'Pride and Prejudice' was written by Jane Austen.",
    reference="Jane Austen",
)

# Sample 3
sample3 = SingleTurnSample(
    user_input="What's the chemical formula for water?",
    retrieved_contexts=["Water has the chemical formula H2O."],
    response="The chemical formula for water is H2O.",
    reference="H2O",
)
```

**Step 3:** Put them in a `Dataset`.

```python
dataset = Dataset(
    "capitals",
    "inmemory",
    data_model=SingleTurnSample,
    data=[sample1, sample2, sample3],
)
len(dataset)  # 3
dataset[0].reference  # 'Berlin'
```

**Step 4:** Score it.

```python
from ragas import metrics_experiment, summarize
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

llm = llm_factory("gpt-6-luna")
exp = metrics_experiment([Faithfulness(llm=llm)]).run(dataset, name="capitals")
summarize(exp)
# {'faithfulness': {'mean': 1.0, 'count': 3, 'missing': 0}}
```

Rows may also be plain dicts with the same field names (`user_input`, `response`, `retrieved_contexts`, `reference`, ...). Use `metrics_experiment(..., column_map={"user_input": "question"})` when your columns are named differently.

## Loading a dataset from Hugging Face Datasets

To start from an existing dataset, such as one on the Hugging Face Hub, go through pandas. `Dataset.from_pandas` builds each sample from a DataFrame row, so turn missing values (NaN) into `None` first:

```python
from datasets import load_dataset

from ragas import Dataset, SingleTurnSample

hf = load_dataset("vibrantlabsai/amnesty_qa", "english_v3")["eval"]
df = hf.to_pandas()
df = df.astype(object).where(df.notna(), None)  # NaN -> None
dataset = Dataset.from_pandas(df, "amnesty_qa", "inmemory", data_model=SingleTurnSample)
```

Ensure that the dataset has the fields your metrics need, such as user inputs, retrieved contexts, responses and references. `datasets` isn't a ragas dependency; install it yourself.

## Saving and loading

Give the dataset a file backend and call `save()`; load it back with its `data_model`:

```python
dataset = Dataset(
    "capitals",
    "local/jsonl",
    root_dir="./data",
    data_model=SingleTurnSample,
    data=[sample1, sample2, sample3],
)
dataset.save()  # ./data/datasets/capitals.jsonl

loaded = Dataset.load(
    "capitals", "local/jsonl", root_dir="./data", data_model=SingleTurnSample
)
```

!!! warning "Prefer JSONL to CSV for datasets you'll score"
    CSV loads a missing text value as `""`, and `metrics_experiment` scores `""` as a value. A sample without a `reference` that went through CSV is scored against an empty reference instead of raising. See [What each backend keeps](../../howtos/customizations/storage_backends.md#what-each-backend-keeps).

Coming from `EvaluationDataset`? See [From `evaluate()` to experiments](../../howtos/migrations/evaluate_to_experiment.md#evaluationdataset-dataset).
