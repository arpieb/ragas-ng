# Cancelling Long-Running Tasks

Scoring a large dataset or generating a large testset can take a long time. This page shows how to stop either one cleanly, and how to bound a run with timeouts.

## Overview

- **Experiments**, including [`metrics_experiment`](../../concepts/experimentation.md#scoring-a-dataset-in-one-call-metrics_experiment): `arun()` is a coroutine. Cancel its task to stop the run.
- **Test set generation**: `generate_with_docs(..., return_executor=True)` returns an executor you can cancel.

## Cancelling an experiment

Run `arun()` as a task and cancel the task:

```py
import asyncio

from ragas import metrics_experiment

run = metrics_experiment(metrics)

task = asyncio.create_task(run.arun(dataset, name="long_run"))
# ... later, e.g. when a request is aborted
task.cancel()

try:
    exp = await task
except asyncio.CancelledError:
    print("The run was cancelled")
```

When the task is cancelled, the rows still running are cancelled too, and their metric and LLM calls stop at their next `await`. **Nothing is saved**: the experiment is written only when every row has finished.

From sync code, `run.run(...)` stops the same way on Ctrl+C: the `KeyboardInterrupt` cancels the run before it propagates.

### A time limit for the whole run

```py
try:
    exp = await asyncio.wait_for(run.arun(dataset, name="nightly"), timeout=1800)
except asyncio.TimeoutError:
    print("The run took longer than 30 minutes and was cancelled")
```

### A time limit per metric call

`RunConfig(timeout=...)` bounds each metric call (and each LLM call's request). A call that takes longer raises `TimeoutError`; the metric scores NaN on that row and the run goes on, unless you set `raise_exceptions=True`:

```py
from ragas.run_config import RunConfig

run = metrics_experiment(metrics, run_config=RunConfig(timeout=60))
```

See [RunConfig](run_config.md).

### Stopping on the first error

`raise_exceptions=True` stops the run at the first metric that raises: the rows still running are cancelled and that error is re-raised. Add `save_partial=True` to save the rows that finished first:

```py
run = metrics_experiment(metrics, raise_exceptions=True, save_partial=True)
```

### In a web application

Keep the task, and cancel it when the client asks:

```py
import asyncio
import uuid

runs: dict[str, asyncio.Task] = {}


async def start(dataset):
    run_id = str(uuid.uuid4())
    runs[run_id] = asyncio.create_task(run.arun(dataset, name=run_id))
    return run_id


def cancel(run_id: str) -> bool:
    task = runs.pop(run_id, None)
    if task is None or task.done():
        return False
    return task.cancel()
```

## Cancelling test set generation

`generate_with_docs()` (and `generate()`) take `return_executor=True`. You get the executor instead of the testset, and can cancel it from another thread:

```py
from ragas.testset.synthesizers.generate import TestsetGenerator

generator = TestsetGenerator(...)

executor = generator.generate_with_docs(
    documents=documents,
    testset_size=100,
    return_executor=True,
)

# Elsewhere:
executor.cancel()
executor.is_cancelled()  # True

# Blocks until the jobs finish or are cancelled
results = executor.results()
```

## Limitations

- Cancellation is cooperative: a call stops at its next `await`. A sync experiment function runs in a worker thread, which can't be stopped; its row finishes in the background and its result is discarded.
- An LLM request already sent may still complete and be billed by the provider.
