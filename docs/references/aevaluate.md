# Sync and async runs

A `metrics_experiment()` (like any `@experiment`) has an async `arun()` and a sync `run()`. Both take the same arguments and return the same `Experiment`.

## In async code: `arun()`

```python
from ragas import metrics_experiment, summarize

run = metrics_experiment([Faithfulness(llm=llm)])


async def score(dataset):
    exp = await run.arun(dataset, name="nightly")
    return summarize(exp)
```

Use `arun()` in async applications and notebooks you drive with `await`. A running `arun()` can be cancelled like any task: `task = asyncio.create_task(run.arun(dataset))`, then `task.cancel()`. The rows still running are cancelled and nothing is saved.

## From sync code: `run()`

```python
exp = run.run(dataset, name="nightly")
```

`run()` runs `arun()` on a shared background event loop, so the same call works in scripts, in Jupyter, and from sync code inside async applications. It doesn't patch your event loop.

!!! note "Async clients in async code"
    If your evaluator LLM wraps an async client (such as `AsyncOpenAI`) that your
    own async code also awaits, call `await run.arun(...)` (or the metrics'
    `ascore()`) instead. Otherwise the client is used from two event loops, which
    can fail with "bound to a different event loop".

See [Scoring a dataset](evaluate.md) for the arguments, and [Experiments](../concepts/experimentation.md) for writing your own experiment function.
