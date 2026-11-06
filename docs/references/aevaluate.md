# Async Evaluation

## aevaluate()

::: ragas.evaluation.aevaluate

## Async Usage

Ragas provides both synchronous and asynchronous evaluation APIs to accommodate different use cases:

### Using aevaluate() (Recommended for Production)

For production async applications, use `aevaluate()` to avoid event loop conflicts:

```python
import asyncio
from ragas import aevaluate


async def evaluate_app():
    result = await aevaluate(dataset, metrics)
    return result


# In your async application
result = await evaluate_app()
```

### Using evaluate() from Sync Code

`evaluate()` runs `aevaluate()` on a shared background event loop, so the same
call works in scripts, in Jupyter, and from sync code inside async
applications. It does not patch your event loop. Earlier versions used
`nest_asyncio`; ragas no longer depends on it, and `allow_nest_asyncio=` is
deprecated and ignored.

```python
result = evaluate(dataset, metrics)
```

!!! note "Async clients in async code"
    If your evaluator LLM wraps an async client (such as `AsyncOpenAI`) that your
    own async code also awaits, call `await aevaluate(...)` (or the metrics'
    `ascore()`) instead. Otherwise the client is used from two event loops, which
    can fail with "bound to a different event loop".
