# Customize Timeouts, Retries and Rate Limits

Ragas retries rate limits and other transient provider errors for you, with
exponential backoff. `RunConfig` controls how hard it tries, how long a job may
take, and how many jobs run at once.

## Retries and Backoff

LLMs built with `llm_factory` retry a failed provider call when the error is
transient:

- HTTP 429 (rate limit), 408, 409 and any 5xx
- timeouts and connection failures

Other errors are never retried. A 400 (a bad parameter, a rejected model) or a
401 fails at once instead of after minutes of backoff. Each retry waits a
random, exponentially growing time, capped at `max_wait` seconds.

Without any configuration an LLM uses `RunConfig()` defaults: up to 10 attempts,
at most 60 seconds between them. Change them for one LLM:

```python
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

llm = llm_factory("gpt-6-luna")
llm.set_run_config(RunConfig(max_retries=5, max_wait=30))
```

`evaluate(run_config=...)` and `TestsetGenerator` apply their `RunConfig` to
every metric's LLM, so for a whole run you only set it once:

```python
from ragas import evaluate
from ragas.metrics.collections import Faithfulness
from ragas.run_config import RunConfig

results = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(llm=llm)],
    run_config=RunConfig(max_workers=8, max_retries=8, max_wait=60),
)
```

To see the retries, set `log_tenacity=True`: each one is then logged at
`WARNING` on the `ragas.llms.retry` logger (at `DEBUG` otherwise).

To retry a different set of errors, pass `exception_types`, for example
`RunConfig(exception_types=(MyGatewayError,))`. Only those are retried then.
The default, `(Exception,)`, means "transient errors" as described above.

!!! note "Client retries stack with ragas retries"
    Provider SDKs retry on their own too. `openai.OpenAI` retries twice by
    default, inside each ragas attempt. To leave retrying to the client, use
    `RunConfig(max_retries=1)`. To leave it to ragas, create the client with
    `max_retries=0`.

## Timeouts

`RunConfig.timeout` (default 180 seconds) is a wall-clock limit on one **job**:
one metric scoring one sample in `evaluate()`, or one step of testset generation.
It covers every LLM call the job makes, **retries and backoff waits included**.
A metric that makes several calls, or a job that backs off on rate limits, can
reach it even though each request is quick. The job then fails with
`TimeoutError` and its score is NaN.

If jobs time out under rate limiting, raise `timeout`, lower `max_workers` so
fewer requests compete for the limit, or lower `max_retries`/`max_wait`.

`timeout` is not a per-request limit. Set that on the client:

```python
import httpx
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    timeout=httpx.Timeout(
        60.0,  # Total timeout per request
        connect=5.0,  # Connection timeout
        read=30.0,  # Read timeout
        write=10.0,  # Write timeout
    ),
    max_retries=0,  # leave retries to ragas
)
llm = llm_factory("gpt-6-luna", client=client)
```

!!! tip "Provider Documentation"
    Each LLM provider has its own client configuration options. Refer to your provider's SDK documentation:

    - [OpenAI Python SDK](https://github.com/openai/openai-python)
    - [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)

## RunConfig Parameters

```python
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=180,  # Max seconds per job, retries included (default: 180)
    max_retries=10,  # Attempts per LLM call (default: 10)
    max_wait=60,  # Max seconds between attempts (default: 60)
    max_workers=16,  # Concurrent jobs (default: 16)
    log_tenacity=False,  # Log each retry at WARNING (default: False)
    seed=42,  # Random seed (default: 42)
)
```
