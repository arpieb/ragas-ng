# Customize Timeouts, Retries and Rate Limits

Ragas retries rate limits and other transient provider errors for you, with
exponential backoff. `RunConfig` controls how hard it tries, how long a job may
take, and how many jobs run at once. To stay under a provider's requests- or
tokens-per-minute limit in the first place, give the LLM a
[`RateLimit`](#requests-and-tokens-per-minute).

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

## Requests and Tokens per Minute

`max_workers` limits how many jobs run at once, not how many requests reach the
provider: one metric or experiment row can make several LLM calls. To keep under
your provider's limit, pass a `RateLimit` to `llm_factory`:

```python
from ragas.llms import llm_factory
from ragas.rate_limit import RateLimit

limit = RateLimit(rpm=500, tpm=200_000)
llm = llm_factory("gpt-6-luna", rate_limit=limit)
```

Each request then waits until the limit has room, instead of being sent and
failing with a 429. It is paced in the LLM, so `evaluate()`, `metric.ascore()`,
testset generation and experiments all go through it.

- **One object is one limit.** Limits usually apply per API key, so give every
  LLM that uses the key the same `RateLimit`. It also holds across threads and
  event loops.
- **Every request counts,** including instructor's validation retries and
  ragas' retries. Retries made inside a client (the OpenAI SDK's own
  `max_retries`) are not seen.
- **`rpm`** is also spread over the minute, at most `ceil(rpm / 60)` requests
  per second, rather than sent in one burst.
- **`tpm`** counts a request before it is sent: the prompt's tokens (from
  `litellm.token_counter`) plus its `max_tokens`, if you set one. The answer's
  length isn't known in advance, so without `max_tokens` leave some room.
- **Embeddings** take one too, usually their own:
  `LiteLLMEmbeddings(model=..., rate_limit=RateLimit(rpm=3000))`.

A 429 can still happen, for example when other programs share the key. It is
then retried as described above.

## Timeouts

`RunConfig.timeout` (default 180 seconds) is a wall-clock limit on one **job**:
one metric scoring one sample in `evaluate()`, or one step of testset generation.
It covers every LLM call the job makes, **retries and backoff waits included**.
A metric that makes several calls, or a job that backs off on rate limits, can
reach it even though each request is quick. The job then fails with
`TimeoutError` and its score is NaN.

If jobs time out under rate limiting, raise `timeout`, lower `max_workers` so
fewer requests compete for the limit, or lower `max_retries`/`max_wait`. Time
spent waiting for a [`RateLimit`](#requests-and-tokens-per-minute) counts too:
with a low limit, lower `max_workers` so jobs don't queue for it inside their
timeout.

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
