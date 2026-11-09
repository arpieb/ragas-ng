# Google Gemini Integration Guide

Use Gemini models as ragas' evaluator and testset generator through [LiteLLM](https://docs.litellm.ai/docs/providers/gemini). You need no Google SDK for the LLM.

## Setup

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
pip install ragas
export GOOGLE_API_KEY="your-google-api-key"   # or GEMINI_API_KEY
```

```python
from ragas.llms import llm_factory

llm = llm_factory("gemini/gemini-3.6-flash")
```

### Vertex AI

With Google Cloud credentials (Application Default Credentials, e.g. `gcloud auth application-default login`):

```python
llm = llm_factory(
    "vertex_ai/gemini-3.6-flash",
    vertex_project="my-project",
    vertex_location="us-central1",
)
```

### Gemini's OpenAI-compatible endpoint

An `OpenAI` client pointed at Gemini's endpoint works too. The client's settings (headers, timeout, proxy) carry over:

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
llm = llm_factory("gemini-3.6-flash", client=client)
```

!!! note "google-genai clients are no longer accepted for LLMs"
    `llm_factory("gemini-...", provider="google", client=genai.Client(...))` raises an
    error naming `llm_factory("gemini/gemini-...")` instead (#182). Every call goes
    through LiteLLM, and instructor's native Gemini mode, with its
    `HARM_CATEGORY_JAILBREAK` safety-settings bug, isn't used any more.
    Embeddings still take a google-genai client (below).

## Models

- **Flash:** `gemini-3.6-flash` (stable, used on this page) and `gemini-3.8-flash` (Google's newest Flash model, with better reasoning for complex evaluations).
- **Flash-Lite:** `gemini-3.5-flash-lite`, the fastest and cheapest, for simple or very high-volume metrics.

Older models are shut down (Gemini 1.x and 2.0) or shut down on 2026-12-01 (2.5). See Google's [models](https://aistudio.google.com/docs/models) and [deprecations](https://aistudio.google.com/docs/deprecations) pages.

## Embeddings

Metrics such as `AnswerCorrectness`, `AnswerRelevancy` and `SemanticSimilarity` also need embeddings. Use Google's:

```python
from ragas.embeddings import GoogleEmbeddings

embeddings = GoogleEmbeddings(model="gemini-embedding-001")  # pip install google-genai
# or, with your own google-genai client:
# from google import genai
# embeddings = GoogleEmbeddings(client=genai.Client(), model="gemini-embedding-001")
```

!!! note "evaluate()'s default embeddings follow the LLM"
    When a metric needs embeddings and you pass none, `evaluate()` picks them
    from the LLM. For `llm_factory("gemini/...")` that's
    `gemini/gemini-embedding-2` through LiteLLM, with the same `GEMINI_API_KEY`
    (or `api_key=`), so a Gemini-only setup needs no `OPENAI_API_KEY`.
    `llm_factory("vertex_ai/...")` gets `vertex_ai/gemini-embedding-2` with the
    LLM's `vertex_project` / `vertex_location`. An LLM built on an OpenAI
    client pointed at Gemini's endpoint, as above, still asks that client for
    an OpenAI embedding model, which Gemini's endpoint doesn't serve, so pass
    `embeddings=` there.

## Example

```python
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerCorrectness, ContextPrecision, Faithfulness

llm = llm_factory("gemini/gemini-3.6-flash")
embeddings = GoogleEmbeddings(model="gemini-embedding-001")

dataset = EvaluationDataset.from_list(
    [
        {
            "user_input": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "retrieved_contexts": [
                "France is in Western Europe. Paris is its capital."
            ],
            "reference": "Paris",
        }
    ]
)

result = evaluate(
    dataset,
    metrics=[
        ContextPrecision(llm=llm),
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ],
)
print(result)
```

Or score one sample with a collections metric:

```python
result = await AnswerCorrectness(llm=llm, embeddings=embeddings).ascore(
    user_input="What is the capital of France?",
    response="Paris",
    reference="Paris is the capital of France.",
)
```

--8<--
running_async.md
--8<--

## Model parameters

```python
llm = llm_factory(
    "gemini/gemini-3.6-flash", temperature=0.5, max_tokens=2048, top_p=0.9
)
```

Without them, ragas sends near-deterministic defaults (`temperature=0.01`, `top_p=0.1`), which LiteLLM drops if the model rejects them. See [LLMs (llm_factory)](../llm-adapters.md#model-arguments).

## Troubleshooting

- **Missing key:** set `GOOGLE_API_KEY` or `GEMINI_API_KEY`, or pass `api_key=` to `llm_factory`.
- **Rate limits:** ragas retries transient errors (rate limits, timeouts, 5xx) with backoff according to `RunConfig`. Lower `RunConfig(max_workers=...)` if you still hit quota limits.
- **Model not available:** check your region and quota, and that the model is on Google's current [models page](https://aistudio.google.com/docs/models).

## Migrating

| Before | Now |
|---|---|
| `llm_factory("gemini-...", provider="google", client=genai.Client(api_key=...))` | `llm_factory("gemini/gemini-3.6-flash")` |
| `llm_factory("gemini-...", provider="google", client=genai.GenerativeModel(...))` (legacy SDK) | `llm_factory("gemini/gemini-3.6-flash")` |
| a google-genai Vertex client | `llm_factory("vertex_ai/gemini-3.6-flash", vertex_project=..., vertex_location=...)` |
| `GoogleEmbeddings(client=genai.Client(), ...)` | unchanged |

## Resources

- [LiteLLM: Gemini](https://docs.litellm.ai/docs/providers/gemini) and [Vertex AI](https://docs.litellm.ai/docs/providers/vertex)
- [Google AI Studio](https://aistudio.google.com)
- [LLMs (llm_factory)](../llm-adapters.md)
