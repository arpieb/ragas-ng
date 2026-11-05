# Caching in Ragas

You can use caching to speed up your evaluations and testset generation by avoiding redundant computations. We use Exact Match Caching to cache the responses from the LLM and Embedding models.

Cache entries are keyed on the input **and** on what produced the answer: the wrapper class, provider, model, model arguments (temperature, max tokens, system prompt) and the client's endpoint (`base_url`). So LLMs or embedding models configured differently can safely share one cache without returning each other's results.

You can use the [DiskCacheBackend][ragas.cache.DiskCacheBackend] which uses a local disk cache to store the cached responses. You can also implement your own custom cacher by implementing the [CacheInterface][ragas.cache.CacheInterface].

## Using Caching with Modern LLMs and Embeddings

The new metrics collections and experiments support caching through a simple interface.

### Quick Start

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache once
cache = DiskCacheBackend()

# Use with LLM factory
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# All LLM calls are now cached!
from pydantic import BaseModel


class Response(BaseModel):
    answer: str


response = llm.generate("Evaluate this...", Response)
```

### Caching with llm_factory

```python
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from openai import OpenAI

# Create cache instance
cache = DiskCacheBackend()

# Create LLM with caching
client = OpenAI(api_key="...")
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# First call - makes API request and caches result
response1 = llm.generate("Evaluate this text", Response)

# Second call - returns cached result instantly
response2 = llm.generate("Evaluate this text", Response)

# Result: Same output, 60x faster, $0 cost
```

### Caching with embedding_factory

```python
from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory
from openai import OpenAI

cache = DiskCacheBackend()
client = OpenAI(api_key="...")

embeddings = embedding_factory("openai", client=client, cache=cache)

# First call - makes API request
vector1 = embeddings.embed_text("Some text to embed")

# Second call - instant cache hit
vector2 = embeddings.embed_text("Some text to embed")

assert vector1 == vector2  # Identical results
```

### Caching in Experiments

Caching is especially powerful in experiments where you run the same evaluation multiple times:

```python
from ragas import experiment, Dataset
from ragas.cache import DiskCacheBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

# Setup cached LLM once
cache = DiskCacheBackend()
llm = llm_factory("gpt-4o-mini", client=client, cache=cache)

# Use in metric
metric = FactualCorrectness(llm=llm)


@experiment()
async def evaluate_model(row):
    score = metric.score(response=row["response"], reference=row["reference"])
    return {**row, "factual_correctness": score.value, "reason": score.reason}


# Load your dataset
dataset = Dataset.from_list(
    [
        {"response": "Paris is the capital of France", "reference": "Paris"},
        {"response": "London is the capital of UK", "reference": "London"},
    ]
)

# First run - makes API calls and caches results
print("First run (populating cache)...")
results1 = await evaluate_model.arun(dataset)
# Takes ~2 seconds for 2 samples

# Second run - uses cache, nearly instant!
print("Second run (using cache)...")
results2 = await evaluate_model.arun(dataset)
# Takes ~0.1 seconds for 2 samples

# Results are identical, but 20x faster!
```

### Cache Management

#### Clearing the Cache

```python
# Clear all cached data
cache = DiskCacheBackend()
cache.cache.clear()
```

#### Setting Size Limits

```python
# Limit cache to 1GB
cache = DiskCacheBackend()
cache.cache.reset("size_limit", 1e9)  # 1GB
cache.cache.reset("cull_limit", 10)  # Remove 10% when full
```

#### Cache Location

By default, cache is stored in `.cache/` directory. You can change this:

```python
cache = DiskCacheBackend(cache_dir="my_custom_cache")
```

### Benefits of Caching

1. **Cost Savings**: Avoid repeated API calls for identical inputs (50-60% savings)
2. **Speed**: Cached calls return nearly instantly (60x+ faster)
3. **Development**: Iterate quickly without waiting for API calls
4. **Reproducibility**: Same inputs always return same results

Cache hits occur when:

- ✅ Same prompt/text (exact match)
- ✅ Same model parameters (temperature, max_tokens, etc.)
- ✅ Same response model/structure (for LLMs)

Cache misses occur when:

- ❌ Different prompt/text
- ❌ Different parameters
- ❌ Different response model

### Anti-Patterns (When NOT to Cache)

- ❌ **Non-deterministic prompts**: If prompts contain random elements or timestamps
- ❌ **High temperature**: If temperature > 0.7 (responses vary too much)
- ❌ **Streaming responses**: Caching doesn't work with streaming
- ❌ **Real-time data**: If responses need to reflect current state

### Environment-Specific Notes

**Notebooks**: Cache persists between cell executions and kernel restarts

**Web Applications**: Share cache across requests for better performance

**Serverless Functions**: Use `/tmp` directory:

```python
cache = DiskCacheBackend(cache_dir="/tmp/.cache")
```

**Distributed Workers**: Cache is process-safe but for high-throughput systems consider implementing a Redis backend via the `CacheInterface`

### Performance Expectations

| Scenario | Time | Cost |
|----------|------|------|
| First run (100 samples) | ~2 minutes | $0.50 |
| Second run (cached) | ~2 seconds | $0.00 |
| **Speedup** | **60x faster** | **100% savings** |

---

## Caching with a LangChain LLM

Removed. `LangchainLLMWrapper` no longer exists -- ragas does not depend on
LangChain. Use `llm_factory()` and `embedding_factory()` as shown above; both
take the same `cache=` argument.

See the [migration guide](../migrations/migrate_off_langchain.md).
