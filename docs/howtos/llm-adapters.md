# LLM Adapters: Using Multiple Structured Output Backends

Ragas supports multiple structured output backends through an adapter pattern. This guide explains how to use different adapters for different LLM providers.

## Overview

Ragas uses adapters to handle structured output from different LLM providers:

- **Instructor Adapter**: Works with OpenAI, Anthropic, Azure, Groq, Mistral, Cohere, and many others
- **LiteLLM Adapter**: Works with all 100+ LiteLLM-supported providers (Gemini, Ollama, vLLM, Bedrock, etc.)

The framework automatically selects the best adapter for your provider, but you can also choose explicitly.

## Quick Start

### Automatic Adapter Selection (Recommended)

Let Ragas auto-detect the best adapter:

```python
from ragas.llms import llm_factory
from openai import OpenAI

# For OpenAI - automatically uses Instructor adapter
client = OpenAI(api_key="...")
llm = llm_factory("gpt-6-luna", client=client)
```

```python
from ragas.llms import llm_factory

# No client - routed through LiteLLM, reads GEMINI_API_KEY
llm = llm_factory("gemini/gemini-3.6-flash")
```

### Explicit Adapter Selection

Choose a specific adapter if you need more control:

```python
from ragas.llms import llm_factory

# Force using Instructor adapter
llm = llm_factory("gpt-6-luna", client=client, adapter="instructor")

# Force using LiteLLM adapter
llm = llm_factory("gemini-3.6-flash", client=client, adapter="litellm")
```

## Auto-Detection Logic

When `adapter="auto"` (default), Ragas uses this logic:

1. **No client**: always the LiteLLM adapter, which reads credentials from the environment
2. **Check client type**: If client is from `litellm` module → use LiteLLM adapter
3. **Check provider**: If provider is `google` or `gemini` and the client is not a
   `google-genai` `Client` → use LiteLLM adapter
4. **Default**: Use Instructor adapter for all other cases

The LiteLLM adapter accepts an OpenAI-compatible client (such as one pointed at a
LiteLLM proxy), a `litellm.Router`, or an instructor-patched client. Other clients
are rejected with a pointer to the no-client form.

```python
from ragas.llms.adapters import auto_detect_adapter

# See which adapter will be used
adapter_name = auto_detect_adapter(client, "google")
print(adapter_name)  # Output: "litellm"

adapter_name = auto_detect_adapter(client, "openai")
print(adapter_name)  # Output: "instructor"
```

## Provider-Specific Examples

### OpenAI

```python
from openai import OpenAI
from ragas.llms import llm_factory

client = OpenAI(api_key="your-key")
llm = llm_factory("gpt-6-luna", client=client)
# Uses Instructor adapter automatically
```

### Anthropic Claude

```python
from anthropic import Anthropic
from ragas.llms import llm_factory

client = Anthropic(api_key="your-key")
llm = llm_factory("claude-sonnet-5-5", provider="anthropic", client=client)
# Uses Instructor adapter automatically
```

### Google Gemini

```python
from google import genai
from ragas.llms import llm_factory

client = genai.Client(api_key="your-key")
llm = llm_factory("gemini-3.6-flash", provider="google", client=client)
# Uses the Instructor adapter (instructor.from_genai)

# Or with no client, through LiteLLM (reads GEMINI_API_KEY)
llm = llm_factory("gemini/gemini-3.6-flash")
```

### Google Gemini (with LiteLLM Proxy - Advanced)

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Requires running: litellm --model gemini-3.6-flash
client = OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000",  # LiteLLM proxy endpoint
)
llm = llm_factory("gemini-3.6-flash", client=client, adapter="litellm")
# Uses LiteLLM adapter explicitly
```

### Local Models (Ollama)

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Ollama exposes OpenAI-compatible API
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
llm = llm_factory("mistral", provider="openai", client=client)
# Uses Instructor adapter
```

### AWS Bedrock

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Use LiteLLM proxy for Bedrock
# Note: Set up LiteLLM with Bedrock credentials first
client = OpenAI(
    api_key="",  # Bedrock uses IAM auth
    base_url="http://0.0.0.0:4000",  # LiteLLM proxy endpoint
)
llm = llm_factory("claude-sonnet-5-5", client=client, adapter="litellm")
```

### Groq

```python
from groq import Groq
from ragas.llms import llm_factory

client = Groq(api_key="your-key")
llm = llm_factory("mixtral-8x7b", provider="groq", client=client)
# Uses Instructor adapter automatically
```

### Mistral

```python
from mistralai import Mistral
from ragas.llms import llm_factory

client = Mistral(api_key="your-key")
llm = llm_factory("mistral-large", provider="mistral", client=client)
# Uses Instructor adapter automatically
```

### Cohere

```python
from cohere import Cohere
from ragas.llms import llm_factory

client = Cohere(api_key="your-key")
llm = llm_factory("command-r-plus", provider="cohere", client=client)
# Uses Instructor adapter automatically
```

## Adapter Selection Guide

Choose your adapter based on your needs:

### Use Instructor Adapter if:
- Using OpenAI, Anthropic, Azure, Groq, Mistral, or Cohere
- Provider is natively supported by Instructor
- You want the most stable, well-tested option
- Provider doesn't require special handling

### Use LiteLLM Adapter if:
- Using Google Gemini
- Using local models (Ollama, vLLM, etc.)
- Using providers with 100+ options (Bedrock, etc.)
- You need maximum provider compatibility
- Auto-detection selects it for your provider

## Working with Adapters Directly

### Get Available Adapters

```python
from ragas.llms.adapters import ADAPTERS

print(ADAPTERS)
# Output: {
#     "instructor": InstructorAdapter(),
#     "litellm": LiteLLMAdapter()
# }
```

### Get Specific Adapter

```python
from ragas.llms.adapters import get_adapter

instructor = get_adapter("instructor")
litellm = get_adapter("litellm")

# Create LLM using adapter directly
llm = instructor.create_llm(client, "gpt-6-luna", "openai")
```

## Advanced Usage

### Model Arguments

All adapters support the same model arguments:

```python
llm = llm_factory(
    "gpt-6-luna",
    client=client,
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)
```

Without them, ragas uses near-deterministic defaults (`temperature=0.01`,
`top_p=0.1`, `max_tokens=1024`). A default that a provider rejects is dropped;
a value you set yourself is always sent:

- **OpenAI reasoning models** (o-series, GPT-5 and later, `codex-mini`, including
  dotted versions such as `gpt-5.2` and IDs like `openai/gpt-5.6-sol` or Bedrock's
  `openai.gpt-6-sol`): `temperature` is sent as 1, `top_p` is dropped, and no
  token cap is sent unless you set `max_tokens`, which is sent as
  `max_completion_tokens`. The cap counts the model's hidden reasoning tokens
  too, so set it generously.
- **Anthropic** (Claude on any provider): neither default is sent, because
  current Claude models reject `temperature` (other than 1) and `top_p`, and
  anthropic SDK 2.6 removed both from `messages.create()`. Per-call sampling
  (`strictness`, `n > 1`) doesn't send a temperature either; those calls use
  Claude's default sampling. A `temperature` or `top_p` you set yourself is sent
  (#177).

Reasoning models are detected from the model name. For an Azure deployment (or
proxy alias) whose name doesn't say what it serves, tell ragas:

```python
llm = llm_factory("eval-deployment", provider="azure", client=client, reasoning=True)
```

### System Prompts

Both adapters support system prompts for models that require specific instructions:

```python
llm = llm_factory(
    "gpt-6-luna",
    client=client,
    system_prompt="You are a helpful assistant that evaluates RAG systems.",
)
```

System prompts are useful when:
- Your LLM requires specific behavior instructions
- You're using fine-tuned models with custom system prompts
- You want to guide the evaluation style across all metrics

The system prompt is prepended to all LLM calls as a system message.

### Custom Instructor Modes

The instructor adapter supports multiple modes for structured output generation. By default, `Mode.JSON` is used, but you can specify a different mode for backends that don't support certain features:

```python
import instructor
from ragas.llms import llm_factory
from openai import OpenAI

# Use MD_JSON mode for backends without response_format support
client = OpenAI(api_key="...", base_url="https://custom-backend")
llm = llm_factory(
    "custom-model", provider="openai", client=client, mode=instructor.Mode.MD_JSON
)
```

`mode` reaches every provider (OpenAI, Anthropic, Google, Mistral, Perplexity,
LiteLLM). Without it:

- OpenAI-style clients use `Mode.JSON`.
- Claude uses JSON too: `Mode.ANTHROPIC_JSON` with an Anthropic client, and
  `Mode.JSON` through LiteLLM. Instructor's default for Claude forces a tool
  call, which `claude-sonnet-5-5` rejects (#184). Pass `mode=` to choose
  another, e.g. `Mode.ANTHROPIC_TOOLS`.
- Other providers keep instructor's default for that provider.

Available instructor modes:
- `Mode.JSON` (default) - Uses OpenAI's response_format parameter
- `Mode.MD_JSON` - Uses markdown JSON in the prompt (fallback for unsupported backends)
- `Mode.TOOLS` - Uses function calling
- `Mode.JSON_SCHEMA` - Uses JSON schema validation

Use `Mode.MD_JSON` when you encounter errors like:
```
Error code: 400 - {'message': 'only pytorch backend can use response_format now'}
```

### Async Support

Both adapters support async operations:

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

async_client = AsyncOpenAI(api_key="...")
llm = llm_factory("gpt-6-luna", client=async_client)

# Async generation
response = await llm.agenerate(prompt, ResponseModel)
```

Sync and async both work with either kind of client. A sync client runs on a
worker thread under `agenerate()`. An async client's sync calls (`generate()`,
`metric.score()`, `embed_text()`) run on one shared background event loop. This
also works inside Jupyter or another running loop, and the client stays bound
to that one loop across calls.

!!! warning "Don't mix both styles on one async client"
    An async client keeps its connections tied to the event loop that opened
    them. Calling it with `await` on your own loop *and* through the sync API
    (which uses ragas' background loop) makes the two loops share those
    connections, which can fail with "bound to a different event loop". In
    async code, use the async methods (`agenerate`, `ascore`, `aembed_text`)
    throughout.

### Custom Providers with LiteLLM

LiteLLM supports many providers beyond what Instructor covers. Use the LiteLLM proxy approach:

```python
from openai import OpenAI
from ragas.llms import llm_factory

# Set up LiteLLM proxy first:
# litellm --model grok-1  (for xAI)
# litellm --model deepseek-chat  (for DeepSeek)
# etc.

client = OpenAI(
    api_key="your-provider-api-key",
    base_url="http://0.0.0.0:4000",  # LiteLLM proxy endpoint
)

# xAI Grok
llm = llm_factory("grok-1", client=client, adapter="litellm")

# DeepSeek
llm = llm_factory("deepseek-chat", client=client, adapter="litellm")

# Together AI
llm = llm_factory("mistral-7b", client=client, adapter="litellm")
```

## Complete Evaluation Example

```python
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerCorrectness,
)

# Initialize LLM with your provider
import google.generativeai as genai

genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-3.6-flash")
llm = llm_factory("gemini-3.6-flash", provider="google", client=client)

# Create evaluation dataset
data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is in Europe. Paris is its capital."]],
    "ground_truth": ["Paris"],
}
dataset = Dataset.from_dict(data)

# Define metrics
metrics = [
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
    Faithfulness(llm=llm),
    AnswerCorrectness(llm=llm),
]

# Evaluate
results = evaluate(dataset, metrics=metrics)
print(results)
```

## Troubleshooting

### "Unknown adapter: xyz"

Make sure you're using a valid adapter name:

```python
# Valid: "instructor" or "litellm"
llm = llm_factory("model", client=client, adapter="instructor")

# Invalid: "dspy" (not yet implemented)
# llm = llm_factory("model", client=client, adapter="dspy")  # Error!
```

### "Failed to initialize provider client"

Ensure:
1. Your client is properly initialized
2. Your API key is valid
3. The provider is supported by the adapter

```python
# Check if adapter can handle your provider
from ragas.llms.adapters import auto_detect_adapter

adapter = auto_detect_adapter(client, "my-provider")
print(f"Will use: {adapter}")
```

### Adapter Mismatch

Auto-detection handles most cases, but explicit selection can help:

```python
# If auto-detection picks the wrong adapter:
llm = llm_factory(
    "model",
    provider="provider-name",
    client=client,
    adapter="litellm",  # Explicit override
)
```

## Migration Guide

### From Text-Only to Structured Output

If you're upgrading from text-only LLM usage:

```python
# Before (removed in the LangChain removal)
# from ragas.llms import LangchainLLMWrapper
# llm = LangchainLLMWrapper(langchain_llm)

# After (new way)
from ragas.llms import llm_factory

llm = llm_factory("gpt-6-luna", client=client)
```

### Switching Providers

To switch from OpenAI to Gemini:

```python
# Before: OpenAI
from openai import OpenAI

client = OpenAI(api_key="...")
llm = llm_factory("gpt-6-luna", client=client)

# After: Gemini (similar code pattern!)
import google.generativeai as genai

genai.configure(api_key="...")
client = genai.GenerativeModel("gemini-3.6-flash")
llm = llm_factory("gemini-3.6-flash", provider="google", client=client)
# Adapter automatically switches to LiteLLM for google provider
```

## See Also

- [Gemini Integration Guide](./integrations/gemini.md) - Detailed Gemini setup
- LLM Factory Reference - Complete API reference
- [Metrics Documentation](../concepts/metrics/index.md) - Using metrics with LLMs
