# LLMs: `llm_factory()` and LiteLLM

Every ragas LLM is created with `llm_factory()`, and every call goes through [LiteLLM](https://docs.litellm.ai/docs/providers), with [instructor](https://python.useinstructor.com/) for structured output. You name the model with LiteLLM's `provider/model` string, and the credentials come from the environment.

```python
from ragas.llms import llm_factory

llm = llm_factory("gpt-6-luna")  # OPENAI_API_KEY
llm = llm_factory("anthropic/claude-sonnet-5-5")  # ANTHROPIC_API_KEY
llm = llm_factory("gemini/gemini-3.6-flash")  # GOOGLE_API_KEY or GEMINI_API_KEY
llm = llm_factory("vertex_ai/gemini-3.6-flash")  # Google Cloud credentials
llm = llm_factory("bedrock/<bedrock-model-id>")  # AWS credentials and region
llm = llm_factory("ollama/qwen3.8:27b")  # a local Ollama server
llm = llm_factory("groq/<model>")  # GROQ_API_KEY
llm = llm_factory("mistral/<model>")  # MISTRAL_API_KEY
```

No provider SDK is needed. See [LiteLLM's provider list](https://docs.litellm.ai/docs/providers) for each provider's prefix and environment variables. Pass credentials explicitly with `api_key=` / `api_base=` if you prefer:

```python
llm = llm_factory("anthropic/claude-haiku-5-5", api_key="sk-ant-...")
llm = llm_factory("ollama/qwen3.8:27b", api_base="http://gpu-box:11434")
```

`provider=` prefixes a bare model name: `llm_factory("llama3", provider="ollama")` is `"ollama/llama3"`. `"openai"`, the default, adds no prefix.

### Embeddings to go with the LLM

Metrics that need embeddings take them when you build them, e.g. `AnswerRelevancy(llm=llm, embeddings=embeddings)`. Neither the metrics nor `metrics_experiment` pick a default. Through LiteLLM, for example:

| LLM | Embeddings |
|---|---|
| bare name or `openai/...` | `embedding_factory("openai", model="text-embedding-3-small")` (`OPENAI_API_KEY`) |
| `gemini/...` | `embedding_factory("litellm", model="gemini/gemini-embedding-2")` |
| `vertex_ai/...` | `embedding_factory("litellm", model="vertex_ai/gemini-embedding-2")`, with your `vertex_project` / `vertex_location` |
| anything else | `embedding_factory("litellm", model="<provider>/<embedding model>")` |

## Bringing your own client

Two kinds of client are accepted:

- **`OpenAI` / `AsyncOpenAI`**, pointed at OpenAI or at any OpenAI-compatible endpoint (vLLM, Ollama, a LiteLLM proxy, Anthropic's or Gemini's OpenAI-compatible endpoints, ...), and **`AzureOpenAI` / `AsyncAzureOpenAI`**. Their settings (base URL, headers, timeout, proxy, `http_client`) carry over. ragas hands LiteLLM a copy of the client, because LiteLLM changes the object it is given. The model becomes `openai/<model>` or `azure/<deployment>`.

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    llm = llm_factory("qwen3.8:27b", client=client)
    ```

    To send embeddings through the same proxy or base URL, pass the client to `embedding_factory("openai", model=..., client=client)` too.

- **A `litellm.Router`**, for load balancing or fallbacks across deployments.

Native SDK clients of other providers (`Anthropic()`, `genai.Client()`, `Mistral()`, `Groq()`, `Cohere()`, boto3) are **no longer accepted**, and neither are instructor-patched clients. `llm_factory()` raises an error naming the model string to use instead (see [Migrating from native clients](#migrating-from-native-clients)). If you need your own instructor client, construct `ragas.llms.InstructorLLM(client=..., model=..., provider=...)` directly.

## Model arguments

```python
llm = llm_factory(
    "ollama/qwen3.8:27b",
    temperature=0.2,
    top_p=0.9,
    max_tokens=8192,
)
```

Without them, ragas sends near-deterministic defaults, `temperature=0.01` and `top_p=0.1`, together with LiteLLM's `drop_params`, so LiteLLM drops either one for a model it knows rejects it. A value **you** set is sent as given, so a provider that rejects it says so. No output cap is sent unless you set `max_tokens`.

ragas adds two rules where LiteLLM's model tables lag behind new models:

- **OpenAI reasoning models** (o-series, GPT-5 and later, `codex-mini`, including dotted versions such as `gpt-5.2` and IDs like `openai/gpt-5.6-sol` or Bedrock's `openai.gpt-6-sol`): `temperature` is sent as 1, `top_p` is dropped, and a `max_tokens` you set is sent as `max_completion_tokens`. The cap counts the model's hidden reasoning tokens too, so set it generously.
- **Claude** (on any provider): no `temperature` or `top_p` unless you set them. Current Claude models reject both (#177).

Per-call sampling (`strictness`, legacy prompts with `n > 1`) sends a temperature with `drop_params`. Where a model rejects it, as Claude does, the samples use the model's default sampling.

Reasoning models are detected from the model name. For an Azure deployment or proxy alias whose name doesn't say what it serves, tell ragas:

```python
llm = llm_factory("eval-deployment", client=azure_client, reasoning=True)
```

## System prompts

```python
llm = llm_factory(
    "gpt-6-luna",
    system_prompt="You are a helpful assistant that evaluates RAG systems.",
)
```

The system prompt is prepended to every call as a system message.

## Structured-output modes

instructor asks the model for structured output in one of several modes:

| Mode | How |
|---|---|
| `Mode.TOOLS` | function calling |
| `Mode.JSON` | `response_format={"type": "json_object"}` |
| `Mode.JSON_SCHEMA` | `response_format` with the JSON schema |
| `Mode.MD_JSON` | the schema in the prompt, with markdown JSON back (for backends without `response_format`) |

Defaults:
- **`Mode.JSON`** for `OpenAI` / Azure clients and for Claude. instructor's tool mode returned `{}` for `Dict` fields on OpenAI, and its forced tool call fails on `claude-sonnet-5-5` (#184).
- **instructor's default** (`Mode.TOOLS`) otherwise.

Choose another with `mode=`:

```python
import instructor

llm = llm_factory("ollama/my-model", mode=instructor.Mode.MD_JSON)
```

Use `Mode.MD_JSON` when a backend fails with errors like `only pytorch backend can use response_format now`. Native instructor modes (`Mode.ANTHROPIC_*`, `Mode.GENAI_*`, ...) only work with native SDK clients and raise an error naming the equivalent.

## Sync and async

```python
llm = llm_factory("gpt-6-luna")
response = await llm.agenerate(prompt, ResponseModel)
response = llm.generate(prompt, ResponseModel)
```

Both work for every LLM. With an `OpenAI` (sync) client, `agenerate()` runs on a worker thread. With an `AsyncOpenAI` client, `generate()`, `metric.score()` and `embed_text()` run on one shared background event loop. That also works inside Jupyter or another running loop.

!!! warning "Don't mix both styles on one async client"
    An async client keeps its connections tied to the event loop that opened
    them. Calling it with `await` on your own loop *and* through the sync API
    (which uses ragas' background loop) makes the two loops share those
    connections, which can fail with "bound to a different event loop". In
    async code, use the async methods (`agenerate`, `ascore`, `aembed_text`)
    throughout.

## Token usage and cost

Every response comes back in OpenAI's shape, Claude and Gemini included, so use `get_token_usage_for_openai` as the usage parser for any `llm_factory()` LLM.

## Migrating from native clients

`llm_factory()` used to wrap each provider's own SDK. It now passes everything through LiteLLM (#182):

| Before | Now |
|---|---|
| `llm_factory("claude-sonnet-5-5", provider="anthropic", client=Anthropic())` | `llm_factory("anthropic/claude-sonnet-5-5")` |
| `llm_factory("gemini-3.6-flash", provider="google", client=genai.Client(api_key=...))` | `llm_factory("gemini/gemini-3.6-flash")` |
| a google-genai Vertex AI client | `llm_factory("vertex_ai/gemini-3.6-flash", vertex_project=..., vertex_location=...)` |
| `llm_factory(m, provider="mistral", client=Mistral(...))` | `llm_factory("mistral/<model>")` |
| `llm_factory(m, provider="groq", client=Groq(...))` | `llm_factory("groq/<model>")` |
| `llm_factory(m, provider="cohere", client=Cohere(...))` | `llm_factory("cohere_chat/<model>")` |
| a boto3 Bedrock client | `llm_factory("bedrock/<model id>", aws_region_name=...)` |
| `oci_genai_factory(model_id=..., compartment_id=...)` | `llm_factory("oci/<model id>", oci_compartment_id=...)` |
| `llm_factory(m, client=OpenAI(...))` | unchanged |
| `llm_factory(m, client=client, adapter="instructor" / "litellm")` | drop `adapter=`; it is ignored, with a warning |
| `mode=instructor.Mode.ANTHROPIC_JSON` / `ANTHROPIC_TOOLS` | `Mode.JSON` / `Mode.TOOLS` |

The key that used to go to the SDK client is read from the provider's environment variable, or pass `api_key=`. There are no provider- or framework-specific LLM wrappers either: use the LiteLLM model string for OCI Generative AI (`llm_factory("oci/<model id>")`, see [OCI Gen AI](integrations/oci_genai.md)), and for a model you reach through LlamaIndex or Haystack, pass its provider's model string rather than the framework's LLM object. See [Behavior changes](migrations/behavior_changes.md#one-path-for-every-llm-litellm-182).

## Troubleshooting

- **`llm_factory() no longer accepts ... clients`**: pass the model string from the error message instead of the client.
- **`AuthenticationError` / missing key**: set the provider's environment variable from [LiteLLM's provider list](https://docs.litellm.ai/docs/providers), or pass `api_key=`.
- **`UnsupportedParamsError: ... does not support temperature=...`**: you set a sampling value the model rejects. Remove it; ragas' own defaults are dropped automatically.
- **Structured output fails to parse**: try `mode=instructor.Mode.JSON` or `Mode.MD_JSON`.
