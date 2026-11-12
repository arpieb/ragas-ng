# OCI Gen AI Integration

This guide shows how to evaluate with Oracle Cloud Infrastructure (OCI) Generative AI models in Ragas.

Ragas reaches OCI through [LiteLLM's OCI provider](https://docs.litellm.ai/docs/providers/oci): pass an `oci/<model id>` model string to `llm_factory()`. There is no OCI-specific wrapper and no OCI SDK to install.

## Setup

### 1. Configure OCI credentials

LiteLLM signs each request with your OCI API key. Set these environment variables:

```bash
export OCI_USER=ocid1.user.oc1..example
export OCI_FINGERPRINT=your_fingerprint
export OCI_TENANCY=ocid1.tenancy.oc1..example
export OCI_KEY_FILE=~/.oci/private_key.pem   # or OCI_KEY with the PEM contents
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
export OCI_REGION=us-chicago-1               # defaults to us-ashburn-1
```

The values are the ones in your `~/.oci/config` profile. You can also pass any of them to `llm_factory()` as keyword arguments (`oci_user=`, `oci_fingerprint=`, `oci_tenancy=`, `oci_key_file=`, `oci_compartment_id=`, `oci_region=`), which LiteLLM forwards with each request.

### 2. Pick a model

Use the model ID from the OCI Generative AI console, prefixed with `oci/`, for example `oci/cohere.command-latest` or `oci/meta.llama-3.3-70b-instruct`. Check the [OCI documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) for the models available in your region.

## Create the evaluator LLM

```python
from ragas.llms import llm_factory

llm = llm_factory("oci/cohere.command-latest")

# Or pass the credentials explicitly instead of through the environment
llm = llm_factory(
    "oci/meta.llama-3.3-70b-instruct",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1..example",
)
```

Ragas asks the model for structured output through tool calling. If a model returns no tool call, use JSON mode instead:

```python
import instructor

llm = llm_factory("oci/cohere.command-latest", mode=instructor.Mode.JSON)
```

For embeddings (needed by metrics such as `AnswerRelevancy`), use an OCI embedding model through LiteLLM as well:

```python
from ragas.embeddings import LiteLLMEmbeddings

embeddings = LiteLLMEmbeddings(model="oci/cohere.embed-english-v3.0")
```

## Score a single sample

```python
from ragas.metrics.collections import Faithfulness

faithfulness = Faithfulness(llm=llm)

result = await faithfulness.ascore(
    user_input="What is the capital of France?",
    response="Paris is the capital of France.",
    retrieved_contexts=["France is a country in Europe. Its capital is Paris."],
)
print(result.value, result.reason)
```

## Evaluate a dataset

```python
from ragas import Dataset, metrics_experiment, summarize
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    Faithfulness,
)

rows = [
    {
        "user_input": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "retrieved_contexts": ["France is a country in Europe. Its capital is Paris."],
        "reference": "Paris",
    },
    {
        "user_input": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
        "retrieved_contexts": ["Romeo and Juliet is a play by William Shakespeare."],
        "reference": "William Shakespeare",
    },
]
dataset = Dataset(
    "oci_eval",
    "inmemory",
    data_model=SingleTurnSample,
    data=[SingleTurnSample(**row) for row in rows],
)

run = metrics_experiment(
    [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
    ]
)
exp = run.run(dataset, name="oci")
print(summarize(exp))
```

## Rate limits

OCI Generative AI enforces per-model rate limits. Pace the requests with `rate_limit=`:

```python
from ragas.rate_limit import RateLimit

llm = llm_factory("oci/cohere.command-latest", rate_limit=RateLimit(rpm=60))
```

## Troubleshooting

- **`Missing required parameters: oci_user, ...`**: one of the credential variables above is unset.
- **Model not found**: the model is not offered in `OCI_REGION`, or your compartment has no access to it.
- **Permission errors**: your user needs the IAM policies for Generative AI in the compartment.

Turn on LiteLLM's debug logging to see the request it sends:

```python
import litellm

litellm._turn_on_debug()
```

See also: [LLMs (`llm_factory`)](../llm-adapters.md) and the [OCI Generative AI docs](https://docs.oracle.com/en-us/iaas/Content/generative-ai/).
