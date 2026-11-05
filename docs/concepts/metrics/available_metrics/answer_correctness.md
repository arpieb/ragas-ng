## Answer Correctness

The assessment of Answer Correctness involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the `ground truth` and the `answer`, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.

Answer correctness encompasses two critical aspects: semantic similarity between the generated answer and the ground truth, as well as factual similarity. These aspects are combined using a weighted scheme to formulate the answer correctness score. Users also have the option to employ a 'threshold' value to round the resulting score to binary, if desired.

!!! note "Embedding Requirement"
    AnswerCorrectness requires embeddings for semantic similarity calculation. When using `evaluate()` without explicitly providing embeddings, Ragas will automatically match the embedding provider to your LLM provider. For example, if you use Gemini as your LLM, Google embeddings will be used automatically (no OpenAI API key needed). You can also provide embeddings explicitly for full control.


!!! example
    **Ground truth**: Einstein was born in 1879 in Germany.

    **High answer correctness**: In 1879, Einstein was born in Germany.

    **Low answer correctness**: Einstein was born in Spain in 1879.


### Example

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import AnswerCorrectness

# Setup LLM and embeddings
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

# Create metric
scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

# Evaluate
result = await scorer.ascore(
    user_input="When was the first super bowl?",
    response="The first superbowl was held on Jan 15, 1967",
    reference="The first superbowl was held on January 15, 1967",
)
print(f"Answer Correctness Score: {result.value}")
```

Output:

```
Answer Correctness Score: 0.95
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        reference="The first superbowl was held on January 15, 1967",
    )
    ```

### Calculation

Answer correctness is a **weighted average** of two parts: a factuality score and the semantic similarity between the response and the reference.

**Factuality.** Both the response and the reference are broken into statements. A single LLM call then sorts them into:

- TP (True Positive): statements in the response that are supported by the reference.
- FP (False Positive): statements in the response that are not in the reference.
- FN (False Negative): statements in the reference that are missing from the response.

In the second example:

- TP: `[Einstein was born in 1879]`
- FP: `[Einstein was born in Spain]`
- FN: `[Einstein was born in Germany]`

The factuality score is the F-beta score of these counts (`beta=1.0` by default, which gives F1):

$$
\text{Precision} = \frac{|TP|}{|TP| + |FP|}, \qquad \text{Recall} = \frac{|TP|}{|TP| + |FN|}
$$

$$
F_\beta = \frac{(1 + \beta^2) \times \text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}
$$

Here that is `1 / (1 + 0.5 × (1 + 1)) = 0.5`.

!!! note "Not the same as FactualCorrectness"
    This factuality score is AnswerCorrectness's own calculation. It is **not** the [FactualCorrectness](./factual_correctness.md) metric. FactualCorrectness decomposes claims with configurable atomicity and coverage and checks each direction (response → reference, reference → response) with separate NLI calls. AnswerCorrectness classifies both statement lists in one call. The two can give different numbers for the same sample.

**Semantic similarity** between the response and the reference is computed as in [Semantic Similarity](./semantic_similarity.md).

**Final score.** The weighted average of the two, with `weights=[0.75, 0.25]` (factuality, similarity) by default:

$$
\text{Answer Correctness} = \frac{w_f \times \text{factuality} + w_s \times \text{similarity}}{w_f + w_s}
$$

Set `weights=[1.0, 0.0]` to use factuality alone (no embeddings needed), or `beta` to weight recall (`beta > 1`) or precision (`beta < 1`) in the factuality score.

## Legacy Metrics API

The following examples use the legacy metrics API pattern. For new projects, we recommend using the collections-based API shown above.

!!! warning "Deprecation Timeline"
    This API will be deprecated in version 0.4 and removed in version 1.0. Please migrate to the collections-based API shown above.

### Example with Dataset

```python
from datasets import Dataset
from ragas.metrics import answer_correctness
from ragas import evaluate

data_samples = {
    "question": ["When was the first super bowl?", "Who won the most super bowls?"],
    "answer": [
        "The first superbowl was held on Jan 15, 1967",
        "The most super bowls have been won by The New England Patriots",
    ],
    "ground_truth": [
        "The first superbowl was held on January 15, 1967",
        "The New England Patriots have won the Super Bowl a record six times",
    ],
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset, metrics=[answer_correctness])
score.to_pandas()
```
