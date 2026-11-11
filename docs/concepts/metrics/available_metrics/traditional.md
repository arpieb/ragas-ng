# Traditional NLP Metrics

## Non LLM String Similarity

`NonLLMStringSimilarity` metric measures the similarity between the reference and the response using traditional string distance measures such as Levenshtein, Hamming, and Jaro. This metric is useful for evaluating the similarity of `response` to the `reference` text without relying on large language models (LLMs). The metric returns a score between 0 and 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

### Example

```python
from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure

# Create metric (no LLM/embeddings needed)
scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN)

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India.",
)
print(f"NonLLM String Similarity Score: {result.value}")
```

Output:

```
NonLLM String Similarity Score: 0.8918918918918919
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    ```

### Configuration

You can choose from available string distance measures from `DistanceMeasure`. Here is an example of using Hamming distance:

```python
scorer = NonLLMStringSimilarity(distance_measure=DistanceMeasure.HAMMING)
```

Available distance measures include:
- `DistanceMeasure.LEVENSHTEIN` (default)
- `DistanceMeasure.HAMMING`
- `DistanceMeasure.JARO`
- `DistanceMeasure.JARO_WINKLER`

## BLEU Score

The `BleuScore` metric is used to evaluate the quality of `response` by comparing it with `reference`. It measures the similarity between the response and the reference based on n-gram precision and brevity penalty. BLEU score was originally designed to evaluate machine translation systems, but it is also used in other natural language processing tasks. BLEU score ranges from 0 to 1, where 1 indicates a perfect match between the response and the reference. This is a non-LLM based metric.

### Example

```python
from ragas.metrics.collections import BleuScore

# Create metric
scorer = BleuScore()

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India.",
)
print(f"BLEU Score: {result.value}")
```

Output:

```
BLEU Score: 0.7071067811865478
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    ```

### Configuration

You can pass additional arguments to the underlying `sacrebleu.corpus_bleu` function using the `kwargs` parameter:

```python
scorer = BleuScore(kwargs={"smooth_method": "exp"})
```

Each sample is scored as one segment, the whole `response` against the whole `reference`. Unsmoothed BLEU is often `0` for short texts that share no 4-gram, so `smooth_method="exp"` is a good choice for short answers. Unless you pass `tokenize`, Chinese text is tokenized with sacreBLEU's `"zh"` tokenizer and Japanese with `"char"`. The default `"13a"` splits on spaces, so it would treat a whole sentence as one word. All other text uses `"13a"`.

## ROUGE Score

The `RougeScore` score is a set of metrics used to evaluate the quality of natural language generations. It measures the overlap between the generated `response` and the `reference` text based on n-gram recall, precision, and F1 score. ROUGE score ranges from 0 to 1, where 1 indicates a perfect match between the response and the reference. This is a non LLM based metric.

### Example

```python
from ragas.metrics.collections import RougeScore

# Create metric (no LLM/embeddings needed)
scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India.",
)
print(f"ROUGE Score: {result.value}")
```

Output:

```
ROUGE Score: 0.8571428571428571
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    ```

### Configuration

You can change the `rouge_type` to `rouge1` or `rougeL` to calculate the ROUGE score based on unigrams or longest common subsequence respectively.

```python
scorer = RougeScore(rouge_type="rouge1")
```

You can change the `mode` to `precision`, `recall`, or `fmeasure` to calculate the ROUGE score based on precision, recall, or F1 score respectively.

```python
scorer = RougeScore(mode="recall")
```

**Languages other than English.** By default (`tokenizer="auto"`), text that is entirely ASCII is tokenized exactly as `rouge_score` does it. Any other text uses a Unicode-aware tokenizer that keeps words in every script and treats each Chinese or Japanese character as a token. `rouge_score`'s own tokenizer drops every character outside `a-z0-9`, so Cyrillic, Greek, Arabic or CJK text would otherwise score `0`. Porter stemming (`use_stemmer=True`) only ever applies to English tokens. For a language that needs real word segmentation (e.g. Thai), pass any object with a `tokenize(text)` method:

```python
scorer = RougeScore(tokenizer=MyThaiSegmenter())
```

## Exact Match

The `ExactMatch` metric checks if the response is exactly the same as the reference text. It is useful in scenarios where you need to ensure that the generated response matches the expected output word-for-word. For example, arguments in tool calls, etc. The metric returns 1 if the response is an exact match with the reference, and 0 otherwise.

### Example

```python
from ragas.metrics.collections import ExactMatch

# Create metric (no LLM/embeddings needed)
scorer = ExactMatch()

# Evaluate
result = await scorer.ascore(reference="Paris", response="India")
print(f"Exact Match Score: {result.value}")
```

Output:

```
Exact Match Score: 0.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(reference="Paris", response="India")
    ```

## String Presence

The `StringPresence` metric checks if the response contains the reference text. It is useful in scenarios where you need to ensure that the generated response contains certain keywords or phrases. The metric returns 1 if the response contains the reference, and 0 otherwise.

### Example

```python
from ragas.metrics.collections import StringPresence

# Create metric (no LLM/embeddings needed)
scorer = StringPresence()

# Evaluate
result = await scorer.ascore(
    reference="Eiffel Tower", response="The Eiffel Tower is located in India."
)
print(f"String Presence Score: {result.value}")
```

Output:

```
String Presence Score: 1.0
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:
    
    ```python
    result = scorer.score(
        reference="Eiffel Tower", response="The Eiffel Tower is located in India."
    )
    ```

## CHRF Score

The `CHRFScore` metric evaluates the similarity between a `response` and a `reference` using **character n-gram F-score**. Unlike BLEU, which emphasizes precision, CHRF accounts for both **precision and recall**, making it more suitable for:

- Morphologically rich languages
- Responses with paraphrasing or flexible wording

CHRF scores range from 0 to 1, where 1 indicates a perfect match between the generated response and the reference. This is a non-LLM-based metric, relying entirely on deterministic comparisons.

### Example

```python
from ragas.metrics.collections import CHRFScore

# Create metric (no LLM/embeddings needed)
scorer = CHRFScore()

# Evaluate
result = await scorer.ascore(
    reference="The Eiffel Tower is located in Paris.",
    response="The Eiffel Tower is located in India.",
)
print(f"CHRF Score: {result.value}")
```

Output:

```
CHRF Score: 0.8048
```

!!! note "Synchronous Usage"
    If you prefer synchronous code, you can use the `.score()` method instead of `.ascore()`:

    ```python
    result = scorer.score(
        reference="The Eiffel Tower is located in Paris.",
        response="The Eiffel Tower is located in India.",
    )
    ```

### Configuration

You can pass additional arguments to the underlying `sacrebleu.corpus_chrf` function using the `kwargs` parameter:

```python
# Customize character and word order
scorer = CHRFScore(kwargs={"char_order": 4, "word_order": 2})

# Customize beta (recall weight)
scorer = CHRFScore(kwargs={"beta": 3})
```
