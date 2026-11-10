# DSPy Optimizer for Advanced Prompt Optimization

The DSPyOptimizer provides state-of-the-art prompt optimization for Ragas metrics using DSPy's MIPROv2 algorithm. It combines instruction and demonstration optimization to find better prompts than simple evolutionary approaches.

## Overview

**DSPyOptimizer** uses MIPROv2 (Multi-prompt Instruction Proposal with Ranked Outcomes) to optimize metric prompts through:

- **Instruction optimization**: Generates and tests multiple prompt variations
- **Demonstration optimization**: Automatically selects effective few-shot examples
- **Combined search**: Explores both instruction and demonstration spaces simultaneously

This typically produces better results than the simpler GeneticOptimizer, especially when you have high-quality annotated data.

Optimizers tune the prompts of the metrics in `ragas.metrics.collections`. Each such metric lists its prompts with `get_prompts()` (a dict of copies, keyed by attribute name) and takes replacements with `set_prompts(**prompts)`. `metric.train(...)` runs an optimizer and applies its result.

## Installation

DSPy is an optional dependency. Install it with:

```bash
# Using uv (recommended)
uv add "ragas[dspy]"

# Using pip
pip install "ragas[dspy]"
```

MIPROv2 in DSPy 3 also needs `optuna`: `pip install "dspy[optuna]"`.

## Basic Usage

### Prerequisites

You need:

1. **Annotated dataset**: Ground truth scores for your metric, and the prompt inputs and outputs behind them
2. **A collections metric with prompts**: most LLM-based metrics in `ragas.metrics.collections`
3. **LLM**: An `llm_factory()` LLM for optimization (gpt-6-luna recommended for cost)

### Quick Start

```python
from openai import OpenAI
from ragas.config import InstructionConfig
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from ragas.optimizers import DSPyOptimizer

# Setup LLM for optimization
client = OpenAI()
llm = llm_factory("gpt-6-luna", client=client)

# Initialize metric
metric = Faithfulness(llm=llm)
print(list(metric.get_prompts()))
# ['statement_generator_prompt', 'nli_statement_prompt']

# Create annotated dataset (see below for format)
dataset = create_annotated_dataset()

# Configure DSPy optimizer
config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,  # Try 10 prompt variations (used when auto=None)
        max_bootstrapped_demos=5,  # Generate up to 5 examples
        max_labeled_demos=5,  # Use up to 5 human annotations
    ),
)

# Optimize the metric's prompts
metric.train(dataset=dataset, instruction_config=config)

# Save optimized prompts for reuse
for name, prompt in metric.get_prompts().items():
    prompt.save(f"optimized_faithfulness_{name}.json")
```

`metric.train()` accepts the annotations either as an object (`dataset=`) or as a JSON file (`metric.train("annotations.json", instruction_config=config)`). For each annotated prompt, the result is a copy of the metric's prompt with MIPROv2's instruction and its chosen demonstrations as examples.

DSPy calls the model your LLM calls: `DSPyOptimizer` builds a `dspy.LM` from the `llm_factory()` LLM's LiteLLM model name, the arguments you gave `llm_factory` (`api_key`, `api_base`, ...) and an `OpenAI` client's key and base URL. DSPy's own cache and retries apply to those calls. For an LLM over a `litellm.Router` or an Azure client, pass a DSPy LM yourself: `DSPyOptimizer(lm=dspy.LM("azure/my-deployment", api_key=..., api_base=...))`.

### Annotated Dataset Format

DSPy optimizer requires ground truth annotations. The keys of `prompts` are the metric's `get_prompts()` names; `prompt_input` and `prompt_output` are that prompt's input and output models as dicts:

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)

# Create prompt annotations
prompt_annotation = PromptAnnotation(
    prompt_input={"context": "...", "statements": ["..."]},
    prompt_output={"statements": [{"statement": "...", "reason": "...", "verdict": 1}]},
    edited_output=None,  # Or corrected output if needed
)

# Create sample with annotations
sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "...", "retrieved_contexts": ["..."]},
    metric_output=0.9,  # Ground truth score
    prompts={"nli_statement_prompt": prompt_annotation},
    is_accepted=True,  # Whether to use in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="faithfulness",
    samples=[sample, ...],  # Need 20-50+ samples for best results
)
```

DSPy scores each prediction against the annotation on one output field: a field named after the metric if the prompt has one, else the prompt's first numeric output field (such as AspectCritic's `verdict`). A prompt with neither, such as Faithfulness' `nli_statement_prompt` (a list of verdicts), gives MIPROv2 no signal; use `GeneticOptimizer`, which scores the whole metric, for those.

## Advanced Configuration

### Optimization Parameters

Control MIPROv2 behavior:

```python
optimizer = DSPyOptimizer(
    auto=None,  # set the search size yourself; "light"/"medium"/"heavy" size it for you
    num_candidates=20,  # More candidates = better prompts, higher cost
    max_bootstrapped_demos=10,  # Auto-generated few-shot examples
    max_labeled_demos=10,  # Human-annotated examples to use
    init_temperature=1.0,  # Exploration temperature (0.0-2.0)
)
```

**Parameter Guide:**

| Parameter | Default | Description | Cost Impact |
|-----------|---------|-------------|-------------|
| `auto` | `"light"` | MIPROv2's search size; overrides `num_candidates` | High |
| `num_candidates` | 10 | Prompt variations to try (with `auto=None`) | High - linear scaling |
| `max_bootstrapped_demos` | 5 | Auto-generated examples | Medium - adds LLM calls |
| `max_labeled_demos` | 5 | Human annotations to use | Low - uses existing data |
| `init_temperature` | 1.0 | Exploration randomness | None - algorithmic only |

With `auto=None`, MIPROv2 runs `num_trials` trials: `InstructionConfig(optimizer_config={"num_trials": 15})`, defaulting to `num_candidates`.

### Cost Optimization

MIPROv2 optimization can be expensive. Reduce costs by:

```python
# Budget-conscious configuration
budget_optimizer = DSPyOptimizer(
    auto=None,
    num_candidates=5,  # Fewer candidates
    max_bootstrapped_demos=2,  # Fewer generated examples
    max_labeled_demos=3,  # More reliance on annotations
    init_temperature=0.5,  # Less exploration
)

# Use cheaper LLM for optimization
cheap_llm = llm_factory("gpt-6-luna", client=client)
config = InstructionConfig(llm=cheap_llm, optimizer=budget_optimizer)
```

**Cost Estimation:**

- ~10-50 LLM calls per candidate
- ~5-10 calls per bootstrapped demo
- Total: `num_candidates * 30 + max_bootstrapped_demos * 7` calls (approximate)

## Comparing with GeneticOptimizer

### When to Use DSPyOptimizer

✅ **Use DSPyOptimizer when:**

- You have 50+ high-quality annotated examples
- You need the best possible metric accuracy
- You can afford 100-500 LLM calls for optimization
- You're optimizing critical production metrics

### When to Use GeneticOptimizer

✅ **Use GeneticOptimizer when:**

- You have limited annotated data (10-20 examples)
- You need faster, cheaper optimization
- You're doing initial prototyping
- Simple instruction-only optimization is sufficient

### Side-by-Side Comparison

```python
from ragas.config import InstructionConfig
from ragas.optimizers import DSPyOptimizer, GeneticOptimizer

# Genetic optimizer - simpler, faster, cheaper (the default optimizer)
genetic_config = InstructionConfig(
    llm=llm,
    optimizer=GeneticOptimizer(),
    optimizer_config={"population_size": 3, "sample_size": 12},
)

# DSPy optimizer - advanced, better results, more expensive
dspy_config = InstructionConfig(
    llm=llm,
    optimizer=DSPyOptimizer(
        num_candidates=10,
        max_bootstrapped_demos=5,
        max_labeled_demos=5,
    ),
)

# Compare results
metric_genetic = Faithfulness(llm=llm)
metric_genetic.train(dataset=dataset, instruction_config=genetic_config)

metric_dspy = Faithfulness(llm=llm)
metric_dspy.train(dataset=dataset, instruction_config=dspy_config)

# Evaluate on holdout set
test_scores_genetic = metric_genetic.batch_score(test_set)
test_scores_dspy = metric_dspy.batch_score(test_set)
```

**Typical Results:**

| Metric | GeneticOptimizer | DSPyOptimizer | Improvement |
|--------|------------------|---------------|-------------|
| Faithfulness | 0.82 | 0.89 | +8.5% |
| Answer Relevancy | 0.75 | 0.84 | +12% |
| Context Precision | 0.78 | 0.86 | +10% |

## Working with Multiple Metrics

Optimize several metrics with the same approach:

```python
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

metrics = {
    "faithfulness": Faithfulness(llm=llm),
    "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
    "context_precision": ContextPrecision(llm=llm),
}

# Optimize each metric
for name, metric in metrics.items():
    print(f"Optimizing {name}...")

    # Load metric-specific dataset
    dataset = load_annotated_dataset(name)

    # Optimize
    metric.train(dataset=dataset, instruction_config=dspy_config)

    # Save
    for prompt_name, prompt in metric.get_prompts().items():
        prompt.save(f"optimized_{name}_{prompt_name}.json")
```

## Troubleshooting

### Import Error

If you get `ImportError: DSPy optimizer requires dspy-ai`:

```bash
# Install the DSPy extra
uv add "ragas[dspy]"
# or
pip install "ragas[dspy]"
```

### Optimization Takes Too Long

Reduce the number of LLM calls:

```python
fast_optimizer = DSPyOptimizer(
    auto=None,
    num_candidates=3,  # Minimum viable
    max_bootstrapped_demos=1,
    max_labeled_demos=3,
)
```

### Poor Results

Common causes:

1. **Insufficient data**: Need 20+ high-quality annotations
2. **Low-quality annotations**: Ensure ground truth scores are accurate
3. **Wrong LLM**: Use a capable model, such as gpt-6.1-sol, for optimization
4. **Bad configuration**: Try default parameters first

### Memory Issues

MIPROv2 can use significant memory for large datasets:

```python
# Optimize on a subset first
subset = dataset.select(list(range(20)))
metric.train(dataset=subset, instruction_config=dspy_config)
```

## Best Practices

### Data Quality

1. **Diverse examples**: Cover edge cases and common scenarios
2. **Accurate labels**: Double-check ground truth scores
3. **Sufficient quantity**: 50+ examples for production metrics

### Optimization Strategy

1. **Start small**: Test with 3-5 candidates first
2. **Iterate**: Gradually increase parameters as needed
3. **Validate**: Always test on a holdout set
4. **Cache**: Save optimized prompts, or pass `DSPyOptimizer(cache=DiskCacheBackend())`, to avoid re-running

### Production Deployment

```python
# 1. Optimize offline
metric = Faithfulness(llm=optimization_llm)
metric.train(dataset=training_dataset, instruction_config=dspy_config)
for name, prompt in metric.get_prompts().items():
    prompt.save(f"production_faithfulness_{name}.json")

# 2. Load in production
production_metric = Faithfulness(llm=production_llm)
production_metric.set_prompts(
    **{
        name: type(prompt).load(f"production_faithfulness_{name}.json")
        for name, prompt in production_metric.get_prompts().items()
    }
)

# 3. Use for evaluation
results = production_metric.batch_score(production_samples)
```

## See Also

- [Optimizers API Reference](../../../references/optimizers.md) - Full API documentation
- Metric Customization - Creating custom metrics
- [DSPy Documentation](https://dspy.ai/) - Learn more about DSPy
