# Optimizers API Reference

Ragas provides optimizers to improve metric prompts through automated optimization. This page documents the available optimizer classes and their configuration.

## Overview

Optimizers use annotated datasets with ground truth scores to refine the prompts of `ragas.metrics.collections` metrics, improving accuracy through:

- **Instruction optimization**: Finding better prompt wording
- **Demonstration optimization**: Selecting effective few-shot examples
- **Search strategies**: Exploring the prompt space efficiently

A collections metric lists its prompts with `get_prompts()` (copies, keyed by attribute name) and takes replacements with `set_prompts(**prompts)`. `Optimizer.optimize()` returns optimized copies of those prompts and never changes the metric; `metric.train()` runs an optimizer and applies the result.

## Training a metric

```python
from ragas.config import DemonstrationConfig, InstructionConfig
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AspectCritic

llm = llm_factory("gpt-6-luna")
metric = AspectCritic(
    name="helpfulness", definition="Is the response helpful?", llm=llm
)

metric.train(
    "annotated_data.json",  # or dataset=SingleMetricAnnotation(...)
    instruction_config=InstructionConfig(llm=llm),  # GeneticOptimizer by default
    demonstration_config=DemonstrationConfig(
        embedding=embedding_factory("openai", model="text-embedding-3-small")
    ),
)
```

`train(path=None, demonstration_config=None, instruction_config=None, *, dataset=None, run_config=None, batch_size=None, raise_exceptions=True)` optimizes the instructions first, then adds demonstrations; a config that is `None` or has `enabled=False` is skipped. The same function is `ragas.optimizers.train_metric(metric, ...)`; the two steps on their own are `optimize_instructions()` and `optimize_demonstrations()`, which return prompts for `set_prompts` without applying them.

The default loss follows the metric's `output_type`: `BinaryMetricLoss` (accuracy) for a binary verdict such as AspectCritic's, `MSELoss` otherwise. Set `InstructionConfig(loss=...)` to choose another.

## Core Classes

::: ragas.optimizers
    options:
        members:
            - Optimizer
            - GeneticOptimizer
            - DSPyOptimizer
            - train_metric
            - optimize_instructions
            - optimize_demonstrations

## GeneticOptimizer

Simple evolutionary optimizer for prompt instructions. It takes no constructor parameters besides `metric` and `llm`, which `train()` sets; its options go in `InstructionConfig.optimizer_config`.

### Options

| Option | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | `int` | 3 | Candidate instructions, including the metric's own |
| `num_demonstrations` | `int` | 3 | Annotations per reverse-engineered instruction |
| `sample_size` | `int` | 12 | Annotations per feedback mutation |

### Usage

```python
from ragas.config import InstructionConfig
from ragas.optimizers import GeneticOptimizer

config = InstructionConfig(
    llm=llm,
    optimizer=GeneticOptimizer(),
    optimizer_config={"population_size": 4},
)
metric.train(dataset=dataset, instruction_config=config)
```

### How it Works

1. Reverse-engineers candidate instructions from batches of annotations, and adds the metric's own
2. Scores each candidate on a copy of the metric and asks the LLM for feedback on the samples it got wrong, then rewrites it
3. Crosses each candidate with the one whose mistakes differ least
4. Keeps the candidate with the best loss

It needs at least 10 annotations. **Pros**: Simple, works with limited data, scores the whole metric. **Cons**: Instruction-only.

## DSPyOptimizer

Advanced optimizer using DSPy's [MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/) algorithm.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto` | `str` | `"light"` | MIPROv2's search size: `"light"`, `"medium"`, `"heavy"` or `None` |
| `num_candidates` | `int` | 10 | Number of prompt variants to try, with `auto=None` |
| `max_bootstrapped_demos` | `int` | 5 | Max auto-generated examples |
| `max_labeled_demos` | `int` | 5 | Max human-annotated examples |
| `init_temperature` | `float` | 1.0 | Exploration temperature (0.0-2.0) |
| `max_errors` | `int` | `None` | Failed calls MIPROv2 tolerates; when `None`, `train(raise_exceptions=...)` decides (see below) |
| `lm` | DSPy LM | `None` | The LM DSPy uses, instead of one built from the LLM |

### Usage

```python
from ragas.config import InstructionConfig
from ragas.optimizers import DSPyOptimizer

optimizer = DSPyOptimizer(
    num_candidates=10,
    max_bootstrapped_demos=5,
    max_labeled_demos=5,
)

config = InstructionConfig(llm=llm, optimizer=optimizer)
metric.train(dataset=dataset, instruction_config=config)
```

### How it Works

1. Generates candidate prompt instructions
2. Bootstraps few-shot demonstrations from data
3. Selects best human-annotated examples
4. Evaluates all combinations on dataset
5. Returns, per annotated prompt, a copy with the best instruction and its demonstrations as examples

DSPy calls the model the `llm_factory()` LLM calls, through a `dspy.LM` over the same LiteLLM model name and credentials. Pass `lm=` for an LLM over a `litellm.Router` or Azure client.

`train()`'s options apply as follows:

- `run_config`: its `max_retries` and `timeout` become the `dspy.LM`'s `num_retries` and `timeout`, unless your `llm_factory()` arguments set them. A DSPy LM passed as `lm=` keeps its own settings.
- `raise_exceptions`: with `max_errors=None`, `True` (the default) stops MIPROv2 at its first failed call, and `False` never stops on errors. DSPy logs each error.
- `batch_size`: not supported. MIPROv2 sizes its own minibatches, and any value raises `NotImplementedError`.

Learn more about DSPy concepts:
- [Signatures](https://dspy.ai/learn/programming/signatures/) - DSPy's approach to defining input/output specifications
- [Optimizers](https://dspy.ai/learn/optimization/optimizers/) - Algorithms for improving prompts and LM weights
- [Modules](https://dspy.ai/learn/programming/modules/) - Building blocks for LLM programs

**Pros**: Better results, combines instructions + demos
**Cons**: Requires DSPy installation, more LLM calls

### Installation

[DSPy](https://dspy.ai/) is an optional dependency:

```bash
# Using uv (recommended)
uv add "ragas-ng[dspy]"

# Using pip
pip install "ragas-ng[dspy]"
```

### Cost Estimation

Approximate LLM calls per optimization:

```
Total calls ≈ num_candidates × 30 + max_bootstrapped_demos × 7
```

Examples:

- Default config (10, 5, 5): ~335 calls
- Budget config (5, 2, 3): ~164 calls
- Aggressive config (20, 10, 10): ~670 calls

## Optimizer Base Class

::: ragas.optimizers.base.Optimizer
    options:
        show_source: false
        members:
            - optimize

## Configuration

Both optimizers are used with `InstructionConfig`; few-shot demonstrations use `DemonstrationConfig`:

```python
from ragas.config import DemonstrationConfig, InstructionConfig

config = InstructionConfig(
    llm=llm,  # an llm_factory() LLM for the optimizer's own calls
    optimizer=optimizer_instance,  # Optimizer to use (GeneticOptimizer by default)
)
demonstrations = DemonstrationConfig(
    embedding=embeddings,  # an embedding_factory() model
    top_k=3,  # examples shown per input
    threshold=0.7,  # minimum cosine similarity
    technique="similarity",  # or "random": a fixed random sample
)

metric.train(
    dataset=dataset, instruction_config=config, demonstration_config=demonstrations
)
```

## Dataset Format

Optimizers require annotated datasets with ground truth scores. The keys of `prompts` are the metric's `get_prompts()` names:

```python
from ragas.dataset_schema import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)

# Create annotated sample
prompt_annotation = PromptAnnotation(
    prompt_input={"user_input": "...", "response": "..."},
    prompt_output={"reason": "...", "verdict": 1},
    edited_output=None,  # Optional: corrected output
)

sample = SampleAnnotation(
    metric_input={"user_input": "...", "response": "..."},
    metric_output=1,  # Ground truth score
    prompts={"prompt": prompt_annotation},  # AspectCritic's prompt is "prompt"
    is_accepted=True,  # Include in optimization
)

# Create dataset
dataset = SingleMetricAnnotation(
    name="helpfulness",
    samples=[sample, ...],  # 20-50+ samples recommended
)
```

## Loss Functions

Optimizers use loss functions to evaluate prompt quality:

```python
from ragas.losses import BinaryMetricLoss, MSELoss

# Mean Squared Error (default for continuous and discrete scores)
loss = MSELoss()

# Accuracy or F1 (default for binary verdicts)
loss = BinaryMetricLoss(metric="f1_score")

# Use with config
config = InstructionConfig(llm=llm, optimizer=optimizer, loss=loss)
```

## Comparison

| Feature | GeneticOptimizer | DSPyOptimizer |
|---------|------------------|---------------|
| Installation | Built-in | Requires `ragas-ng[dspy]` |
| Optimization Target | Instructions only | Instructions + Demos |
| Min Dataset Size | 10+ samples | 20+ samples |
| Typical LLM Calls | 100-500 | 200-700 |
| Accuracy Improvement | +5-8% | +8-12% |
| Best For | Quick optimization | Production metrics |

## See Also

- [DSPy Optimizer Guide](../howtos/customizations/optimizers/index.md) - Detailed usage
- Metric Customization - Creating metrics
- [Prompt API Reference](./prompt.md) - Understanding prompts

## Additional Resources

**DSPy Documentation:**
- [DSPy Official Documentation](https://dspy.ai/) - Complete guide to DSPy
- [MIPROv2 API Reference](https://dspy.ai/api/optimizers/MIPROv2/) - Detailed MIPROv2 documentation
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/) - Guide to all DSPy optimizers
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy) - Source code and examples

**Research Papers:**
- [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695) - MIPROv2 paper
