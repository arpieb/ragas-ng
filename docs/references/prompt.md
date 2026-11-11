# Prompt API Reference

The prompt system in Ragas provides a flexible and type-safe way to define prompts for LLM-based metrics and other components. This page documents the core prompt classes and their usage.

## Overview

Ragas uses a modular prompt architecture based on the `BasePrompt` class. Prompts can be:

- **Input/Output Models**: Pydantic BaseModel classes that define the structure of prompt inputs and outputs
- **Prompt Classes**: Inherit from `BasePrompt` to define instructions, examples, and prompt generation logic
- **String Prompts**: Simple text-based prompts for backward compatibility

## Core Classes

::: ragas.prompt
    options:
        members:
            - BasePrompt
            - StringPrompt
            - InputModel
            - OutputModel
            - PydanticPrompt
            - BoolIO
            - StringIO

## Metrics Collections Prompts

Modern metrics in Ragas use specialized prompt classes. Each metric module contains:

- **Input Model**: Defines what data the prompt needs (e.g., `StatementGeneratorInput`)
- **Output Model**: Defines the expected LLM response structure (e.g., `StatementGeneratorOutput`)
- **Prompt Class**: Inherits from `BasePrompt` to generate the prompt string with examples and instructions

### Example: Faithfulness Metric Prompts

```python
from ragas.metrics.collections.faithfulness.util import (
    StatementGeneratorInput,
    StatementGeneratorPrompt,
)

# The prompt class combines input/output models with instructions and examples
prompt = StatementGeneratorPrompt()

# Create input data
input_data = StatementGeneratorInput(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
)

# Generate the prompt string for the LLM
prompt_string = prompt.to_string(input_data)

# The LLM's reply is parsed into StatementGeneratorOutput (a list of statements)
```

### Available Metric Prompts

See the individual metric documentation for details on their prompts:

- [Faithfulness](../concepts/metrics/available_metrics/faithfulness.md)
- [Context Recall](../concepts/metrics/available_metrics/context_recall.md)
- [Context Precision](../concepts/metrics/available_metrics/context_precision.md)
- [Answer Correctness](../concepts/metrics/available_metrics/answer_correctness.md)
- [Factual Correctness](../concepts/metrics/available_metrics/factual_correctness.md)
- [Noise Sensitivity](../concepts/metrics/available_metrics/noise_sensitivity.md)

## Customization

For detailed guidance on customizing prompts for metrics, see [Modifying prompts in metrics](../howtos/customizations/metrics/modifying-prompts-metrics.md).
