# API References

This section contains detailed API documentation for all core components of Ragas. The documentation is organized into the following sections:

## Core Components

- [Prompt](prompt.md) - Core prompt management and templating
- [LLMs](llms.md) - Language model interfaces and configurations
- [Embeddings](embeddings.md) - Embedding model interfaces and utilities
- [Tokenizers](tokenizers.md) - Tokenizer interfaces for text splitting
- [RunConfig](run_config.md) - Evaluation runtime configuration
- [RateLimit](rate_limit.md) - Requests and tokens per minute for LLM calls
- [Executor](executor.md) - Execution engine for concurrent jobs
- [Cache](cache.md) - Caching mechanisms for LLM calls

## Evaluation

- [Schemas](evaluation_schema.md) - Samples, messages, datasets and experiments
- [Metrics](metrics.md) - Available metrics and their implementations
- [metrics_experiment()](evaluate.md) - Score metrics over a dataset; `summarize()` and `check()`
- [Sync and async runs](aevaluate.md) - `run()` and `arun()`

## Testset Generation

- [Schemas](testset_schema.md) - Data structures for test data
- [Graph](graph.md) - Knowledge graph creation and management
- [Transforms](transforms.md) - Data transformation utilities
- [Synthesizers](synthesizers.md) - Test data generation components
- [Generation](generate.md) - Test data generation API

## Integrations

- [Integrations](integrations.md) - APIs for external tool integrations