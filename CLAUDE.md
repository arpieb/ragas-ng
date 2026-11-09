# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ragas is an evaluation toolkit for Large Language Model (LLM) applications. It provides objective metrics for evaluating LLM applications, test data generation capabilities, and integrations with popular LLM frameworks.

This repository is **ragas-ng**, a hard fork of `vibrantlabsai/ragas`. The package and import name are still `ragas`. The fork removes vendor lock-in (no LangChain; `openai` is not a core dependency, and `llm_factory()` routes through LiteLLM) and fixes upstream's reported bugs. Changes that affect users are recorded in `docs/howtos/migrations/`: `behavior_changes.md` for API and behavior, and `score_changes.md` for metric scores.

The repository contains:

1. **Ragas Library** (`src/ragas/`) - metrics, test data generation, `evaluate()`, datasets, experiments and storage backends
2. **Examples** (`examples/`, the `ragas-examples` workspace package) - the `ragas quickstart` templates and standalone example scripts

## Development Environment Setup

### Installation

Choose the appropriate installation based on your needs:

```bash
# RECOMMENDED: Minimal dev setup (fast)
make install-minimal

# FULL: Complete dev environment (comprehensive ML stack)
make install

# OR manual installation (uv creates and manages .venv for you):
uv sync --locked --group dev-minimal   # minimal
uv sync --locked --group dev           # full
```

### Installation Methods Explained

Both paths are `uv sync` against the committed `uv.lock`, so every environment
(local, CI, Docker) resolves to byte-identical dependency versions.

- **Minimal setup**: `[dependency-groups].dev-minimal` — lint, type-check and the
  full unit test suite. This is what CI installs.
- **Full setup**: `[dependency-groups].dev` — includes `dev-minimal` plus the heavy
  ML/tracing stack and all runtime extras.
- `dev-minimal` is uv's **default group**, so a bare `uv sync` or `uv run <tool>`
  gets the fast environment without pulling the full ML stack.
- Dev tooling lives in dependency groups, *not* in `[project.optional-dependencies]`,
  so it is never published as an extra on PyPI.

### Changing dependencies

`uv.lock` is committed and CI verifies it with `uv lock --check`. After editing
dependencies in `pyproject.toml`:

```bash
make lock          # re-resolve and update uv.lock
make lock-check    # verify uv.lock matches pyproject.toml (what CI runs)
```

Commit the resulting `uv.lock` alongside the `pyproject.toml` change.

### Workspace Structure

The project uses a UV workspace configuration for managing multiple packages:

```bash
# Install
uv sync

# Install examples separately
uv sync --package ragas-examples

# Build specific workspace package
uv build --package ragas-examples
```

**Workspace Members:**
- `ragas` (main package) - Located in `src/ragas/`
- `ragas-examples` (examples package) - Located in `examples/`

The workspace ensures consistent dependency versions across packages and enables editable installs of workspace members.

## Common Commands

### Commands (from root directory)

```bash
# Setup and installation  
make install-minimal # Minimal dev setup (recommended; what CI installs)
make install        # Full dev environment (adds the ML/tracing stack and all extras)

# Code quality
make format         # Format and lint all code
make type           # Type check all code
make check          # Quick health check (format + type, no tests)

# Testing
make test           # Run all unit tests
make test-e2e       # Run end-to-end tests

# CI/Build
make run-ci         # Run complete CI pipeline
make clean          # Clean all generated files

# Documentation
make build-docs     # Build all documentation
make serve-docs     # Serve documentation locally

# Benchmarks
make benchmarks     # Run performance benchmarks
make benchmarks-docker # Run benchmarks in Docker
```

### Testing

```bash
# Run all tests (from root)
make test

# Run specific test (using pytest -k flag)
make test k="test_name"

# Run end-to-end tests
make test-e2e

# Direct pytest commands for more control
uv run pytest tests/unit -k "test_name"
uv run pytest tests/unit -v
```

### Documentation

```bash
# Build all documentation (from root)
make build-docs

# Serve documentation locally
make serve-docs
```

### Model retirements

Model IDs named in `src`, `examples` and `docs` are tracked in `scripts/model_registry.toml`, with shutdown dates and replacements. `tests/unit/test_retired_models.py` fails on a retired ID and warns 60 days ahead. `uv run python scripts/replace_model_ids.py --check|--apply [--within DAYS]` applies the replacements. See `scripts/README.md`.

### Benchmarks

```bash
# Run all benchmarks locally
make benchmarks

# Run benchmarks in Docker
make benchmarks-docker
```

## Project Architecture

The repository has the following structure:

```sh
/                          # Main ragas project
├── src/ragas/             # Source code
├── tests/                 # unit/, e2e/, docs/ and benchmarks/ tests
├── examples/              # Example code
├── pyproject.toml         # Build config
├── docs/                  # Documentation
├── scripts/               # Build/CI scripts
├── Makefile               # Build commands
└── README.md              # Repository overview
```

### Ragas Core Components

The Ragas core library provides metrics, test data generation and evaluation functionality for LLM applications:

1. **Metrics** - Various metrics for evaluating LLM applications including:

   - AspectCritic
   - AnswerCorrectness
   - ContextPrecision
   - ContextRecall
   - Faithfulness
   - and many more

2. **Test Data Generation** - Automatic creation of test datasets for LLM applications

3. **Integrations** (`src/ragas/integrations/`) - LlamaIndex, Griptape, Amazon Bedrock, R2R, Swarm, Helicone and AG-UI, plus MLflow tracing; Haystack and OCI Gen AI are LLM wrappers in `ragas.llms`. Spans are emitted with the OpenTelemetry API, so any OTel backend (e.g. Langfuse) works without vendor code. LangChain is banned: ruff rejects its imports and `tests/unit/test_langchain_boundary.py` checks it at runtime

### Datasets, Experiments and Backends

`Dataset`, the `@experiment` decorator and the storage backends are part of the main package (the former `ragas.experimental` namespace no longer exists):

```python
from ragas import Dataset, experiment
from ragas.backends import get_registry  # local/csv, local/jsonl, inmemory, gdrive
```

- `Dataset(name, backend="local/csv", root_dir="./data")`: a backend by name plus its constructor arguments, or a backend instance.
- `@experiment(Model, backend=..., backend_kwargs=..., max_workers=16, raise_exceptions=False, save_partial=False)`, then `await fn.arun(dataset, ...)`. `name`, `backend`, `backend_kwargs`, `max_workers`, `raise_exceptions` and `save_partial` belong to `arun`; other keyword arguments go to the experiment function.
- User guide: `docs/howtos/customizations/storage_backends.md`. Contributor notes: `src/ragas/backends/README.md`.

### LLMs

`llm_factory()` (`src/ragas/llms/base.py`) returns an `InstructorLLM` over LiteLLM for every model (#182). The client may be `None`, an `OpenAI` / `Azure` client (passed to LiteLLM as a copy), or a `litellm.Router`. Native SDK clients raise. Per-provider parameter rules live in `InstructorLLM._map_provider_params`: the OpenAI reasoning-model and Claude rules, plus `_map_litellm_defaults`. Tests use LiteLLM's `mock_response`, so no network is needed (`tests/unit/llms/test_llm_factory.py`).

## Debugging Logs

To view debug logs for any module:

```python
import logging

# Configure logging for a specific module (example with evaluation)
eval_logger = logging.getLogger('ragas.evaluation')
eval_logger.setLevel(logging.DEBUG)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
eval_logger.addHandler(console_handler)
```

## Memories

- whenever you create such docs put in in /\_experiments because that is gitignored and you can use it as a scratchpad or tmp directory for storing these
- always use uv to run python and python related commandline tools like isort, ruff, pyright etc. This is because we are using uv to manage the .venv and dependencies.
- The project uses two dependency groups, both synced from the committed `uv.lock`:
  - **Minimal setup**: `[dependency-groups].dev-minimal` for fast development (uv's default group)
  - **Full setup**: `[dependency-groups].dev` for comprehensive development (full ML stack)
- `uv.lock` is committed. Run `make lock` after changing dependencies and commit the result.
- Use `make install-minimal` for most development tasks, `make install` for full ML stack work
- if the user asks you to save a plan, save it into the plan/ directory with an appropriate file name.
