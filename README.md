<h1 align="center">
  <img style="vertical-align:middle" height="200" src="docs/_static/imgs/logo.png">
</h1>
<p align="center">
  <i>Objective, provider-neutral evaluation for LLM applications</i>
</p>

<p align="center">
    <a href="https://github.com/arpieb/ragas-ng/actions/workflows/ci.yaml">
        <img alt="CI" src="https://github.com/arpieb/ragas-ng/actions/workflows/ci.yaml/badge.svg">
    </a>
    <a href="https://www.python.org/">
        <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-1f425f.svg?color=purple">
    </a>
    <a href="./LICENSE">
        <img alt="License Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-green.svg">
    </a>
</p>

## About this fork

**ragas-ng is a hard fork of [`vibrantlabsai/ragas`](https://github.com/vibrantlabsai/ragas)**, taken after the upstream repository went seven months without updates. It has two goals:

1. Remove lock-in to any single frontier-lab vendor or framework wherever possible.
2. Fix the functional issues reported against the original repository.

Changes so far:

- **No LangChain.** `import ragas` loads no LangChain modules, and nothing LangChain-related is installed as a dependency. The LangChain, LangSmith, LangGraph and Opik integrations are gone. See the [LangChain migration guide](docs/howtos/migrations/migrate_off_langchain.md).
- **Provider-neutral by default.** `openai` is no longer a declared dependency. `llm_factory()` and `embedding_factory()` work without a client and route through [LiteLLM](https://github.com/BerriAI/litellm), so any provider LiteLLM supports works out of the box.
- **Native tracing.** The LangChain callback system has been replaced by a native run tree plus the OpenTelemetry API. Spans cost nothing until you configure an OpenTelemetry SDK.
- **Collections metrics everywhere.** Metrics have been ported to `ragas.metrics.collections`, including the non-LLM context metrics, AspectCritic, SimpleCriteriaScore and FaithfulnesswithHHEM, and `evaluate()` accepts them directly. The legacy metric singletons in `ragas.metrics` still work, but they are deprecated.
- **Upstream bugs fixed, with migration notes.** Fixes that change behavior or scores are listed in [Behavior changes](docs/howtos/migrations/behavior_changes.md) and [Score changes](docs/howtos/migrations/score_changes.md), so you can see what differs from upstream before upgrading.
- **Modern toolchain.** Requires Python 3.11+. Uses uv with a committed lockfile, and the docs build in CI with `mkdocs --strict`.

The Python package and import name are still `ragas`, so existing code keeps importing the same way.

## :shield: Installation

This fork is not published to PyPI: `pip install ragas` installs the **upstream** package. Install from GitHub instead:

```bash
pip install git+https://github.com/arpieb/ragas-ng
# or
uv add git+https://github.com/arpieb/ragas-ng
```

Optional features ship as extras, for example `tracing`, `gdrive`, `oci`, `ag-ui`, `dspy`, or `all`:

```bash
pip install "ragas[all] @ git+https://github.com/arpieb/ragas-ng"
```

## :fire: Quickstart

### Evaluate your LLM app

`DiscreteMetric` lets you judge any aspect of an output with an LLM. This example needs no vendor SDK: `llm_factory` routes through LiteLLM and reads credentials from the environment.

```python
import asyncio

from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Any LiteLLM model string works: "gpt-4o-mini", "anthropic/claude-sonnet-5-5",
# "gemini/gemini-3.8-flash", "ollama/qwen3.8:27b", ...
llm = llm_factory("gpt-4o-mini")

metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate and captures key information.

Response: {response}

Answer with only 'accurate' or 'inaccurate'.""",
)


async def main():
    score = await metric.ascore(llm=llm, response="The summary of the text is...")
    print(f"Score: {score.value}")  # 'accurate' or 'inaccurate'
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())
```

> **Note**: Set the API key your provider needs, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. You can still pass an explicit client, such as `llm_factory("gpt-4o", client=AsyncOpenAI())`, if you prefer.

### Start from a template project

`ragas quickstart` copies a complete example project:

```bash
ragas quickstart                          # list available templates
ragas quickstart rag_eval -o ./my-project # create a project
```

Templates: `rag_eval`, `improve_rag`, `agent_evals`, `llamaIndex_agent_evals`, `text2sql`, `workflow_eval`, `prompt_evals`, `judge_alignment`, `benchmark_llm`. The template sources live in [`examples/ragas_examples`](examples/ragas_examples).

## Documentation

The documentation lives in [`docs/`](docs/) and is kept current with this fork. To build and browse it locally:

```bash
make serve-docs
```

Upstream's hosted docs at docs.ragas.io describe the original project and will not match this fork on LangChain, provider setup or the metrics APIs.

## Getting help

- **Questions**: ask in [GitHub Discussions](https://github.com/arpieb/ragas-ng/discussions/categories/q-a).
- **Bugs and feature requests**: open a [GitHub issue](https://github.com/arpieb/ragas-ng/issues).

## Development

```bash
git clone https://github.com/arpieb/ragas-ng.git
cd ragas-ng
make install-minimal   # lint, type-check and the full unit test suite (what CI installs)
make check             # format + type check
make test              # unit tests
```

Run `make install` for the full ML stack, which the e2e tests need. Some e2e and docs tests also call real providers and need an API key, such as `OPENAI_API_KEY`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request against `main`.

## 🔍 No usage analytics

ragas-ng collects no usage data. The upstream analytics module, which sent anonymized usage events to upstream's endpoint, has been removed, so ragas sends no telemetry and no longer writes a persistent user ID to disk. `RAGAS_DO_NOT_TRACK` is no longer needed.

## Acknowledgements and citation

ragas-ng is built on the work of the original Ragas authors and contributors at VibrantLabs, and is distributed under the same [Apache-2.0 license](./LICENSE). If you use Ragas in research, please cite the original project:

```
@misc{ragas2024,
  author       = {VibrantLabs},
  title        = {Ragas: Supercharge Your LLM Application Evaluations},
  year         = {2024},
  howpublished = {\url{https://github.com/vibrantlabsai/ragas}},
}
```
